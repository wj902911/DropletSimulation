#include <glad/glad.h> 
#include <GLFW\glfw3.h>

#include <iostream>
#include <format>
#include "shader.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <filesystem>
#include <cmath>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include <chrono>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Camera.h>
#include <dropletsSimCore.h>
#include <dropletsSimCore_GPU.h>
#include <mesh.h>
#include <openGLText.h>
#include <colorBar.h>

//#define DRAW_SQUARE

//#define USE_IMGUI

#define USE_GPU
#define TWO_DIM
//#define SHOW_SPRING
//#define DPhPC
//#define SHOW_DROPLET_CONCENTRATION_COLOR
//#define SHOW_FLOOR
//#define SHOW_WALL
//#define ONLY_DRAW_DROPLET

#ifdef USE_IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

std::mutex data_mutex;
std::atomic<bool> stopFlag(false);

struct SaveTask {
    std::vector<unsigned char> pixels;
    std::string filename;
    int width;
    int height;
};

std::queue<SaveTask> saveQueue;
std::mutex queueMutex;
std::condition_variable queueCondition;

std::atomic<bool> stopWorker(false);

std::string getCurrentTimeForFileName() {
    // Get the current time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // Format the time to a string for the filename
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

__global__ void renewDropletColor_kernel(const Droplet_GPU* droplets, 
                                         glm::vec3* dropletColor, 
                                         int numDroplets, 
                                         glm::vec3 fixedColor, 
                                         glm::vec3 unfixedColor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numDroplets)
	{
		dropletColor[idx] = droplets[idx].fixed ? fixedColor : unfixedColor;
	}

}

void renewDropletColor(const thrust::device_vector<Droplet_GPU>& droplets, 
                       thrust::device_vector < glm::vec3>& dropletColor, 
                       glm::vec3 fixedColor, 
                       glm::vec3 unfixedColor)
{
    int numDroplets = droplets.size();
	dropletColor.resize(numDroplets);
	int blockSize = 1024;
	int numBlocks = (numDroplets + blockSize - 1) / blockSize;
	renewDropletColor_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(droplets.data()),
															thrust::raw_pointer_cast(dropletColor.data()),
															numDroplets,
															fixedColor,
															unfixedColor);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	// Wait for the kernel to complete, then check for runtime errors
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}

__global__ void model_matrix_kernel(double3* d_dropletPos_d3,
	                                double* d_dropletRadius_d,
	                                glm::mat4* d_model,
									double initialDropletRadius,
									int num_droplets)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < num_droplets)
	{
		glm::vec3 pos_glm(d_dropletPos_d3[idx].x, d_dropletPos_d3[idx].y, d_dropletPos_d3[idx].z);
		d_model[idx] = glm::mat4(1.0f);
		d_model[idx] = glm::translate(d_model[idx], pos_glm);
		d_model[idx] = glm::scale(d_model[idx], glm::vec3(d_dropletRadius_d[idx] / initialDropletRadius));
	}
}

__global__ void spring_model_matrix_kernel(double3* d_springPos_d3,
                                            double* d_springRot_d,
                                            double* d_springLambda_d,
                                         glm::mat4* d_model,
                                                int numConnectors)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numConnectors)
    {
        glm::vec3 pos_glm(d_springPos_d3[idx].x, d_springPos_d3[idx].y, d_springPos_d3[idx].z);
        d_model[idx] = glm::mat4(1.0f);
        d_model[idx] = glm::translate(d_model[idx], pos_glm);
        d_model[idx] = glm::rotate(d_model[idx], static_cast<float>(d_springRot_d[idx]), glm::vec3(0.0f, 0.0f, 1.0f));
        d_model[idx] = glm::scale(d_model[idx], glm::vec3(d_springLambda_d[idx], 1.0f, 1.0f));
    }
}

std::vector<glm::mat4> getModelMatrices_GPU(thrust::device_vector<double3> d_dropletPos_d3,
	                                   thrust::device_vector<double> d_dropletRadius_d,
	                                   double initialDropletRadius)
{
	thrust::device_vector<glm::mat4> d_model(d_dropletPos_d3.size());
	int num_droplets = d_dropletPos_d3.size();
	int blockSize = 1024;
	int numBlocks = (num_droplets + blockSize - 1) / blockSize;
	model_matrix_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_dropletPos_d3.data()),
		                                               thrust::raw_pointer_cast(d_dropletRadius_d.data()),
		                                               thrust::raw_pointer_cast(d_model.data()),
		                                               initialDropletRadius,
		                                               num_droplets);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	// Wait for the kernel to complete, then check for runtime errors
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	std::vector<glm::mat4> h_model(num_droplets);
	thrust::copy(d_model.begin(), d_model.end(), h_model.begin());
	return h_model;
}

std::vector<glm::mat4> getSpringModelMatrices_GPU(thrust::device_vector<double3> d_springPos_d3,
    thrust::device_vector<double> d_springRot_d,
    thrust::device_vector<double> d_springLam_d)
{
    int num_springs = d_springPos_d3.size();
    thrust::device_vector<glm::mat4> d_model(num_springs);
    int blockSize = 1024;
    int numBlocks = (num_springs + blockSize - 1) / blockSize;
    spring_model_matrix_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_springPos_d3.data()),
		                                                      thrust::raw_pointer_cast(d_springRot_d.data()),
		                                                      thrust::raw_pointer_cast(d_springLam_d.data()),
		                                                      thrust::raw_pointer_cast(d_model.data()),
		                                                      num_springs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    // Wait for the kernel to complete, then check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
    }
    std::vector<glm::mat4> h_model(num_springs);
    thrust::copy(d_model.begin(), d_model.end(), h_model.begin());
    return h_model;
}

std::vector<glm::mat4> getModelMatrices(thrust::host_vector<double3>& h_dropletPos_d3,
	                                   thrust::host_vector<double>& h_dropletRadius_d,
	                                   double initialDropletRadius)
{
	std::vector<glm::mat4> model(h_dropletPos_d3.size());
	for (int i = 0; i < h_dropletPos_d3.size(); i++)
	{
		glm::vec3 pos_glm(h_dropletPos_d3[i].x, h_dropletPos_d3[i].y, h_dropletPos_d3[i].z);
		model[i] = glm::mat4(1.0f);
		model[i] = glm::translate(model[i], pos_glm);
		model[i] = glm::scale(model[i], glm::vec3(h_dropletRadius_d[i] / initialDropletRadius));
	}
	return model;
}

void frameSaverWorker() {
    while (!stopWorker) {
        SaveTask task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [] { return !saveQueue.empty() || stopWorker; });

            if (stopWorker && saveQueue.empty()) {
                return;
            }

            task = std::move(saveQueue.front());
            saveQueue.pop();
        }

        // Save the image using stb_image_write
        if (stbi_write_png(task.filename.c_str(), task.width, task.height, 3, task.pixels.data(), task.width * 3)) {
            std::cout << "Saved frame to " << task.filename << std::endl;
        }
        else {
            std::cerr << "Failed to save frame to " << task.filename << "!" << std::endl;
        }
    }
}

void saveFrameAsync(const std::string& directory, int frameNumber, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);  // 3 channels (RGB)

    // Read the pixels from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip the image vertically (optional)
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width * 3; ++x) {
            std::swap(pixels[y * width * 3 + x], pixels[(height - 1 - y) * width * 3 + x]);
        }
    }

    // Create the filename
    std::string filename = directory + "/frame_" + std::to_string(frameNumber) + ".png";

    // Push the save task to the queue
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        saveQueue.push(SaveTask{ std::move(pixels), std::move(filename), width, height });
    }
    queueCondition.notify_one(); // Notify the worker thread
}

std::vector<glm::vec3> convertToGLMVec3(const Eigen::MatrixX2d& mat)
{
	std::vector<glm::vec3> vec;
	vec.reserve(mat.rows());
	for (int i = 0; i < mat.rows(); i++)
	{
		vec.push_back(glm::vec3(mat(i, 0), mat(i, 1), 0.0f));
	}
	return vec;
}

__global__ void convertToGLMVec3_kernel(double3* d_dropletPos_d3,
	                                  glm::vec3* d_dropletPos,
                                             int num_droplets)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < num_droplets)
	{
		d_dropletPos[idx] = glm::vec3(d_dropletPos_d3[idx].x, d_dropletPos_d3[idx].y, d_dropletPos_d3[idx].z);
	}
}

void convertToGLMVec3_GPU(thrust::device_vector<double3>& d_dropletPos_d3,
	                              std::vector<glm::vec3>& h_dropletPos)
{
    thrust::device_vector<glm::vec3> d_dropletPos(d_dropletPos_d3.size());
	int num_droplets = d_dropletPos_d3.size();
	int blockSize = 1024;
	int numBlocks = (num_droplets + blockSize - 1) / blockSize;
	convertToGLMVec3_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_dropletPos_d3.data()),
		                                                   thrust::raw_pointer_cast(d_dropletPos.data()),
		                                                   num_droplets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    // Wait for the kernel to complete, then check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
    }
    thrust::copy(d_dropletPos.begin(), d_dropletPos.end(), h_dropletPos.begin());
}

void saveFrame(const std::string& directory, int frameNumber, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);  // 3 channels (RGB)

    // Read the pixels from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip the image vertically (optional)
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width * 3; ++x) {
            std::swap(pixels[y * width * 3 + x], pixels[(height - 1 - y) * width * 3 + x]);
        }
    }

    // Create the filename
    std::string filename = directory + "/frame_" + std::to_string(frameNumber) + ".png";

    // Save the image using stb_image_write
    if (stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3)) {
        std::cout << "Saved frame " << frameNumber << " to " << filename << std::endl;
    }
    else {
        std::cerr << "Failed to save frame " << frameNumber << "!" << std::endl;
    }
}


__global__ void getDropletColor_kernel(double* d_dropletC_d,
	                                glm::vec3* d_dropletColor,
	                                    double minConcentration,
	                                    double maxConcentration,
	                                       int num_droplets)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < num_droplets)
	{
		d_dropletColor[idx] = mapToColor(d_dropletC_d[idx], minConcentration, maxConcentration);
	}
}

__global__ void getSpringColor_kernel(double* d_springForce_d, 
                                   glm::vec3* d_springColor,
                                       double minForce,
                                       double maxForce,
                                          int numSpring)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numSpring)
	{
        d_springColor[idx] = mapToColor(d_springForce_d[idx], minForce, maxForce);
	}
}

void getDropletColor(thrust::device_vector<double>& d_dropletC_d,
	              thrust::device_vector<glm::vec3>& d_dropletColor,
	                                         double minConcentration,
	                                         double maxConcentration)
{
	int num_droplets = d_dropletC_d.size();
    d_dropletColor.resize(num_droplets);
	int blockSize = 1024;
	int numBlocks = (num_droplets + blockSize - 1) / blockSize;
	getDropletColor_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_dropletC_d.data()),
		                                                  thrust::raw_pointer_cast(d_dropletColor.data()),
		                                                  minConcentration,
		                                                  maxConcentration,
		                                                  num_droplets);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	// Wait for the kernel to complete, then check for runtime errors
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}

void getSpringColor(thrust::device_vector<double>& d_springForce_d,
	             thrust::device_vector<glm::vec3>& d_springColor,
	                                        double minForce,
	                                        double maxForce)
{
	int numSpring = d_springForce_d.size();
	d_springColor.resize(numSpring);
	int blockSize = 1024;
	int numBlocks = (numSpring + blockSize - 1) / blockSize;
	getSpringColor_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_springForce_d.data()),
		                                                 thrust::raw_pointer_cast(d_springColor.data()),
		                                                 minForce,
		                                                 maxForce,
		                                                 numSpring);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	// Wait for the kernel to complete, then check for runtime errors
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

#ifndef TWO_DIM
// camera
glm::vec3 cameraPos = glm::vec3(0.0f, 3.0f, 10.0f);
Camera camera(cameraPos);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
#endif // !TWO_DIM



// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

bool startPhysics = false;
bool rightButtonPressed = false;
bool saveSingleFrame = false;
fs::path directory_singleFrame = "output_frames_single";
int singleFrameNumber = 0;

int main()
{
    fs::path directory = "output_frames";
    fs::path data_directory = "output_data";

    // Create the directory if it doesn't exist using filesystem
    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }
    if (!fs::exists(directory_singleFrame)) {
        fs::create_directories(directory_singleFrame);
    }
    if (!fs::exists(data_directory)) {
		fs::create_directories(data_directory);
	}

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Droplets Simulation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
#ifndef TWO_DIM
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
#endif
    glfwSetKeyCallback(window, key_callback);

#ifndef TWO_DIM
    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
#endif

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
#ifndef TWO_DIM
    glEnable(GL_DEPTH_TEST);
#endif
    // Set OpenGL options
    //glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // build and compile our shader program
    // ------------------------------------


#ifndef USE_GPU
    DropletsSimCore dropletsSimCore;
    dropletsSimCore.initSimulation();
    Eigen::MatrixX2d dropletPos_Eigen;
    Eigen::VectorXd dropletRadius_Eigen;
    dropletsSimCore.getCurrentDropletPositionAndRadius(dropletPos_Eigen, dropletRadius_Eigen);
    int numDroplets = dropletPos_Eigen.rows();
    std::vector<glm::vec3> dropletPos = convertToGLMVec3(dropletPos_Eigen);
#else
    DropletsSimCore_GPU dropletsSimCore;
    dropletsSimCore.initSimulation();

    //double2 center = dropletsSimCore.getGlobalCenter();
    double2 center = dropletsSimCore.params_.boundingBoxCenter;
    //double2 size = dropletsSimCore.getGlobalSize();
    double2 size = make_double2(dropletsSimCore.params_.boundingBoxWidth, dropletsSimCore.params_.boundingBoxHeight);
    //size = make_double2(size.x * dropletsSimCore.params_.cameraDistanceRatio, size.y * dropletsSimCore.params_.cameraDistanceRatio);
#ifndef TWO_DIM
    glm::vec3 sizeGLM(size.x, size.y, 0.0f);
    float boundingRadius = glm::length(sizeGLM) * 0.5f;
    float fov = glm::radians(camera.m_Zoom);
    float distance = dropletsSimCore.params_.cameraDistanceRatio * boundingRadius / std::tan(fov / 2.0f);
    cameraPos = glm::vec3(center.x, center.y, distance);
    camera.setPos(cameraPos);
    camera.setSpeed(distance * 0.2);
#else
    float margin = 0.4* size.x * dropletsSimCore.params_.cameraDistanceRatio;
    float windowLeft = center.x - size.x / 2.0 - margin;
    float windowRight = center.x + size.x / 2.0 + margin;
    float windowAspectRatio = static_cast<float>(SCR_WIDTH) / static_cast<float>(SCR_HEIGHT);
    float windowHeight = (size.x + margin * 2.0) / windowAspectRatio;
    float windowTop = center.y + windowHeight * 0.5;
    float windowBottom = center.y - windowHeight * 0.5;
#endif

    thrust::device_vector<double3> d_dropletPos_d3;
    thrust::device_vector<double> d_dropletRadius_d;
    thrust::device_vector<double> d_dropletC_d;
    thrust::device_vector<glm::vec3> d_dropletColor;
#ifndef SHOW_DROPLET_CONCENTRATION_COLOR
    renewDropletColor(dropletsSimCore.droplets_, d_dropletColor, glm::vec3(0.3, 0.3, 0.3), glm::vec3(0.7, 0.7, 0.7));
    std::vector<glm::vec3> dropletColor(dropletsSimCore.num_droplets_);
    thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), dropletColor.begin());
#endif
    //dropletColor[19] = glm::vec3(0.3f, 0.3f, 0.3f);

    //for(int i = 0; i < dropletsSimCore.params_.numDroplets_y; i++)
    //    for (int j = 0; j < dropletsSimCore.params_.numDroplets_x; j++)
    //        if(j == 0 || j == dropletsSimCore.params_.numDroplets_x - 1)
    //           dropletColor[dropletsSimCore.getIndexofDroplet(i * dropletsSimCore.params_.numDroplets_x + j)] = glm::vec3(0.3f, 0.3f, 0.3f);
    //thrust::device_vector<glm::mat4> d_model;
    std::vector<glm::mat4> model;
    //dropletsSimCore.getCurrentDropletPositionRadiusConcentration(d_dropletPos_d3, d_dropletRadius_d, d_dropletC_d);
    int everageNumConnectionsPerDroplet = dropletsSimCore.getAverageNumberOfConnections();
    dropletsSimCore.getAllDropletPositionAsD3(d_dropletPos_d3);
    dropletsSimCore.getAllDropletRadius(d_dropletRadius_d);
    model = getModelMatrices_GPU(d_dropletPos_d3, d_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);
    //thrust::host_vector<double3> h_dropletPos_d3(d_dropletPos_d3.size());
    //thrust::copy(d_dropletPos_d3.begin(), d_dropletPos_d3.end(), h_dropletPos_d3.begin());
    //thrust::host_vector<double> h_dropletRadius_d(d_dropletRadius_d.size());
    //thrust::copy(d_dropletRadius_d.begin(), d_dropletRadius_d.end(), h_dropletRadius_d.begin());
    //model= getModelMatrices(h_dropletPos_d3, h_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
    thrust::pair<double, double> concentrationRange = dropletsSimCore.getConcentration(d_dropletC_d);
    double maxConcentration = concentrationRange.second, initialMaxConcentration = concentrationRange.second;
    double minConcentration = concentrationRange.first, initialMinConcentration = concentrationRange.first;
#endif
    //thrust::copy(d_model.begin(), d_model.end(), model.begin());
    //std::vector<glm::vec3> dropletPos(numDroplets);
    //convertToGLMVec3_GPU(d_dropletPos_d3, dropletPos);
    //std::vector<double> h_dropletRadius_d(d_dropletRadius_d.size());
    //thrust::copy(d_dropletRadius_d.begin(), d_dropletRadius_d.end(), h_dropletRadius_d.begin());
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
    std::vector<glm::vec3> dropletColor(d_dropletRadius_d.size(), glm::vec3(0.75f, 0.34f, 0.0f));
    getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
    thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), dropletColor.begin());
#endif



    //double maxVelocity = 0.0;
    //int maxVelocityIndex = 0;
    //double totalKineticEnergy = 0.0;
    //double endDistance = 0.0;
#endif
    

#ifndef TWO_DIM
    DropletMesh_3D droplet_mesh(dropletsSimCore.params_.initialDropletRadius, 10, 20, dropletColor, model);
#else
    DropletMesh_2D droplet_mesh(dropletsSimCore.params_.initialDropletRadius, 100, dropletColor, model);

#ifdef SHOW_SPRING
    int numConnectors = dropletsSimCore.getNumConnectors();
    float springWidth = dropletsSimCore.params_.initialDropletRadius * 0.1f;
    float springLength = dropletsSimCore.params_.SpringRestLengthRatio * dropletsSimCore.params_.initialDropletRadius * 2.0f;
    thrust::device_vector<double3> d_springPos_d3;
    thrust::device_vector<double> d_springRot_d;
    thrust::device_vector<double> d_springLam_d;
    thrust::device_vector<double> d_springf_d;
    thrust::pair < double, double > springForceRange = thrust::make_pair(0.0, 0.0);
    thrust::device_vector<glm::vec3> d_springColor(numConnectors);
    std::vector<glm::vec3> springColor(1, glm::vec3(0.0f, 0.0f, 1.0f));
    std::vector<glm::mat4> springModel = { glm::mat4(1.0f) };
    if (numConnectors > 0)
    {
        springForceRange = dropletsSimCore.getAllSpringConfiguration(d_springPos_d3, d_springRot_d, d_springLam_d, d_springf_d);
        getSpringColor(d_springf_d, d_springColor, springForceRange.first, springForceRange.second);
        springColor.resize(numConnectors);
        thrust::copy(d_springColor.begin(), d_springColor.end(), springColor.begin());
        springModel = getSpringModelMatrices_GPU(d_springPos_d3, d_springRot_d, d_springLam_d);
    }
    SpringMesh_2D spring_mesh(springWidth, springLength, springColor, springModel);
#endif // SHOW_SPRING

#ifdef SHOW_FLOOR
    glm::mat4 floorModel = glm::mat4(1.0f);
    double floorWidth = size.x * 1.2;
    double floorThickness = dropletsSimCore.params_.mean_radius * 0.2;
    floorModel = glm::translate(floorModel, glm::vec3(center.x - floorWidth * 0.5, -floorThickness, 0.0f));
    std::vector<glm::mat4> floorModels = { floorModel };
    FloorMesh_2D floor_mesh(floorThickness, floorWidth, floorModels);
#endif // SHOW_FLOOR

#ifdef SHOW_WALL
    glm::mat4 wallModel_left = glm::mat4(1.0f);
    glm::mat4 wallModel_right = glm::mat4(1.0f);
	double wallHeight = size.y;
	double wallThickness = dropletsSimCore.params_.mean_radius * 0.2;
    wallModel_left = glm::translate(wallModel_left, glm::vec3(dropletsSimCore.params_.leftWallX - wallThickness, 0.0, 0.0f));
    wallModel_right = glm::translate(wallModel_right, glm::vec3(dropletsSimCore.params_.rightWallX + wallThickness, 0.0, 0.0f));
	std::vector<glm::mat4> wallModels = { wallModel_left, wallModel_right };
    FloorMesh_2D wall_mesh(wallHeight, wallThickness, wallModels);
#endif // SHOW_WALL

#endif
    OpenGLText textRenderer(static_cast<GLfloat>(SCR_WIDTH), static_cast<GLfloat>(SCR_HEIGHT));

    //glm::vec2 colorBarAnchorPos(static_cast<GLfloat>(SCR_WIDTH) - 10.0f, 10.0f);
    float colorBarWidth = 20.0f;
    float colorBarHeight = 200.0f;
    //glm::vec3 colorBarMinColor = mapToColor(initialMinConcentration, initialMinConcentration, initialMaxConcentration);
    //glm::vec3 colorBarMaxColor = mapToColor(initialMaxConcentration, initialMinConcentration, initialMaxConcentration);
    //glm::vec3 colorBarMiddleColor = mapToColor((initialMaxConcentration + initialMaxConcentration) * 0.5, initialMinConcentration, initialMaxConcentration);

#ifdef SHOW_SPRING
    colorBar ColorBar(colorBarWidth,
                      colorBarHeight,
                      static_cast<GLfloat>(SCR_WIDTH),
                      static_cast<GLfloat>(SCR_HEIGHT),
                      "Spring force (N)",
                      springForceRange.first,
                      springForceRange.second);
#endif
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
    float anchorX = static_cast<float>(SCR_WIDTH) - 10.0f;
    float anchorY = 500.0f;
    colorBar ColorBarConcentration(anchorX,
                                   anchorY, 
                                   colorBarWidth,
								   colorBarHeight,
								   static_cast<GLfloat>(SCR_WIDTH),
								   static_cast<GLfloat>(SCR_HEIGHT),
								   "Concentration",
								   minConcentration,
								   maxConcentration);
#endif

    //enableDebugging();

    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

#ifdef USE_IMGUI
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Adjust scale for high DPI screens
    io.FontGlobalScale = 2.0f; // Adjust this scale factor as needed

    ImGui::StyleColorsDark();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
#endif

    double saveInterval = 10.0;
    int frameToSave = 0;
    //int frame = 0;
    double lastSaveTime = 0.0;
    double stepTime = 0.0;
    double displacement = 0.0;
    double force_to_save = 0.0;
    bool initialFrameSaved = false;

    /*std::string timestamp = getCurrentTimeForFileName();*/
    //std::string numDroplet_str = std::to_string(dropletsSimCore.params_.numDroplets_x) + "x" + std::to_string(dropletsSimCore.params_.numDroplets_y) + "_";
    std::string diameterEnlargeRatio_str = std::to_string(dropletsSimCore.params_.diameterEnlargeRatio);
    std::string settlingTime_str = std::to_string(dropletsSimCore.params_.settlingTime);
    std::string loadingTime_str = std::to_string(dropletsSimCore.params_.increaseTime);
    std::string maxTensileStrain_str = std::to_string(dropletsSimCore.params_.maxTensileStrain);
    std::string dropletType;
    switch (dropletsSimCore.params_.dropletType)
    {
    case DropletType::DPhPC:
    {
        dropletType = "DPhPC";
        break;
    }
    case DropletType::PB_PEO:
    {
        dropletType = "PB_PEO";
        break;
    }
    }
    std::string displacementFileName = dropletType + "_displacement_er"+ diameterEnlargeRatio_str +"_st" + settlingTime_str + "_lt" + loadingTime_str + "_ms" + maxTensileStrain_str + ".txt";
    std::ofstream displacementFile(data_directory / displacementFileName);
    displacementFile << displacement << std::endl;
    displacementFile.close();
    std::string forceFileName = dropletType + "_force_er" + diameterEnlargeRatio_str + "_st" + settlingTime_str + "_lt" + loadingTime_str + "_ms" + maxTensileStrain_str + ".txt";
    std::ofstream forceFile(data_directory / forceFileName);
    forceFile << force_to_save << std::endl;
    forceFile.close();
    
    //thrust::pair<double, double> raddiusChangeSpeedRange;
    //double maxRadiusIncreaseSpeed = 0.0;
    //double maxRadiusDecreaseSpeed = 0.0;

#ifndef USE_GPU
    std::thread physicsThread(physicsLoop,
                              std::ref(dropletsSimCore), 
                              std::ref(dropletPos_Eigen), 
                              std::ref(dropletRadius_Eigen), 
                              std::ref(stepTime));
#else
#if 1
    std::thread physicsThread(&DropletsSimCore_GPU::physicsLoop,
							  std::ref(dropletsSimCore),
							  std::ref(startPhysics),
                              std::ref(stopFlag),
                              std::ref(data_mutex));
#else
    std::thread physicsThread(physicsLoop,
							  std::ref(dropletsSimCore),
							  std::ref(d_dropletPos_d3),
                              std::ref(d_dropletRadius_d),
                              std::ref(d_dropletC_d),
                              std::ref(concentrationRange),
                              std::ref(raddiusChangeSpeedRange),
                              std::ref(maxVelocity),
                              std::ref(maxVelocityIndex),
							  std::ref(stepTime),
                              std::ref(totalKineticEnergy),
                              std::ref(endDistance));
#endif
#endif

    std::thread saverThread(frameSaverWorker);

    double numSprings_old = dropletsSimCore.getNumConnectors();
    double numDroplets_old = dropletsSimCore.num_droplets_;
    //int frame = 0;
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

#ifdef USE_IMGUI
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()!)
        ImGui::ShowDemoWindow();

        // render
        // ------
        ImGui::Render();
#endif

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

#ifndef USE_GPU
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            dropletPos = convertToGLMVec3(dropletPos_Eigen);
            stepTime_s << std::fixed << stepTime;
            for (int i = 0; i < numDroplets; i++)
            {
                model = glm::mat4(1.0f);
                model = glm::translate(model, dropletPos[i]);
                model = glm::scale(model, glm::vec3(dropletRadius_Eigen[i] / dropletsSimCore.getSimParameters().initialDropletRadius));
                lightingShader.setMat4("model", model);
                glDrawElements(GL_TRIANGLES, indicesVec.size(), GL_UNSIGNED_INT, 0);
            }
        }//end of lock_guard
#else
        //std::vector<double> h_dropletRadius_d(d_dropletRadius_d.size());
        if (startPhysics)
        {
            //cout << frame++ << endl;

            {
                std::lock_guard<std::mutex> lock(data_mutex);

                //convertToGLMVec3_GPU(d_dropletPos_d3, dropletPos);
                //thrust::copy(d_dropletRadius_d.begin(), d_dropletRadius_d.end(), h_dropletRadius_d.begin());
                //thrust::copy(d_model.begin(), d_model.end(), model.begin());
                //thrust::copy(d_dropletPos_d3.begin(), d_dropletPos_d3.end(), h_dropletPos_d3.begin());
                //thrust::copy(d_dropletRadius_d.begin(), d_dropletRadius_d.end(), h_dropletRadius_d.begin());
                stepTime = dropletsSimCore.getTime();
                displacement = dropletsSimCore.getDisplacement();
                //std::cout << 1 << std::endl;
                dropletsSimCore.getAllDropletPositionAsD3(d_dropletPos_d3);
                dropletsSimCore.getAllDropletRadius(d_dropletRadius_d);
                everageNumConnectionsPerDroplet = dropletsSimCore.getAverageNumberOfConnections();
                //std::cout << 2 << std::endl;
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
                concentrationRange = dropletsSimCore.getConcentration(d_dropletC_d);
#endif
#ifdef SHOW_SPRING
                numConnectors = dropletsSimCore.getNumConnectors();
               // std::cout << 3 << std::endl;
                if (numConnectors > 0)
                {
                    springForceRange = dropletsSimCore.getAllSpringConfiguration(d_springPos_d3, d_springRot_d, d_springLam_d, d_springf_d);
                }
               // std::cout << 4 << std::endl;
#endif
                if (stepTime - lastSaveTime >= dropletsSimCore.params_.saveFrameInterval && dropletsSimCore.params_.startApplyDisplacement)
                {
                    //int index = dropletsSimCore.getIndexofDroplet(dropletsSimCore.num_droplets_ - 1);
                    double2 force = dropletsSimCore.getTotalForce();
                    if(!dropletsSimCore.params_.isShear)
                        force_to_save = force.x;
                    else
                        force_to_save = force.y;
                }
                //std::cout << 5 << std::endl;
#ifndef SHOW_DROPLET_CONCENTRATION_COLOR
                renewDropletColor(dropletsSimCore.droplets_, d_dropletColor, glm::vec3(0.3, 0.3, 0.3), glm::vec3(0.7, 0.7, 0.7));
                //std::cout << 6 << std::endl;
#endif
                //maxRadiusIncreaseSpeed = raddiusChangeSpeedRange.second;
                //maxRadiusDecreaseSpeed = raddiusChangeSpeedRange.first;
            }//end of lock_guard
            droplet_mesh.modelMatrices = getModelMatrices_GPU(d_dropletPos_d3, d_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);
            //std::cout << 7 << std::endl;
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
            minConcentration = concentrationRange.first;
            maxConcentration = concentrationRange.second;
            getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
#endif
            thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), droplet_mesh.colors.begin());
            //std::cout << 8 << std::endl;
#ifdef SHOW_SPRING
            if (numConnectors > 0)
            {
                spring_mesh.modelMatrices = getSpringModelMatrices_GPU(d_springPos_d3, d_springRot_d, d_springLam_d);
                ColorBar.setMinMaxValues(springForceRange.first, springForceRange.second);
                getSpringColor(d_springf_d, d_springColor, springForceRange.first, springForceRange.second);
                //model = getModelMatrices(h_dropletPos_d3, h_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);
                //dropletColor = std::vector<glm::vec3>(d_dropletRadius_d.size(), glm::vec3(0.75f, 0.34f, 0.0f));
                //thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), droplet_mesh.colors.begin());
                //if (numSprings_old != d_springPos_d3.size())
                    //cout << "spring size changed" << endl;
                spring_mesh.colors.resize(d_springPos_d3.size());
                //std::cout << 9 << std::endl;
                thrust::copy(d_springColor.begin(), d_springColor.end(), spring_mesh.colors.begin());
            }
#endif
            //if (startPhysics)
                //dropletColor[maxVelocityIndex] = glm::vec3(0.75f, 0.34f, 0.0f);

            //colorBarMinColor = mapToColor(minConcentration, initialMinConcentration, initialMaxConcentration);
            //colorBarMaxColor = mapToColor(maxConcentration, initialMinConcentration, initialMaxConcentration);
        }

#ifndef TWO_DIM
        droplet_mesh.Draw(glm::perspective(glm::radians(camera.m_Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, distance * 0.001f, distance * 10.0f),
                          camera.GetViewMatrix());
#else
        if (numDroplets_old != dropletsSimCore.num_droplets_)
		{
            std::cout << numDroplets_old - dropletsSimCore.num_droplets_ << " droplets are removed" << std::endl;
			droplet_mesh.updateBuffers();
			numDroplets_old = dropletsSimCore.num_droplets_;
		}
        droplet_mesh.Draw(glm::ortho(windowLeft, windowRight, windowBottom, windowTop, -1.0f, 1.0f));

        //std::cout << 10 << std::endl;
#ifdef SHOW_SPRING
        if (numSprings_old != d_springPos_d3.size())
        {
            spring_mesh.updateBuffers();
            numSprings_old = d_springPos_d3.size();
        }
        if (numConnectors > 0)
            spring_mesh.Draw(glm::ortho(windowLeft, windowRight, windowBottom, windowTop, -1.0f, 1.0f));
#endif
        //std::cout << 11 << std::endl;
#endif
#ifdef SHOW_SPRING
        ColorBar.setColorRange(ColorBar.minValue, ColorBar.maxValue);
        ColorBar.draw();
        //std::cout << 12 << std::endl;
#endif
#ifdef SHOW_DROPLET_CONCENTRATION_COLOR
        ColorBarConcentration.setMinMaxValues(minConcentration, maxConcentration);
#ifndef ONLY_DRAW_DROPLET
        ColorBarConcentration.draw();
#endif
#endif

#ifdef SHOW_FLOOR
        if (dropletsSimCore.params_.floorEnabled)
            floor_mesh.Draw(glm::ortho(windowLeft, windowRight, windowBottom, windowTop, -1.0f, 1.0f));
#endif

#ifdef SHOW_WALL
		if (dropletsSimCore.params_.wallEnabled)
			wall_mesh.Draw(glm::ortho(windowLeft, windowRight, windowBottom, windowTop, -1.0f, 1.0f));
#endif

#endif // !USE_GPU
        //texts on top left corner
        std::vector<std::string> topLeftTexts;
        float aline_x =10.0f;
        float aline_y = SCR_HEIGHT - 10.0f;
        std::ostringstream stepTime_s;
        stepTime_s.precision(4);
        stepTime_s << std::fixed << stepTime;
        topLeftTexts.push_back("Simulation time: " + stepTime_s.str() + " s");
        std::ostringstream currentFrame_s;
        currentFrame_s.precision(0);
        currentFrame_s << std::fixed << currentFrame;
        topLeftTexts.push_back("Real time: " + currentFrame_s.str() + " s");
#ifndef TWO_DIM
#ifndef ONLY_DRAW_DROPLET
        std::ostringstream cmPos_s;
        cmPos_s.precision(6);
        cmPos_s << std::fixed << camera.m_Position.x << " " << camera.m_Position.y << " " << camera.m_Position.z;
        topLeftTexts.push_back("Camera position: " + cmPos_s.str());
#endif
#endif
        textRenderer.RenderMultilineText(topLeftTexts, glm::vec2(aline_x, aline_y), anchorPosType::TOP_LEFT, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
        //std::cout << 13 << std::endl;

        
        //texts on top right corner
        aline_x = SCR_WIDTH - 10.0f;
        aline_y = SCR_HEIGHT - 10.0f;
        std::vector<std::string> topRightTexts;
        /*
#if 0
        std::ostringstream maxConcentration_s;
        maxConcentration_s << std::scientific;
        maxConcentration_s.precision(6);
        maxConcentration_s << maxConcentration;
        topRightTexts.push_back("Max concentration: " + maxConcentration_s.str());
        std::ostringstream minConcentration_s;
        minConcentration_s << std::scientific;
        minConcentration_s.precision(6);
        minConcentration_s << minConcentration;
        topRightTexts.push_back("Min concentration: " + minConcentration_s.str());
#endif
        std::ostringstream velocity_s;
        velocity_s << std::scientific;
        velocity_s.precision(3);
        velocity_s << maxVelocity;
        topRightTexts.push_back("Max velocity: " + velocity_s.str() + " m/s");
        std::ostringstream totalKineticEnergy_s;
        totalKineticEnergy_s << std::scientific;
        totalKineticEnergy_s.precision(3);
        totalKineticEnergy_s << totalKineticEnergy;
        topRightTexts.push_back("Total kinetic energy: " + totalKineticEnergy_s.str() + " J");
        std::ostringstream distance_s;
        distance_s << std::scientific;
        distance_s.precision(3);
        distance_s << endDistance;
        topRightTexts.push_back("End distance: " + distance_s.str() + " m");
        std::ostringstream maxRadiusIncreaseSpeed_s;
        maxRadiusIncreaseSpeed_s << std::scientific;
        maxRadiusIncreaseSpeed_s.precision(3);
        maxRadiusIncreaseSpeed_s << maxRadiusIncreaseSpeed;
        topRightTexts.push_back("Max radius increase speed: " + maxRadiusIncreaseSpeed_s.str() + " m/s");
        std::ostringstream maxRadiusDecreaseSpeed_s;
        maxRadiusDecreaseSpeed_s << std::scientific;
        maxRadiusDecreaseSpeed_s.precision(3);
        maxRadiusDecreaseSpeed_s << maxRadiusDecreaseSpeed;
        topRightTexts.push_back("Max radius decrease speed: " + maxRadiusDecreaseSpeed_s.str() + " m/s");
        */
        std::ostringstream displacement_s;
        displacement_s << std::scientific;
        displacement_s.precision(3);
        displacement_s << displacement;
        topRightTexts.push_back("Displacement: " + displacement_s.str() + " m");
#ifndef ONLY_DRAW_DROPLET
        textRenderer.RenderMultilineText(topRightTexts, glm::vec2(aline_x, aline_y), anchorPosType::TOP_RIGHT, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
#endif
        //std::cout << 14 << std::endl;


        //texts on bottom left corner
        aline_x = 10.0f;
        aline_y = 10.0f;
        std::vector<std::string> bottomLeftTexts;
        std::ostringstream spWidth_s;
        spWidth_s.precision(3);
        spWidth_s << std::scientific << dropletsSimCore.params_.boundingBoxWidth;
        bottomLeftTexts.push_back("Specimen width: " + spWidth_s.str() + " m");
        std::ostringstream spHeight_s;
        spHeight_s.precision(3);
        spHeight_s << std::scientific << dropletsSimCore.params_.boundingBoxHeight;
        bottomLeftTexts.push_back("Specimen height: " + spHeight_s.str() + " m");
        //std::ostringstream numDroplets_x_s;
        //numDroplets_x_s.precision(0);
       // numDroplets_x_s << std::fixed << dropletsSimCore.params_.numDroplets_x;
        //bottomLeftTexts.push_back("Number of droplets x: " + numDroplets_x_s.str());
        //std::ostringstream numDroplets_y_s;
       // numDroplets_y_s.precision(0);
        //numDroplets_y_s << std::fixed << dropletsSimCore.params_.numDroplets_y;
        //bottomLeftTexts.push_back("Number of droplets y: " + numDroplets_y_s.str());
        //std::ostringstream radius_s;
        //radius_s << std::scientific;
        //radius_s.precision(3);
        //radius_s << dropletsSimCore.params_.initialDropletRadius;
        //bottomLeftTexts.push_back("Droplet radius: " + radius_s.str() + " m");
        //std::ostringstream mass_s;
        //mass_s << std::scientific;
        //mass_s.precision(3);
        //mass_s << dropletsSimCore.params_.dropletMass;
        //bottomLeftTexts.push_back("Droplet mass: " + mass_s.str() + " kg");
        std::ostringstream stiffness_s;
        stiffness_s << std::scientific;
        stiffness_s.precision(3);
        stiffness_s << dropletsSimCore.params_.springStiffness;
        bottomLeftTexts.push_back("Spring stiffness: " + stiffness_s.str() + " N/m");
        //std::ostringstream damping_s;
        //damping_s << std::scientific;
        //damping_s.precision(3);
        //damping_s << dropletsSimCore.params_.globalDamping;
        //bottomLeftTexts.push_back("Damping: " + damping_s.str() + " Ns/m");
        //std::ostringstream SpringRestLengthRatio_s;
        //SpringRestLengthRatio_s.precision(2);
        //SpringRestLengthRatio_s << std::fixed << dropletsSimCore.params_.SpringRestLengthRatio;
        //bottomLeftTexts.push_back("Spring rest length ratio: " + SpringRestLengthRatio_s.str());
        std::ostringstream toltalSeperationLengthRatio_s;
        toltalSeperationLengthRatio_s.precision(2);
        toltalSeperationLengthRatio_s << std::fixed << dropletsSimCore.params_.toltalSeperationLengthRatio;
        bottomLeftTexts.push_back("Toltal seperation length ratio: " + toltalSeperationLengthRatio_s.str());
        //std::ostringstream criticalConcentrationDifference_s;
        //criticalConcentrationDifference_s.precision(3);
        //criticalConcentrationDifference_s << std::fixed << dropletsSimCore.params_.criticalConcentrationDifference;
        //bottomLeftTexts.push_back("Critical concentration difference: " + criticalConcentrationDifference_s.str());
        std::ostringstream timeStep_s;
        timeStep_s.precision(3);
        timeStep_s << std::scientific << dropletsSimCore.params_.timeStep;
        bottomLeftTexts.push_back("Time step: " + timeStep_s.str() + " s");
        //std::ostringstream permeability_s;
        //permeability_s << std::scientific;
        //permeability_s.precision(3);
        //permeability_s << dropletsSimCore.params_.permeability;
        //bottomLeftTexts.push_back("Current permeability: " + permeability_s.str() + " m/s");
        std::ostringstream gamma_m_s;
        gamma_m_s.precision(6);
        gamma_m_s << std::fixed << dropletsSimCore.params_.gamma_m;
        bottomLeftTexts.push_back("Gamma_m: " + gamma_m_s.str() + " N/m");
        std::ostringstream theta_eq_s;
        theta_eq_s.precision(1);
        theta_eq_s << std::fixed << dropletsSimCore.params_.theta_eq;
        bottomLeftTexts.push_back("Theta_eq: " + theta_eq_s.str() + " Degree");
        std::ostringstream numConnections_s;
        numConnections_s << dropletsSimCore.getNumConnectors();
        bottomLeftTexts.push_back("Number of connections: " + numConnections_s.str());
        std::ostringstream averageNumConnectionsPerDroplet_s;
        averageNumConnectionsPerDroplet_s << everageNumConnectionsPerDroplet;
        bottomLeftTexts.push_back("Average connections per droplet: " + averageNumConnectionsPerDroplet_s.str());
#ifndef ONLY_DRAW_DROPLET
        textRenderer.RenderMultilineText(bottomLeftTexts, glm::vec2(aline_x, aline_y), anchorPosType::BOTTOM_LEFT, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
#endif
        //std::cout << 15 << std::endl;

        /*
        if (maxConcentration - minConcentration < dropletsSimCore.params_.criticalConcentrationDifference)
        {
            startPhysics = false;
            std::ostringstream cd_s;
            cd_s.precision(4);
            cd_s << std::fixed << dropletsSimCore.params_.criticalConcentrationDifference;
            std::vector<std::string> topCenterTexts;
            topCenterTexts.push_back("Concentration difference is less than " + cd_s.str() + ".");
            topCenterTexts.push_back("Simulation stoped.");
            textRenderer.RenderMultilineText(topCenterTexts, glm::vec2(SCR_WIDTH / 2.0f, SCR_HEIGHT-10.0f), anchorPosType::TOP_CENTER, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
        }
        */
        if (dropletsSimCore.getTime() >= dropletsSimCore.params_.settlingTime+ dropletsSimCore.params_.compressionTime+ dropletsSimCore.params_.increaseTime)
        {
            startPhysics = false;
            std::ostringstream md_s;
            md_s.precision(3);
            md_s << std::scientific << dropletsSimCore.params_.maxDisplacement;
            std::vector<std::string> topCenterTexts;
            topCenterTexts.push_back("Displacement applied reach " + md_s.str() + " m.");
            topCenterTexts.push_back("Simulation stoped.");
            textRenderer.RenderMultilineText(topCenterTexts, glm::vec2(SCR_WIDTH / 2.0f, SCR_HEIGHT - 10.0f), anchorPosType::TOP_CENTER, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
        }
        
        
        //std::cout << 16 << std::endl;

#ifdef USE_IMGUI
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Save the frame to disk
        if (dropletsSimCore.params_.saveFrames)
        {
            if ((stepTime - lastSaveTime >= dropletsSimCore.params_.saveFrameInterval || !initialFrameSaved) && startPhysics/* && stepTime < 0.02*/)
            {
                int width, height;
                glfwGetFramebufferSize(window, &width, &height);
                saveFrameAsync(directory.string(), frameToSave++, width, height);
                if (!initialFrameSaved)
					initialFrameSaved = true;
            }
        }
        if (dropletsSimCore.params_.outputData)
        {
            if (stepTime - lastSaveTime >= dropletsSimCore.params_.saveFrameInterval && startPhysics && dropletsSimCore.params_.startApplyDisplacement)
            {
                displacementFile.open(data_directory / displacementFileName, std::ios::app);
                displacementFile << displacement << std::endl;
                displacementFile.close();
                forceFile.open(data_directory / forceFileName, std::ios::app);
                forceFile << force_to_save << std::endl;
                forceFile.close();
            }
        }
        if(stepTime - lastSaveTime >= dropletsSimCore.params_.saveFrameInterval && startPhysics)
            lastSaveTime = stepTime;
        //std::cout << 17 << std::endl;
    }
    // Cleanup
#ifdef USE_IMGUI
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopWorker = true;
    }
    queueCondition.notify_one();
    saverThread.join();

    // Signal the physics loop to stop
    stopFlag.store(true);

    // Properly join the physics thread to ensure it has finished
    if (physicsThread.joinable()) {
        physicsThread.join();
    }
    
    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
#ifndef TWO_DIM
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        rightButtonPressed = true;
        //glfwSetCursorPosCallback(window, mouse_callback);
    }
    else
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        rightButtonPressed = false;
        //glfwSetCursorPosCallback(window, NULL);
    }
#endif
}
#ifndef TWO_DIM
// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    if (rightButtonPressed)
        camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
#endif

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
	{
		startPhysics = !startPhysics;
        std::cout << "startPhysics = " << startPhysics << std::endl;
    }
    if (key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		saveFrameAsync(directory_singleFrame.string(), singleFrameNumber++, SCR_WIDTH, SCR_HEIGHT);
	}
#ifndef TWO_DIM
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
	{
        camera.resetCamera(cameraPos);
	}
#endif
}