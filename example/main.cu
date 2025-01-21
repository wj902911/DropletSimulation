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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

std::vector<glm::mat4> getModelMatrices_GPU(thrust::device_vector<double3> d_dropletPos_d3,
    thrust::device_vector<double> d_dropletRadius_d,
    double initialDropletRadius);
void getDropletColor(thrust::device_vector<double>& d_dropletC_d,
    thrust::device_vector<glm::vec3>& d_dropletColor,
    double minConcentration,
    double maxConcentration);


__global__ void model_matrix_kernel(double3* d_dropletPos_d3,
    double* d_dropletRadius_d,
    glm::mat4* d_model,
    double initialDropletRadius,
    int num_droplets);
__global__ void getDropletColor_kernel(double* d_dropletC_d,
    glm::vec3* d_dropletColor,
    double minConcentration,
    double maxConcentration,
    int num_droplets);

const unsigned int SCR_WIDTH = 1920; //window width
const unsigned int SCR_HEIGHT = 1080; //window height

int num_droplets_x = 20;

// camera
glm::vec3 cameraPos = glm::vec3(0.0f, 3.0f, 10.0f);
Camera camera(cameraPos);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

bool startPhysics = false;
bool rightButtonPressed = false;
std::mutex data_mutex;
std::atomic<bool> stopFlag(false);

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

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
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    DropletsSimCore_GPU dropletsSimCore(num_droplets_x);
    dropletsSimCore.initSimulation();

    //setup camera
    double2 center = dropletsSimCore.params_.boundingBoxCenter;
    double2 size = make_double2(dropletsSimCore.params_.boundingBoxWidth, dropletsSimCore.params_.boundingBoxHeight);
    glm::vec3 sizeGLM(size.x, size.y, 0.0f);
    float boundingRadius = glm::length(sizeGLM) * 0.5f;
    float fov = glm::radians(camera.m_Zoom);
    float distance = dropletsSimCore.params_.cameraDistanceRatio * boundingRadius / std::tan(fov / 2.0f);
    cameraPos = glm::vec3(center.x, center.y, distance);
    camera.setPos(cameraPos);
    camera.setSpeed(distance * 0.2);

    thrust::device_vector<double3> d_dropletPos_d3;
    thrust::device_vector<double> d_dropletRadius_d;
    thrust::device_vector<double> d_dropletC_d;
    thrust::device_vector<glm::vec3> d_dropletColor;
    std::vector<glm::mat4> model;

    dropletsSimCore.getAllDropletPositionAsD3(d_dropletPos_d3);
    dropletsSimCore.getAllDropletRadius(d_dropletRadius_d);
    model = getModelMatrices_GPU(d_dropletPos_d3, d_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);

    thrust::pair<double, double> concentrationRange = dropletsSimCore.getConcentration(d_dropletC_d);
    double maxConcentration = concentrationRange.second, initialMaxConcentration = concentrationRange.second;
    double minConcentration = concentrationRange.first, initialMinConcentration = concentrationRange.first;
    std::vector<glm::vec3> dropletColor(d_dropletRadius_d.size(), glm::vec3(0.75f, 0.34f, 0.0f));
    getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
    thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), dropletColor.begin());

    DropletMesh_3D droplet_mesh(dropletsSimCore.params_.initialDropletRadius, 10, 20, dropletColor, model);
    OpenGLText textRenderer(static_cast<GLfloat>(SCR_WIDTH), static_cast<GLfloat>(SCR_HEIGHT));

    float colorBarWidth = 20.0f;
    float colorBarHeight = 200.0f;
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

    std::thread physicsThread(&DropletsSimCore_GPU::physicsLoop,
        std::ref(dropletsSimCore),
        std::ref(startPhysics),
        std::ref(stopFlag),
        std::ref(data_mutex));

    double stepTime = 0.0;

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

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

        if (startPhysics)
        {
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                stepTime = dropletsSimCore.getTime();
                dropletsSimCore.getAllDropletPositionAsD3(d_dropletPos_d3);
                dropletsSimCore.getAllDropletRadius(d_dropletRadius_d);
                concentrationRange = dropletsSimCore.getConcentration(d_dropletC_d);
            }
            droplet_mesh.modelMatrices = getModelMatrices_GPU(d_dropletPos_d3, d_dropletRadius_d, dropletsSimCore.params_.initialDropletRadius);
            minConcentration = concentrationRange.first;
            maxConcentration = concentrationRange.second;
            getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
            thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), droplet_mesh.colors.begin());
        }
        droplet_mesh.Draw(glm::perspective(glm::radians(camera.m_Zoom),
            (float)SCR_WIDTH / (float)SCR_HEIGHT,
            distance * 0.001f,
            distance * 10.0f),
            camera.GetViewMatrix());
        ColorBarConcentration.setMinMaxValues(minConcentration, maxConcentration);
        ColorBarConcentration.draw();

        std::vector<std::string> topLeftTexts;
        float aline_x = 10.0f;
        float aline_y = SCR_HEIGHT - 10.0f;
        std::ostringstream stepTime_s;
        stepTime_s.precision(4);
        stepTime_s << std::fixed << stepTime;
        topLeftTexts.push_back("Simulation time: " + stepTime_s.str() + " s");
        std::ostringstream currentFrame_s;
        currentFrame_s.precision(0);
        currentFrame_s << std::fixed << currentFrame;
        topLeftTexts.push_back("Real time: " + currentFrame_s.str() + " s");
        std::ostringstream permeability_s;
        permeability_s << std::scientific;
        permeability_s.precision(3);
        permeability_s << dropletsSimCore.params_.permeability;
        topLeftTexts.push_back("Current permeability: " + permeability_s.str() + " m/s");
        std::ostringstream timeStep_s;
        timeStep_s.precision(3);
        timeStep_s << std::scientific << dropletsSimCore.params_.timeStep;
        topLeftTexts.push_back("Time step: " + timeStep_s.str() + " s");
        textRenderer.RenderMultilineText(topLeftTexts, glm::vec2(aline_x, aline_y), anchorPosType::TOP_LEFT, 10.0f, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

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
}

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