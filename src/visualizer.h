#pragma once

#include <glad/glad.h> 
#include <GLFW\glfw3.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Camera.h>
#include <mesh.h>
#include <openGLText.h>
#include <colorBar.h>

#include <dropletsSimCore_GPU.h>

namespace fs = std::filesystem;

struct SaveTask {
	std::vector<unsigned char> pixels;
	std::string filename;
	int width;
	int height;
};

__global__ inline void model_matrix_kernel(double3* d_dropletPos_d3,
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

__global__ inline void getDropletColor_kernel(double* d_dropletC_d,
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

class Visualizer
{
public:
	Visualizer(DropletsSimCore_GPU* dropSimCore)
	:simCore(dropSimCore),
	 dropletColor(simCore->num_droplets_)
	{
		if (!fs::exists(directory)) {
			fs::create_directories(directory);
		}
		if (!fs::exists(directory_singleFrame)) {
			fs::create_directories(directory_singleFrame);
		}


		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

		window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Droplets Simulation", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
		}
		glfwMakeContextCurrent(window);
		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); });
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);
		glfwSetKeyCallback(window, key_callback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
		}

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		text = OpenGLText(SCR_WIDTH, SCR_HEIGHT);
		colBar = colorBar(cbWidth, cbHeight, SCR_WIDTH, SCR_HEIGHT, 1, -1);

		double2 center = simCore->getGlobalCenter();
		double2 size = simCore->getGlobalSize();
		glm::vec3 sizeGLM(size.x, size.y, 0.0f);
		float boundingRadius = glm::length(sizeGLM) * 0.5f;
		float fov = glm::radians(camera.m_Zoom);
		float distance = simCore->params_.cameraDistanceRatio * boundingRadius / std::tan(fov / 2.0f);
		camera = Camera(glm::vec3(center.x, center.y, distance));
		camera.setSpeed(distance * 0.2);
		
		simCore->getAllDropletPositionAsD3(d_dropletPos_d3);
		simCore->getAllDropletRadius(d_dropletRadius_d);
		std::vector<glm::vec3> dpColors(simCore->num_droplets_);
		std::vector<glm::mat4> model = getModelMatrices_GPU(d_dropletPos_d3, d_dropletRadius_d, simCore->params_.initialDropletRadius);
		thrust::pair<double, double> concentrationRange = simCore->getConcentration(d_dropletC_d);
		colBar.maxValue = initialMaxConcentration = concentrationRange.second;
		colBar.minValue = initialMinConcentration = concentrationRange.first;
		getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
		thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), dpColors.begin());
		mesh.push_back(std::make_shared<DropletMesh_3D>(DropletMesh_3D(simCore->params_.initialDropletRadius, 10, 20, dpColors, model)));
		
	}

	~Visualizer()
	{
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void run(std::mutex& data_mutex)
	{
		while (!glfwWindowShouldClose(window))
		{
			float currentFrame = static_cast<float>(glfwGetTime());
			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;

			processInput(window);

			int display_w, display_h;
			glfwGetFramebufferSize(window, &display_w, &display_h);
			glViewport(0, 0, display_w, display_h);
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			{
				std::lock_guard<std::mutex> lock(data_mutex);
				getCurrentMesh();
			}

			for (int i = 0; i < mesh.size(); i++)
			{
				mesh[i]->Draw(glm::perspective(glm::radians(camera.m_Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, camera.m_Position.z * 0.1f, camera.m_Position.z * 10.0f),
					camera.GetViewMatrix());
			}

			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}

	void getCurrentMesh()
	{
		simCore->getAllDropletPositionAsD3(d_dropletPos_d3);
		simCore->getAllDropletRadius(d_dropletRadius_d);

		mesh[0]->modelMatrices = getModelMatrices_GPU(d_dropletPos_d3, 
			                                          d_dropletRadius_d, 
			                                          simCore->params_.initialDropletRadius);

		thrust::pair<double, double> concentrationRange = simCore->getConcentration(d_dropletC_d);
		colBar.maxValue = concentrationRange.second;
		colBar.minValue = concentrationRange.first;
		getDropletColor(d_dropletC_d, d_dropletColor, initialMinConcentration, initialMaxConcentration);
		thrust::copy(d_dropletColor.begin(), d_dropletColor.end(), mesh[0]->colors.begin());
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

	static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
	{
		Visualizer* visualizer = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
		
		float xpos = static_cast<float>(xposIn);
		float ypos = static_cast<float>(yposIn);

		if (visualizer->firstMouse)
		{
			visualizer->lastX = xpos;
			visualizer->lastY = ypos;
			visualizer->firstMouse = false;
		}

		float xoffset = xpos - visualizer->lastX;
		float yoffset = visualizer->lastY - ypos;
		visualizer->lastX = xpos;
		visualizer->lastY = ypos;

		visualizer->camera.ProcessMouseMovement(xoffset, yoffset);
	}

	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		Visualizer* visualizer = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
		visualizer->camera.ProcessMouseScroll(static_cast<float>(yoffset));
	}

	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		Visualizer* visualizer = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));

		if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
		{
			visualizer->startPhysics = !visualizer->startPhysics;
			std::cout << "startPhysics = " << visualizer->startPhysics << std::endl;
		}
		if (key == GLFW_KEY_P && action == GLFW_PRESS)
		{
			visualizer->saveFrameAsync(visualizer->directory_singleFrame.string(), visualizer->singleFrameNumber++, visualizer->SCR_WIDTH, visualizer->SCR_HEIGHT);
		}
		if (key == GLFW_KEY_R && action == GLFW_PRESS)
		{
			visualizer->camera.resetCamera(visualizer->camera.m_Position);
		}
	}

	bool startPhysics = false;
	bool rightButtonPressed = false;
	bool saveSingleFrame = false;
	float deltaTime = 0.0f;
	float lastFrame = 0.0f;

	fs::path directory_singleFrame = "output_frames_single";
	fs::path directory = "output_frames";
	int singleFrameNumber = 0;


	const unsigned int SCR_WIDTH = 1920;
	const unsigned int SCR_HEIGHT = 1080;
	GLFWwindow* window;

	Camera camera;
	float lastX = SCR_WIDTH / 2.0f;
	float lastY = SCR_HEIGHT / 2.0f;
	bool firstMouse = true;

	std::vector<std::shared_ptr<Mesh>> mesh;
	DropletsSimCore_GPU* simCore;

	OpenGLText text;

	GLfloat cbWidth = static_cast<GLfloat>(SCR_WIDTH) / 96.f;
	GLfloat cbHeight = static_cast<GLfloat>(SCR_HEIGHT) / 3.f;
	colorBar colBar;
	

	std::queue<SaveTask> saveQueue;
	std::mutex queueMutex;
	std::condition_variable queueCondition;

	thrust::device_vector<double3> d_dropletPos_d3;
	thrust::device_vector<double> d_dropletRadius_d;
	thrust::device_vector<double> d_dropletC_d;
	thrust::device_vector<glm::vec3> d_dropletColor;
	std::vector<glm::vec3> dropletColor;

	double initialMinConcentration = 0.0;
	double initialMaxConcentration = 1.0;
};
