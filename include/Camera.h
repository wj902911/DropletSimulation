#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement
{
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 5.0f;
const float SENSITIVITY = 0.02f;
const float ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
	// Camera Attributes
	glm::vec3 m_Position;
	glm::vec3 m_Front;
	glm::vec3 m_Up;
	glm::vec3 m_Right;
	glm::vec3 m_WorldUp;
	// Euler Angles
	float m_Yaw;
	float m_Pitch;
	// Camera options
	float m_MovementSpeed;
	float m_MouseSensitivity;
	float m_Zoom;

	// Constructor with vectors
	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : 
		m_Front(glm::vec3(0.0f, 0.0f, -1.0f)), 
		m_MovementSpeed(SPEED), 
		m_MouseSensitivity(SENSITIVITY), 
		m_Zoom(ZOOM)
	{
		m_Position = position;
		m_WorldUp = up;
		m_Yaw = yaw;
		m_Pitch = pitch;
		updateCameraVectors();
	}
	// Constructor with scalar values
	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : 
		m_Front(glm::vec3(0.0f, 0.0f, -1.0f)), 
		m_MovementSpeed(SPEED),
		m_MouseSensitivity(SENSITIVITY),
		m_Zoom(ZOOM)
	{
		m_Position = glm::vec3(posX, posY, posZ);
		m_WorldUp = glm::vec3(upX, upY, upZ);
		m_Yaw = yaw;
		m_Pitch = pitch;
		updateCameraVectors();
	}

	void resetCamera(glm::vec3 pos)
	{
		m_Position = pos;
		m_Front = glm::vec3(0.0f, 0.0f, -1.0f);
		m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
		m_Right = glm::vec3(1.0f, 0.0f, 0.0f);
		m_WorldUp = glm::vec3(0.0f, 1.0f, 0.0f);
		m_Yaw = YAW;
		m_Pitch = PITCH;
		m_MouseSensitivity = SENSITIVITY;
		m_Zoom = ZOOM;
	}

	void setPos(glm::vec3 pos) 
	{
		m_Position = pos;
	}

	void setSpeed(float speed)
	{
		m_MovementSpeed = speed;
	}

	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix()
	{
		return glm::lookAt(m_Position, m_Position + m_Front, m_Up);
	}

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void ProcessKeyboard(Camera_Movement direction, float deltaTime)
	{
		float velocity = m_MovementSpeed * deltaTime;
		if (direction == FORWARD)
			m_Position += m_Front * velocity;
		if (direction == BACKWARD)
			m_Position -= m_Front * velocity;
		if (direction == LEFT)
			m_Position -= m_Right * velocity;
		if (direction == RIGHT)
			m_Position += m_Right * velocity;
		if (direction == UP)
			m_Position += m_Up * velocity;
		if (direction == DOWN)
			m_Position -= m_Up * velocity;
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
	{
		xoffset *= m_MouseSensitivity;
		yoffset *= m_MouseSensitivity;

		m_Yaw += xoffset;
		m_Pitch += yoffset;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (m_Pitch > 89.0f)
				m_Pitch = 89.0f;
			if (m_Pitch < -89.0f)
				m_Pitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Euler angles
		updateCameraVectors();
	}

	// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void ProcessMouseScroll(float yoffset)
	{
		if (m_Zoom >= 1.0f && m_Zoom <= 45.0f)
			m_Zoom -= yoffset;
		if (m_Zoom <= 1.0f)
			m_Zoom = 1.0f;
		if (m_Zoom >= 45.0f)
			m_Zoom = 45.0f;
	}

private:
	// Calculates the front vector from the Camera's (updated) Euler Angles
	void updateCameraVectors()
	{
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
		front.y = sin(glm::radians(m_Pitch));
		front.z = sin(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
		m_Front = glm::normalize(front);
		// Also re-calculate the Right and Up vector
		m_Right = glm::normalize(glm::cross(m_Front, m_WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
		m_Up = glm::normalize(glm::cross(m_Right, m_Front));
	}
};
#endif
