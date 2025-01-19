#pragma once

#include "mesh.h"
#include "shader.h"
#include "openGLText.h"
#include <cuda_runtime.h>

#if 0
void checkOpenGLError(const char* function) {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		switch (err) {
		case GL_INVALID_ENUM:
			std::cerr << "OpenGL Error (GL_INVALID_ENUM): " << function << std::endl;
			break;
		case GL_INVALID_VALUE:
			std::cerr << "OpenGL Error (GL_INVALID_VALUE): " << function << std::endl;
			break;
		case GL_INVALID_OPERATION:
			std::cerr << "OpenGL Error (GL_INVALID_OPERATION): " << function << std::endl;
			break;
		case GL_STACK_OVERFLOW:
			std::cerr << "OpenGL Error (GL_STACK_OVERFLOW): " << function << std::endl;
			break;
		case GL_STACK_UNDERFLOW:
			std::cerr << "OpenGL Error (GL_STACK_UNDERFLOW): " << function << std::endl;
			break;
		case GL_OUT_OF_MEMORY:
			std::cerr << "OpenGL Error (GL_OUT_OF_MEMORY): " << function << std::endl;
			break;
		default:
			std::cerr << "Unknown OpenGL Error: " << function << std::endl;
		}
	}
}

class ColorBarMesh : public Mesh<Vertex_with_color>
{
public:
	shader Shader = shader(VERTEX_COLOR_2D);
	//unsigned int colorVBO = 0;

	ColorBarMesh() = default;

	ColorBarMesh(const vector<Vertex_with_color>& vertices, const vector<unsigned int>& indices)
	{
		this->vertices = vertices;
		this->indices = indices;

		setupMesh();
	}

	void Draw()
	{
		Shader.use();

		checkOpenGLError("After setting up mesh");

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex_with_color), &vertices[0], GL_STATIC_DRAW);

		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);

		checkOpenGLError("After drawing elements");

		glBindVertexArray(0);
	}

	void setupMesh()
	{
		// create buffers/arrays
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		// load data into vertex buffers
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		// A great thing about structs is that their memory layout is sequential for all its items.
		// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
		// again translates to 3/2 floats which translates to a byte array.
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex_with_color), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		// set the vertex attribute pointers
		// vertex Positions
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_with_color), (void*)0);
		/*
		Shader.use();
		Shader.setVec3("objectColor", colors[0]);
		// create buffers/arrays
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);
		//glGenBuffers(1, &colorVBO);

		glBindVertexArray(VAO);
		// load data into vertex buffers
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		// A great thing about structs is that their memory layout is sequential for all its items.
		// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
		// again translates to 3/2 floats which translates to a byte array.
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

		// set the vertex attribute pointers
		// vertex Positions
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
		*/
		

		// vertex colors
		//glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
		//glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex_with_color), (void*)offsetof(Vertex_with_color, Color));

		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glBindVertexArray(0);
	}
	
};

class colorBar
{
public:
	ColorBarMesh mesh;
	OpenGLText text;

	colorBar(glm::vec2 bottomRightCorner,
				 float width,
				 float height,
			   GLfloat window_width,
			   GLfloat window_height,
			 glm::vec3 upperColor,
			 glm::vec3 lowerColor)
		: text(OpenGLText(window_width, window_height))
	{
		vector<Vertex_with_color> vertices;
		vertices.push_back(Vertex_with_color(glm::vec3(bottomRightCorner.x - width, bottomRightCorner.y, 0.0f), lowerColor));
		vertices.push_back(Vertex_with_color(glm::vec3(bottomRightCorner.x, bottomRightCorner.y, 0.0f), lowerColor));
		vertices.push_back(Vertex_with_color(glm::vec3(bottomRightCorner.x, bottomRightCorner.y + height, 0.0f), upperColor));
		vertices.push_back(Vertex_with_color(glm::vec3(bottomRightCorner.x - width, bottomRightCorner.y + height, 0.0f), upperColor));

		vector<unsigned int> indices = { 0, 1, 2, 2, 3, 0 };

		mesh = ColorBarMesh(vertices, indices);

		mesh.Shader.use();
		mesh.Shader.setMat4("projection", glm::ortho(0.0f, window_width, 0.0f, window_height));
	}

	void draw(glm::vec3 upperColor, glm::vec3 lowerColor)
	{
		mesh.vertices[2].Color = upperColor;
		mesh.vertices[3].Color = upperColor;
		mesh.vertices[0].Color = lowerColor;
		mesh.vertices[1].Color = lowerColor;
		mesh.Draw();
	}
};
#endif

__host__ __device__
glm::vec3 mapToColor(double value, double min, double max)
{
#if 0
	double ratio = 0.0;
	if (max == min)
		ratio = 2.0;
	else
		ratio = 4.0 * (value - min) / (max - min);
#else
	double ratio = 4.0 * (value - min) / (max - min);
#endif
	if (ratio < 1.0)
		return glm::vec3(0.0, ratio, 1.0);
	else if (ratio < 2.0)
		return glm::vec3(0.0, 1.0, 2.0 - ratio);
	else if (ratio < 3.0)
		return glm::vec3(ratio - 2.0, 1.0, 0.0);
	else
		return glm::vec3(1.0, 4.0 - ratio, 0.0);
}

#if 0
struct ColorVertex
{
	glm::vec3 Position;
	glm::vec3 Color;

	ColorVertex(glm::vec3 position, glm::vec3 color)
		: Position(position), Color(color) { }
};

class colorBarMsh
{
public:
	std::vector<ColorVertex> vertices;
	std::vector<unsigned int> indices;

	shader Shader = shader(VERTEX_COLOR_2D);
	GLuint colorBarVAO, colorBarVBO, colorBarEBO;
	GLfloat initial_minValue = 0.0f, initial_maxValue = 0.0f;

	colorBarMsh(GLfloat width,
			 GLfloat height,
		     GLfloat window_width,
		     GLfloat window_height,
		      double minValue,
		      double maxValue)
		: initial_minValue(minValue), initial_maxValue(maxValue)
	{
		
		for (int i = 0; i < 5; i++)
		{
			vertices.push_back(ColorVertex(glm::vec3(window_width - 10.0f - width, 10.0f + i * height * 0.25f, 0.0f),
				                           mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue)));
			vertices.push_back(ColorVertex(glm::vec3(window_width - 10.0f, 10.0f + i * height * 0.25f, 0.0f),
				                           mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue)));
		}
		

		indices = {
			0, 1, 3,
			3, 2, 0,
			2, 3, 5,
			5, 4, 2,
			4, 5, 7,
			7, 6, 4,
			6, 7, 9,
			9, 8, 6
		};

		setupMesh(window_width, window_height);
	}

	void draw(float minValue, float maxValue)
	{
		Shader.use();
		glBindVertexArray(colorBarVAO);

		updateColor(minValue, maxValue);

		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
		//glDrawElements(GL_TRIANGLES, 24, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	void setupMesh(GLfloat window_width, GLfloat window_height)
	{
		Shader.use();
		Shader.setMat4("projection", glm::ortho(0.0f, window_width, 0.0f, window_height));

		glGenVertexArrays(1, &colorBarVAO);
		glGenBuffers(1, &colorBarVBO);
		glGenBuffers(1, &colorBarEBO);

		glBindVertexArray(colorBarVAO);

		glBindBuffer(GL_ARRAY_BUFFER, colorBarVBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(ColorVertex), &vertices[0], GL_STATIC_DRAW);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, colorBarEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ColorVertex), (void*)0);
		//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ColorVertex), (void*)offsetof(ColorVertex, Color));
		//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void updateColor(double minValue, double maxValue)
	{
		for (int i = 0; i < 5; i++)
		{
			vertices[2 * i].Color = mapToColor(minValue + i * (maxValue - minValue) * 0.25, initial_minValue, initial_maxValue);
			vertices[2 * i + 1].Color = mapToColor(minValue + i * (maxValue - minValue) * 0.25, initial_minValue, initial_maxValue);
		}

		glBindBuffer(GL_ARRAY_BUFFER, colorBarVBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(ColorVertex), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
};
#endif

class colorBarMesh
{
public:
	std::vector<Vertex> vertices;
	std::vector<glm::vec3> colors;
	std::vector<unsigned int> indices;

	shader Shader;
	GLuint colorBarVAO = 0, colorBarVBO = 0, colorBarEBO = 0;
	GLfloat initial_minValue = 0.0f, initial_maxValue = 0.0f;

	colorBarMesh() = default;

	colorBarMesh(GLfloat width,
		GLfloat height,
		GLfloat window_width,
		GLfloat window_height,
		double minValue,
		double maxValue)
		: initial_minValue(minValue), initial_maxValue(maxValue)
	{
		Shader = shader(VERTEX_COLOR_2D);

		for (int i = 0; i < 5; i++)
		{
			vertices.push_back(Vertex(glm::vec3(window_width - 10.0f - width, 10.0f + i * height * 0.25f, 0.0f)));
			colors.push_back(mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue));
			vertices.push_back(Vertex(glm::vec3(window_width - 10.0f, 10.0f + i * height * 0.25f, 0.0f)));
			colors.push_back(mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue));
		}

		indices = 
		{
			0, 1, 3,
			3, 2, 0,
			2, 3, 5,
			5, 4, 2,
			4, 5, 7,
			7, 6, 4,
			6, 7, 9,
			9, 8, 6
		};

		setupMesh(window_width, window_height);
	}

	colorBarMesh(GLfloat anchorX,
		         GLfloat anchorY,
		         GLfloat width,
		         GLfloat height,
		         GLfloat window_width,
		         GLfloat window_height,
		          double minValue,
		          double maxValue)
		: initial_minValue(minValue), initial_maxValue(maxValue)
	{
		Shader = shader(VERTEX_COLOR_2D);

		for (int i = 0; i < 5; i++)
		{
			vertices.push_back(Vertex(glm::vec3(anchorX - width, anchorY + i * height * 0.25f, 0.0f)));
			colors.push_back(mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue));
			vertices.push_back(Vertex(glm::vec3(anchorX, anchorY + i * height * 0.25f, 0.0f)));
			colors.push_back(mapToColor(minValue + i * (maxValue - minValue) * 0.25, minValue, maxValue));
		}

		indices =
		{
			0, 1, 3,
			3, 2, 0,
			2, 3, 5,
			5, 4, 2,
			4, 5, 7,
			7, 6, 4,
			6, 7, 9,
			9, 8, 6
		};

		setupMesh(window_width, window_height);
	}

	void draw(float minValue, float maxValue)
	{
		Shader.use();
		glBindVertexArray(colorBarVAO);

		for (int i = 0; i < 5; i++)
		{
			colors[2 * i] = mapToColor(minValue + i * (maxValue - minValue) * 0.25, initial_minValue, initial_maxValue);
			colors[2 * i + 1]= mapToColor(minValue + i * (maxValue - minValue) * 0.25, initial_minValue, initial_maxValue);
		}

		glBindBuffer(GL_ARRAY_BUFFER, colorBarVBO);
		glBufferSubData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), colors.size() * sizeof(glm::vec3), &colors[0]);

		glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	void setupMesh(GLfloat window_width, GLfloat window_height)
	{
		Shader.use();
		Shader.setMat4("projection", glm::ortho(0.0f, window_width, 0.0f, window_height));

		glGenVertexArrays(1, &colorBarVAO);
		glGenBuffers(1, &colorBarVBO);
		glGenBuffers(1, &colorBarEBO);

		glBindVertexArray(colorBarVAO);

		glBindBuffer(GL_ARRAY_BUFFER, colorBarVBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex) + colors.size() * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Vertex), &vertices[0]);
		glBufferSubData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), colors.size() * sizeof(glm::vec3), &colors[0]);
	
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)(vertices.size() * sizeof(Vertex)));

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, colorBarEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
};

class colorBarFrame
{
public:
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	shader Shader;
	GLuint frameVAO = 0, frameVBO = 0, frameEBO = 0;

	colorBarFrame() = default;

	colorBarFrame(GLfloat width,
				  GLfloat height,
				  GLfloat window_width,
				  GLfloat window_height)
	{
		Shader = shader(UNIFORM_COLOR_2D_WITHOUT_MODEL);
		for (int i = 0; i < 5; i++)
		{
			vertices.push_back(Vertex(glm::vec3(window_width - 15.0f - width, 10.0f + i * height * 0.25f, 0.1f)));
			vertices.push_back(Vertex(glm::vec3(window_width - 10.0f - width, 10.0f + i * height * 0.25f, 0.1f)));
			vertices.push_back(Vertex(glm::vec3(window_width - 10.0f, 10.0f + i * height * 0.25f, 0.1f)));
		}

		indices = 
		{
			0, 2,
			3, 4,
			6, 7,
			9, 10,
			12, 14,
			1, 13,
			2, 14
		};

		setupMesh(window_width, window_height);
	}

	colorBarFrame(GLfloat anchorX,
		          GLfloat anchorY,
		          GLfloat width,
				  GLfloat height,
				  GLfloat window_width,
				  GLfloat window_height)
	{
		Shader = shader(UNIFORM_COLOR_2D_WITHOUT_MODEL);
		for (int i = 0; i < 5; i++)
		{
			vertices.push_back(Vertex(glm::vec3(anchorX - 5.0f - width, anchorY + i * height * 0.25f, 0.1f)));
			vertices.push_back(Vertex(glm::vec3(anchorX - width, anchorY + i * height * 0.25f, 0.1f)));
			vertices.push_back(Vertex(glm::vec3(anchorX, anchorY + i * height * 0.25f, 0.1f)));
		}

		indices =
		{
			0, 2,
			3, 4,
			6, 7,
			9, 10,
			12, 14,
			1, 13,
			2, 14
		};

		setupMesh(window_width, window_height);
	}

	void draw()
	{
		Shader.use();
		glBindVertexArray(frameVAO);

		glDrawElements(GL_LINES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);

		glBindVertexArray(0);
	}

	void setupMesh(GLfloat window_width, GLfloat window_height)
	{
		Shader.use();
		Shader.setMat4("projection", glm::ortho(0.0f, window_width, 0.0f, window_height));
		Shader.setVec3("objectColor", glm::vec3(0.0f, 0.0f, 0.0f));

		glGenVertexArrays(1, &frameVAO);
		glGenBuffers(1, &frameVBO);
		glGenBuffers(1, &frameEBO);

		glBindVertexArray(frameVAO);

		glBindBuffer(GL_ARRAY_BUFFER, frameVBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frameEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

		glBindVertexArray(0);

		glLineWidth(2.0f);
	}
};

class colorBar
{
public:
	colorBarMesh mesh;
	colorBarFrame frame;
	OpenGLText text;
	string title;

	double minValue = 0.0, maxValue = 0.0;

	colorBar() {}

	colorBar(float width,
		    float height,
		  GLfloat window_width,
		  GLfloat window_height,
		   string ttl,
		   double minValue,
		   double maxValue)
	: minValue(minValue), maxValue(maxValue), title(ttl),
	  text(OpenGLText(window_width, window_height)), 
	  mesh(colorBarMesh(width, height, window_width, window_height, minValue, maxValue)),
	  frame(colorBarFrame(width, height, window_width, window_height))
	{
	}

	colorBar(float anchorX,
		     float anchorY,
		     float width,
		     float height,
		   GLfloat window_width,
		   GLfloat window_height,
		    string ttl,
		    double minValue,
		    double maxValue)
	: minValue(minValue), maxValue(maxValue), title(ttl),
	  text(OpenGLText(window_width, window_height)), 
	  mesh(colorBarMesh(anchorX, anchorY, width, height, window_width, window_height, minValue, maxValue)),
	  frame(colorBarFrame(anchorX, anchorY, width, height, window_width, window_height))
	{
	}

	void draw()
	{
		mesh.draw(minValue, maxValue);
		frame.draw();
		double valueStride= (maxValue - minValue) / 4.0;
		std::vector<std::string> labels;
		std::ostringstream lablesStream;
		lablesStream.precision(3);
		for(int i = 0; i < 5; i++)
		{
			lablesStream << std::scientific << minValue + i * valueStride;
			labels.push_back(lablesStream.str());
			lablesStream.str("");
		}
		glm::vec2 titleAnchorPos = glm::vec2(mesh.vertices[1].Position.x, mesh.vertices[mesh.vertices.size() - 1].Position.y + 30.0f);
		glm::vec2 titlePosition = text.getTextPosition(title, titleAnchorPos, BOTTOM_RIGHT, 0.5f);
		text.RenderText(title, titlePosition.x, titlePosition.y, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
		float lableSpacing = (mesh.vertices[2].Position.y - mesh.vertices[0].Position.y);
#if 0
		if (glIsEnabled(GL_BLEND)) {
			std::cout << "Blending is enabled" << std::endl;
		}
		else {
			std::cout << "Blending is disabled" << std::endl;
		}
#endif
		for (int i = 0; i < 5; i++)
		{
			glm::vec2 anchorPos = glm::vec2(mesh.vertices[0].Position.x - 10.0f, mesh.vertices[0].Position.y + i * lableSpacing);
			glm::vec2 lablePosition = text.getTextPosition(labels[i], anchorPos, BOTTOM_RIGHT,0.5f);
			text.RenderText(labels[i], lablePosition.x, lablePosition.y, 0.5f, glm::vec3(0.3, 0.7f, 0.9f));
		}
	}

	void setColorRange(double minValue, double maxValue)
	{
		this->mesh.initial_minValue = minValue;
		this->mesh.initial_maxValue = maxValue;
	}

	void setMinMaxValues(double minValue, double maxValue)
	{
		this->minValue = minValue;
		this->maxValue = maxValue;
	}
};
