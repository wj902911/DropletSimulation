#pragma once

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shader.h>

#include <vector>
#include <string>

using namespace std;

struct Vertex 
{
public:
    glm::vec3 Position = glm::vec3(0.0f, 0.0f, 0.0f);

    Vertex(glm::vec3 pos) : Position(pos) {}

    //virtual bool hasNormal() { return false; }
    //virtual bool hasTexture() { return false; }
};

struct Vertex_with_color : public Vertex
{
    public:
	glm::vec3 Color = glm::vec3(0.0f, 0.0f, 0.0f);

	Vertex_with_color(glm::vec3 pos, glm::vec3 color) : Vertex(pos), Color(color) {}
};

struct Vertex_with_normal: public Vertex 
{
public:
    glm::vec3 Normal = glm::vec3(0.0f, 0.0f, 0.0f);

    Vertex_with_normal(glm::vec3 pos, glm::vec3 norm) : Vertex(pos), Normal(norm) {}

    //bool hasNormal() override { return true; } 
    //bool hasTexture() override { return false; }
};

struct Vertex_with_texture: public Vertex 
{
public:
    glm::vec2 TexCoords = glm::vec2(0.0f, 0.0f);

    Vertex_with_texture(glm::vec3 pos, glm::vec2 tex) : Vertex(pos), TexCoords(tex) {}

    //bool hasNormal() override { return false; }
    //bool hasTexture() override { return true; }
};

struct Vertex_with_normal_and_texture: public Vertex 
{
public:
    glm::vec3 Normal = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec2 TexCoords = glm::vec2(0.0f, 0.0f);

	Vertex_with_normal_and_texture(glm::vec3 pos, glm::vec3 norm, glm::vec2 tex) : Vertex(pos), Normal(norm), TexCoords(tex) {}

	//bool hasNormal() override { return true; }
	//bool hasTexture() override { return true; }
};

struct Texture {
    unsigned int id;
    string type;
    string path;
};

//template <class T>
class Mesh {
public:
    // mesh Data
    vector<glm::vec3> vertices;
    vector<unsigned int> indices;
    vector<glm::mat4> modelMatrices;
    vector<glm::vec3> colors;
    //vector<Texture>      textures;
    unsigned int VAO = 0;
    unsigned int VBO = 0, EBO = 0, modelVBO = 0;

    Mesh() = default;
    // constructor
    Mesh(vector<glm::vec3> vertices, vector<unsigned int> indices, vector<glm::mat4> mds, vector<glm::vec3> cls)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->modelMatrices = mds;
        this->colors = cls;
        //this->textures = textures;

        // now that we have all the required data, set the vertex buffers and its attribute pointers.
        setupMesh();
    }

    virtual ~Mesh()
    {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteBuffers(1, &modelVBO);
    }

    // render the mesh
    virtual void Draw(const glm::mat4& projection = glm::mat4(1.0f),
                      const glm::mat4& view = glm::mat4(1.0f)) = 0;
    /*
    void Draw(shader& shader)
    {
        // draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // always good practice to set everything back to defaults once configured.
        //glActiveTexture(GL_TEXTURE0);
    }
    */
    

    // render data 
    // initializes all the buffer objects/arrays
    virtual void setupMesh() = 0;
    /*
    virtual void setupMesh()
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    }
    */
    
};

class DropletMesh_3D : public Mesh
{
public:
    vector<glm::vec3> normals;
    float radius;
    shader Shader = shader(UNIFORM_COLOR_3D_INSTANCE);
    GLuint colorVBO = 0;

	DropletMesh_3D(float r, int stacks, int slices, const std::vector<glm::vec3>& cls, const std::vector<glm::mat4>& mds)
    {
        radius = r;
        vertices.clear();
        indices.clear();
        int numVertices = 2.0 + slices * (stacks - 1);
        vertices.reserve(numVertices);

        vertices.push_back(glm::vec3(0.0f, 0.0f, radius));
        normals.push_back(glm::vec3(0.0f, 0.0f, 1.0f));

        for (int i = 1; i < stacks; ++i)
        {
			float phi = glm::pi<float>() * i / stacks;
            for (int j = 0; j < slices; ++j)
            {
                float theta = 2.0f * glm::pi<float>() * j / slices;

                float x = radius * sinf(phi) * cosf(theta);
                float y = radius * sinf(phi) * sinf(theta);
                float z = radius * cosf(phi);

                float nx = x / radius;
                float ny = y / radius;
                float nz = z / radius;

                vertices.push_back(glm::vec3(x, y, z));
                normals.push_back(glm::vec3(nx, ny, nz));
            }
		}

        vertices.push_back(glm::vec3(0.0f, 0.0f, -radius));
        normals.push_back(glm::vec3(0.0f, 0.0f, -1.0f));

        indices.reserve(3 * slices + 6 * slices * (stacks - 2) + 3 * slices);

        for (int i = 0; i < slices; ++i)
        {
            indices.push_back(0);
            indices.push_back(i + 1);
            indices.push_back((i + 1) % slices + 1);
        }
        for (int i = 0; i < stacks - 2; ++i)
        {
            for (int j = 0; j < slices; ++j)
            {
                indices.push_back(1 + i * slices + j);
                indices.push_back(1 + i * slices + (j + 1) % slices);
                indices.push_back(1 + (i + 1) * slices + j);

                indices.push_back(1 + (i + 1) * slices + j);
                indices.push_back(1 + i * slices + (j + 1) % slices);
                indices.push_back(1 + (i + 1) * slices + (j + 1) % slices);
            }
        }
        for (int i = 0; i < slices; ++i)
        {
            indices.push_back(1 + (stacks - 2) * slices + i);
            indices.push_back(1 + (stacks - 2) * slices + (i + 1) % slices);
            indices.push_back(numVertices - 1);
        }

        colors=cls;
        modelMatrices=mds;

        setupMesh();
    }

    void Draw(const glm::mat4& projection = glm::mat4(1.0f),
              const glm::mat4& view = glm::mat4(1.0f)) override
	{
		Shader.use();
        Shader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
        Shader.setVec3("direction", 0.0f, 0.0f, -1.0f);
        Shader.setMat4("projection", projection);
        Shader.setMat4("view", view);
        glBindVertexArray(VAO);

        
        // Update model matrix VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0]);

        // Update color VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.size() * sizeof(glm::vec3), &colors[0]);

        glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());
        
        /*
        for (int i = 0; i < colors.size(); ++i)
		{
			Shader.setVec3("objectColor", colors[i]);
            //glm::mat4 model = glm::mat4(1.0f);
            //model = glm::translate(model, dropletPos[i]);
            //model = glm::scale(model, glm::vec3(currentRadius[i] / radius));
            Shader.setMat4("model", modelMatrices[i]);
            glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
		}
        */

        glBindVertexArray(0);
    }

private:
    void setupMesh() override
	{
		// create buffers/arrays
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);
        glGenBuffers(1, &modelVBO);
        glGenBuffers(1, &colorVBO);

		glBindVertexArray(VAO);
		// load data into vertex buffers
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		// A great thing about structs is that their memory layout is sequential for all its items.
		// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
		// again translates to 3/2 floats which translates to a byte array.
        glBufferData(GL_ARRAY_BUFFER, (vertices.size() + normals.size()) * sizeof(glm::vec3), nullptr, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(glm::vec3), &vertices[0]);
        glBufferSubData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), normals.size() * sizeof(glm::vec3), &normals[0]);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		// set the vertex attribute pointers
		// vertex Positions
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

		// vertex normals
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)(vertices.size() * sizeof(glm::vec3)));

        
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW); 

        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(2 + i);
            glVertexAttribDivisor(2 + i, 1); // Update every instance
        }

        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);

        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(6);
        glVertexAttribDivisor(6, 1); // Update every instance
        
        glBindVertexArray(0);
    }
};

class DropletMesh_2D : public Mesh
{
public:
    float radius;
    shader Shader = shader(UNIFORM_COLOR_2D_INSTANCE_WITHBOOL);
    GLuint modelVBO = 0, colorVBO = 0, boundaryVAO = 0, boundaryEBO = 0;
    vector<unsigned int> boundaryIndices;
    glm::vec3 boundaryColor = glm::vec3(0.0f, 0.0f, 0.0f);
    //using Mesh::setupMesh;

    DropletMesh_2D(float r, int slices, const std::vector<glm::vec3>& cls, const std::vector<glm::mat4>& mds)
    {
        radius = r;
        vertices.clear();
        indices.clear();

        int numVertices = slices + 2;

        vertices.reserve(numVertices);

        vertices.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
        for (int i = 0; i < slices; ++i)
        {
			float theta = 2.0f * glm::pi<float>() * i / slices;
			float x = radius * cosf(theta);
			float y = radius * sinf(theta);
			vertices.push_back(glm::vec3(x, y, 0.0f));
        }

        indices.reserve(3 * slices);

        for (int i = 0; i < slices; ++i)
		{
			indices.push_back(0);
			indices.push_back(i + 1);
			indices.push_back((i + 1) % slices + 1);
		}

        boundaryIndices.reserve(2 * slices);
        for (int i = 0; i < slices; ++i)
		{
			boundaryIndices.push_back(i + 1);
			boundaryIndices.push_back((i + 1) % slices + 1);
		}

        colors = cls;
        modelMatrices = mds;

        setupMesh();
    }

    void Draw(const glm::mat4& projection = glm::mat4(1.0f),
              const glm::mat4& view = glm::mat4(1.0f)) override
    {
        Shader.use();
        Shader.setMat4("projection", projection);
        glBindVertexArray(VAO);

        // Update model matrix VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0]);

        // Update color VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.size() * sizeof(glm::vec3), &colors[0]);

        Shader.setBool("isBoundary", false);
        glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());
        
        glBindVertexArray(boundaryVAO);

        Shader.setBool("isBoundary", true);
        Shader.setVec3("boundaryColor", boundaryColor);

        glDrawElementsInstanced(GL_LINES, static_cast<unsigned int>(boundaryIndices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());


        glBindVertexArray(0);
    }

    void setupMesh() override
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glGenBuffers(1, &modelVBO);
        glGenBuffers(1, &colorVBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // A great thing about structs is that their memory layout is sequential for all its items.
        // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
        // again translates to 3/2 floats which translates to a byte array.
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);

        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(5);
        glVertexAttribDivisor(5, 1); // Update every instance

        glBindVertexArray(0);

        glGenVertexArrays(1, &boundaryVAO);
        glGenBuffers(1, &boundaryEBO);

        glBindVertexArray(boundaryVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaryEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundaryIndices.size() * sizeof(unsigned int), &boundaryIndices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);

        // Model matrix attributes
        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindVertexArray(0);
    }

    void updateBuffers()
    {

        // Reallocate model matrix VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

        // Reallocate color VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);
    }
};

class SpringMesh_2D : public Mesh
{
public:
    float width, length;
    shader Shader = shader(UNIFORM_COLOR_2D_INSTANCE_WITHBOOL);
    GLuint modelVBO = 0, colorVBO = 0, boundaryVAO = 0, boundaryEBO = 0;
    vector<unsigned int> boundaryIndices;
    glm::vec3 boundaryColor = glm::vec3(1.0f, 1.0f, 1.0f);

    SpringMesh_2D() = default;

    SpringMesh_2D(float wth, float lth, const std::vector<glm::vec3>& cls, const std::vector<glm::mat4>& mds)
    : width(wth), length(lth)
    {
        vertices.clear();
        indices.clear();

        vertices.reserve(4);
        vertices.push_back(glm::vec3(0.0f, -width / 2.0f, 0.0f));
        vertices.push_back(glm::vec3(length, -width / 2.0f, 0.0f));
        vertices.push_back(glm::vec3(0.0f, width / 2.0f, 0.0f));
        vertices.push_back(glm::vec3(length, width / 2.0f, 0.0f));

        indices.reserve(6);
        indices.push_back(0);
        indices.push_back(1);
        indices.push_back(2);
        indices.push_back(2);
        indices.push_back(1);
        indices.push_back(3);

        boundaryIndices.reserve(8);
        boundaryIndices.push_back(0);
        boundaryIndices.push_back(1);
        boundaryIndices.push_back(1);
        boundaryIndices.push_back(3);
        boundaryIndices.push_back(3);
        boundaryIndices.push_back(2);
        boundaryIndices.push_back(2);
        boundaryIndices.push_back(0);

        colors = cls;
        modelMatrices = mds;

        setupMesh();
    }

    void Draw(const glm::mat4& projection = glm::mat4(1.0f),
        const glm::mat4& view = glm::mat4(1.0f)) override
    {
        Shader.use();
        Shader.setMat4("projection", projection);
        glBindVertexArray(VAO);

        // Update model matrix VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0]);

        // Update color VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.size() * sizeof(glm::vec3), &colors[0]);

        Shader.setBool("isBoundary", false);
        glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());

        
        glBindVertexArray(boundaryVAO);

        Shader.setBool("isBoundary", true);
        Shader.setVec3("boundaryColor", boundaryColor);

        //glDrawElementsInstanced(GL_LINES, static_cast<unsigned int>(boundaryIndices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());
        
        
        glBindVertexArray(0);
    }

    void setupMesh() override
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glGenBuffers(1, &modelVBO);
        glGenBuffers(1, &colorVBO);
        //glGenBuffers(1, &boundaryEBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
        
        // load data into element buffer for boundary lines
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaryEBO);
        //glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundaryIndices.size() * sizeof(unsigned int), &boundaryIndices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);
        
        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);

        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(5);
        glVertexAttribDivisor(5, 1); // Update every instance

        glBindVertexArray(0);

        
        glGenVertexArrays(1, &boundaryVAO);
        glGenBuffers(1, &boundaryEBO);

        glBindVertexArray(boundaryVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaryEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundaryIndices.size() * sizeof(unsigned int), &boundaryIndices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);

        // Model matrix attributes
        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindVertexArray(0);
        
    }

    void updateBuffers()
    {

        // Reallocate model matrix VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

        // Reallocate color VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);
    }
};

class FloorMesh_2D : public Mesh
{
public:
    shader Shader = shader(UNIFORM_COLOR_2D_INSTANCE_WITHBOOL);
    float width, length;
    GLuint modelVBO = 0, colorVBO = 0, boundaryVAO = 0, boundaryEBO = 0;
    vector<unsigned int> boundaryIndices;
    glm::vec3 boundaryColor = glm::vec3(1.0f, 1.0f, 1.0f);

    FloorMesh_2D() = default;
    FloorMesh_2D(float wth, float lth, const std::vector<glm::mat4>& mds)
    {
        width = wth;
		length = lth;
		vertices.clear();
		indices.clear();

		vertices.reserve(4);
		vertices.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
		vertices.push_back(glm::vec3(length, 0.0f, 0.0f));
		vertices.push_back(glm::vec3(0.0f, width, 0.0f));
		vertices.push_back(glm::vec3(length, width, 0.0f));

		indices.reserve(6);
		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);
		indices.push_back(2);
		indices.push_back(1);
		indices.push_back(3);

		boundaryIndices.reserve(8);
		boundaryIndices.push_back(0);
		boundaryIndices.push_back(1);
		boundaryIndices.push_back(1);
		boundaryIndices.push_back(3);
		boundaryIndices.push_back(3);
		boundaryIndices.push_back(2);
		boundaryIndices.push_back(2);
		boundaryIndices.push_back(0);

        colors = std::vector<glm::vec3>(mds.size(), glm::vec3(0.3f, 0.3f, 0.3f));
		modelMatrices = mds;

		setupMesh();
    }

    void Draw(const glm::mat4& projection = glm::mat4(1.0f),
        const glm::mat4& view = glm::mat4(1.0f)) override
    {
        Shader.use();
        Shader.setMat4("projection", projection);
        glBindVertexArray(VAO);

        // Update model matrix VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0]);

        // Update color VBO with new data
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.size() * sizeof(glm::vec3), &colors[0]);

        Shader.setBool("isBoundary", false);
        glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());


        glBindVertexArray(boundaryVAO);

        Shader.setBool("isBoundary", true);
        Shader.setVec3("boundaryColor", boundaryColor);

        glDrawElementsInstanced(GL_LINES, static_cast<unsigned int>(boundaryIndices.size()), GL_UNSIGNED_INT, 0, modelMatrices.size());


        glBindVertexArray(0);
    }

    void setupMesh() override
    {
        // create buffers/arrays
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glGenBuffers(1, &modelVBO);
        glGenBuffers(1, &colorVBO);
        //glGenBuffers(1, &boundaryEBO);

        glBindVertexArray(VAO);
        // load data into vertex buffers
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // load data into element buffer for boundary lines
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaryEBO);
        //glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundaryIndices.size() * sizeof(unsigned int), &boundaryIndices[0], GL_STATIC_DRAW);

        // set the vertex attribute pointers
        // vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);

        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(5);
        glVertexAttribDivisor(5, 1); // Update every instance

        glBindVertexArray(0);


        glGenVertexArrays(1, &boundaryVAO);
        glGenBuffers(1, &boundaryEBO);

        glBindVertexArray(boundaryVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaryEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundaryIndices.size() * sizeof(unsigned int), &boundaryIndices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);

        // Model matrix attributes
        for (unsigned int i = 0; i < 4; i++) {
            glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
            glEnableVertexAttribArray(1 + i);
            glVertexAttribDivisor(1 + i, 1); // Update every instance
        }

        glBindVertexArray(0);
    }

    void updateBuffers()
    {

        // Reallocate model matrix VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO);
        glBufferData(GL_ARRAY_BUFFER, modelMatrices.size() * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

        // Reallocate color VBO with the new size
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_DYNAMIC_DRAW);
    }
};