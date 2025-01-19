#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h> // include glad to get all the required OpenGL headers

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
  
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

enum ShaderType
{
	LIGHT,
    UNIFORM_COLOR_3D,
    UNIFORM_COLOR_3D_INSTANCE,
    TEXT,
    UNIFORM_COLOR_2D,
    UNIFORM_COLOR_2D_WITHOUT_MODEL,
    UNIFORM_COLOR_2D_INSTANCE,
    UNIFORM_COLOR_2D_INSTANCE_WITHBOOL,
    VERTEX_COLOR_2D,
};
  

class shader
{
public:
    // the program ID
    unsigned int ID = 0;
  
    // constructor reads and builds the shader
    shader()=default;
    shader(const char* vertexPath, const char* fragmentPath);
    shader(ShaderType type);
    // use/activate the shader
    void use();
    // utility uniform functions
    void setBool(const std::string &name, bool value) const;  
    void setInt(const std::string &name, int value) const;   
    void setFloat(const std::string &name, float value) const;
    void setVec3(const std::string& name, float x, float y, float z) const;
    void setVec3(const std::string& name, glm::vec3 vec3, int count = 1) const;
    void setMat4(const std::string& name, glm::mat4 value, int count = 1) const;
private:
    void checkCompileErrors(unsigned int shader, std::string type);
};
  
#endif