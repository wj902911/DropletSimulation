#pragma once

#include "shader.h"

shader::shader(const char* vertexPath, const char* fragmentPath)
    {
        // 1. retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        try 
        {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode   = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        }
        catch (std::ifstream::failure& e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }
        const char* vShaderCode = vertexCode.c_str();
        const char * fShaderCode = fragmentCode.c_str();
        // 2. compile shaders
        unsigned int vertex, fragment;
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");
        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

shader::shader(ShaderType type)
{
    const char* vShaderCode = nullptr;
    const char* fShaderCode = nullptr;
    switch (type)
    {
    case ShaderType::UNIFORM_COLOR_3D:
    {
        vShaderCode = 
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in vec3 aNormal;\n"

            "out vec3 FragPos;\n"
            "out vec3 Normal;\n"

            "uniform mat4 model;\n"
            "uniform mat4 view;\n"
            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
            "   FragPos = vec3(model * vec4(aPos, 1.0));\n"
            "   Normal = mat3(transpose(inverse(model))) * aNormal; \n"
            "}\0";

        fShaderCode = 
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec3 FragPos;\n"
            "in vec3 Normal;\n"

            "uniform vec3 objectColor;\n"
            "uniform vec3 lightColor;\n"
            "uniform vec3 direction;\n"

            "void main()\n"
            "{\n"
            "   float ambientStrength = 0.1;\n"
            "   vec3 ambient = ambientStrength * lightColor;\n"

            "   vec3 norm = normalize(Normal);\n"
            "   vec3 lightDir = normalize(-direction);\n"
            "   float diff = max(dot(norm, lightDir), 0.0);\n"
            "   vec3 diffuse = diff * lightColor;\n"

            "   vec3 result = (ambient + diffuse) * objectColor;\n"
            "   FragColor = vec4(result, 1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::UNIFORM_COLOR_3D_INSTANCE:
    {
        vShaderCode =
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in vec3 aNormal;\n"
            "layout (location = 2) in mat4 instanceModel;\n"
            "layout (location = 6) in vec3 instanceColor;\n"

            "out vec3 FragPos;\n"
            "out vec3 Normal;\n"
            "out vec3 objectColor;\n"

            "uniform mat4 view;\n"
            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * view * instanceModel * vec4(aPos, 1.0);\n"
            "   FragPos = vec3(instanceModel * vec4(aPos, 1.0));\n"
            "   Normal = mat3(transpose(inverse(instanceModel))) * aNormal; \n"
            "   objectColor = instanceColor;\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec3 FragPos;\n"
            "in vec3 Normal;\n"
            "in vec3 objectColor;\n"

            "uniform vec3 lightColor;\n"
            "uniform vec3 direction;\n"

            "void main()\n"
            "{\n"
            "   float ambientStrength = 0.1;\n"
            "   vec3 ambient = ambientStrength * lightColor;\n"

            "   vec3 norm = normalize(Normal);\n"
            "   vec3 lightDir = normalize(-direction);\n"
            "   float diff = max(dot(norm, lightDir), 0.0);\n"
            "   vec3 diffuse = diff * lightColor;\n"

            "   vec3 result = (ambient + diffuse) * objectColor;\n"
            "   FragColor = vec4(result, 1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::LIGHT:
    {
        vShaderCode = 
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"

            "uniform mat4 model;\n"
            "uniform mat4 view;\n"
            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
            "}\0";

        fShaderCode = 
            "#version 330 core\n"
            "out vec4 FragColor;\n"
            
            "void main()\n"
            "{\n"
            "    FragColor = vec4(1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::TEXT:
    {
        vShaderCode = 
            "#version 330 core\n"
            "layout (location = 0) in vec4 vertex;\n"
            "out vec2 TexCoords;\n"

            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);\n"
            "   TexCoords = vertex.zw;\n"
            "}\0";
        fShaderCode = 
            "#version 330 core\n"
			"in vec2 TexCoords;\n"
			"out vec4 color;\n"

			"uniform sampler2D text;\n"
			"uniform vec3 textColor;\n"

			"void main()\n"
			"{\n"
			"   vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);\n"
			"   color = vec4(textColor, 1.0) * sampled;\n"
			"}\n\0";
        break;
    }
    case ShaderType::UNIFORM_COLOR_2D:
    {
        vShaderCode = 
			"#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"

            "uniform mat4 model;\n"
            //"uniform mat4 view;\n"
            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            //"   gl_Position = projection * view * model * vec4(aPos, 0.0, 1.0);\n"
            "   gl_Position = projection * model * vec4(aPos, 1.0);\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
			"out vec4 FragColor;\n"

			"uniform vec3 objectColor;\n"

			"void main()\n"
			"{\n"
			"   FragColor = vec4(objectColor, 1.0);\n"
			"}\n\0";
        break;
    }
    case ShaderType::UNIFORM_COLOR_2D_WITHOUT_MODEL:
    {
        vShaderCode =
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"

            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * vec4(aPos, 1.0);\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "uniform vec3 objectColor;\n"

            "void main()\n"
            "{\n"
            "   FragColor = vec4(objectColor, 1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::UNIFORM_COLOR_2D_INSTANCE:
    {
        vShaderCode =
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in mat4 instanceModel;\n"
            "layout (location = 5) in vec3 instanceColor;\n"

            "out vec3 objectColor;\n"

            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * instanceModel * vec4(aPos, 1.0);\n"
            "   objectColor = instanceColor;\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec3 objectColor;\n"

            "void main()\n"
            "{\n"
            "   FragColor = vec4(objectColor, 1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::UNIFORM_COLOR_2D_INSTANCE_WITHBOOL:
    {
        vShaderCode =
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in mat4 instanceModel;\n"
            "layout (location = 5) in vec3 instanceColor;\n"

            "out vec3 objectColor;\n"

            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * instanceModel * vec4(aPos, 1.0);\n"
            "   objectColor = instanceColor;\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec3 objectColor;\n"

            "uniform vec3 boundaryColor;\n"
            "uniform bool isBoundary;\n"

            "void main()\n"
            "{\n"
            "   if(isBoundary)\n"
            "       FragColor = vec4(boundaryColor, 1.0);\n"
            "   else\n"
            "       FragColor = vec4(objectColor, 1.0);\n"
            "}\n\0";
        break;
    }
    case ShaderType::VERTEX_COLOR_2D:
    {
        vShaderCode =
            "#version 330 core\n"
            "layout (location = 0) in vec3 aPos;\n"
            "layout (location = 1) in vec3 aColor;\n"

            "out vec3 color;\n"

            "uniform mat4 projection;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = projection * vec4(aPos, 1.0);\n"
            "   color = aColor;\n"
            "}\0";

        fShaderCode =
            "#version 330 core\n"
            "out vec4 FragColor;\n"

            "in vec3 color;\n"

            "void main()\n"
            "{\n"
            "   FragColor = vec4(color, 1.0);\n"
            "}\n\0";
        break;
    }
    }
    

    unsigned int vertex, fragment;
    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void shader::use()
{
    glUseProgram(ID);
}

void shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); 
}

void shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value); 
}

void shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
}

void shader::setVec3(const std::string& name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void shader::setVec3(const std::string& name, glm::vec3 vec3, int count) const
{
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), count, glm::value_ptr(vec3));
}

void shader::setMat4(const std::string& name, glm::mat4 value, int count) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), count, GL_FALSE, glm::value_ptr(value));
}

void shader::checkCompileErrors(unsigned int shader, std::string type)
{
    int success;
    char infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}