#pragma once

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>
#include <map>

#include <freetype2/ft2build.h>
#include FT_FREETYPE_H 

#include "shader.h"

/// Holds all state information relevant to a character as loaded using FreeType
struct Character {
    GLuint TextureID;   // ID handle of the glyph texture
    glm::ivec2 Size;    // Size of glyph
    glm::ivec2 Bearing;  // Offset from baseline to left/top of glyph
    GLuint Advance;    // Horizontal offset to advance to next glyph
};

enum anchorPosType
{
	TOP_LEFT,
	TOP_CENTER,
	TOP_RIGHT,
	//CENTER_LEFT,
	//CENTER_CENTER,
	//CENTER_RIGHT,
	BOTTOM_LEFT,
	BOTTOM_CENTER,
	BOTTOM_RIGHT
};

class OpenGLText
{
public:
    shader textShader;
	std::map<GLchar, Character> Characters;
	GLuint VAOText = 0, VBOText = 0;

	OpenGLText()=default;

    OpenGLText(GLfloat window_width, GLfloat window_height)
    {
		textShader= shader(ShaderType::TEXT);
		textShader.use();
		glm::mat4 textProjection = glm::ortho(0.0f, window_width, 0.0f, window_height);
		textShader.setMat4("projection", textProjection);

        FT_Library ft;
        if (FT_Init_FreeType(&ft))
            std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;

        // Load font as face
        FT_Face face;
        if (FT_New_Face(ft, "resource/fonts/arial.ttf", 0, &face))
            std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;

        // Set size to load glyphs as
        FT_Set_Pixel_Sizes(face, 0, 48);

        // Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Load first 128 characters of ASCII set
        for (GLubyte c = 0; c < 128; c++)
		{
			// Load character glyph 
			if (FT_Load_Char(face, c, FT_LOAD_RENDER))
			{
				std::cout << "ERROR::FREETYPE: Failed to load Glyph" << std::endl;
				continue;
			}
			// Generate texture
			GLuint texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RED,
				face->glyph->bitmap.width,
				face->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE,
				face->glyph->bitmap.buffer
			);
			// Set texture options
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			// Now store character for later use
			Character character = {
				texture,
				glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
				glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
				face->glyph->advance.x
			};
			Characters.insert(std::pair<GLchar, Character>(c, character));
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// Destroy FreeType once we're finished
		FT_Done_Face(face);
		FT_Done_FreeType(ft);

		// Configure VAO/VBO for texture quads
		glGenVertexArrays(1, &VAOText);
		glGenBuffers(1, &VBOText);
		glBindVertexArray(VAOText);
		glBindBuffer(GL_ARRAY_BUFFER, VBOText);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
    }

	~OpenGLText()
	{
		glDeleteVertexArrays(1, &VAOText);
		glDeleteBuffers(1, &VBOText);
	}

	void RenderText(std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color)
	{
		textShader.use();
		textShader.setVec3("textColor", color);
		glActiveTexture(GL_TEXTURE0);
		glBindVertexArray(VAOText);

		// Iterate through all characters
		std::string::const_iterator c;
		for (c = text.begin(); c != text.end(); c++)
		{
			Character ch = Characters[*c];

			GLfloat xpos = x + ch.Bearing.x * scale;
			GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

			GLfloat w = ch.Size.x * scale;
			GLfloat h = ch.Size.y * scale;
			// Update VBO for each character
			GLfloat vertices[6][4] = {
				{ xpos,     ypos + h,   0.0, 0.0 },
				{ xpos,     ypos,       0.0, 1.0 },
				{ xpos + w, ypos,       1.0, 1.0 },

				{ xpos,     ypos + h,   0.0, 0.0 },
				{ xpos + w, ypos,       1.0, 1.0 },
				{ xpos + w, ypos + h,   1.0, 0.0 }
			};
			// Render glyph texture over quad
			glBindTexture(GL_TEXTURE_2D, ch.TextureID);
			// Update content of VBO memory
			glBindBuffer(GL_ARRAY_BUFFER, VBOText);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			// Render quad
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
			x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64)
		}
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GLfloat getTextLength(std::string text, GLfloat scale)
	{
		GLfloat length = 0.0f;
		for (char c : text)
		{
			Character ch = Characters[c];
			length += (ch.Advance >> 6) * scale;
		}
		return length;
	}

	GLfloat getTextMaxBearingY(std::string text, GLfloat scale)
	{
		GLfloat maxBearingY = 0.0f;
		for (char c : text)
		{
			Character ch = Characters[c];
			if (ch.Bearing.y > maxBearingY)
				maxBearingY = ch.Bearing.y;
		}
		return maxBearingY * scale;
	}

	glm::vec2 getTextPosition(std::string text, glm::vec2 anchorPos, anchorPosType type, GLfloat scale)
	{
		glm::vec2 textPos = glm::vec2(0.0f, 0.0f);
		GLfloat textLength = 0;
		GLfloat textMaxBearingY = 0;

		switch (type)
		{
		case TOP_LEFT:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textPos = glm::vec2(anchorPos.x, anchorPos.y - textMaxBearingY);
			break;
		case TOP_CENTER:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength / 2, anchorPos.y - textMaxBearingY);
			break;
		case TOP_RIGHT:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength, anchorPos.y - textMaxBearingY);
			break;
		/*
		case CENTER_LEFT:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textPos = glm::vec2(anchorPos.x, anchorPos.y - textMaxBearingY / 2);
			break;
		case CENTER_CENTER:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength / 2, anchorPos.y - textMaxBearingY / 2);
			break;
		case CENTER_RIGHT:
			textMaxBearingY = getTextMaxBearingY(text, scale);
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength, anchorPos.y - textMaxBearingY / 2);
			break;
		*/
		
		case BOTTOM_LEFT:
			textPos = glm::vec2(anchorPos.x, anchorPos.y);
			break;
		case BOTTOM_CENTER:
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength / 2, anchorPos.y);
			break;
		case BOTTOM_RIGHT:
			textLength = getTextLength(text, scale);
			textPos = glm::vec2(anchorPos.x - textLength, anchorPos.y);
			break;
		}

		return textPos;
	}

	void RenderMultilineText(std::vector<std::string> texts, glm::vec2 anchorPos, anchorPosType type, GLfloat lineSpacing, GLfloat scale, glm::vec3 color)
	{
		glm::vec2 anchorPosPerline = anchorPos;
		for (std::string text : texts)
		{
			glm::vec2 textPos = getTextPosition(text, anchorPosPerline, type, scale);
			RenderText(text, textPos.x, textPos.y, scale, color);
			GLfloat textMaxBearingY = getTextMaxBearingY(text, scale);
			if (type == TOP_LEFT || type == TOP_CENTER || type == TOP_RIGHT)
				anchorPosPerline.y -= textMaxBearingY + lineSpacing;
			else
				anchorPosPerline.y += textMaxBearingY + lineSpacing;
		}
	}
};