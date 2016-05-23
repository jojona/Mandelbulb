#ifndef CUDAPBO_H
#define CUDAPBO_H

#include <glew.h>
#include "glm.hpp"

void createPBO(GLuint*);
void deletePBO(GLuint* pbo);
void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);

void initCuda(int argc, char** argv);
void cleanupCuda();
void runCuda(glm::mat3 rot, glm::vec3 campos);

#endif CUDAPBO_H