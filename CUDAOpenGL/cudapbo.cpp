// includes
#include <windows.h>
#include <stdio.h>

#include <glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "kernel.h"
#include "constants.h"
#include "cudapbo.h"

extern bool* keyPressed;

GLuint pbo = 0;
GLuint textureID = 0;
struct cudaGraphicsResource * cuda_pbo;

void createPBO(GLuint* pbo) {
	if (pbo) {


		// set up vertex data parameter
		int num_texels = window_width * window_height;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof(GLubyte) * num_values;

		// Generate a buffer ID called a PBO (Pixel Buffer Object)
		glGenBuffers(1, pbo);
		// Make this the current UNPACK buffer (OpenGL is state-based)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		// Allocate data for the buffer. 4-channel 8-bit image
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW);

		// Register buffer
		cudaGraphicsGLRegisterBuffer(&cuda_pbo, *pbo, cudaGraphicsMapFlagsWriteDiscard);
	}
}

void deletePBO(GLuint* pbo) {
	if (pbo) {
		// Unregister this buffer object with CUDA
		cudaGraphicsUnregisterResource(cuda_pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = NULL;
	}
}

void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y) {
	// Enable Texturing
	glEnable(GL_TEXTURE_2D);

	// Generate a texture identifier
	glGenTextures(1, textureID);

	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_2D, *textureID);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Create texture data (4-component unsigned byte)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = NULL;
}

void cleanupCuda() {
	delete[] keyPressed;

	if (pbo) deletePBO(&pbo);
	if (textureID) deleteTexture(&textureID);
}

// Run the Cuda part of the computation
void runCuda(glm::mat3 rot, glm::vec3 campos, float focalLength, LOD l) {
	uchar4 *dptr = NULL;

	// Map the buffer object
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_pbo, 0);

	// Get Address for kernel 
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo);

	launchKernel(dptr, window_width, window_height, focalLength, rot, campos, l);

	// Unmap the buffer object
	cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

}

void initCuda(int argc, char** argv) {
	cudaGLSetGLDevice(0);

	createPBO(&pbo);
	createTexture(&textureID, window_width, window_height);
}