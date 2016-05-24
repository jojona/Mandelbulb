#ifndef KERNEL_H
#define KERNEL_H

#define GLM_FORCE_CUDA
#include <glm.hpp>

void checkCUDAError(const char *msg);

__device__ float DE(glm::vec3 pos);
__device__ float DESphere1(glm::vec3);
__device__ float DETetredon(glm::vec3 pos);
__device__ float DEMandelbulb1(glm::vec3 p);
__device__ float DEMandelbulb2(glm::vec3 pos);

__device__ bool BoundingSphere(glm::vec3 dir, glm::vec3 pos);
__device__ bool PlaneFloor(glm::vec3 dir, glm::vec3 pos);

__device__ void color(uchar4* pixels, bool hit, unsigned int steps, glm::vec3 dir, glm::vec3 pos, unsigned int index);

extern "C" void launchKernel(uchar4* pixels, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 pos);

__global__ void primaryRay(unsigned char*, float*, unsigned int w, unsigned int h, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);
__global__ void secondaryRay(uchar4* pixels, unsigned char*, float*, unsigned int w, unsigned int h, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);
__global__ void singleRay(uchar4* pixels, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 pos);
__global__ void setUp(float epsilon, unsigned int fractalIterations, unsigned int raymarchsteps, unsigned int amountPrimary);
#endif KERNEL_H