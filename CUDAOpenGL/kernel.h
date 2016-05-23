#ifndef KERNEL_H
#define KERNEL_H

#define GLM_FORCE_CUDA
#include <glm.hpp>

void checkCUDAError(const char *msg);

__global__ void kernel(uchar4* pos, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 campos);
extern "C" void launch_kernel(uchar4* pos, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 campos);

__device__ void pixel(uchar4* pos, unsigned int index, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 campos);

__device__ void Spectrumbackground(uchar4& pos, int x, int y, int width, int height);

__device__ float RayMarching(glm::vec3 pos, glm::vec3 dir);

__device__ float DESphere1(glm::vec3);
__device__ float DETetredon(glm::vec3 pos);
__device__ float DEMandelbulb1(glm::vec3 p);
__device__ float DEMandelbulb2(glm::vec3 pos);

__device__ bool BoundingSphere(glm::vec3 dir, glm::vec3 pos);

__device__ bool PlaneFloor(glm::vec3 dir, glm::vec3 pos);

extern "C" void launchKernel2(uchar4* pixels, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 pos);
__global__ void primaryRay(unsigned char*, float*, unsigned int, unsigned int, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);
__global__ void secondaryRay(uchar4* pixel, unsigned char*, float*, unsigned int, unsigned int, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);

#endif KERNEL_H