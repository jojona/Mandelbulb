#ifndef KERNEL_H
#define KERNEL_H

#define GLM_FORCE_CUDA
#include <glm.hpp>
#include "constants.h"

void checkCUDAError(const char *msg);

__device__ float DE(const glm::vec3& pos);
__device__ float DESphere1(const glm::vec3& pos);
__device__ float DETetredon(const glm::vec3& pos);
__device__ float DEMandelbulb1(const glm::vec3& pos);
__device__ float DEMandelbulb2(const glm::vec3& pos);

__device__ bool BoundingSphere(const glm::vec3& dir, const glm::vec3& pos);
__device__ glm::vec3 PlaneFloor(const glm::vec3& dir, const glm::vec3& pos);

__device__ void color(uchar4* pixels, bool hit, unsigned int steps, glm::vec3 rayDir, glm::vec3 rayOrigin, glm::vec3 position, int index);

extern "C" void launchKernel(uchar4* pixels, unsigned int width, unsigned int height, float focalLength, glm::mat3 rot, glm::vec3 pos, LOD l);

__global__ void primaryRay(unsigned char*, float*, unsigned int w, unsigned int h, float focalLength, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);
__global__ void secondaryRay(uchar4* pixels, unsigned char*, float*, unsigned int w, unsigned int h, float focalLength, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rot, glm::vec3 pos);
__global__ void singleRay(uchar4* pixels, unsigned int width, unsigned int height, float focalLength, glm::mat3 rot, glm::vec3 pos);
__global__ void setUp(LOD l);//(float epsilon, unsigned int fractalIterations, unsigned int raymarchsteps, unsigned int amountPrimary);


__device__ glm::vec3 light(const glm::vec3& lightPos, const glm::vec3& lightColor, const glm::vec3& position, const glm::vec3& normal, bool calcShadow);
__device__ bool shadow(const glm::vec3& lightPos, const glm::vec3& position);



template<typename T>
__device__ inline float length(const T& x) {
	return sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
}

template<typename T>
__device__ inline float dot(const T& x, const T& y) {
	return x.x*y.x + x.y*y.y + x.z*y.z;
}

template<typename T>
__device__ inline float dot(const T& x) {
	return x.x*x.x + x.y*x.y + x.z*x.z;
}

template<typename T>
__device__ inline T cross(const T& x, const T& y) {
	return T(x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.y * x.x);
}

template<typename T>
__device__ inline T normalize(const T& x) {
	return T(x.x, x.y, x.z) / length(x);
}


#endif KERNEL_H