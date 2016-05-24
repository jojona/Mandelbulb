#include <windows.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "kernel.h"
//#include "constants.h"

__device__ float EpsilonRaymarch = 0;
__device__ unsigned int MaxRaymarchSteps = 0;
__device__ unsigned int FractalIterations = 0;
__device__ bool PrimaryRays = true;
__device__ unsigned int PrimarySize = 0;

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
	}
}

__device__ float DE(glm::vec3 pos) {
	return DEMandelbulb2(pos);
	//return DEMandelbulb1(pos);
	//return DETetredon(pos);
	//return DESphere1(pos);
}

/*
 * Distance Estiamtor for a Sphere in origo with radius 3
 */
__device__ float DESphere1(glm::vec3 pos) {
	float distance = glm::length(pos);
	if (distance > 3) {
		distance = distance - 3;
	} else {
		distance = 0;
	}
	return distance;
}

/*
 * Distance estimator for a Tetredon.
 */
__device__ float DETetredon(glm::vec3 z) {
	float Scale = 2.f;

	glm::vec3 a1(1, 1, 1);
	glm::vec3 a2(-1, -1, 1);
	glm::vec3 a3(1, -1, -1);
	glm::vec3 a4(-1, 1, -1);
	glm::vec3 c;
	int n = 0;
	float dist, d;
	while (n < FractalIterations) {
		c = a1; dist = glm::length(z - a1);
		d = glm::length(z - a2); if (d < dist) { c = a2; dist = d; }
		d = glm::length(z - a3); if (d < dist) { c = a3; dist = d; }
		d = glm::length(z - a4); if (d < dist) { c = a4; dist = d; }
		z = Scale*z - c*(Scale - 1.0f);
		n++;
	}

	return glm::length(z) * pow(Scale, float(-n));
}

/*
* Distance estimator for a Mandelbulb. Version 1
*/
__device__ float DEMandelbulb1(glm::vec3 p) {
	glm::vec3 z = p;
	float r = 0.0f;
	float dr = 1.0f;
	for (int i = 0; i < FractalIterations; ++i) {
		r = glm::length(z);
		if (r > 8.0f) break;

		float theta = glm::acos(z.y / r);
		float phi = glm::atan(z.x, z.z);
		float r7 = r*r*r*r*r*r*r; // glm::pow(r, 7.0f);

		dr = r7 * 8.0f * dr + 1.0f;

		float zr = r7*r;
		theta = theta*8.0f;
		phi = phi * 8.0f;

		z = p + zr*glm::vec3(glm::sin(phi)*glm::sin(theta), glm::cos(theta), glm::sin(theta) * glm::cos(phi));
	}
	return 0.5f*glm::log(r)*r / dr;
}

/*
 * Distance estimator for a Mandelbulb. Version 2
 */
__device__ float DEMandelbulb2(glm::vec3 pos) {
	glm::vec3 zz = pos;
	float m = glm::dot(zz, zz);

	float dz = 1.0f;

	for (int i = 0; i < FractalIterations; ++i) {
		float m2 = m*m;
		float m4 = m2*m2;
		dz = 8.0f*glm::sqrt(m4*m2*m)*dz + 1.0f;

		float x = zz.x; float x2 = zz.x*zz.x; float x4 = x2*x2;
		float y = zz.y; float y2 = zz.y* zz.y; float y4 = y2*y2;
		float z = zz.z; float z2 = zz.z*zz.z; float z4 = z2*z2;

		float k3 = x2 + z2;
		float k2 = 1.f / glm::sqrt(k3*k3*k3*k3*k3*k3*k3);
		float k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
		float k4 = x2 - y2 + z2;

		zz.x = pos.x + 64.0f*x*y*z*(x2 - z2)*k4*(x4 - 6.0f*x2*z2 + z4)*k1*k2;
		zz.y = pos.y + -16.0f*y2*k3*k4*k4 + k1*k1;
		zz.z = pos.z + -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2;

		m = glm::dot(zz, zz);
		if (m > 1000.0f)
			break;
	}

	return 0.25f*glm::log(m)*glm::sqrt(m) / dz;
}

__device__ bool BoundingSphere(glm::vec3 dir, glm::vec3 pos) {
	return !(glm::length(glm::cross(dir, -pos)) > 1.2f);
}

__device__ bool PlaneFloor(glm::vec3 dir, glm::vec3 pos) {
	float denom = glm::dot(glm::vec3(0, 1, 0), dir);
	if (denom > 0.0001f) // Only visible from above
	{
		float t = glm::dot(glm::vec3(0, 1.1f, 0) - pos, (glm::vec3(0, 1, 0))) / denom;
		if (t >= 0) {
			float distanceFromOrigo = glm::length(pos + t * dir);
			if (distanceFromOrigo > 3 && distanceFromOrigo < 7) {
				return true;
			}
		}
	}
	return false;
}

__device__ void color(uchar4* pixels, bool hit, unsigned int steps, glm::vec3 direction, glm::vec3 pos, unsigned int index) {
	// Draw color to pixels
	if (hit) {
		float color = MaxRaymarchSteps - steps;
		float maxColor = MaxRaymarchSteps;
		//float color = 1.5f + distance - glm::length(position);
		//float maxColor = 1.5f;
		//float color = glm::length(pos) -0.65f;
		//float maxColor = 0.5f;

		//float color = MaxRaymarchSteps - (steps * (1 - glm::length(pos) / 1.5f));
		//float maxColor = MaxRaymarchSteps;

		/*
		glm::vec3 zz = pos;
		float m = glm::dot(zz, zz);

		glm::vec3 orbittrap(abs(zz.x), abs(zz.y), abs(zz.z)); // DO not glm::abs here

		float dz = 1.0f;

		for (int i = 0; i < FractalIterations; ++i) {
			float m2 = m*m;
			float m4 = m2*m2;
			dz = 8.0f*glm::sqrt(m4*m2*m)*dz + 1.0f;

			float x = zz.x; float x2 = zz.x*zz.x; float x4 = x2*x2;
			float y = zz.y; float y2 = zz.y* zz.y; float y4 = y2*y2;
			float z = zz.z; float z2 = zz.z*zz.z; float z4 = z2*z2;

			float k3 = x2 + z2;
			float k2 = 1.f / glm::sqrt(k3*k3*k3*k3*k3*k3*k3);
			float k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
			float k4 = x2 - y2 + z2;

			zz.x = pos.x + 64.0f*x*y*z*(x2 - z2)*k4*(x4 - 6.0f*x2*z2 + z4)*k1*k2;
			zz.y = pos.y + -16.0f*y2*k3*k4*k4 + k1*k1;
			zz.z = pos.z + -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2;

			orbittrap = glm::min(orbittrap, glm::vec3(abs(zz.x), abs(zz.y), abs(zz.z))); // Do not glm::abs here

			m = glm::dot(zz, zz);
			if (m > 1000.0f)
				break;

			if (glm::length(glm::vec3(x, y, z)) > 2) {
				dz = i;
				break;
			}
		}
		//color = FractalIterations - dz;
		//maxColor = FractalIterations;
		//printf("%f %f %f\n", orbittrap.x, orbittrap.y, orbittrap.z);

		*/
		//bool orbitcolor = false; // TODO 

		pixels[index].w = 0;
		//if (orbitcolor) {
			// Orbittrap
			//pixels[index].x = (int)(orbittrap.x * 255.f) & 0xff;
			//pixels[index].y = (int)(orbittrap.y * 255.f) & 0xff;
			//pixels[index].z = (int)(orbittrap.z * 255.f) & 0xff;
		//} else {
			pixels[index].x = (int)(color*255.f / maxColor) & 0xff;
			pixels[index].y = (int)(color*255.f / maxColor) & 0xff;
			pixels[index].z = (int)(color*255.f / maxColor) & 0xff;
		//}
	} else {
		//if (PlaneFloor(direction, pos)) {
			pixels[index].w = 0;
			pixels[index].x = 255 & 0xff;
			pixels[index].y = 0 & 0xff;
			pixels[index].z = 0 & 0xff;
		//} else {
			pixels[index].w = 0;
			pixels[index].x = 0;
			pixels[index].y = 0;
			pixels[index].z = 0;
		//}
	}
}


extern "C" void launchKernel(uchar4* pixels, unsigned int width, unsigned int height, glm::mat3 rot, glm::vec3 pos) {

	// Change these values if close or far away from the bulb
	// TODO calculate LOD
	unsigned int lod = 4;
	unsigned int primRays;

	if (lod == 1) {
		setUp << <1, 1 >> >(.01f, 5, 120, 9);
		primRays = 9;
	} else if (lod == 2) {
		setUp << <1, 1 >> >(.0005f, 10, 60, 5);
		primRays = 5;
	} else if (lod == 3) {
		setUp << <1, 1 >> >(.0005f, 10, 60, 3);
		primRays = 3;
	} else if (lod == 4) {
		setUp << <1, 1 >> >(0.0005f, 6, 60, 1);
		primRays = 1;
	} else {
		printf("Undefined LOD");
	}

	cudaThreadSynchronize();

	int blockThreads = 256;
	int totalThreads = height * width;
	int totalBlocks = totalThreads % blockThreads == 0 ? totalThreads / blockThreads : totalThreads / blockThreads + 1;

	if (lod != 4) {
		// Allocate raymarchSteps and raymarchDistance
		unsigned char* raymarchSteps;
		float * raymarchDistance;

		unsigned int primaryWidth = width % primRays == 0 ? width / primRays : width / primRays + 1;
		unsigned int primaryHeight = height % primRays == 0 ? height / primRays : height / primRays + 1;
		unsigned int primarySize = primaryWidth * primaryHeight;

		cudaMalloc((void**)&raymarchSteps, sizeof(unsigned char) * primarySize); // Do only once?
		cudaMalloc((void**)&raymarchDistance, sizeof(float) * primarySize); // Do only once?

		int blockThreadsPrimary = 256;
		int totalThreadsPrimary = primarySize;
		int totalBlocksPrimary = totalThreadsPrimary % blockThreadsPrimary == 0 ? totalThreadsPrimary / blockThreadsPrimary : totalThreadsPrimary / blockThreadsPrimary + 1;

		primaryRay << <totalBlocksPrimary, blockThreadsPrimary >> >(raymarchSteps, raymarchDistance, width, height, primaryWidth, primaryHeight, rot, pos);

		cudaThreadSynchronize(); // Make sure all primary rays are done

		secondaryRay << <totalBlocks, blockThreads >> >(pixels, raymarchSteps, raymarchDistance, width, height, primaryWidth, primaryHeight, rot, pos);

		cudaThreadSynchronize(); // Synchronize secondary rays

		cudaFree(raymarchSteps); // Do only once?
		cudaFree(raymarchDistance); // Do only once?
	} else {
		singleRay << <totalBlocks, blockThreads >> >(pixels, width, height, rot, pos);

		cudaThreadSynchronize(); // Synchronize secondary rays
	}
}


__global__ void primaryRay(unsigned char* raymarchSteps, float* raymarchDistance, unsigned int width, unsigned int height, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rotation, glm::vec3 position) {
	// Calculate pixel index, x, y 
	const unsigned int index = blockIdx.x * blockDim.x + (threadIdx.x);
	if (index >= primaryHeight*primaryWidth) {
		return;
	}
	int squareRadius = PrimarySize / 2;

	const unsigned int x = squareRadius + PrimarySize * (index % primaryWidth);
	const unsigned int y = squareRadius + PrimarySize * (index / primaryWidth);

	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), height / 2.f);
	direction = rotation*direction;
	direction = glm::normalize(direction);

	glm::vec3 secondDir(x + squareRadius - (width / 2.f), y + squareRadius - (height / 2.f), height / 2.f);
	secondDir = rotation*secondDir;
	secondDir = glm::normalize(secondDir);
	glm::vec3 origin(position);

	float distance = 0;
	int steps = 0;
	// Check bounding sphere
	if (BoundingSphere(direction, position)) {
		// Raymarch as long as all neighbouring rays fit
		//// Only check the corner ray Chapter 4 drive report
		float de = 0.0f; // Maybe create an Estimate or calculation of first circle
		float d = de;
		position += de * direction;

		for (int i = 0; i < MaxRaymarchSteps; ++i) {
			de = DE(position);
			d += de;

			// Check if all rays are inside here
			if (glm::length(glm::cross(secondDir, position - origin)) > de) {
				de = 0.0f; // TODO change to boolean
			}

			if (de <= EpsilonRaymarch) {
				distance = d;
				steps = i;
				break;
			}
			position += de * direction;
		}

	}

	// Save result 
	raymarchSteps[index] = steps;
	raymarchDistance[index] = distance;
}

__global__ void secondaryRay(uchar4* pixels, unsigned char* raymarchSteps, float* raymarchDistance, unsigned int width, unsigned int height, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rotation, glm::vec3 position) {
	// Calculate pixel index, x, y
	const unsigned int index = blockIdx.x * blockDim.x + (threadIdx.x);
	if (index >= width * height) {
		return;
	}

	int secondarySteps = 0;

	const unsigned int x = index % width;
	const unsigned int y = index / width;
	const unsigned int primaryIndex = x / PrimarySize + (y / PrimarySize) * primaryWidth;

	// Calculate start position from primary ray
	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), height / 2);
	direction = rotation*direction;
	direction = glm::normalize(direction);

	int steps = raymarchSteps[primaryIndex];
	float distance = raymarchDistance[primaryIndex];
	glm::vec3 pos = position + direction * distance;
	bool hit = false;

	if (steps != 0) {
		// Raymarch until eps
		for (int i = steps; i < MaxRaymarchSteps; ++i) {
			secondarySteps++;
			float de = DE(pos);
			distance += de;
			if (de <= EpsilonRaymarch) {
				hit = true;
				steps = i;
				break;
			}
			pos += de * direction;
		}
	}

	color(pixels, hit, steps, direction, pos, index);


}

__global__ void singleRay(uchar4* pixels, unsigned int width, unsigned int height, glm::mat3 rotation, glm::vec3 position) {
	// Calculate pixel index, x, y
	const unsigned int index = blockIdx.x * blockDim.x + (threadIdx.x);
	if (index >= width * height) {
		return;
	}

	const unsigned int x = index % width;
	const unsigned int y = index / width;


	// Calculate start position from primary ray
	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), height / 2);
	direction = rotation*direction;
	direction = glm::normalize(direction);

	glm::vec3 pos = position;
	bool hit = false;
	unsigned int steps = 0;
	float distance = 0;

	if (BoundingSphere(direction, position)) {
		// Raymarch until eps
		for (int i = steps; i < MaxRaymarchSteps; ++i) {
			float de = DE(pos);
			distance += de;
			if (de <= EpsilonRaymarch) {
				hit = true;
				steps = i;
				break;
			}
			pos += de * direction;
		}
	}


	color(pixels, hit, steps, direction, position, index);
}

__global__ void setUp(float epsilon, unsigned int fractalIterations, unsigned int raymarchsteps, unsigned int priSize) {
	EpsilonRaymarch = epsilon;
	FractalIterations = fractalIterations;
	MaxRaymarchSteps = raymarchsteps;
	if (priSize == 1) {
		PrimarySize = 1;
		PrimaryRays = false;
	} else {
		PrimarySize = priSize;
		PrimaryRays = true;
	}
}

/*
CUDA time event

float time;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
setUp<< <1, 1>> >(.0005f, 10, 60);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
printf("Time %f", time);
*/

/*
Spectrum background
pixel.w = 0;
pixel.x = (256 * x / (width)) & 0xff;
pixel.y = (256 * y / (height)) & 0xff;
pixel.z = 10;

*/