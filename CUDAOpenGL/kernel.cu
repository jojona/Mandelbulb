#include <windows.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "kernel.h"
#include "constants.h"

#define CAMERALIGHT 0
#define CIRCLETINT 0

__device__ float EpsilonRaymarch = 0;
__device__ unsigned int MaxRaymarchSteps = 0;
__device__ unsigned int FractalIterations = 0;
__device__ bool PrimaryRays = false;
__device__ unsigned int PrimarySize = 0;
__device__ unsigned int iteration = 0;


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
	}
}

__device__ float DE(const glm::vec3& pos) {
	//return DEMandelbulb2(pos);
	return DEMandelbulb1(pos);
	//return DETetredon(pos);
	//return DESphere1(pos);
}

/*
 * Distance Estiamtor for a Sphere in origo with radius 3
 */
__device__ float DESphere1(const glm::vec3& pos) {
	float distance = length(pos);
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
__device__ float DETetredon(const glm::vec3& pos) {
	glm::vec3 z(pos);
	float Scale = 2.f;

	glm::vec3 a1(1, 1, 1);
	glm::vec3 a2(-1, -1, 1);
	glm::vec3 a3(1, -1, -1);
	glm::vec3 a4(-1, 1, -1);
	glm::vec3 c;
	int n = 0;
	float dist, d;
	while (n < FractalIterations) {
		c = a1; dist = length(z - a1);
		d = length(z - a2); if (d < dist) { c = a2; dist = d; }
		d = length(z - a3); if (d < dist) { c = a3; dist = d; }
		d = length(z - a4); if (d < dist) { c = a4; dist = d; }
		z = Scale*z - c*(Scale - 1.0f);
		n++;
	}

	return length(z) * pow(Scale, float(-n));

}

/*
* Distance estimator for a Mandelbulb. Version 1
*/
__device__ float DEMandelbulb1(const glm::vec3& p) {
	glm::vec3 z = p;
	float r = 0.0f;
	float dr = 1.0f;
	float power = 8.f;
	for (int i = 0; i < FractalIterations; ++i) {
		r = length(z);
		if (r > 100.f) break;

		// To polar coordianates
		float theta = acosf(z.y / r);
		float phi = atan2f(z.x, z.z);
		float r7 = glm::pow(r, power - 1);

		// Derivative
		dr = r7 * power * dr + 1.0f;

		// "Squaring" the length
		float zr = r7*r;
		// "Double" the angle
		theta = theta*power;
		phi = phi * power;

		// From polar coordianates
		z = p + zr*glm::vec3(sinf(phi)*sinf(theta), cosf(theta), sinf(theta) * cosf(phi));
	}
	return 0.5f*logf(r)*r / dr;
}

/*
 * Distance estimator for a Mandelbulb. Version 2
 */
__device__ float DEMandelbulb2(const glm::vec3& pos) {
	glm::vec3 zz = pos;
	float m = dot(zz);

	float dz = 1.0f;

	for (int i = 0; i < FractalIterations; ++i) {
		float m2 = m*m;
		float m4 = m2*m2;
		dz = 8.0f*sqrtf(m4*m2*m)*dz + 1.0f;

		float x = zz.x; float x2 = zz.x*zz.x; float x4 = x2*x2;
		float y = zz.y; float y2 = zz.y* zz.y; float y4 = y2*y2;
		float z = zz.z; float z2 = zz.z*zz.z; float z4 = z2*z2;

		float k3 = x2 + z2;
		float k2 = 1.f / sqrtf(k3*k3*k3*k3*k3*k3*k3);
		float k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
		float k4 = x2 - y2 + z2;

		zz.x = pos.x + 64.0f*x*y*z*(x2 - z2)*k4*(x4 - 6.0f*x2*z2 + z4)*k1*k2;
		zz.y = pos.y + -16.0f*y2*k3*k4*k4 + k1*k1;
		zz.z = pos.z + -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2;

		m = dot(zz);
		if (m > 1000.0f)
			break;
	}

	return 0.25f*logf(m)*sqrtf(m) / dz;
}

__device__ bool BoundingSphere(const glm::vec3& dir, const glm::vec3& pos) {
	float rSquared = 1.2f * 1.2f;
	if (dot(pos) <= rSquared) {
		return true;
	} else if (dot(pos, dir) <= 0) {
		glm::vec3 v = pos - dir * dot(pos, dir) / dot(dir);
		if (dot(v) <= rSquared) {
			return true;
		}
	}
	return false;
}

__device__ glm::vec3 PlaneFloor(const glm::vec3& dir, const glm::vec3& pos) {
	float denom = dot(glm::vec3(0, 1, 0), dir);
	if (denom > 0.0001f) { // Only visible from above
		float t = dot(glm::vec3(0, 1.5f, 0) - pos, (glm::vec3(0, 1, 0))) / denom;
		if (t >= 0) {

			glm::vec3 collision = pos + t * dir;
			if (((int)floorf(collision.x) % 2 == 0 || (int)floorf(collision.z) % 2 == 0) && !((int)floorf(collision.x) % 2 == 0 && (int)floorf(collision.z) % 2 == 0)) {
				return glm::vec3(100.f, 100.f, 100.f);
			} else {
				return glm::vec3(50.f, 50.f, 50.f);
			}

			/*
			float distanceFromOrigo = length(pos + t * dir);
			if (distanceFromOrigo > 3 && distanceFromOrigo < 7) {
			return true;
			}
			*/
		}
	}
	return glm::vec3(0, 0, 0);
}

__device__ void color(uchar4* pixels, bool hit, unsigned int steps, glm::vec3 rayDir, glm::vec3 rayOrigin, glm::vec3 position, int index) {
	// Draw color to pixels
	if (hit) {
		float normalEpsilon = 0.000005f; // TODO find a good epsilon for normal
		glm::vec3 normal(DE(position + glm::vec3(normalEpsilon, 0, 0)) - DE(position - glm::vec3(normalEpsilon, 0, 0)),
			DE(position + glm::vec3(0, normalEpsilon, 0)) - DE(position - glm::vec3(0, normalEpsilon, 0)),
			DE(position + glm::vec3(0, 0, normalEpsilon)) - DE(position - glm::vec3(0, 0, normalEpsilon)));
		normal = normalize(normal);

		glm::vec3 lightPower(0.f, 0.f, 0.f);

		// Global illumination
		lightPower += glm::vec3(0.1f, 0.1f, 0.1f);

		// Light 1, 2, 3
		// Light side
		lightPower += light(glm::vec3(2.0f, 0.f, -2.f), glm::vec3(1.f, 1.f, 1.f) * 24.f, position, normal, true);
		// Light below
		lightPower += light(glm::vec3(0.f, 2.f, 0.5f), glm::vec3(1.f, 1.f, 1.f)* 5.f, position, normal, true);
		// Light above
		lightPower += light(glm::vec3(0.f, -2.f, 0.5f), glm::vec3(1.f, 1.f, 1.f)* 5.f, position, normal, true);

#if CIRCLETINT
		float tintFactor = 5.f;
		if (iteration % 1000 < 600) {
			lightPower += light(glm::vec3(3.0f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.f) *tintFactor, position, normal, true);
			lightPower += light(glm::vec3(0.0f, 1.f, 3.f), glm::vec3(1.f, 0.f, 0.f) * tintFactor, position, normal, true);
			lightPower += light(glm::vec3(0.0f, 1.f, -3.f), glm::vec3(1.f, 0.f, 0.f) * tintFactor, position, normal, true);
			lightPower += light(glm::vec3(-3.0f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.f) * tintFactor, position, normal, true);
		} else {
			lightPower += light(glm::vec3(3.0f, 1.f, 0.f), glm::vec3(0.f, 1.f, 0.f)* tintFactor, position, normal, true);
			lightPower += light(glm::vec3(0.0f, 1.f, 3.f), glm::vec3(0.f, 1.f, 0.f)* tintFactor, position, normal, true);
			lightPower += light(glm::vec3(0.0f, 1.f, -3.f), glm::vec3(0.f, 1.f, 0.f)* tintFactor, position, normal, true);
			lightPower += light(glm::vec3(-3.0f, 1.f, 0.f), glm::vec3(0.f, 1.f, 0.f)* tintFactor, position, normal, true);
		}
#endif

#if CAMERALIGHT
		// Use camera as a light
		lightPower += light(rayOrigin, glm::vec3(1.f, 1.f, 1.f) * 5.f, position, normal, false);
#endif	
		// Clamping
		lightPower = glm::min(glm::vec3(1.f, 1.f, 1.f), lightPower);
		lightPower = glm::max(glm::vec3(0.f, 0.f, 0.f), lightPower);

		/* // Raymarch step coloring
		float color = MaxRaymarchSteps - steps;
		float maxColor = MaxRaymarchSteps;
		pixels[index].w = 0;
		pixels[index].x = (int)(color*255.f / maxColor) & 0xff;
		pixels[index].y = (int)(color*255.f / maxColor) & 0xff;
		pixels[index].z = (int)(color*255.f / maxColor) & 0xff;
		*/

		pixels[index].w = 0;
		pixels[index].x = lightPower.x * 255.f;
		pixels[index].y = lightPower.y * 255.f;
		pixels[index].z = lightPower.z * 255.f;
	} else {
		glm::vec3 col = PlaneFloor(rayDir, rayOrigin);
		pixels[index].w = 0;
		pixels[index].x = (int)col.x & 0xff;
		pixels[index].y = (int)col.y & 0xff;
		pixels[index].z = (int)col.z & 0xff;
		/*
		if (PlaneFloor(rayDir, rayOrigin)) {
		if (iteration % 1000 < 600) {
		pixels[index].w = 0;
		pixels[index].x = 100 & 0xff;
		pixels[index].y = 0 & 0xff;
		pixels[index].z = 0 & 0xff;
		} else {
		pixels[index].w = 0;
		pixels[index].x = 0 & 0xff;
		pixels[index].y = 100 & 0xff;
		pixels[index].z = 0 & 0xff;
		}
		} else {
		pixels[index].w = 0;
		pixels[index].x = 0;
		pixels[index].y = 0;
		pixels[index].z = 0;
		}
		*/

	}
}

__device__ glm::vec3 light(const glm::vec3& lightPos, const glm::vec3& lightColor, const glm::vec3& position, const glm::vec3& normal, bool calcShadow) {
	if (calcShadow && (shadow(lightPos, position))) {
		return glm::vec3(0.f, 0.f, 0.f);
	}
	glm::vec3 radius(lightPos - position);
	return glm::vec3(lightColor * fmaxf(dot(normal, normalize(radius)), 0) / (4 * CUDART_PI_F * length(radius)*length(radius)));
}

__device__ bool shadow(const glm::vec3& lightPos, const glm::vec3& position) {
	float de = 0.0f;
	float d = de;

	glm::vec3 pos(lightPos);
	glm::vec3 direction(position - lightPos);

	glm::vec3 dir = normalize(direction);

	bool hit = false;

	for (int i = 0; i < MaxRaymarchSteps; ++i) {
		de = DE(pos);
		d += de;
		pos += de * dir;
		if (de <= EpsilonRaymarch) {
			hit = true;
			break;
		}
	}

	if (!hit) {
		return false;
	}
	return (length(direction) - 2 * EpsilonRaymarch > d);
}



extern "C" void launchKernel(uchar4* pixels, unsigned int width, unsigned int height, float focalLength, glm::mat3 rot, glm::vec3 pos, LOD l) { // TODO Dela upp lod i 3 param.

	// setUp:
	// float epsilon, 
	// int fractalIterations, 
	// int raymarchsteps, 
	// int priSize
	setUp << <1, 1 >> >(l);
	unsigned int primRays = l.primRays;

	cudaThreadSynchronize();

	int blockThreads = 256;
	int totalThreads = height * width;
	int totalBlocks = totalThreads % blockThreads == 0 ? totalThreads / blockThreads : totalThreads / blockThreads + 1;

	if (primRays > 1) {
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

		primaryRay << <totalBlocksPrimary, blockThreadsPrimary >> >(raymarchSteps, raymarchDistance, width, height, focalLength, primaryWidth, primaryHeight, rot, pos);

		cudaThreadSynchronize(); // Make sure all primary rays are done

		secondaryRay << <totalBlocks, blockThreads >> >(pixels, raymarchSteps, raymarchDistance, width, height, focalLength, primaryWidth, primaryHeight, rot, pos);

		cudaThreadSynchronize(); // Synchronize secondary rays

		cudaFree(raymarchSteps); // Do only once?
		cudaFree(raymarchDistance); // Do only once?
	} else {
		singleRay << <totalBlocks, blockThreads >> >(pixels, width, height, focalLength, rot, pos);

		cudaThreadSynchronize(); // Synchronize secondary rays
	}
}


__global__ void primaryRay(unsigned char* raymarchSteps, float* raymarchDistance, unsigned int width, unsigned int height, float focalLength, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rotation, glm::vec3 position) {
	// Calculate pixel index, x, y 
	const unsigned int index = blockIdx.x * blockDim.x + (threadIdx.x);
	if (index >= primaryHeight*primaryWidth) {
		return;
	}
	int squareRadius = PrimarySize / 2;

	const unsigned int x = squareRadius + PrimarySize * (index % primaryWidth);
	const unsigned int y = squareRadius + PrimarySize * (index / primaryWidth);

	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), focalLength);
	direction = rotation*direction;
	direction = normalize(direction);

	glm::vec3 secondDir(x + squareRadius - (width / 2.f), y + squareRadius - (height / 2.f), height / 2.f);
	secondDir = rotation*secondDir;
	secondDir = normalize(secondDir);
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
			position += de * direction;

			// Check if all rays are inside here
			if (length(cross(secondDir, position - origin)) > de) {
				de = 0.0f; // TODO change to boolean
			}

			if (de <= EpsilonRaymarch) {
				distance = d;
				steps = i;
				break;
			}
		}

	}

	// Save result 
	raymarchSteps[index] = steps;
	raymarchDistance[index] = distance;
}

__global__ void secondaryRay(uchar4* pixels, unsigned char* raymarchSteps, float* raymarchDistance, unsigned int width, unsigned int height, float focalLength, unsigned int primaryWidth, unsigned int primaryHeight, glm::mat3 rotation, glm::vec3 position) {
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
	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), focalLength);
	direction = rotation*direction;
	direction = normalize(direction);

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
			pos += de * direction;
			if (de <= EpsilonRaymarch) {
				hit = true;
				steps = i;
				break;
			}
		}
	}

	color(pixels, hit, steps, direction, position, pos, index);


}

__global__ void singleRay(uchar4* pixels, unsigned int width, unsigned int height, float focalLength, glm::mat3 rotation, glm::vec3 position) {
	// Calculate pixel index, x, y
	const unsigned int index = blockIdx.x * blockDim.x + (threadIdx.x);
	if (index >= width * height) {
		return;
	}

	const unsigned int x = index % width;
	const unsigned int y = index / width;


	// Calculate start position from primary ray
	glm::vec3 direction(x - (width / 2.f), y - (height / 2.f), focalLength);
	direction = rotation*direction;
	direction = normalize(direction);

	glm::vec3 pos = position;
	bool hit = false;
	unsigned int steps = 0;
	float distance = 0;

	if (BoundingSphere(direction, position)) {
		// Raymarch until eps
		for (int i = steps; i < MaxRaymarchSteps; ++i) {
			float de = DE(pos);
			distance += de;
			pos += de * direction;
			if (de <= EpsilonRaymarch) {
				hit = true;
				steps = i;
				break;
			}
		}
	} else {
		// Show bounding sphere
		/*
		pixels[index].w = 0;
		pixels[index].x = 255;
		pixels[index].y = 255;
		pixels[index].z = 255;
		return;
		*/
	}
	color(pixels, hit, steps, direction, position, pos, index);
}

__global__ void setUp(LOD l) {//(float epsilon, unsigned int fractalIterations, unsigned int raymarchsteps, unsigned int priSize) {
	EpsilonRaymarch = l.epsilon;
	FractalIterations = l.fractalIterations;
	MaxRaymarchSteps = l.raymarchsteps;
	if (l.primRays == 1) {
		PrimarySize = 1;
		PrimaryRays = false;
	} else {
		PrimarySize = l.primRays;
		PrimaryRays = true;
	}
	iteration++;
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