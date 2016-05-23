// includes
#include <windows.h>
#include <stdio.h>

#include <glew.h>
#include <freeglut.h>

#include "cudapbo.h"
#include "openGL.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernel.h"

int main(int argc, char** argv) {
	initGL(argc, argv);

	initCuda(argc, argv);

	glutMainLoop();
}

