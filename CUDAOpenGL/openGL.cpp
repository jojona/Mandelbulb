#include <windows.h>
#include <stdio.h>
#include <iostream>

#include <glew.h>
#include <freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "constants.h"
#include "openGL.h"
#include "cudapbo.h"
#include "glm.hpp"

extern GLuint textureID;
extern GLuint pbo;

int frames;
long lasttime;

// Camera position and controls
glm::vec3 cameraPosition;
glm::mat3 rotationMatrix;
glm::vec3 right;
glm::vec3 up;
glm::vec3 forward;
glm::vec2 mousePos;
glm::vec2 mouseSpeed;
bool mouseDown;
float yaw;
float pitch;

// Constants
float sprintSpeed = 3;
const float rotSpeed = 0.001;
const float mouseStillDistance = 5;
const float moveSpeed = .2f;

bool* keyPressed;

void initGL(int argc, char **argv) {
	// Create a window and GL context (also register callbacks)
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(300, 300);
	glutCreateWindow("Raymarching");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialFunc(keySpecial);
	glutSpecialUpFunc(keySpecialUp);

	glutMotionFunc(motion);
	glutMouseFunc(mouse);
	glutIdleFunc(idle);
	glutCloseFunc(cleanupCuda);

	glutIgnoreKeyRepeat(1); // Needed?

	keyPressed = new bool[256];
	for (size_t i = 0; i < 256; ++i) {
		keyPressed[i] = false;
	}

	resetCamera();

	// check for necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		std::cout << "Error glew" << std::endl;
	}

	// Setup our viewport and viewing modes
	glViewport(0, 0, window_width, window_height);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

	lasttime = glutGet(GLUT_ELAPSED_TIME);
}

void display() {
	update();

	// run CUDA kernel
	runCuda(rotationMatrix, cameraPosition);

	// Create a texture from the buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// bind texture from PBO
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Note: NULL indicates the data resides in device memory
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Draw a single Quad with texture coordinates for each vertex.
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glutSwapBuffers();

	frames++;
}

// Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y) {
	if (key == 27) {
		glutDestroyWindow(glutGetWindow());
	} else if (key == 'r') {
		resetCamera();
	} else if (key == '1') {
		resetCamera();
		std::cout << "View Front" << std::endl;
	} else if (key == '2') {
		setView(glm::vec3(0, -2.f, .001f), 0, 90 * 0.0174532925f);
		std::cout << "View Up" << std::endl;
	} else if (key == '3') {
		setView(glm::vec3(-2.f, 0, 0), -90 * 0.0174532925f, 0);
		std::cout << "View Side" << std::endl;
	} else if (key == '4') {
		setView(glm::vec3(0, 2.f, .001f), 0, -90 * 0.0174532925f);
		std::cout << "View Down" << std::endl;
	} else if (key == 'p') {
		printf("Cameraview (%f, %f, %f) yaw: %f, pitch: %f\n", cameraPosition.x, cameraPosition.y, cameraPosition.z, yaw, pitch);
	} else if (key == '5') {
		setView(glm::vec3(0.514507f, 0.193910f, -1.154963f), 0.306770f, 0.073186f);
	} else if (key == '6') {
		setView(glm::vec3(0.020500f, 0.197362f, -0.961050f), 0.117f, -0.151f);
	} else if (key == '7') {
		setView(glm::vec3(0.391448f, 0.976358f, 0.322076f), 33.917992f, -0.978999f);
	} else {
		keyPressed[key] = true;
	}
}

void keyboardUp(unsigned char key, int x, int y) {
	keyPressed[key] = false;
}

void keySpecial(int key, int, int) {

}

void keySpecialUp(int key, int, int) {

}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		mouseDown = true;
		mousePos.x = x;
		mousePos.y = y;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		mouseDown = false;
		mouseSpeed.x = 0; mouseSpeed.y = 0;
	}
}

void motion(int x, int y) {
	if (mouseDown) {
		if (x - mouseStillDistance < mousePos.x || x + mouseStillDistance > mousePos.x) {
			mouseSpeed.x = rotSpeed * (mousePos.x - x);
		} else {
			mouseSpeed.x = 0;
		}

		if (y - mouseStillDistance < mousePos.y || y + mouseStillDistance > mousePos.y) {
			mouseSpeed.y = rotSpeed * (y - mousePos.y);
		} else {
			mouseSpeed.y = 0;
		}

	}
}

void idle() {
	fps();
	glutPostRedisplay();
}

void fps() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	if (lasttime + 1000 < time) {
		printf("Fps: %d \n", frames * 1000 / (time - lasttime));

		frames = 0;
		lasttime = time;
	}
}

void calculateRotation() {
	glm::mat3 r1(cos(yaw), 0, sin(yaw), 0, 1, 0, -sin(yaw), 0, cos(yaw)); // yaw
	glm::mat3 r2(1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch)); // pitch

	rotationMatrix = r1*r2;

	right = glm::vec3(rotationMatrix[0][0], rotationMatrix[0][1], rotationMatrix[0][2]);
	up = glm::vec3(rotationMatrix[1][0], rotationMatrix[1][1], rotationMatrix[1][2]);
	forward = glm::vec3(rotationMatrix[2][0], rotationMatrix[2][1], rotationMatrix[2][2]);

	glm::normalize(right);
	glm::normalize(up);
	glm::normalize(forward);

}

/*
 * Update the camera
 */
void update() {
	glm::vec3 movement(0, 0, 0);
	bool sprint = false;

	if (keyPressed['a']) {
		// left 
		movement -= right * moveSpeed;
	}
	if (keyPressed['d']) {
		//right
		movement += right * moveSpeed;
	}
	if (keyPressed['w']) {
		//Forward
		movement += forward * moveSpeed;
	}
	if (keyPressed['s']) {
		//Backward
		movement -= forward * moveSpeed;
	}
	if (keyPressed['q']) {
		//Up
		movement -= up * moveSpeed;
	}
	if (keyPressed['e']) {
		//Down
		movement += up * moveSpeed;
	}
	if (keyPressed['z']) {
		sprint = true;
	}

	yaw += mouseSpeed.x;
	pitch += mouseSpeed.y;

	sprintSpeed = glm::length(cameraPosition) / 10;
	sprintSpeed = sprintSpeed > 3.0f ? 3.0f : sprintSpeed;

	calculateRotation();
	cameraPosition += sprint ? movement * sprintSpeed : movement;
}

/*
 * Reset the camera position and make all keys unpressed.
 */
void resetCamera() {
	setView(glm::vec3(0, 0, -2.f), 0, 0);
}

/*
 * Changes the camera position to pos with given rotationsangles degrees.
 */
void setView(glm::vec3 pos, float yawangle, float pitchangle) {
	cameraPosition = pos;
	yaw = yawangle;
	pitch = pitchangle;

	calculateRotation();
	for (size_t i = 0; i < 256; ++i) {
		keyPressed[i] = false;
	}
	mousePos = glm::vec2(0, 0);
	mouseSpeed = glm::vec2(0, 0);
	mouseDown = false;

}
