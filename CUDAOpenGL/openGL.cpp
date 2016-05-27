#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

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
long lastUpdateTime;

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
float focalLength = window_height / 2.f;
float moveSpeed = 0.001f;
float sprintSpeed = 3.f;
LOD l;
float epsDelta = 0.0002f;
float rotSpeed = 0.00001f;
bool orbit = false;
float orbTime = 0;
glm::vec3 savedUp;
float r;

// Constants
const float mouseStillDistance = 5.f;
const unsigned int fracItDelta = 1;
const unsigned int raymarchDelta = 10;
const unsigned int priDelta = 2;
const float orbVel = 0.001f;


bool* keyPressed;




void initGL(int argc, char **argv) {
	// Create a window and GL context (also register callbacks)
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(300, 10);
	glutCreateWindow("Raymarching");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialFunc(keySpecial);
	glutSpecialUpFunc(keySpecialUp);

	glutMotionFunc(motion);
	glutMouseFunc(mouse);
	glutMouseWheelFunc(mouseWheel);

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

	l.epsilon = 0.001f;
	l.fractalIterations = 10;
	l.raymarchsteps = 120;
	l.primRays = 1;
}

void display() {
	update();

	// run CUDA kernel
	runCuda(rotationMatrix, cameraPosition, focalLength, l);

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
		l.epsilon = 0.001f;
		l.fractalIterations = 10;
		l.raymarchsteps = 120;
		l.primRays = 1;
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
		setView(glm::vec3(0, 0, -9.5f), 0, 0);
	} else if (key == '<'){
		if(!orbit){
			calculateRotation();
			savedUp = up;
			r = glm::sqrt(cameraPosition.x*cameraPosition.x + cameraPosition.z*cameraPosition.z);//glm::sqrt(glm::dot(cameraPosition, cameraPosition));
			orbTime = 0;
			
			float z = cameraPosition.z;
			float x = cameraPosition.x;
			std::cout << "x/z" << x/z << " x: " << x << " z: " << z << " r: " << r << std::endl;
			if (x < -0.001f){ // TODO THIS IS SHIT
				orbTime = atan(z/x) + 3.1415927f;
				
			} else if (x > 0.001f){
				orbTime = atan(z/x);
			}else {
				if(z > 0.f){
					orbTime = 3.1415927f/2;
				}else{
					orbTime = -3.1415927f / 2;
				}
			}

			std::cout << "orbTime: " << orbTime << std::endl;
			

		} else {
			yaw = orbTime + 3.1415927f / 2;
			calculateRotation();
		}
		orbit = !orbit;

	}


	// Upper 4 keys: improve accuracy
	// Lower 4 keys: decrease accuracy
	else if (key == 'h'){
		if(l.epsilon <= epsDelta){
			epsDelta *= 0.1f;
		}
		l.epsilon = (l.epsilon > epsDelta) ?  l.epsilon - epsDelta : l.epsilon;
		std::cout << "epsilon: " << l.epsilon << "\t epsDelta: " << epsDelta << std::endl;
	} else if (key == 'b'){
		if(l.epsilon >= epsDelta*10.f){
			epsDelta *= 10.f;
		}
		l.epsilon += epsDelta;
		std::cout << "epsilon: " << l.epsilon << "\t epsDelta:" << epsDelta << std::endl;
	} 

	else if (key == 'j'){
		l.fractalIterations += fracItDelta;
		std::cout << "fractalIterations: " << l.fractalIterations << std::endl;
	} else if (key == 'n'){
		l.fractalIterations = (l.fractalIterations > fracItDelta) ?  l.fractalIterations - fracItDelta : l.fractalIterations;
		std::cout << "fractalIterations: " << l.fractalIterations << std::endl;

	} 

	else if (key == 'k'){
		l.raymarchsteps += raymarchDelta;
		std::cout << "raymarchsteps: " << l.raymarchsteps << std::endl;
	} else if (key == 'm'){
		l.raymarchsteps = (l.raymarchsteps > raymarchDelta) ?  l.raymarchsteps - raymarchDelta : l.raymarchsteps;
		std::cout << "raymarchsteps: " << l.raymarchsteps << std::endl;
	} 

	else if (key == 'l'){
		l.primRays = (l.primRays > priDelta) ?  l.primRays - priDelta : l.primRays;
		std::cout << "primRays: " << l.primRays << std::endl;
	} else if (key == ','){
		l.primRays += priDelta;
		std::cout << "primRays: " << l.primRays << std::endl;
	}

	else if (key == 'o'){
		std::cout << "LOD: " << std::endl 
		<< "epsilon:\t \t" << l.epsilon << std::endl 
		<< "fractalIterations:\t" << l.fractalIterations << std::endl 
		<< "raymarchsteps:\t \t" << l.raymarchsteps << std::endl 
		<< "primRays:\t \t" << l.primRays << std::endl;
	} 

	else {
		keyPressed[key] = true;
	}
	// l.epsilon = 0.001f;
	// l.fractalIterations = 10;
	// l.raymarchsteps = 120;
	// l.priSize = 1;
	// std::cout << "key: " << key << std::endl;

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

void mouseWheel(int button, int direction, int x, int y){
	if(direction > 0){
		//Up
		sprintSpeed *= 0.91f;
		moveSpeed *= 0.91f;
		focalLength *= 1.1f;
		rotSpeed *= 0.91f;
	}else{
		//Down
		sprintSpeed *= 1.1f;
		moveSpeed *= 1.1f;
		focalLength *= 0.91f;
		rotSpeed *= 1.1f;
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

	long time = glutGet(GLUT_ELAPSED_TIME);

	float dt = time - lastUpdateTime;
	lastUpdateTime = time;

	glm::vec3 movement(0, 0, 0);
	bool sprint = false;
	bool slow = false;

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
	if (keyPressed['x'] ||keyPressed[' ']) {
		slow = true;
	}

	float slowSpeed = glm::length(cameraPosition) / 10;
	slowSpeed = slowSpeed > 1.0f ? 1.0f : slowSpeed;


	movement = (sprint ? movement * sprintSpeed : movement);
	movement = (slow ? movement * slowSpeed : movement);

	// Update camera
	if(orbit){
		
		cameraPosition = glm::vec3(r*glm::cos(orbTime), cameraPosition.y, r*glm::sin(orbTime)); // TODO THIS IS SHIT

		up = glm::vec3(0,1,0);// glm::normalize(savedUp); 
		forward = glm::normalize(-cameraPosition);
		right = glm::normalize(glm::cross(up, forward));
		up = glm::normalize(glm::cross(forward, right));

		rotationMatrix = glm::mat3(right.x, right.y, right.z,
			up.x, up.y, up.z,
			forward.x, forward.y, forward.z);

		orbTime += dt * orbVel;
	} else {
		yaw += mouseSpeed.x * dt;
		pitch += mouseSpeed.y * dt;
		calculateRotation();
		cameraPosition += movement *dt;
	}
}

void printVec3(glm::vec3 v){
	std::cout << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
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
