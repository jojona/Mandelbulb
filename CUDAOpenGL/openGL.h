#ifndef OPENGL_H
#define OPENGL_H

#include <glm.hpp>

void initGL(int argc, char **argv);
void display();
void keyboard(unsigned char, int, int);
void keyboardUp(unsigned char, int, int);
void keySpecial(int, int, int);
void keySpecialUp(int, int, int);

void mouse(int, int, int, int);
void idle();
void motion(int, int);
void fps();
void update();

void calculateRotation();
void resetCamera();

void setView(glm::vec3 pos, float yawangle, float pitchangle);



#endif OPENGL_H