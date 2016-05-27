#ifndef CONSTANTS_H
#define CONSTANTS_H

/*
const unsigned int window_width = 256;
const unsigned int window_height = 256;
//*/
//*
const unsigned int window_width = 1280;
const unsigned int window_height = 720;
//*/
/*
const unsigned int window_width = 512;
const unsigned int window_height = 512;
//*/



struct LOD{
	float epsilon;
	unsigned int fractalIterations;
	unsigned int raymarchsteps;
	unsigned int primRays;
};

#endif CONSTANTS_H