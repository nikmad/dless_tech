// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>
#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

typedef struct{
//float x[298];
//float y[79];
//float x[277];
//float y[101];
//float x[575];
//float y[96];
float x[511];
float y[72];
float dist;
float theta1; 
float theta2; 
}linestruct;

typedef struct{
   float forwardNoise;
   float turnNoise;
   float senseNoise;
}noiseStruct;

typedef struct{
   int NEIGHBORHOOD;
   int NUM_GOOD_LINES;
   int NUM_BEST_LINES;
   int imgrows;
   int imgcols;
}wtKernParams;

//void move_line(linestruct *, noiseStruct *, float, float, int, int);
float gaussian(float, float, float);
float cyclicWorld(float, float);

#endif
