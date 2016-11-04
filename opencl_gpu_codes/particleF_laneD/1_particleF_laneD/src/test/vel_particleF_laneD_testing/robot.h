#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

typedef struct{
float theta; //radian
float *x;
float *y;
float dist;
float forwardNoise;
float turnNoise;
float senseNoise;
float linvel; //linear velocity
float angvel; //angular velocity
}linestruct;

void move_line(linestruct *, int, int, float);
float gaussian(float, float, float);
float cyclicWorld(float, float);

#endif
