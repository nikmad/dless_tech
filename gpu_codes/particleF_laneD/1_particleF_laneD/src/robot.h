#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

//The Robot Structure
//typedef struct{
//	float xCord;
//	float yCord;
//	float orientation;	//This is angle in radians.
//	float forwardNoise;
//	float turnNoise;
//	float senseNoise;
//	MatrixXf landmarks;
//} Robot;

//The line structure
typedef struct{
float theta; //radian
float *x;
float *y;
float dist;
//These noises should be replaced with meaningful
//numbers in future using for forward noise, say
//the carś wheel efficiency in moving by desired
//distance if say there is some slip with road.
//Also, these noise params can be put in a
//different structure which can be shared by all 
//the lines instead of each line having its own.
//Because we may usually never specify a separate 
//noise for each line.
float forwardNoise;
float turnNoise;
float senseNoise;
}linestruct;


	
//Accessor Methods
//void set_rob(Robot *,float, float, float, float);
//Vector3f get_rob(Robot *);
//void setNoise_rob(Robot *, float, float, float);
//void setLandmarks_rob(Robot *, MatrixXf);

//Other Methods
//void init_rob(Robot *, float, float);
void move_line(linestruct *, float, float, int, int);
//VectorXf sense_rob(Robot *);
//float measurementProb_rob(Robot *, VectorXf);
float gaussian(float, float, float);
float cyclicWorld(float, float);

#endif
