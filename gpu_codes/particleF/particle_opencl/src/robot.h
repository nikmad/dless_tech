#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

//The Robot Class
typedef struct{
	float xCord;
	float yCord;
	float orientation;	//This is angle in radians.
	float forwardNoise;
	float turnNoise;
	float senseNoise;
} Robot;
	
//Accessor Methods
void set_rob(Robot *,float, float, float, float);
Vector3f get_rob(Robot *);
void setNoise_rob(Robot *, float, float, float);
void setLandmarks_rob(Robot *, MatrixXf, MatrixXf);

//Other Methods
void init_rob(Robot *, float, float);
void move_rob(Robot *, float, float, float);
VectorXf sense_rob(Robot *, MatrixXf);
float measurementProb_rob(Robot *, VectorXf, MatrixXf);
float gaussian(float, float, float);
float cyclicWorld(float, float);

#endif
