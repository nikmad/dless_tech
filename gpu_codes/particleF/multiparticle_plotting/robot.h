#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

//The Robot Class
typedef struct{
	double xCord;
	double yCord;
	double orientation;	//This is angle in radians.
	double forwardNoise;
	double turnNoise;
	double senseNoise;
	MatrixXd landmarks;
} Robot;
	
//Accessor Methods
void set_rob(Robot *,double, double, double, double);
Vector3d get_rob(Robot *);
void setNoise_rob(Robot *, double, double, double);
void setLandmarks_rob(Robot *, MatrixXd);

//Other Methods
void init_rob(Robot *, double, double);
void move_rob(Robot *, double, double, double);
VectorXd sense_rob(Robot *);
double measurementProb_rob(Robot *, VectorXd);
double gaussian(double, double, double);
double cyclicWorld(double, double);

#endif
