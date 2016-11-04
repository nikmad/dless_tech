#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//#include <eigen3/Eigen/Dense>
#include "RNG.h"

#include "robot.h"

using namespace std;
using namespace Eigen;

//void init_rob(Robot *particle, float world_size, float a)
//{
//	int c1, c2, c3;
//
//	srand(a);
//	c1 = rand()%10000;
//	line->xCord = ((float)c1/10000.0000) * world_size;
//
//	srand(a+1.324);
//	c2 = rand()%10000;
//	line->yCord = ((float)c2/10000.0000) * world_size;
//
//	srand(a+23343.33323);
//	c3 = rand()%10000;
//	line->orientation = ((float)c3/10000.0000) * 2 * PI;
//	
//	line->forwardNoise = 0.0;
//	line->turnNoise = 0.0;
//	line->senseNoise = 0.0;
//}
//
//void set_rob(Robot *particle, float new_x, float new_y, float new_orientation, float world_size)
//{
//	if ( new_x<0 || new_x>=world_size)
//	printf("X coordinate is out of bounds\n");
//	if ( new_y<0 || new_y>=world_size)
//	printf("Y coordinate is out of bounds\n");
//	if ( new_orientation<0 || new_orientation>=2.0*PI)
//	printf("Orientation is out of bounds\n");
//	line->xCord = new_x;
//	line->yCord = new_y;
//	line->orientation = new_orientation;
//}
//
//Vector3f get_rob(Robot *particle)
//{
//	Vector3f coordinates(line->xCord, line->yCord, line->orientation);
//	return coordinates;
//}
//
//void setNoise_rob(Robot *particle, float new_f_noise, float new_t_noise, float new_s_noise)
//{
//	//makes it possible to change the noise parameters.
//	//this is often useful in particle filters.
//	line->forwardNoise = new_f_noise;
//	line->turnNoise = new_t_noise;
//	line->senseNoise = new_s_noise;
//}

//void sense_rob(linestruct *line, int imgrows, int imgcols, int NEIGHBORHOOD)
//{
//
//for(int glord=0;glord<imgrows;glord++)//glord for goodline coordinate
//{
//if(line->x[glord]>=NEIGHBORHOOD && line->x[glord] <= imgcols-NEIGHBORHOOD){  
// 	for(int m=line->x[glord]-NEIGHBORHOOD; m<line->x[glord]+NEIGHBORHOOD; m++)
// 	{
// 		line->[gl].dist += img_grad.at<uchar>(glord,m);
// 	}
//	line->[gl].dist += bgauss;
// }
// else if(line->[gl].x[glord] > imgcols-NEIGHBORHOOD){  
// 	for(int m=line->[gl].x[glord]-NEIGHBORHOOD; m<imgcols; m++)
// 	{
// 		line->[gl].dist += img_grad.at<uchar>(glord,m);
// 	}
//	line->[gl].dist += bgauss;
// }
// else if(line->[gl].x[glord]<NEIGHBORHOOD){
// 	for(int m=0; m<line->[gl].x[glord]+NEIGHBORHOOD; m++)
// 	{
// 		line->[gl].dist += img_grad.at<uchar>(glord,m);
// 	}
//	line->[gl].dist += bgauss;
// }
////measurement << sense_rob(&myrobot);
//}


void move_line(linestruct *line, float turnAngle, float moveDistance, int imgrows, int imgcols) 
{
	RNG rng(time(NULL));
//	if(moveDistance < 0){printf("Robot can't move backwards!!!\n");}
	line->theta = line->theta + turnAngle + rng.gaussian(line->turnNoise);
	line->theta = cyclicWorld(line->theta, PI);
//Assumption: moveDistance is always the shift in x-intercept of the line
//We specify the overall rotation of the line
//by which the 'y' coordinate should automatically change.
	moveDistance = moveDistance + rng.gaussian(line->forwardNoise);
	float xincept = line->x[0]+moveDistance;
	float r=0.0;
	for(int l=0; l<imgrows; l++)
	{
		line->y[l] = l; 
		r = line->y[l]/sin(line->theta);	
		line->x[l] = abs((int)(xincept + r * (cos(line->theta))));
		//if(l==0)
		line->x[l] = cyclicWorld(line->x[l], imgcols);
	}

//	line->yCord = line->yCord + (moveDistance * sin(line->orientation));
//	line->yCord = cyclicWorld(line->yCord, world_size);
}

//float measurementProb_rob(Robot *particle, VectorXf measurement)
//{
//	//Calculates how likely a measurement should be
//	//which is an essential step
//	float prob = 1.0;
//	float dist;
//	for (int i=0; i < line->landmarks.rows(); i++)
//	{
//	dist = sqrt(pow(( line->xCord - line->landmarks(i,0)),2)+pow(( line->yCord - line->landmarks(i,1)),2));
//	prob *= gaussian(dist, line->senseNoise, measurement[i]);
//	}
//	return prob;
//}

//void setLandmarks_rob(Robot *particle, MatrixXf new_landmarks)
//{
//	line->landmarks = new_landmarks;
//}

//Subfunction below calculates a%b where a and b are floats. 
//But modulus operator "%" in C++ only works for integers.
//Hence the implementation had to be indirect in case of floats. 
float cyclicWorld(float a, float b)
{
	if (a>=0)
	{
		return a-b*(int)(a/b);
	}
	else
	{
		return a+b*(1+(int)(abs(a/b)));
	}	
}

float gaussian(float mu, float sigma, float x)
{
	return (1/sqrt(2.*PI*pow(sigma,2)))*exp(-0.5*pow((x-mu),2)/pow(sigma,2));
	//Observe the expression (x-mu)^2 above. Greater this difference, lesser 
	//the value of ¨return¨. So when we cal this 'gaussian' function from 
	//'measurement_prob' function, this difference is nothing but the difference
	//between the distances measured to each of the landmarks from robot and 
	//each particle. So, farther a particle from robot, smaller the value of 'prob'
	//which represents the weight of that particle.  
}

