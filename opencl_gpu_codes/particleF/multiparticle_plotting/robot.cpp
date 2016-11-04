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

void init_rob(Robot *particle, double world_size, double a)
{
	int c1, c2, c3;

	srand(a);
	c1 = rand()%10000;
	particle->xCord = ((double)c1/10000.0000) * world_size;

	srand(a+1.324);
	c2 = rand()%10000;
	particle->yCord = ((double)c2/10000.0000) * world_size;

	srand(a+23343.33323);
	c3 = rand()%10000;
	particle->orientation = ((double)c3/10000.0000) * 2 * PI;
	
	particle->forwardNoise = 0.0;
	particle->turnNoise = 0.0;
	particle->senseNoise = 0.0;
}

void set_rob(Robot *particle, double new_x, double new_y, double new_orientation, double world_size)
{
	if ( new_x<0 || new_x>=world_size)
	printf("X coordinate is out of bounds\n");
	if ( new_y<0 || new_y>=world_size)
	printf("Y coordinate is out of bounds\n");
	if ( new_orientation<0 || new_orientation>=2.0*PI)
	printf("Orientation is out of bounds\n");
	particle->xCord = new_x;
	particle->yCord = new_y;
	particle->orientation = new_orientation;
}

Vector3d get_rob(Robot *particle)
{
	Vector3d coordinates(particle->xCord, particle->yCord, particle->orientation);
	return coordinates;
}

void setNoise_rob(Robot *particle, double new_f_noise, double new_t_noise, double new_s_noise)
{
	//makes it possible to change the noise parameters.
	//this is often useful in particle filters.
	particle->forwardNoise = new_f_noise;
	particle->turnNoise = new_t_noise;
	particle->senseNoise = new_s_noise;
}

VectorXd sense_rob(Robot *particle)
{
	double b;
	RNG rng(time(NULL));
	VectorXd dist(particle->landmarks.rows());
	for (int i=0; i < particle->landmarks.rows(); i++)
	{
	dist[i] = sqrt(pow(( particle->xCord - particle->landmarks(i,0)),2)+pow(( particle->yCord - particle->landmarks(i,1)),2));
	b = rng.gaussian( particle->senseNoise );
	dist[i] = dist[i]+b;
	}
	return dist;
}

void move_rob(Robot *particle, double turnAngle, double moveDistance, double world_size) 
{
	RNG rng(time(NULL));

	if(moveDistance < 0){printf("Robot can't move backwards!!!\n");}

	particle->orientation = particle->orientation + turnAngle + rng.gaussian(particle->turnNoise);
	particle->orientation = cyclicWorld(particle->orientation, 2*PI);

	moveDistance = moveDistance + rng.gaussian(particle->forwardNoise);

	particle->xCord = particle->xCord + (moveDistance * cos(particle->orientation));
	particle->xCord = cyclicWorld(particle->xCord, world_size); //cyclic truncation.

	particle->yCord = particle->yCord + (moveDistance * sin(particle->orientation));
	particle->yCord = cyclicWorld(particle->yCord, world_size);
}

double measurementProb_rob(Robot *particle, VectorXd measurement)
{
	//Calculates how likely a measurement should be
	//which is an essential step
	double prob = 1.0;
	double dist;
	for (int i=0; i < particle->landmarks.rows(); i++)
	{
	dist = sqrt(pow(( particle->xCord - particle->landmarks(i,0)),2)+pow(( particle->yCord - particle->landmarks(i,1)),2));
	prob *= gaussian(dist, particle->senseNoise, measurement[i]);
	}
	return prob;
}

void setLandmarks_rob(Robot *particle, MatrixXd new_landmarks)
{
	particle->landmarks = new_landmarks;
}

//Subfunction below calculates a%b where a and b are doubles. 
//But modulus operator "%" in C++ only works for integers.
//Hence the implementation had to be indirect in case of doubles. 
double cyclicWorld(double a, double b)
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

double gaussian(double mu, double sigma, double x)
{
	return (1/sqrt(2.*PI*pow(sigma,2)))*exp(-0.5*pow((x-mu),2)/pow(sigma,2));
	//Observe the expression (x-mu)^2 above. Greater this difference, lesser 
	//the value of ¨return¨. So when we cal this 'gaussian' function from 
	//'measurement_prob' function, this difference is nothing but the difference
	//between the distances measured to each of the landmarks from robot and 
	//each particle. So, farther a particle from robot, smaller the value of 'prob'
	//which represents the weight of that particle.  
}

