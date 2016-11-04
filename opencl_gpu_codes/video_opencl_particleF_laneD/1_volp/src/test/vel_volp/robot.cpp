// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "RNG.h"

#include "robot.h"

using namespace std;
using namespace Eigen;
void move_line(linestruct *line, noiseStruct *noise, float turnAngle, float moveDistance, int imgrows, int imgcols) 
{
	RNG rng(time(NULL));
	line->theta = line->theta + turnAngle + rng.gaussian(noise->turnNoise);
	line->theta = cyclicWorld(line->theta, PI);
	
   moveDistance = moveDistance + rng.gaussian(noise->forwardNoise);
	float xincept = line->x[0]+moveDistance;
	float r=0.0;
	for(int l=0; l<imgrows; l++)
	{
		line->y[l] = l; 
		r = line->y[l]/sin(line->theta);	
		line->x[l] = abs((int)(xincept + r * (cos(line->theta))));
		line->x[l] = cyclicWorld(line->x[l], imgcols);
	}

}

float cyclicWorld(float a, float b)
{
	if (a>=0)
		return a-b*(int)(a/b);
	else
		return a+b*(1+(int)(abs(a/b)));
}

float gaussian(float mu, float sigma, float x)
{
	return (1/sqrt(2.*PI*pow(sigma,2)))*exp(-0.5*pow((x-mu),2)/pow(sigma,2));
}

