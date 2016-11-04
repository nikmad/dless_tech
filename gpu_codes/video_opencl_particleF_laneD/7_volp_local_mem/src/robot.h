// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>
#include <eigen3/Eigen/Dense>

#ifndef ROBOT_H_INCLUDED
#define ROBOT_H_INCLUDED
 
#define PI 3.141592653589793  

using namespace Eigen;

struct lineStruct{
float x[109];
float y[109];
float dist;
float theta; 
};

struct createLineParams{
   int imgrows;
   int imgcols;  
   int NUM_LINES;
   int NEIGHBORHOOD;
};

struct noiseStruct{
   float forwardNoise;
   float turnNoise;
   float senseNoise;
};

struct wtKernParams{
   int NEIGHBORHOOD;
   int NUM_GOOD_LINES;
   int NUM_BEST_LINES;
   int imgrows;
   int imgcols;
   ulong framme_seed;
};

void move_line(struct lineStruct *, struct noiseStruct *, float, float, int, int);
float gaussian(float, float, float);
float cyclicWorld(float, float);

#endif
