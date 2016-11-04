#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "RNG.h"
#include <eigen3/Eigen/Dense>
#include "robot_class.h"

using namespace Eigen;
using namespace std;

#define PI 3.141592653589793
typedef unsigned long int ULONG;

double eval(MatrixXd, Vector3d, double, ULONG); 

int main(void)
{
	system("rm particleCord.txt");
	system("rm myrobotCord.txt");
	system("rm out.png");
	MatrixXd landmarks(4,2);
	landmarks << 20,20,
	80,80,
	20,80,
	80,20;
	double world_size = 100.0;
	ULONG N = 10000;
	int iterations = 10;
	double errorInitial;
	double error[iterations];
	Robot particle[N];
	Robot myrobot;
	double srandSeed[N];
	VectorXd measurement(landmarks.rows());
	VectorXd w(N); //Particle weights
	MatrixXd particleCord(N,3);
	Vector3d myrobotCord;

	//Initializing myrobot:	
	myrobot.setLandmarks(landmarks);
	myrobot.init(world_size, time(NULL));
	myrobot.move(0.1, 1.0, world_size);
	measurement << myrobot.sense();
	
	for(int i=0; i<N; i++)
	{
		//Number below is used to seed the random number generator.
	 	srandSeed[i] = (double)i; 
		particle[i].init(world_size, srandSeed[i]);
		particle[i].setNoise(0.05, 0.05, 5.0);
		particle[i].setLandmarks(landmarks);
	}

	//Evaluating the error before any particle filtering algorithm is used:
	for (int i=0; i<N; i++)
	{
		particleCord.row(i) = particle[i].get();
	}
	myrobotCord = myrobot.get();
  	errorInitial = eval(particleCord, myrobotCord, world_size, N);	
		
	//The loop below is for changing the number of iterations 
	//for which the entire particle filter has to run so that
	//results can converge much better.
	for(int topIndex=0; topIndex<iterations; topIndex++) 
	{
		//Moving the robot and all the particles
		myrobot.move(0.1, 1.0, world_size);
		measurement << myrobot.sense();
		//double turn1,dist1;
		for(int i=0; i<N; i++)
		{
		//	int c1 = rand()%10000;
		//	turn1 = ((double)c1/10000.0000) * 2 * PI;
		//	int c2 = rand()%10000;
		//	dist1 = ((double)c2/10000.0000)*5;
	
		//	particle[i].move(turn1, dist1, world_size);
			particle[i].move(0.1, 1.0, world_size);
		}
		
		//Computing the weight of each particle
		for(int i=0; i<N; i++)
		{
			w[i] = particle[i].measurement_prob(measurement);
		}
		
		//Normalizing the weights.
		double tWeight = 0.0;
		for(int i=0; i<N; i++)
		{
			tWeight = tWeight + w[i];
		}
		
		for(int i=0; i<N; i++)
		{
			w[i] = w[i]/tWeight;
		}

		srand(time(NULL));
		int index = rand() % N;
		int beta1; 
		double beta = 0.0;
		double mw = 0.0; //max. value out of w[1...N]	
		Robot particle1[N];
		
		//Evaluating the maximum weight
		for (int i=0; i<N; i++)
		{
			if(w[i] > mw)
			{
				mw = w[i];
			}	
		}
		
		//RESAMPLING STEP
		for (int i=0; i<N; i++)
		{
			beta1 = rand()%N;
			beta += ((double)beta1/N) * ( 0.5 * mw );
			while (beta > w[index])
			{
				beta -= w[index];
				index = (index + 1) % N;
			}
			particle1[i] = particle[index];
		}
		
		//Setting the noise and landmarks for the new particles
		for (int i=0; i<N; i++)
		{
			particle[i] = particle1[i];
			particle[i].setNoise(0.05, 0.05, 5.0);
			particle[i].setLandmarks(landmarks);
		}
	
	
         	 FILE* file = fopen("particleCord.txt", "a"); // open a file for writing
	 	 char output[255]; // a buffer to hold the output text

		//Obtaining the coordinates of particles and robot
		for (int i=0; i<N; i++)
		{
			particleCord.row(i) = particle[i].get();
	 	 	sprintf(output, "%10.9f %10.9f\n", particleCord(i,0), particleCord(i,1)); // fill the buffer with some information
	 		fputs(output, file); // write that buffer into the file
		}
	 	
		fclose(file); // c
		
		myrobotCord = myrobot.get();
         	FILE* file1 = fopen("myrobotCord.txt", "a"); 
	 	char output1[255]; 
	 	sprintf(output1, "%10.9f %10.9f\n", myrobotCord(0), myrobotCord(1));
	 	fputs(output1, file1); 
		fclose(file1); 

		//Comparing the coordinates of each particle to those of robot
		error[topIndex] = eval(particleCord, myrobotCord, world_size, N); 
	}
	


	//Printing the coordinates of particles and robot.
	for (int i=0; i<100; i++)
	{
		printf("xCord = %10.9f | yCord = %10.9f | orientation = %10.9f \n", 
		particleCord(i,0), particleCord(i,1), particleCord(i,2));
	}
	cout << "My robot:" << endl;
	printf("xCord = %10.9f | yCord = %10.9f | orientation = %10.9f \n", 
	myrobotCord(0), myrobotCord(1), myrobotCord(2));

	//Printing the error at each iteration.
	for (int i=0; i<iterations; i++)
	{
	cout << error[i] << endl;
	}	

	return 0;
}

//The formulae used below for dx and dy are a bit tricky but simply common sense. 
//Since the world is cyclic, a particle that falls off the laft edge of the robot's
//world appears on the right edge. So on a world of size 100(in both x, y dirct.)
//if robot is at x = 1 and a particle is at x = -1, the 'cyclicWorld' subprogram
//in robot_class turns this -1 into x = 99. Now when the distance b/w robot and 
//this particle is evaluated, we get 98 instead of a mere 2. So the code below 
//resolves this issue by diving the x-direction into 2 parts i.e., world_size/2
//and if the robot and particle are more than half the world_size, the particle is
//taken to be in the same half as the robot instead of it being in the other half.  
double eval(MatrixXd particleCord, Vector3d myrobotCord, double world_size, ULONG N) 
{
	double sum = 0.0, dx, dy, err;
	for (int i=0; i<N; i++)
	{
		dx = cyclicWorld(particleCord(i,0)-myrobotCord(0)+world_size/2.0, 
		world_size) - world_size/2.0;
		
		dy = cyclicWorld(particleCord(i,1)-myrobotCord(1)+world_size/2.0, 
		world_size) - world_size/2.0;
		
		err = sqrt(dx*dx + dy*dy);
		sum += err;
	}
return sum/(double)N;
}









