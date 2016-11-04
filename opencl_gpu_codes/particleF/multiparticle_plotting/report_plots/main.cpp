#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "RNG.h"
//#include <eigen3/Eigen/Dense>
#include "robot.h"

using namespace Eigen;
using namespace std;


#define PI 3.141592653589793
typedef unsigned long int ULONG;

double eval(MatrixXd, Vector3d, double, ULONG); 

int main(void)
{
	clock_t t1_clock, t2_clock;
	t1_clock = clock();
	system("rm particleCord.txt");
	system("rm myrobotCord.txt");
	system("rm out1.png");
	system("rm out2.png");
	MatrixXd landmarks(4,2);
	landmarks << 20,20,
	80,80,
	20,80,
	80,20;
	double world_size = 100.0;
	ULONG N = 10000;
	int iterations = 40;
	double errorInitial;
	double error[iterations];
	Robot particle[N];
	Robot myrobot;
//	cout << sizeof(particle) << endl;
	cout << sizeof(myrobot) << endl;
	double srandSeed[N];
	VectorXd measurement(landmarks.rows());
	VectorXd w(N); //Particle weights
	MatrixXd particleCord(N,3);
	Vector3d myrobotCord;

	//Initializing myrobot:	
	setLandmarks_rob(&myrobot,landmarks);
	init_rob(&myrobot, world_size, time(NULL));
//	move_rob(&myrobot, 0.1, 1.0, world_size);
	measurement << sense_rob(&myrobot);
//	printf("%f\n", myrobot.xCord);
	
   
	myrobotCord = get_rob(&myrobot);
   FILE* file2 = fopen("myrobotCord.txt", "a"); 
	char output2[255]; 
	sprintf(output2, "%10.9f %10.9f\n", myrobotCord(0), myrobotCord(1));
	fputs(output2, file2); 
	fclose(file2); 
   	
	for(int i=0; i<N; i++)
	{
		//Number below is used to seed the random number generator.
	 	srandSeed[i] = (double)i; 
		init_rob(&particle[i], world_size, srandSeed[i]);
		//printf("%f\n", particle[i].xCord);
		setNoise_rob(&particle[i], 0.05, 0.05, 5.0);
		setLandmarks_rob(&particle[i], landmarks);
	//	get_rob(&particle[i]);
	}
        //New modifications.
	 	 FILE* file = fopen("particleCord.txt", "a"); // open a file for writing
	 	 char output[255]; // a buffer to hold the output text

		//Obtaining the coordinates of particles and robot
		for (int i=0; i<N; i++)
		{
			particleCord.row(i) = get_rob(&particle[i]);
	 	 	sprintf(output, "Iteration=1:\n"); // fill the buffer with some information
	 	 	sprintf(output, "%10.9f %10.9f %10.9f %10.9f\n", particleCord(i,0), particleCord(i,1), particleCord(i,2), 0.0); // fill the buffer with some information
	 		fputs(output, file); // write that buffer into the file
		}
	 	
		fclose(file); 

	//Evaluating the error before any particle filtering algorithm is used:
	for (int i=0; i<N; i++)
	{
		particleCord.row(i) = get_rob(&particle[i]);
	}
	myrobotCord = get_rob(&myrobot);
  	errorInitial = eval(particleCord, myrobotCord, world_size, N);	
		
	//The loop below is for changing the number of iterations 
	//for which the entire particle filter has to run so that
	//results can converge much better.
	for(int topIndex=0; topIndex<iterations; topIndex++) 
	{
		//Moving the robot and all the particles
		move_rob(&myrobot, 0.5, 5.0, world_size);
		measurement << sense_rob(&myrobot);
		double turn1,dist1;
      for(int i=0; i<N; i++)
      {
          int c1 = rand()%10000;
          turn1 = ((double)c1/10000.0000)*1.0;
          int c2 = rand()%10000;
          dist1 = ((double)c2/10000.0000)*10.0;
//        particle[i].move(turn1, dist1, world_size);
			 move_rob(&particle[i], turn1, dist1, world_size);
		}
		
		//Computing the weight of each particle
		for(int i=0; i<N; i++)
		{
			w[i] = measurementProb_rob(&particle[i], measurement);
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

       FILE* file1 = fopen("particleCord.txt", "a"); // open a file for writing
	 	 char output1[255]; // a buffer to hold the output text

		//Obtaining the coordinates of particles and robot
		for (int i=0; i<N; i++)
		{
			particleCord.row(i) = get_rob(&particle[i]);
	 	 	sprintf(output1, "%10.9f %10.9f %10.9f %10.9f\n", particleCord(i,0), particleCord(i,1), particleCord(i,2), w[i]); // fill the buffer with some information
	 		fputs(output1, file1); // write that buffer into the file
		}
	 	
		fclose(file1); // c
		
      
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
			beta += ((double)beta1/N) * ( 2 * mw );
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
			setNoise_rob(&particle[i], 0.05, 0.05, 5.0);
			setLandmarks_rob(&particle[i], landmarks);
		}
	
	
   //    FILE* file1 = fopen("particleCord.txt", "a"); // open a file for writing
	// 	 char output1[255]; // a buffer to hold the output text

	//	//Obtaining the coordinates of particles and robot
	//	for (int i=0; i<N; i++)
	//	{
	//		particleCord.row(i) = get_rob(&particle[i]);
	// 	 	sprintf(output1, "%10.9f %10.9f %10.9f %10.9f\n", particleCord(i,0), particleCord(i,1), particleCord(i,2), w[i]); // fill the buffer with some information
	// 		fputs(output1, file1); // write that buffer into the file
	//	}
	// 	
	//	fclose(file1); // c
		
		myrobotCord = get_rob(&myrobot);
      file2 = fopen("myrobotCord.txt", "a"); 
	 	sprintf(output2, "%10.9f %10.9f\n", myrobotCord(0), myrobotCord(1));
	 	fputs(output2, file2); 
		fclose(file2); 

		//Comparing the coordinates of each particle to those of robot
		error[topIndex] = eval(particleCord, myrobotCord, world_size, N); 
	}

	//Printing the coordinates of particles and robot.
	for (int i=0; i<1000; i++)
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
	
	t2_clock = clock();
    	float diff ((float)t2_clock-(float)t1_clock);
    	cout<<"Total Running Time = " << diff/CLOCKS_PER_SEC << "\n"<< endl;
     
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
