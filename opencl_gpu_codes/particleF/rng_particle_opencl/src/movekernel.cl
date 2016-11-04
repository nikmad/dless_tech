#include "rng_own.h"

__kernel void movekernel(__global const Robot *particle,
		  	__constant float *measurement,
			__constant float *landmarks,
			__constant MoveKernelParams *move_params,
		  	__global Robot *particle_output,
		  	__global float *weight_output)
{
	int gid = get_global_id(0);

   //Generating required random numbers.
	__private float turnmax = 1.0f, distmax = 10.0f; 
 	__private float rn1, rn2, rn3, rn4; //random numbers.
	__private int num_digits;
	__local mwc64x_state_t rng;
	__local ulong samplesPerStream;
	
	samplesPerStream = 50;
	MWC64X_SeedStreams(&rng,0xf00dcafe, samplesPerStream);
	//rn1	
	__private uint num = MWC64X_NextUint(&rng);
	num_digits = findn(num);
	rn1 = num/pow(10.0f, (float)(num_digits));
	//rn2
	num = MWC64X_NextUint(&rng);
	num_digits = findn(num);
	rn2 = num/pow(10.0f, (float)(num_digits));
	//gaussian rn3 & rn4
	rn3 = rng_gaussian(particle[gid].turnNoise, &rng);
	rn4 = rng_gaussian(particle[gid].forwardNoise, &rng);
     
   //Moving
	particle_output[gid].orientation = particle[gid].orientation + 
					rn1 * turnmax + rn3;
	particle_output[gid].orientation = 
		cyclicWorld(particle_output[gid].orientation,2*M_PI_F);
	
	float dist_local;
	
	dist_local = rn2 * distmax + rn4;
	
	particle_output[gid].xCord = particle[gid].xCord + 
			(dist_local * cos(particle_output[gid].orientation));
	particle_output[gid].xCord = 
		cyclicWorld(particle_output[gid].xCord, move_params->world_size); 
	
	particle_output[gid].yCord = particle[gid].yCord + 
		(dist_local * sin(particle_output[gid].orientation));
	particle_output[gid].yCord = 
		cyclicWorld(particle_output[gid].yCord, move_params->world_size);

  	particle_output[gid].forwardNoise = particle[gid].forwardNoise;
	particle_output[gid].turnNoise = particle[gid].turnNoise;
	particle_output[gid].senseNoise = particle[gid].senseNoise;

//Measurement Probability
	float prob = 1.0f;
	float dist;
	for (int i=0; i < move_params->landmarks_rows; i++)
	{
	dist = sqrt(pow(( particle_output[gid].xCord - 
		landmarks[i*move_params->landmarks_cols]),2)+
		pow(( particle_output[gid].yCord - 
		landmarks[i*move_params->landmarks_cols+1]),2));

	prob *= gaussian(dist, particle_output[gid].senseNoise,
			measurement[i]);
	}
	weight_output[gid] = prob;
}


