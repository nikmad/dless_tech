typedef struct{
	float xCord;
	float yCord;
	float orientation;	
	float forwardNoise;
	float turnNoise;
	float senseNoise;
} Robot;

typedef struct{
	float world_size;
	int landmarks_rows;
	int landmarks_cols; 
} MoveKernelParams; 
	
float cyclicWorld(float a, float b)
{
	if (a>=0)
	{
		return a-b*(int)(a/b);
	}
	else
	{
		return a+b*(1+(int)(fabs(a/b)));
	}	
}

float gaussian(float mu, float sigma, float x)
{
	return (1/sqrt(2*M_PI_F*pow(sigma,2)))*
		exp(- 0.5f*pow((x-mu),2)/pow(sigma,2));
}
	
__kernel void movekernel(__global const Robot *particle,
		  	__constant float *turn1,
		  	__constant float *dist1,
		  	__constant float *rngturn,
		  	__constant float *rngfwd,
		  	__constant float *measurement,
			__constant float *landmarks,
			__constant MoveKernelParams *move_params,
		  	__global Robot *particle_output,
		  	__global float *weight_output)
{
	int gid = get_global_id(0);

//Moving
	particle_output[gid].orientation = particle[gid].orientation + 
					turn1[gid] + rngturn[gid];
	particle_output[gid].orientation = 
		cyclicWorld(particle_output[gid].orientation,2*M_PI_F);
	
	__private float dist_local = dist1[gid] + rngfwd[gid];
	
	particle_output[gid].xCord = particle[gid].xCord + 
			(dist_local * cos(particle[gid].orientation));
	particle_output[gid].xCord = 
		cyclicWorld(particle_output[gid].xCord, move_params->world_size); 
	
	particle_output[gid].yCord = particle[gid].yCord + 
		(dist_local * sin(particle[gid].orientation));
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
//	w_output[gid] = 0.0f;
}


