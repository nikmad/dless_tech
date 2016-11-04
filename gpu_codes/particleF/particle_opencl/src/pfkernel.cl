typedef struct{
	float xCord;
	float yCord;
	float orientation;	//This is angle in radians.
	float forwardNoise;
	float turnNoise;
	float senseNoise;
} Robot;
	

__kernel void pfkernel( __global Robot *particle_new,
			__global const Robot *particle,
			__private float fNoise,
			__private float tNoise,
			__private float sNoise)
{
    __private size_t gid = get_global_id(0);
 
  	particle_new[gid].xCord = particle[gid].xCord;
	particle_new[gid].yCord = particle[gid].yCord;
	particle_new[gid].orientation = particle[gid].orientation;
  	particle_new[gid].forwardNoise = fNoise;
	particle_new[gid].turnNoise = tNoise;
	particle_new[gid].senseNoise = sNoise;
}


