/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
// OPENCL KERNEL FOR EVALUATING WEIGHTS OF PARTICLES
#include "kernel_header.h"

//--------------------------
typedef struct{
float x[298];
float y[79];
//float x[277];
//float y[101];
//float x[575];
//float y[96];
//float x[600];
//float y[200];
float dist;
float theta; 
}linestruct;
//--------------------------
linestruct move_line_kernel(linestruct line, float turnAngle, 
   float moveDistance, int imgrows, int imgcols, float rngturn, float rngfwd) 
{
	line.theta = line.theta + turnAngle + rngturn;
	line.theta = cyclicWorld_kernel(line.theta, M_PI_F);
	
   moveDistance = moveDistance + rngfwd;
	float xincept = line.x[0]+moveDistance;
	float r=0.0f;
	for(int l=0; l<imgrows; l++)
	{
		line.y[l] = l; 
		r = line.y[l]/sin(line.theta);	
		line.x[l] = abs((int)(xincept + r * (cos(line.theta))));
		line.x[l] = cyclicWorld_kernel(line.x[l], imgcols);
	}
   return line;
}
//--------------------------
__kernel void goodLineWeightKernel(
              __global const linestruct *good_line,
              __global const linestruct *best_line,
              __global const int *img_grad,
              __constant const noiseStruct *allnoises,
              __constant const wtKernParams *params,
              __global linestruct *good_line_out,
              __global float *w_pf_out)
{       
    int idx_dim1 = get_global_id(0);
    __private float prob_dist = 0.0f;
    
    good_line_out[idx_dim1].dist = good_line[idx_dim1].dist; 
    good_line_out[idx_dim1].theta = good_line[idx_dim1].theta; 
     for(int l=0; l < params->imgrows; l++)
    {
    	good_line_out[idx_dim1].x[l] = good_line[idx_dim1].x[l]; 
    	good_line_out[idx_dim1].y[l] = good_line[idx_dim1].y[l]; 
    }

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
	 rn3 = rng_gaussian(allnoises->turnNoise, &rng);
	 rn4 = rng_gaussian(allnoises->forwardNoise, &rng);
    
    good_line_out[idx_dim1] = move_line_kernel( good_line_out[idx_dim1], rn1*turnmax, rn2*distmax, 
      params->imgrows, params->imgcols, rn3, rn4);
	 
    for(int l=0; l < params->imgrows; l++)
	 {	
	 	if(good_line_out[idx_dim1].x[l] >= params->NEIGHBORHOOD && 
         good_line_out[idx_dim1].x[l] <= params->imgcols - params->NEIGHBORHOOD){  
	     	for(int m = good_line_out[idx_dim1].x[l] - params->NEIGHBORHOOD; 
            m < good_line_out[idx_dim1].x[l] + params->NEIGHBORHOOD; m++)
	     		{
	     			good_line_out[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 	      }
	 	}
	  	else if(good_line_out[idx_dim1].x[l] > params->imgcols - params->NEIGHBORHOOD){  
	 		for(int m = good_line_out[idx_dim1].x[l] - params->NEIGHBORHOOD; m < params->imgcols; m++)
	 		{
	 			good_line_out[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 	else if(good_line_out[idx_dim1].x[l] < params->NEIGHBORHOOD){
	 		for(int m=0; m < good_line_out[idx_dim1].x[l] + params->NEIGHBORHOOD; m++)
	 		{
	 			good_line_out[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 }//end l
	 
	 w_pf_out[idx_dim1] = 0.0f;	
	 
    //Computing the weight of each particle
	 for(int gl=0; gl < params->NUM_BEST_LINES; gl++)
    {	
	    prob_dist = gaussian_kernel((good_line_out[idx_dim1].theta), allnoises->turnNoise,
         (best_line[gl].theta));	
	   // prob_dist += gaussian_kernel( good_line_out[idx_dim1].x[0], 
      //   allnoises->senseNoise, best_line[gl].x[0] );
	   // prob_dist += gaussian_kernel( good_line_out[idx_dim1].x[ params->imgrows - 1], 
      //   allnoises->senseNoise, best_line[gl].x[ params->imgrows - 1]);
	    	
       if( prob_dist > w_pf_out[idx_dim1])
	    {
	      w_pf_out[idx_dim1] = prob_dist;
	      //bl_points[gl]++;
	    }
	 } //CLOSING gl-LOOP.	

}//END KERNEL

