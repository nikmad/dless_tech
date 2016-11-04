/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
#include "kernel_header.h"

typedef struct{
//float x[511];
//float x[448];
//float y[72];
//float y[144];
//float y[240];
//float y[109];
float x[144];
float y[144];
//float x[432];
//float y[80];
float dist;
float theta; 
} lineStruct;

__kernel void goodLineWeightKernel(__global const lineStruct * restrict good_line,
                          __global const lineStruct * restrict best_line,
                          __global const int * restrict img_grad,
                          __constant noiseStruct * restrict allnoises,
                          __constant wtKernParams * restrict params,
                          __global lineStruct * restrict good_line_out,
                          __global float * restrict w_pf_out)
{       
   size_t idx_dim1 = get_global_id(0);
   
 	float turnmax = 0.05f, distmax = 2.0f; 
	float rn_turn = 0.0f, rn_dist = 0.0f, rn_gaussturn = 0.0f, rn_gaussdist = 0.0f; //random numbers.
	int num_digits = 0;
	mwc64x_state_t rng;
	ulong perStreamOffset = 40;
	uint num = 892346354U;
   ulong baseOffset = params->framme_seed;

	MWC64X_SeedStreams(&rng, baseOffset, perStreamOffset);
	
   //rn_turn	
	num = MWC64X_NextUint(&rng);
	num_digits = findn(num);
	rn_turn = (num/pow(10.0f, (float)(num_digits))) * turnmax - turnmax/2;
   
   //rn_dist
	num = MWC64X_NextUint(&rng);
	num_digits = findn(num);
	rn_dist = (num/pow(10.0f, (float)(num_digits))) * distmax - distmax/2;
	
   //gaussian rn_gaussturn & rn_gaussdist
	rn_gaussturn = rng_gaussian(allnoises->turnNoise, &rng);
	rn_gaussdist = rng_gaussian(allnoises->forwardNoise, &rng);
   //------------------------------ 
    float prob_dist;
    prob_dist = 0.0f;
   
    good_line_out[idx_dim1].dist  = good_line[idx_dim1].dist  ;
    good_line_out[idx_dim1].theta = good_line[idx_dim1].theta ;

    for(int l=0; l < params->imgrows; l++)
	 {
	 	good_line_out[idx_dim1].x[l] = good_line[idx_dim1].x[l]; 
	 	good_line_out[idx_dim1].y[l] = good_line[idx_dim1].y[l]; 
	 }

   //--------------------------------------------
   //             Moving the line
   //--------------------------------------------

	good_line_out[idx_dim1].theta += turnmax + rn_gaussturn;
	//good_line_out[idx_dim1].theta = cyclicWorld_kernel(good_line_out[idx_dim1].theta, M_PI_F);
	
	float xincept = good_line_out[idx_dim1].x[0] + distmax + rn_gaussdist;
	float r=0.0f;
	for(int l=0; l<params->imgrows; l++)
	{
		good_line_out[idx_dim1].y[l] = l; 
		r = good_line_out[idx_dim1].y[l]/sin(good_line_out[idx_dim1].theta);	
		good_line_out[idx_dim1].x[l] = abs((int)(xincept + r * (cos(good_line_out[idx_dim1].theta))));
		//good_line_out[idx_dim1].x[l] = cyclicWorld_kernel(good_line_out[idx_dim1].x[l], params->imgcols);
	}
   //--------------------------------------------
	 
    good_line_out[idx_dim1].dist = 0.0f;

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
	    prob_dist = gaussian_kernel(good_line_out[idx_dim1].theta, allnoises->turnNoise,
         best_line[gl].theta);	
	   // prob_dist *= gaussian_kernel( good_line_out[idx_dim1].x[0], 
      //   allnoises->senseNoise, best_line[gl].x[0] );
	    prob_dist *= gaussian_kernel( good_line_out[idx_dim1].x[ params->imgrows - 1], 
         allnoises->forwardNoise, best_line[gl].x[ params->imgrows - 1]);
	    	
       if( prob_dist > w_pf_out[idx_dim1])
	    {
	      w_pf_out[idx_dim1] = prob_dist;
	    }
	 } //CLOSING gl-LOOP.	
}//END KERNEL
