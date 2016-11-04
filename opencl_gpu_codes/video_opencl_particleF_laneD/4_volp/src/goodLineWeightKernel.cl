/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
#include "kernel_header.h"

typedef struct{
//float x[511];
float x[448];
//float y[72];
//float y[144];
float y[109];
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
   
 	float turnmax = 1.0f, distmax = 30.0f; 
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
	rn_turn = (num/pow(10.0f, (float)(num_digits))) * turnmax;
   
   //rn_dist
	num = MWC64X_NextUint(&rng);
	num_digits = findn(num);
	rn_dist = (num/pow(10.0f, (float)(num_digits))) * distmax;
	
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

   if(good_line_out[idx_dim1].x[0] <= (float)params->imgcols/2.0f)
   {
      rn_turn -= turnmax/3.0f;
      rn_dist -= distmax/3.0f;
   }
	
   good_line_out[idx_dim1].theta += rn_turn + rn_gaussturn;
   
	float xincept = fabs(good_line_out[idx_dim1].x[0] + rn_dist + rn_gaussdist);

   if(good_line_out[idx_dim1].x[0] <= (float)params->imgcols/2.0f)
   {
      //VISUAL xincept = cyclicWorld_kernel(xincept, (float)params->imgcols/4.0f) + (float)params->imgcols/4.0f; 
      xincept = cyclicWorld_kernel(xincept, (float)params->imgcols/6.0f) + 2.0f * (float)params->imgcols/6.0f; 
	   
      good_line_out[idx_dim1].theta = cyclicWorld_kernel(good_line_out[idx_dim1].theta, 50.0f*M_PI_F/180.0f) + 10.0f * M_PI_F/18.0f;
   }
   else
   {
      //VISUAL xincept = cyclicWorld_kernel(xincept-(float)params->imgcols/2.0f, 2.0*(float)params->imgcols/8.0f) + 5.0f*(float)params->imgcols/8.0f; 
      xincept = cyclicWorld_kernel(xincept-(float)params->imgcols/2.0f, 1.0*(float)params->imgcols/8.0f) + (float)params->imgcols/2.0f; 
	
      good_line_out[idx_dim1].theta = cyclicWorld_kernel(good_line_out[idx_dim1].theta, 50.0f*M_PI_F/180.0f) + M_PI_F/6.0f;
   }

	float r;
   r = 0.0f;

   if(good_line_out[idx_dim1].theta < M_PI_F/2)
   {
   	for(int l=0; l<params->imgrows; l++)
   	{
   		good_line_out[idx_dim1].y[l] = l; 
   		r = good_line_out[idx_dim1].y[l]/sin(good_line_out[idx_dim1].theta);	
   		good_line_out[idx_dim1].x[l] = (xincept + r * (cos(good_line_out[idx_dim1].theta)));
   	}
   }
   else
   {
   	for(int l=0; l<params->imgrows; l++)
   	{
   		good_line_out[idx_dim1].y[l] = l; 
   		r = good_line_out[idx_dim1].y[l]/sin( M_PI_F - good_line_out[idx_dim1].theta );	
   		good_line_out[idx_dim1].x[l] = xincept - r * (cos( M_PI_F - good_line_out[idx_dim1].theta ));
   	}
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
	   // prob_dist *= gaussian_kernel( good_line_out[idx_dim1].x[ params->imgrows - 1], 
      //   allnoises->senseNoise, best_line[gl].x[ params->imgrows - 1]);
	    	
       if( prob_dist > w_pf_out[idx_dim1])
	    {
	      w_pf_out[idx_dim1] = prob_dist;
	    }
	 } //CLOSING gl-LOOP.	
}//END KERNEL
