/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
#include "kernel_header.h"

typedef struct{
float x[109];
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
//   size_t GID = get_global_id(0);
   size_t GrID = get_group_id(0);
   size_t LID = get_local_id(0);
   int LSIZE = get_local_size(0);

   __local linestruct lines_local[1000]; 
    
    size_t idx = GrID*50 + LID;

    lines_local[LID].dist  = good_line[idx].dist  ;
    lines_local[LID].theta = good_line[idx].theta ;

    for(int l=0; l < params->imgrows; l++)
	 {
	 	lines_local[LID].x[l] = good_line[idx].x[l]; 
	 	lines_local[LID].y[l] = good_line[idx].y[l]; 
	 }
 	
    barrier(CLK_LOCAL_MEM_FENCE);
   
   //------------------------------ 
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
   
    //lines_local[LID].dist  = good_line[LID].dist  ;
    //lines_local[LID].theta = good_line[LID].theta ;

    //for(int l=0; l < params->imgrows; l++)
	 //{
	 //	lines_local[LID].x[l] = good_line[LID].x[l]; 
	 //	lines_local[LID].y[l] = good_line[LID].y[l]; 
	 //}

   //--------------------------------------------
   //             Moving the line
   //--------------------------------------------

   if(lines_local[LID].x[0] <= (float)params->imgcols/2.0f)
   {
      rn_turn -= turnmax/3.0f;
      rn_dist -= distmax/3.0f;
   }
	
   lines_local[LID].theta += rn_turn + rn_gaussturn;
   
	float xincept = fabs(lines_local[LID].x[0] + rn_dist + rn_gaussdist);

   if(lines_local[LID].x[0] <= (float)params->imgcols/2.0f)
   {
      //VISUAL xincept = cyclicWorld_kernel(xincept, (float)params->imgcols/4.0f) + (float)params->imgcols/4.0f; 
      xincept = cyclicWorld_kernel(xincept, (float)params->imgcols/6.0f) + 2.0f * (float)params->imgcols/6.0f; 
	   
      lines_local[LID].theta = cyclicWorld_kernel(lines_local[LID].theta, 50.0f*M_PI_F/180.0f) + 10.0f * M_PI_F/18.0f;
   }
   else
   {
      //VISUAL xincept = cyclicWorld_kernel(xincept-(float)params->imgcols/2.0f, 2.0*(float)params->imgcols/8.0f) + 5.0f*(float)params->imgcols/8.0f; 
      xincept = cyclicWorld_kernel(xincept-(float)params->imgcols/2.0f, 1.0*(float)params->imgcols/8.0f) + (float)params->imgcols/2.0f; 
	
      lines_local[LID].theta = cyclicWorld_kernel(lines_local[LID].theta, 50.0f*M_PI_F/180.0f) + M_PI_F/6.0f;
   }

	float r;
   r = 0.0f;

   if(lines_local[LID].theta < M_PI_F/2)
   {
   	for(int l=0; l<params->imgrows; l++)
   	{
   		lines_local[LID].y[l] = l; 
   		r = lines_local[LID].y[l]/sin(lines_local[LID].theta);	
   		lines_local[LID].x[l] = (xincept + r * (cos(lines_local[LID].theta)));
   	}
   }
   else
   {
   	for(int l=0; l<params->imgrows; l++)
   	{
   		lines_local[LID].y[l] = l; 
   		r = lines_local[LID].y[l]/sin( M_PI_F - lines_local[LID].theta );	
   		lines_local[LID].x[l] = xincept - r * (cos( M_PI_F - lines_local[LID].theta ));
   	}
   }
   //--------------------------------------------
	 
    lines_local[LID].dist = 0.0f;

    for(int l=0; l < params->imgrows; l++)
	 {	
	 	if(lines_local[LID].x[l] >= params->NEIGHBORHOOD && 
         lines_local[LID].x[l] <= params->imgcols - params->NEIGHBORHOOD){  
	     	for(int m = lines_local[LID].x[l] - params->NEIGHBORHOOD; 
            m < lines_local[LID].x[l] + params->NEIGHBORHOOD; m++)
	     		{
	     			lines_local[LID].dist += img_grad[l* params->imgcols + m];
	 	      }
	 	}
	  	else if(lines_local[LID].x[l] > params->imgcols - params->NEIGHBORHOOD){  
	 		for(int m = lines_local[LID].x[l] - params->NEIGHBORHOOD; m < params->imgcols; m++)
	 		{
	 			lines_local[LID].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 	else if(lines_local[LID].x[l] < params->NEIGHBORHOOD){
	 		for(int m=0; m < lines_local[LID].x[l] + params->NEIGHBORHOOD; m++)
	 		{
	 			lines_local[LID].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 }//end l
  
  w_pf_out[idx] = 0.0f;	
  
  //Computing the weight of each particle
	 for(int gl=0; gl < params->NUM_BEST_LINES; gl++)
    {	
	    prob_dist = gaussian_kernel(lines_local[LID].theta, allnoises->turnNoise,
         best_line[gl].theta);	
	   // prob_dist *= gaussian_kernel( lines_local[LID].x[0], 
      //   allnoises->senseNoise, best_line[gl].x[0] );
	   // prob_dist *= gaussian_kernel( lines_local[LID].x[ params->imgrows - 1], 
      //   allnoises->senseNoise, best_line[gl].x[ params->imgrows - 1]);
	    	
       if( prob_dist > w_pf_out[idx])
	    {
	      w_pf_out[idx] = prob_dist;
	    }
	 } //CLOSING gl-LOOP.	
    
    barrier(CLK_LOCAL_MEM_FENCE);
  
    good_line_out[idx].dist   = lines_local[LID].dist  ;
    good_line_out[idx].theta  = lines_local[LID].theta ;

    for(int l=0; l < params->imgrows; l++)
	 {
      good_line_out[idx].x[l]  = lines_local[LID].x[l];
      good_line_out[idx].y[l]  = lines_local[LID].y[l];
	 }
}//END KERNEL
