/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
#include "kernel_header.h"

typedef struct{
float x[511];
float y[72];
float dist;
float theta; 
} linestruct;

//------------------------- KERNEL 2 ---------------------------       

__kernel void goodLineWeightKernel(__global linestruct * restrict good_line,
                          __global linestruct * restrict best_line,
                          __global int * restrict img_grad,
                          __global float * restrict turnArray,
                          __global float * restrict distArray,
                          __global float * restrict rngturnArray,
                          __global float * restrict rngfwdArray,
                          __global noiseStruct * restrict allnoises,
                          __global wtKernParams * restrict params,
                          __global linestruct * restrict good_line_out,
                          __global float * restrict w_pf_out)
{       
    size_t idx_dim1 = get_global_id(0);
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

	good_line_out[idx_dim1].theta += turnArray[idx_dim1] + rngturnArray[idx_dim1];
   
   distArray[idx_dim1] += rngfwdArray[idx_dim1];

	float xincept = fabs(good_line_out[idx_dim1].x[0] + distArray[idx_dim1]);

   if(good_line_out[idx_dim1].x[0] <= (float)params->imgcols/2.0f)
   {
      xincept = cyclicWorld_kernel(xincept, (float)params->imgcols/4.0f) + (float)params->imgcols/4.0f; 
	   
      good_line_out[idx_dim1].theta = cyclicWorld_kernel(good_line_out[idx_dim1].theta, 50.0f*M_PI_F/180.0f) + 10.0f * M_PI_F/18.0f;
   }
   else
   {
      xincept = cyclicWorld_kernel(xincept-(float)params->imgcols/2.0f, 3.0*(float)params->imgcols/8.0f) + (float)params->imgcols/2.0f; 
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

