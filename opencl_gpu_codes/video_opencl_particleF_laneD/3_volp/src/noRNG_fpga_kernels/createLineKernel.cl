/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
#include "kernel_header.h"

typedef struct{
float x[511];
float y[72];
float dist;
float theta; 
} linestruct;

//------------------------- KERNEL 1 ---------------------------       

__kernel void createLineKernel(__global createLineParams * restrict cline_params,
                               __global int * restrict img_grad, 
                               __global float * restrict xadd, 
                               __global linestruct * restrict line_output)
{
   size_t ln = get_global_id(0);
   
   float r; 

	line_output[ln].dist = 0.0f;
	
   line_output[ln].theta = ln * (M_PI_F)/cline_params->NUM_LINES;
	line_output[ln].theta = cyclicWorld_kernel((float)line_output[ln].theta, (float)(M_PI_F));
	
   for(int l=0; l<cline_params->imgrows; l++)
	{
		line_output[ln].y[l] = l; 
		r = line_output[ln].y[l]/sin(line_output[ln].theta);	
		
		line_output[ln].x[l] = fabs((xadd[ln] + r * (cos(line_output[ln].theta))));
      line_output[ln].x[l] = cyclicWorld_kernel((float)(line_output[ln].x[l]), (float)(cline_params->imgcols));
	      
     if(line_output[ln].x[l] >= (cline_params->NEIGHBORHOOD) && line_output[ln].x[l] <= (cline_params->imgcols) - (cline_params->NEIGHBORHOOD)){  
	  for(int m=line_output[ln].x[l]-(cline_params->NEIGHBORHOOD); m < line_output[ln].x[l] + (cline_params->NEIGHBORHOOD); m++)
	  {
	      line_output[ln].dist += img_grad[l*(cline_params->imgcols) + m];
	  }
	  }
	  else if(line_output[ln].x[l] > (cline_params->imgcols) - (cline_params->NEIGHBORHOOD)){  
	  for(int m=line_output[ln].x[l] - (cline_params->NEIGHBORHOOD); m < cline_params->imgcols; m++)
	  {
	     	line_output[ln].dist += img_grad[l*(cline_params->imgcols) + m];
	  }
	  }
	  else if(line_output[ln].x[l] < (cline_params->NEIGHBORHOOD)){
	  for(int m=0; m < line_output[ln].x[l] + cline_params->NEIGHBORHOOD; m++)
	  {
	   	line_output[ln].dist += img_grad[l* cline_params->imgcols + m];
	  }
	  }
	}
}
