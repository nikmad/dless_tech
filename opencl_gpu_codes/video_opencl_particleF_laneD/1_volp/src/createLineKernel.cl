/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/

typedef struct{
//float x[298];
//float y[79];
//float x[277];
//float y[101];
//float x[575];
//float y[96];
float x[511];
float y[72];
float dist;
float theta; 
}linestruct;

typedef struct{
   int imgrows;
   int imgcols;  
   int NUM_LINES;
   int NEIGHBORHOOD;
}createLineParams;

float cyclicWorld(float a, float b)
{
	if (a>=0)
		return a-b*(int)(a/b);
	else
		return a+b*(1+(int)(fabs(a/b)));
}

__kernel void createLineKernel(__global createLineParams *cline_params,
                               __global int *img_grad, 
                               __global float *xadd, 
                               __global linestruct *line_output)
{
   int ln = get_global_id(0);
   
   float r; 

	line_output[ln].dist = 0.0f;
	
   line_output[ln].theta = ln * (M_PI_F)/cline_params->NUM_LINES;
	line_output[ln].theta = cyclicWorld((float)line_output[ln].theta, (float)(M_PI_F));
	
   for(int l=0; l<cline_params->imgrows; l++)
	{
		line_output[ln].y[l] = l; 
		r = line_output[ln].y[l]/sin(line_output[ln].theta);	
		
		line_output[ln].x[l] = fabs((xadd[ln] + r * (cos(line_output[ln].theta))));
      line_output[ln].x[l] = cyclicWorld((float)(line_output[ln].x[l]), (float)(cline_params->imgcols));
	      
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
