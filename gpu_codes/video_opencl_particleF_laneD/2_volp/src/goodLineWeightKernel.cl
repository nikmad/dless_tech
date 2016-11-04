/*(C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>*/
// OPENCL KERNEL FOR EVALUATING WEIGHTS OF PARTICLES

// 19.09.2013: Added a logic to avoid lines having improbable 
//             angles or the positions for the good-lines. In
//             case they somehow fall in these improbable regions,
//             they are re-adjusted to fall back in to meaningful 
//             regions. 

//----------------------------
typedef struct{
   float forwardNoise;
   float turnNoise;
   float senseNoise;
}noiseStruct;
//----------------------------
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
//----------------------------
typedef struct{
   int NEIGHBORHOOD;
   int NUM_GOOD_LINES;   
   int NUM_BEST_LINES;   
   int imgrows;
   int imgcols;
}wtKernParams;
//----------------------------
float cyclicWorld_kernel(float a, float b)
{
	if (a>=0)
		return a-b*(int)(a/b);
	else
		return a+b*(1+(int)(fabs(a/b)));
}
//----------------------------
float gaussian_kernel(float mu, float sigma, float x)
{
	return (1/sqrt(2.0f*M_PI_F*pow(sigma,2)))*exp(-0.5f*pow((x-mu),2)/pow(sigma,2));
}
//----------------------------
linestruct move_line_kernel(linestruct line, float turnAngle, 
   float moveDistance, int imgrows, int imgcols, float rngturn, float rngfwd) 
{
	line.theta = line.theta + turnAngle + rngturn;
	//line.theta = cyclicWorld_kernel(line.theta, M_PI_F);
   moveDistance = moveDistance + rngfwd;
	float xincept = fabs(line.x[0]+moveDistance);
  // xincept = (cyclicWorld_kernel(xincept, imgcols/2);

   if(line.x[0] <= imgcols/2)
   {
      //if(xincept < imgcols/5){
      //xincept = xincept + (float)imgcols/4.0f;}
      xincept = cyclicWorld_kernel(xincept, imgcols/4); 
//      if(xincept < imgcols/4)
//      {
         xincept = xincept + (float)imgcols/4.0f;
//      }
	   
      line.theta = cyclicWorld_kernel(line.theta, 50.0f*M_PI_F/180.0f) + 10.0f * M_PI_F/18.0f;
   }
   else
   {
      xincept = cyclicWorld_kernel(xincept-imgcols/2, 3*imgcols/8) + (float)imgcols/2.0f; 
	   line.theta = cyclicWorld_kernel(line.theta, 50.0f*M_PI_F/180.0f) + M_PI_F/6.0f;
   }

	float r=0.0f;
   if(line.theta < M_PI_F/2)
   {
   	for(int l=0; l<imgrows; l++)
   	{
   		line.y[l] = l; 
   		r = line.y[l]/sin(line.theta);	
   		line.x[l] = (xincept + r * (cos(line.theta)));
   	}
   }
   else
   {
   	for(int l=0; l<imgrows; l++)
   	{
   		line.y[l] = l; 
   		r = line.y[l]/sin( M_PI_F - line.theta );	
   		//line.x[l] = abs((int)(xincept - r * (cos( M_PI_F - line.theta ))));
   		line.x[l] = xincept - r * (cos( M_PI_F - line.theta ));
   	}

   }

//	for(int l=0; l<imgrows; l++)
//	{
//		line.y[l] = l; 
//		r = line.y[l]/sin(line.theta);	
//		line.x[l] = abs((int)(xincept + r * (cos(line.theta))));
////		line.x[l] = cyclicWorld_kernel(line.x[l], imgcols);
//	}

   return line;
}       
//------------------------- ACTUAL KERNEL BEGINS ---------------------------       

__kernel void goodLineWeightKernel(
              __global linestruct *good_line,
              __global linestruct *best_line,
              __global int *img_grad,
              __global float *turnArray,
              __global float *distArray,
              __global float *rngturnArray,
              __global float *rngfwdArray,
              __global noiseStruct *allnoises,
              __global wtKernParams *params,
              __global linestruct *good_line_out,
              __global float *w_pf_out)
{       
    int idx_dim1 = get_global_id(0);
    float prob_dist = 0.0f;
	
    good_line[idx_dim1] = move_line_kernel( good_line[idx_dim1], turnArray[idx_dim1], distArray[idx_dim1], 
      params->imgrows, params->imgcols, rngturnArray[idx_dim1], rngfwdArray[idx_dim1]);
	 
	 good_line[idx_dim1].dist = 0.0f;
	 
    for(int l=0; l < params->imgrows; l++)
	 {	
	 	if(good_line[idx_dim1].x[l] >= params->NEIGHBORHOOD && 
         good_line[idx_dim1].x[l] <= params->imgcols - params->NEIGHBORHOOD){  
	     	for(int m = good_line[idx_dim1].x[l] - params->NEIGHBORHOOD; 
            m < good_line[idx_dim1].x[l] + params->NEIGHBORHOOD; m++)
	     		{
	     			good_line[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 	      }
	 	}
	  	else if(good_line[idx_dim1].x[l] > params->imgcols - params->NEIGHBORHOOD){  
	 		for(int m = good_line[idx_dim1].x[l] - params->NEIGHBORHOOD; m < params->imgcols; m++)
	 		{
	 			good_line[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 	else if(good_line[idx_dim1].x[l] < params->NEIGHBORHOOD){
	 		for(int m=0; m < good_line[idx_dim1].x[l] + params->NEIGHBORHOOD; m++)
	 		{
	 			good_line[idx_dim1].dist += img_grad[l* params->imgcols + m];
	 		}
	 	}
	 }//end l
	 
	 w_pf_out[idx_dim1] = 0.0f;	
	 
    //Computing the weight of each particle
	 for(int gl=0; gl < params->NUM_BEST_LINES; gl++)
    {	
	    prob_dist = gaussian_kernel(good_line[idx_dim1].theta, allnoises->turnNoise,
         best_line[gl].theta);	
	   // prob_dist *= gaussian_kernel( good_line[idx_dim1].x[0], 
      //   allnoises->senseNoise, best_line[gl].x[0] );
	   // prob_dist *= gaussian_kernel( good_line[idx_dim1].x[ params->imgrows - 1], 
      //   allnoises->senseNoise, best_line[gl].x[ params->imgrows - 1]);
	    	
       if( prob_dist > w_pf_out[idx_dim1])
	    {
	      w_pf_out[idx_dim1] = prob_dist;
	      //bl_points[gl]++;
	    }
	 } //CLOSING gl-LOOP.	
    
    for(int l=0; l < params->imgrows; l++)
	 {
	 	good_line_out[idx_dim1].x[l] = good_line[idx_dim1].x[l]; 
	 	good_line_out[idx_dim1].y[l] = good_line[idx_dim1].y[l]; 
	 }
    
    good_line_out[idx_dim1].dist = good_line[idx_dim1].dist; 
    good_line_out[idx_dim1].theta = good_line[idx_dim1].theta; 

}//END KERNEL

