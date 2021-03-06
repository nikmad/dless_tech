// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

//28.09.2013: Added new code to save the output video to a file.

// 25.09.2013: Added clFinish() event command at several required points so
//      that unintentionally the program doesn't enter a piece of code without
//      finishing all the pre-requisites until then.

// 19.09.2013: (1) Added code to skip a few frames for displaying the detected
//      lanes in order to avoid the jerkiness in the output. However,
//      computation is obviously carried on each and every frame. Just 
//      that every step is not show for better visualization.
//      (2) Gaussian Blurring + Sobel filtering is introduced for gradient
//      image estimation, and removed the manual img_grad computation
//      that was followed before.

// 18.09.2013: Modified by rearranging the conditional loops of how the 
//             equal number of good lines are ensured in each half of ROI,
//             in lane detection step. 

// 17.09.2013: Added a logic to ensure equal number of lines to have their 
//             x-intercept in either halves of the ROI for the lane tracking 
//             step. However, this doesn't ensure that they have meaningful angles.

// 11.09.2013: Modified the method for extracting good lines by removing
//       GOOD_LINE_DIST criteria. Now good lines can be spaced in anyway 
//       except that half the lines should be in 1st half and half in 2nd.
//       However, this assurance of half the lines in each half of the ROI
//       is only for "Image processing step"/lane-detection step but there 
//       might still be skewing of all lines in a single half in 
//       particle-filter step.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "RNG.h"
#include "robot.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "clll.h"

using namespace Eigen;
using namespace std;

#define CV_LOAD_IMAGE_GRAYSCALE 0
#define CV_LOAD_IMAGE_COLOR 1
#define CV_WINDOW_NORMAL 0

float TURNMAX = 1.5;
float DISTMAX = 30.0;

typedef struct{
   int imgrows;
   int imgcols;  
   int NUM_LINES;
   int NEIGHBORHOOD;
}createLineParams;

typedef struct{
   float GOOD_LINE_DIST;
   int NUM_LINES;   
   int NUM_GOOD_LINES;
   int count;
   int imgrows;
   int imgcols;
   float dist_val_thisLine;
}goodParams;

bool CreateMemCreateLine(cl_context, cl_mem *, createLineParams,
   int *, lineStruct *, int, int, int);

void CleanupCreateLineKernel(cl_context, cl_command_queue,
   cl_program, cl_kernel, cl_mem *);

typedef unsigned long int ULONG;

bool CreateMemWt(cl_context, cl_mem *,
   lineStruct *, lineStruct *, int *, 
   noiseStruct *, wtKernParams *, 
   lineStruct *, float *);

void CleanupWtKernel(cl_context, 
   cl_command_queue, 
   cl_program, 
   cl_kernel, 
   cl_mem *);

int main( int argc, char** argv )
{
   int BEST_LINE_DIST = atoi(argv[1]);
	int NUM_BEST_LINES = atoi(argv[2]);
 
   string input_video_file = "../../../../1_dependencies/video/clip4"; 
   
   cv::VideoCapture capture(input_video_file);
   cv::Mat img_org_full;
   if( !capture.isOpened() )
       throw "Error when reading steam_avi";
   
   cv::VideoWriter output_cap("output.avi",  
            capture.get(CV_CAP_PROP_FOURCC),
            capture.get(CV_CAP_PROP_FPS), 
            cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), 
            capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
   
       if (!output_cap.isOpened())
       {
           std::cout << "!!! Output video could not be opened" << std::endl;
           return -1;
       }
  
   cv::namedWindow( "Lane Detection", CV_WINDOW_AUTOSIZE);
   //cv::moveWindow( "Lane Detection", 200, 300);
 
   double num_frames;
   num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

   int total_v_frames = 0;
   int Iframes = 1000;//Take accurate measurement after every Iframes.
   int NUM_GOOD_LINES = 300;
   int iterations = 1;
   int NUM_LINES = 3000;
   int NEIGHBORHOOD = 3;

   lineStruct *good_line;
   good_line = new lineStruct [NUM_GOOD_LINES];
   
   lineStruct *best_line;
   best_line = new lineStruct [NUM_BEST_LINES];
  
   int best_line_idx[NUM_BEST_LINES];
   float good_dist_vect[NUM_GOOD_LINES];
   
   int good_line_idx[NUM_GOOD_LINES];
   float dist_vect[NUM_LINES];
   int count =0;
   
   int x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0;  
   //int x1=0, y1=0, x2=0, y2=0;  
   //int x21=0, y21=0, x22=0, y22=0, x23=0, y23=0, x24=0, y24=0;  
   int x21=0, y21=0, x22=0, y22=0;  
   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_tmp[NUM_BEST_LINES];
   
   for(int framme=0; framme < num_frames; framme++)
   {
      clock_t t1_clock, t2_clock;
      t1_clock = clock();
    
      cout << "Frame Number of the Video = " << total_v_frames+1 << endl;
      capture >> img_org_full;
   
      cv::Rect rect;
      int ROI_START_X = (int)(0.20*img_org_full.cols);// % of image cropped in the right
      int ROI_START_Y = (int)(0.80*img_org_full.rows);// % of image cropped in the top
      rect = cv::Rect(0,ROI_START_Y,img_org_full.cols-ROI_START_X,img_org_full.rows-ROI_START_Y);
      
      cv::Mat img_gray_full;
      cv::cvtColor(img_org_full, img_gray_full, CV_RGB2GRAY, 1); 
      cv::Mat img_gray = img_gray_full(rect);
 
      cv::Mat img_gray_blur(img_gray.rows, img_gray.cols, CV_8UC1);  
	   cv::Mat img_grad(img_gray.rows, img_gray.cols, CV_8UC1); //gradient matrix
     
      cv::Size kernSize = cv::Size( 3, 3 );
      cv::GaussianBlur(img_gray,img_gray_blur,
                    kernSize, 0.4,0.4,
                    cv::BORDER_DEFAULT);
      
      cv::Sobel(img_gray, img_grad, -1,
                   3, 3,  
                   7, 1, 0,
                   cv::BORDER_DEFAULT);
      
      int *img_grad_array;
      img_grad_array = new int [img_grad.rows*img_grad.cols]; 	
   
   	float GOOD_LINE_DIST = img_grad.cols/(4*NUM_GOOD_LINES);
      
      createLineParams cline_params;
      cline_params.NEIGHBORHOOD = NEIGHBORHOOD;
      cline_params.NUM_LINES = NUM_LINES;
      cline_params.imgrows = img_grad.rows;
      cline_params.imgcols = img_grad.cols;
   	
   
   //WHILE INTRO SOBEL//	for(int row=0; row < img_grad.rows; row++) {
   //WHILE INTRO SOBEL//	for(int col=0; col < img_grad.cols; col++) {
   //WHILE INTRO SOBEL//	img_grad.at<uchar>(row, col) =  (uchar)abs(
   //WHILE INTRO SOBEL//					(int)img_gray.at<uchar>(row, col+1) - 
   //WHILE INTRO SOBEL//			 	     	(int)img_gray.at<uchar>(row, col));
   //WHILE INTRO SOBEL//	}
   //WHILE INTRO SOBEL//	}
   	
   	double maxVal=0; 
   	double minVal=0;
   	cv::minMaxLoc(img_grad, &minVal, &maxVal, 0, 0);
   
   	cout << "MaxValue: " << maxVal << endl;
   	
   	for(int row=0; row < img_grad.rows; row++) {
   	for(int col=0; col < img_grad.cols; col++) {
   	if((int)img_grad.at<uchar>(row, col) < 0.1*maxVal) 
   	img_grad.at<uchar>(row, col) = 0;
   	}
   	}
   
      for(int l=0; l<img_grad.rows; l++)
      for(int m=0; m<img_grad.cols; m++)
         {
            img_grad_array[l*img_grad.cols+m] = (int)img_grad.at<uchar>(l,m);      
         }
   	
   	cout << img_grad.rows << endl;
   	cout << img_grad.cols << endl;
   	
      if(total_v_frames%Iframes == 0)
      {
         cout << "Im here in if" << endl;
         lineStruct *line;
         line = new lineStruct [NUM_LINES];
         
         lineStruct *line_new;
         line_new = new lineStruct [NUM_LINES];
         
         //OpenCL parameters
         cl_context context = 0;
         cl_command_queue commandQueue = 0;
         cl_program program = 0;
         cl_device_id device = 0;
         cl_kernel kernel = 0;
         cl_int errNum;
         cl_mem memCreateLine[3] = {0, 0, 0};
         
         // Create an OpenCL context on first available platform
         context = CreateContext();
         if (context == NULL)
         {
              cerr << "Failed to create OpenCL context." <<  endl;
             return 1;
         }
         
         // Create a command-queue on the first device available
         // on the created context
         commandQueue = CreateCommandQueue(context, &device);
         if (commandQueue == NULL)
         {
             cerr << "Failed to create commandQueue." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);
             return 1;
         }
         
         //_________________________________________________________________
         //
         // KERNEL1 BEGIN: "OpenCL-ling" CREATING LINES
         //_________________________________________________________________
         
         //int c2;
         //float *xadd;
         //xadd = new float [NUM_LINES];
         //
         //for(int ln=0; ln<NUM_LINES; ln++)
         //{
         //   srand(ln*10000+20*ln*ln*ln);
         //   c2 = rand()%10000;
         //   xadd[ln] = (float)(((float)c2/10000.0000) * img_grad.cols);
         //}
         
         //-----------
         program = CreateProgram(context, device, "createLineKernel.cl");
         if (program == NULL)
         {
              cerr << "Failed: CreateProgram of createLineKernel" <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program,
             	 kernel, memCreateLine);
             return 1;
         }
         //-----------
         //Unload the compiler when program building is over.
         clUnloadCompiler(); 
         //-----------
         // Create OpenCL kernel
         kernel = clCreateKernel(program, "createLineKernel", NULL);
         if (kernel == NULL)
         {
              cerr << "Failed: clCreateKernel of createLineKernel" <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);
             return 1;
         }
         //-----------
         //Creating Memory Objects
         if (!CreateMemCreateLine(context, memCreateLine, cline_params,
            img_grad_array, line_new, img_grad.rows, 
            img_grad.cols, NUM_LINES))
         {
             cerr << "Failed: CreateMemCreateLine" <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
               kernel, memCreateLine);
             return 1;
         }
         //-----------
         //Enqueuing the created memory objects.
         errNum = clEnqueueWriteBuffer( 
         		commandQueue, 
         		memCreateLine[0], 
         		CL_TRUE, 
         		0,
         		sizeof(createLineParams), 
         		(void *)&cline_params, 
         		0,
         		NULL, 
         		NULL);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error: enqueuing cline_params buffer." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);          
             return 1;
         }
         //-----------
         errNum = clEnqueueWriteBuffer( 
         		commandQueue, 
         		memCreateLine[1], 
         		CL_TRUE, 
         		0,
         		img_grad.rows*img_grad.cols*sizeof(int), 
         		(void *)img_grad_array, 
         		0,
         		NULL, 
         		NULL);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error: enqueuing img_grad_array buffer." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);          
             return 1;
         }
         //-----------
         //errNum = clEnqueueWriteBuffer( 
         //		commandQueue, 
         //		memCreateLine[2], 
         //		CL_TRUE, 
         //		0,
         //		NUM_LINES*sizeof(float), 
         //		(void *)xadd, 
         //		0,
         //		NULL, 
         //		NULL);
         //
         //if (errNum != CL_SUCCESS)
         //{
         //    cerr << "Error: enqueuing xadd buffer." <<  endl;
         //    CleanupCreateLineKernel(context, commandQueue, program, 
         //    	kernel, memCreateLine);          
         //    return 1;
         //}
         //-----------
         //Setting the kernel arguments
         errNum  = clSetKernelArg(kernel, 0, 
             sizeof(cl_mem),(void *)&memCreateLine[0]);
         errNum |= clSetKernelArg(kernel, 1, 
             sizeof(cl_mem),(void *)&memCreateLine[1]);
         //errNum |= clSetKernelArg(kernel, 2, 
         //    sizeof(cl_mem),(void *)&memCreateLine[2]);
         errNum |= clSetKernelArg(kernel, 2, 
             sizeof(cl_mem),(void *)&memCreateLine[2]);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error setting kernel arguments." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);    
             return 1;
         }
         //-----------
         // Queue the kernel up for execution across the array
         
        // size_t globalWorkSize[1] = { NUM_LINES };
        // size_t localWorkSize[1] = { 1 };
         
         size_t globalWorkSize[1] = { NUM_LINES };
         //size_t localWorkSize[1] = { (int)(NUM_LINES/50)  };
         size_t localWorkSize[1] = { 250 };
        
         errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                         globalWorkSize, localWorkSize,
                                         0, NULL, NULL);
         if (errNum != CL_SUCCESS)
         {
              cerr << "Error queuing kernel for execution." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);
             return 1;
         }
         //-----------
         clFinish(commandQueue);
         //-----------
          
         // Read the output buffer back to the Host
         errNum = clEnqueueReadBuffer(commandQueue, memCreateLine[2], 
             	CL_TRUE, 0, NUM_LINES * sizeof(lineStruct), 
             	line_new, 0, NULL, NULL);
         
         if (errNum != CL_SUCCESS)
         {
              cerr << "Error reading result buffer." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);
             return 1;
         }
         //-----------
         //Write the output to a file
         for (int i = 0; i < NUM_LINES; i++)
         {
             line[i] = line_new[i];	
         }
         
         //-----------
         clFinish(commandQueue);
         //-----------
         cout << "line[0].x[0] = " << line[0].x[0] << endl;
                     
         CleanupCreateLineKernel(context, commandQueue, program, 
         	kernel, memCreateLine);
         
         //Writing the OUTPUT FILE
         //FILE* file_lines_ocl = fopen("lines_opencl.txt", "a"); 
         //char output_lines_ocl[255]; 
         //
         //for (int i=0; i<NUM_LINES; i++)
         //{
         //sprintf(output_lines_ocl, "%10.9f    %10.9f\n", 
         //	line[i].dist, line[i].theta); 
         //fputs(output_lines_ocl, file_lines_ocl); 
         //}
         //fclose(file_lines_ocl);
         cout << "Lines created succefully." << endl;
         //________________________________________________________________
         //
         // KERNEL1 END: "OpenCL-ling" CREATING LINES
         //________________________________________________________________
         
         //************* GOOD LINES (C STYLE BELOW) ***************
         
         for(int ln=0; ln<NUM_LINES; ln++)
         {
         dist_vect[ln] = line[ln].dist;
         }
         
         std::vector<float> myvector(dist_vect, dist_vect+NUM_LINES-1);
         std::sort (myvector.begin(), myvector.end()); 
         count =0;
         int fhalf = 0, shalf = 0; 
         for( std::vector<float>::const_iterator i = myvector.end(); i != myvector.begin(); i--)
         {
            if(*i != (float)0 && count <= NUM_GOOD_LINES){
         	   for(int ln=0; ln<NUM_LINES; ln++)
         	   {
                  if(dist_vect[ln] == *i)
         	   	{
                    // if(fhalf <= NUM_GOOD_LINES/2 && shalf <= NUM_GOOD_LINES/2)
                    // {
                    //    		good_line_idx[count] = ln;
                    //    		count++;

                    //          if(line[ln].x[0] <= img_grad.cols/2)
                    //          {
                    //             fhalf++;
                    //          }
                    //          if(line[ln].x[0] > img_grad.cols/2)
                    //          {
                    //             shalf++;
                    //          }
                    // }//end if line[ln].x[0]
                  
                     //if(line[ln].x[0] <= img_grad.cols/2 && fhalf <= NUM_GOOD_LINES/2 && shalf >= NUM_GOOD_LINES/2)
                     if(line[ln].x[0] <= img_grad.cols/2 && fhalf <= NUM_GOOD_LINES/2)
                     {
                  	   		good_line_idx[count] = ln;
                  	   		count++;
                              fhalf++;
                     }
                     //if(line[ln].x[0] > img_grad.cols/2 && shalf <= NUM_GOOD_LINES/2 && fhalf >= NUM_GOOD_LINES/2)
                     if(line[ln].x[0] > img_grad.cols/2 && shalf <= NUM_GOOD_LINES/2 )
                     {
                  	   		good_line_idx[count] = ln;
                  	   		count++;
                              shalf++;
                     }
                  } //end if(distvect)
               }//END for 'ln'
             }
         }//end i
         
         
         for(int gl=0; gl<NUM_GOOD_LINES; gl++)
         {
         good_line[gl] = line[good_line_idx[gl]];
         }
         cout << "Good lines extracted successfully!" << endl;
         //*********** BEST LINES (C STYLE) ********************
         
         for(int ln=0; ln<NUM_GOOD_LINES; ln++)
         {
         good_dist_vect[ln] = good_line[ln].dist;
         }
         
         std::vector<float> mygoodvector(good_dist_vect, good_dist_vect+NUM_GOOD_LINES-1);
         std::sort (mygoodvector.begin(), mygoodvector.end()); 
         
         count = 0;
         bool intersect = 0;
         for( std::vector<float>::const_iterator i = mygoodvector.end(); i != mygoodvector.begin(); i--)
         {
         	if(*i != (float)0 ) 
         	for(int ln=0; ln<NUM_GOOD_LINES; ln++)
         	{
         		bool intersect =0;
               if(good_dist_vect[ln] == *i && count<NUM_BEST_LINES)
         		{
         			if(count == 0){
               			best_line_idx[count] = ln;
               			count++;
         			}
         			else 
         				if (count > 0)
         				{
         			      for(int kl=0; kl<count; kl++){
         		               	if(abs( good_line[ln].x[img_grad.rows-1] -
         		               		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
         		               		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                                    )
         		               		{
         		               		   if( kl<count-1 ) 
         		               		      continue;
                                       else{ 
                  	   	               best_line_idx[count] = ln;
                  	   	              	count++;
                                            }
         		               		}
         		               	if(abs( good_line[ln].x[img_grad.rows-1] -
         		               		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) < 
         		               		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                                    )         break;
         		               }
                     }
         		}
         	}//END ln
         }
         
         
         for(int bl=0; bl<NUM_BEST_LINES; bl++)
         {
         best_line[bl] = good_line[best_line_idx[bl]];
         }
         cout << "Best lines extracted successfully!" << endl;
         total_v_frames++;
      } //END total_video_frames =0 step.
      
     // if(total_v_frames%20 == 0)
      else
      { //IF VIDEO FRAME > 0
         cout << "Im here in else" << endl;
         //___________________________________________________________________
         //
         //                          PARTICLE FILTER
         //___________________________________________________________________
         
         noiseStruct *allnoises;
         allnoises = new noiseStruct [1];
         
         allnoises->forwardNoise = 0.05;
         allnoises->turnNoise = 0.05;
         allnoises->senseNoise = 100.5;
         
         float turn1 =0.0, dist1 = 0.0;
         int c1 = 0;
         
         float w_pf[NUM_GOOD_LINES];
         
         for(int topIndex=0; topIndex<iterations; topIndex++) 
         {
         	float weight_pf = 0.0;
         	float prob_dist = 0.0;
            	
            //************* PARTICLE WEIGHTS (OPENCL STYLE BELOW) *********************
            //_________________________________________________________________
            //
            // KERNEL3 BEGIN: "OpenCL-ling" EVALUATION OF PARTICLE WEIGHTS
            //_________________________________________________________________
            
            //float *turnArray;
            //turnArray = new float [NUM_GOOD_LINES];
            //float *distArray;
            //distArray = new float [NUM_GOOD_LINES];
            //float *rngturnArray;
            //rngturnArray = new float [NUM_GOOD_LINES];
            //float *rngfwdArray;
            //rngfwdArray = new float [NUM_GOOD_LINES];
            //
            wtKernParams wtparams;
         
            lineStruct *good_line_new;
            good_line_new = new lineStruct [NUM_GOOD_LINES];
         
            float *w_pf_new;
            w_pf_new = new float [NUM_GOOD_LINES];
           
            //int c2 =0; 
            ////-----------
            //for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            //{
            //   c1 = rand()%10000;
            //   turnArray[gl] = ((float)c1/10000.0000)*TURNMAX - TURNMAX/2;
            //   //turnArray[gl] = ((float)c1/10000.0000)*TURNMAX;
            //         
            //   c2 = rand()%10000;
            //   distArray[gl] = ((float)c2/10000.0000)*DISTMAX - DISTMAX/2;
            //   //distArray[gl] = ((float)c2/10000.0000)*DISTMAX;
         
            //   RNG rng(10090329*gl + time(NULL)*gl*gl*gl + 2346*gl);
         	//   rngturnArray[gl] = rng.gaussian(allnoises->turnNoise);
         	//   rngfwdArray[gl] = rng.gaussian(allnoises->forwardNoise);
         	//}
            
            wtparams.NEIGHBORHOOD = NEIGHBORHOOD;
            wtparams.NUM_GOOD_LINES = NUM_GOOD_LINES;
            wtparams.NUM_BEST_LINES = NUM_BEST_LINES;
            wtparams.imgrows = img_grad.rows;  
            wtparams.imgcols = img_grad.cols;
            
            //srand(framme*10000+20*framme*framme*framme + 342895* time(NULL) + time(NULL));
            srand((framme*13434976+24385*framme*framme*framme)*time(NULL));
            wtparams.framme_seed = rand()%100000000;
            //-----------
            cl_context context_wt = 0;
            cl_command_queue commandQueue_wt = 0;
            cl_program program_wt = 0;
            cl_device_id device_wt = 0;
            cl_kernel kernel_wt = 0;
            cl_int errNum_wt;
            cl_mem mem_wt[7] = {0,0,0,0,0,0,0};
            //-----------
            context_wt = CreateContext();
            if (context_wt == NULL)
            {
                 cerr << "Failed to create OpenCL context_wt." <<  endl;
                return 1;
            }
            //-----------
            commandQueue_wt = CreateCommandQueue(context_wt, &device_wt);
            if (commandQueue_wt == NULL)
            {
                cerr << "Failed to create commandQueue_wt." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);
                return 1;
            }
            //-----------
            program_wt = CreateProgram(context_wt, device_wt, "goodLineWeightKernel.cl");
            if (program_wt == NULL)
            {
                 cerr << "Failed: CreateProgram of goodLineWeightKernel" <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt,
                	 kernel_wt, mem_wt);
                return 1;
            }
            //-----------
            //Unload the compiler when program building is over.
            clUnloadCompiler(); 
            //-----------
            // Create OpenCL kernel
            kernel_wt = clCreateKernel(program_wt, "goodLineWeightKernel", NULL);
            if (kernel_wt == NULL)
            {
                 cerr << "Failed: clCreateKernel of goodLineWeightKernel" <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);
                return 1;
            }
            //-----------
            //Creating Memory Objects
            if (!CreateMemWt(context_wt, mem_wt, good_line, best_line, 
                  img_grad_array, allnoises, &wtparams, good_line_new, w_pf_new))
            {
                cerr << "Failed: CreateMemWt mem_wt" <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                  kernel_wt, mem_wt);
                return 1;
            }
            //-----------
            //Enqueuing the created memory objects.
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[0], 
            		CL_TRUE, 
            		0,
            		NUM_GOOD_LINES*sizeof(lineStruct), 
            		(void *)good_line, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing good_line buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[1], 
            		CL_TRUE, 
            		0,
            		NUM_BEST_LINES*sizeof(lineStruct), 
            		(void *)best_line, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing best_line buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[2], 
            		CL_TRUE, 
            		0,
                  img_grad.rows*img_grad.cols*sizeof(int), 
         		   (void *)img_grad_array, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing img_grad_array buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            //errNum_wt = clEnqueueWriteBuffer( 
            //		commandQueue_wt, 
            //		mem_wt[3], 
            //		CL_TRUE, 
            //		0,
            //      NUM_GOOD_LINES*sizeof(float), 
         	//	   (void *)turnArray, 
            //		0,
            //		NULL, 
            //		NULL);
            //
            //if (errNum_wt != CL_SUCCESS)
            //{
            //    cerr << "Error: enqueuing turnArray buffer." <<  endl;
            //    CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            //    	kernel_wt, mem_wt);          
            //    return 1;
            //}
            ////-----------
            //errNum_wt = clEnqueueWriteBuffer( 
            //		commandQueue_wt, 
            //		mem_wt[4], 
            //		CL_TRUE, 
            //		0,
            //      NUM_GOOD_LINES*sizeof(float), 
         	//	   (void *)distArray, 
            //		0,
            //		NULL, 
            //		NULL);
            //
            //if (errNum_wt != CL_SUCCESS)
            //{
            //    cerr << "Error: enqueuing distArray buffer." <<  endl;
            //    CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            //    	kernel_wt, mem_wt);          
            //    return 1;
            //}
            ////-----------
            //errNum_wt = clEnqueueWriteBuffer( 
            //		commandQueue_wt, 
            //		mem_wt[5], 
            //		CL_TRUE, 
            //		0,
            //      NUM_GOOD_LINES*sizeof(float), 
         	//	   (void *)rngturnArray, 
            //		0,
            //		NULL, 
            //		NULL);
            //
            //if (errNum_wt != CL_SUCCESS)
            //{
            //    cerr << "Error: enqueuing rngturnArray buffer." <<  endl;
            //    CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            //    	kernel_wt, mem_wt);          
            //    return 1;
            //}
            ////-----------
            //errNum_wt = clEnqueueWriteBuffer( 
            //		commandQueue_wt, 
            //		mem_wt[6], 
            //		CL_TRUE, 
            //		0,
            //      NUM_GOOD_LINES*sizeof(float), 
         	//	   (void *)rngfwdArray, 
            //		0,
            //		NULL, 
            //		NULL);
            //
            //if (errNum_wt != CL_SUCCESS)
            //{
            //    cerr << "Error: enqueuing rngfwdArray buffer." <<  endl;
            //    CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            //    	kernel_wt, mem_wt);          
            //    return 1;
            //}
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[3], 
            		CL_TRUE, 
            		0,
                  sizeof(noiseStruct), 
         		   (void *)allnoises, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing allnoises buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[4], 
            		CL_TRUE, 
            		0,
                  sizeof(wtKernParams), 
         		   (void *)&wtparams, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing wtparams buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            //Setting the kernel arguments
            errNum_wt  = clSetKernelArg(kernel_wt, 0, 
                sizeof(cl_mem),(void *)&mem_wt[0]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 1, 
                sizeof(cl_mem),(void *)&mem_wt[1]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 2, 
                sizeof(cl_mem),(void *)&mem_wt[2]);
            
            //errNum_wt |= clSetKernelArg(kernel_wt, 3, 
            //    sizeof(cl_mem),(void *)&mem_wt[3]);
            //
            //errNum_wt |= clSetKernelArg(kernel_wt, 4, 
            //    sizeof(cl_mem),(void *)&mem_wt[4]);
            //
            //errNum_wt |= clSetKernelArg(kernel_wt, 5, 
            //    sizeof(cl_mem),(void *)&mem_wt[5]);
            //
            //errNum_wt |= clSetKernelArg(kernel_wt, 6, 
            //    sizeof(cl_mem),(void *)&mem_wt[6]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 3, 
                sizeof(cl_mem),(void *)&mem_wt[3]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 4, 
                sizeof(cl_mem),(void *)&mem_wt[4]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 5, 
                sizeof(cl_mem),(void *)&mem_wt[5]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 6, 
                sizeof(cl_mem),(void *)&mem_wt[6]);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error setting kernel arguments." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);    
                return 1;
            }
            //-----------
            // Queue the kernel up for execution across the array
            //size_t localWorkSize_wt[1] = { (int)(NUM_GOOD_LINES/50) };
            //size_t globalWorkSize_wt[1] = { localWorkSize_wt[0]*50 };
            size_t globalWorkSize_wt[1] = { NUM_GOOD_LINES };
            size_t localWorkSize_wt[1] = { 50 };
            
            errNum_wt = clEnqueueNDRangeKernel(commandQueue_wt, kernel_wt, 1, NULL,
                                            globalWorkSize_wt, localWorkSize_wt,
                                            0, NULL, NULL);
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error queuing weight kernel for execution." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);    
                return 1;
            }
            //-----------
            clFinish(commandQueue_wt);
            //-----------
            // Read the output buffer back to the Host
            //OUTPUT 1
            errNum_wt = clEnqueueReadBuffer(
                           commandQueue_wt, 
                           mem_wt[5],
                           CL_TRUE, 0, 
                           NUM_GOOD_LINES * sizeof(lineStruct), 
                           good_line_new, 
                           0, NULL, NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error reading good_line_new buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);    
                return 1;
            }
            //OUTPUT 2
            errNum_wt = clEnqueueReadBuffer(
                           commandQueue_wt, 
                           mem_wt[6],
                           CL_TRUE, 0, 
                           NUM_GOOD_LINES * sizeof(float),
                           w_pf_new, 
                           0, NULL, NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error reading w_pf_new buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);    
                return 1;
            }
            //-----------
            //ASSIGNING THE OUTPUT VARIABLES TO THE LOCAL VARIABLES
            for (int gl = 0; gl < NUM_GOOD_LINES; gl++)
            {
                good_line[gl] = good_line_new[gl];	
                w_pf[gl] = w_pf_new[gl];
            //    cout << "w_pf: " << w_pf[gl] << endl; 
            //    cout << "x[0]: " << good_line[gl].x[0] << endl; 
            }
            //-----------
            clFinish(commandQueue_wt);
            //-----------
            //CLEANING UP
            CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            	kernel_wt, mem_wt);
            
            //------------------------
            //float *turnArray;
            //turnArray = new float [NUM_GOOD_LINES];
            //float *distArray;
            //distArray = new float [NUM_GOOD_LINES];
            //float *rngturnArray;
            //rngturnArray = new float [NUM_GOOD_LINES];
            //float *rngfwdArray;
            //rngfwdArray = new float [NUM_GOOD_LINES];
            //
            //for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            //{
            //   turnArray[gl] = good_line[gl].x[0];
            //   distArray[gl] = good_line[gl].x[1];
            //   rngturnArray[gl] = good_line[gl].x[2];
            //   rngfwdArray[gl] = good_line[gl].x[3];
         	//}
            ////Sorting RNG arrays
            //std::vector<float> turnArray_vector(turnArray, turnArray+NUM_GOOD_LINES-1);
            //std::sort (turnArray_vector.begin(), turnArray_vector.end()); 
            //
            //std::vector<float> distArray_vector(distArray, distArray+NUM_GOOD_LINES-1);
            //std::sort (distArray_vector.begin(), distArray_vector.end()); 
            //
            //std::vector<float> rngturnArray_vector(rngturnArray, rngturnArray+NUM_GOOD_LINES-1);
            //std::sort (rngturnArray_vector.begin(), rngturnArray_vector.end()); 
            //
            //std::vector<float> rngfwdArray_vector(rngfwdArray, rngfwdArray+NUM_GOOD_LINES-1);
            //std::sort (rngfwdArray_vector.begin(), rngfwdArray_vector.end()); 
           
            ////Writing RNG arrays to OUTPUT FILE
            //FILE* file_rngs = fopen("rng_numbers.txt", "a"); 
            //char output_rngs[255]; 
            //
            //for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            //{
            //   sprintf(output_rngs, "%7.5f    %7.5f    %7.5f    %7.5f\n", 
            //   	turnArray_vector[gl], distArray_vector[gl], rngturnArray_vector[gl], rngfwdArray_vector[gl]); 
            //   fputs(output_rngs, file_rngs); 
         	//}
            //
            //fclose(file_rngs);
            //------------------------
            
            
            //________________________________________________________________
            //
            //KERNEL3 END: "OpenCL-ling" EVALUATION OF PARTICLE WEIGHTS
            //________________________________________________________________
         
            int fhalf = 0;
            int shalf = 0;

            int fidx=0, sidx=0;

            for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            {
               if(good_line[gl].x[0] <= img_grad.cols/2)
               {
                  fhalf++;
               }
               else shalf++;
            }
            

            float w_pf_fhalf[fhalf];
            float w_pf_shalf[shalf];

            lineStruct good_line_fhalf[fhalf];
            lineStruct good_line_shalf[shalf];

            for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            {
               if(good_line[gl].x[0] <= img_grad.cols/2)
               {
                  w_pf_fhalf[fidx] = w_pf[gl];
                 // cout << "w_pf_fhalf: " << w_pf_fhalf[fidx] << endl; 
                  good_line_fhalf[fidx] = good_line[gl];
                  fidx++;
               }
               else 
               {
                  w_pf_shalf[sidx] = w_pf[gl];
                  //cout << "w_pf_shalf: " << w_pf_shalf[sidx] << endl; 
                  good_line_shalf[sidx] = good_line[gl];
                  sidx++;
               }
            }
            
                       
            //SELECTING THE LINES BASED ON WEIGHTS		
         	//Normalizing the weights.
         	float tWeight_fhalf = 0.0;
         
         	for(int i=0; i < fhalf; i++)
               tWeight_fhalf = tWeight_fhalf + w_pf_fhalf[i]; 	
         	for(int i=0; i < fhalf; i++)
               w_pf_fhalf[i] = w_pf_fhalf[i]/tWeight_fhalf; 
            
            //FHALF    
         	srand(time(NULL));
         	int index = rand() % fhalf;
         	float beta1 = 0.0; 
         	float beta = 0.0;
         	float mw = 0.0; //max. value out of w[1...N]	
         	lineStruct *good_line_fhalf1;
         	good_line_fhalf1 = new lineStruct [fhalf];
         	
         	//Evaluating the maximum weight
         	for (int i=0; i<fhalf; i++)
         	{
         		if(w_pf_fhalf[i] > mw)
         		{
         			mw = w_pf_fhalf[i];
         		}	
         	}
         
            cout << "mw_fhalf = " << mw << endl;
         	
         	//RESAMPLING STEP
         	for (int i=0; i < fhalf; i++)
         	{
         		beta1 = rand()%fhalf;
         		beta += ((float)beta1/fhalf) * ( 2 * mw );
         		while (beta > w_pf_fhalf[index])
         		{
         			beta -= w_pf_fhalf[index];
         			index = (index + 1) % fhalf;
         		}
         		good_line_fhalf1[i] = good_line_fhalf[index];
         	}
         	
         	//Setting the noise and landmarks for the new particles
         	for (int i=0; i<fhalf; i++)
         	{
         		good_line[i] = good_line_fhalf1[i];
         	}
         
            
            //SHALF    
         	float tWeight_shalf = 0.0;
         	
            for(int i=0; i < shalf; i++)
               tWeight_shalf = tWeight_shalf + w_pf_shalf[i]; 	
         	for(int i=0; i < shalf; i++)
               w_pf_shalf[i] = w_pf_shalf[i]/tWeight_shalf; 
         	
            cout << "fhalf = " << fhalf << endl;
            cout << "shalf = " << shalf << endl;
           
           if(shalf != 0)
            {
            srand(time(NULL));
         	index = rand() % shalf;
         	beta1 = 0.0; 
         	beta = 0.0;
         	mw = 0.0; //max. value out of w[1...N]	
         	lineStruct *good_line_shalf1;
         	good_line_shalf1 = new lineStruct [shalf];
         	
         	//Evaluating the maximum weight
         	for (int i=0; i<shalf; i++)
         	{
         		if(w_pf_shalf[i] > mw)
         		{
         			mw = w_pf_shalf[i];
         		}	
         	}
         
         	
         	//RESAMPLING STEP
         	for (int i=0; i < shalf; i++)
         	{
         		beta1 = rand()%shalf;
         		beta += ((float)beta1/shalf) * ( 2 * mw );
         		while (beta > w_pf_shalf[index])
         		{
         			beta -= w_pf_shalf[index];
         			index = (index + 1) % shalf;
         		}
         		good_line_shalf1[i] = good_line_shalf[index];
         	}
         	
            cout << "mw_shalf = " << mw << endl;
         	//Setting the noise and landmarks for the new particles
         	for (int i=0; i < shalf; i++)
         	{
         		good_line[fhalf+i] = good_line_shalf1[i];
         	}
            }

//TEST1         	//Evaluating the maximum weight
//TEST1         	for (int i=0; i<NUM_GOOD_LINES; i++)
//TEST1         	{
//TEST1         		if(w_pf[i] > mw)
//TEST1         		{
//TEST1         			mw = w_pf[i];
//TEST1         		}	
//TEST1         	}
//TEST1         		cout << "mw = " << mw << endl;
//TEST1         	
//TEST1         	//RESAMPLING STEP
//TEST1         	for (int i=0; i<NUM_GOOD_LINES; i++)
//TEST1         	{
//TEST1         		beta1 = rand()%NUM_GOOD_LINES;
//TEST1         		beta += ((float)beta1/NUM_GOOD_LINES) * ( 2 * mw );
//TEST1         		while (beta > w_pf[index])
//TEST1         		{
//TEST1         			beta -= w_pf[index];
//TEST1         			index = (index + 1) % NUM_GOOD_LINES;
//TEST1         		}
//TEST1         		good_line1[i] = good_line[index];
//TEST1         	}
//TEST1         	
//TEST1         	//Setting the noise and landmarks for the new particles
//TEST1         	for (int i=0; i<NUM_GOOD_LINES; i++)
//TEST1         	{
//TEST1         		good_line[i] = good_line1[i];
//TEST1         	}

          
            for(int ln=0; ln<NUM_GOOD_LINES; ln++)
            {
            good_dist_vect[ln] = good_line[ln].dist;
            }
            
            std::vector<float> mygoodvector(good_dist_vect, good_dist_vect+NUM_GOOD_LINES-1);
            std::sort (mygoodvector.begin(), mygoodvector.end()); 
            
            count = 0;
            for( std::vector<float>::const_iterator i = mygoodvector.end(); i != mygoodvector.begin(); i--)
            {
            	if(*i != (float)0 ) 
            	for(int ln=0; ln<NUM_GOOD_LINES; ln++)
            	{
                  bool intersect = 0;
            		if(good_dist_vect[ln] == *i && count<NUM_BEST_LINES)
            		{
            			if(count == 0){
                  			best_line_idx[count] = ln;
                  			count++;
            			}
            			else 
            				if (count > 0)
            				{
            	            for(int kl=0; kl<count; kl++){
         		            	if(abs( good_line[ln].x[img_grad.rows-1] -
         		            		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
         		            		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                                 )
         		            		{
         		            		   if( kl<count-1 ) 
         		            		      continue;
                                    else{ 
                  		               best_line_idx[count] = ln;
                  		              	count++;
                                         }
         		            		}
         		            	if(abs( good_line[ln].x[img_grad.rows-1] -
         		            		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) < 
         		            		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                                 )         break;
         		            }
            				}
            		}
            	}//END ln
            }
            
            for(int bl=0; bl<NUM_BEST_LINES; bl++)
            {
               best_line[bl] = good_line[best_line_idx[bl]];
               //for(int meanidx=0; meanidx<4; meanidx++)
               //{
               //   int xincept = (good_line[best_line_idx[bl]].x[0] +
               //                  good_line[best_line_idx[bl]+1].x[0] +
               //                  good_line[best_line_idx[bl]+2].x[0] +
               //                  good_line[best_line_idx[bl]+3].x[0])/4;

               //   best_line[bl].theta = 
               //                  (good_line[best_line_idx[bl]].theta +
               //                   good_line[best_line_idx[bl]+1].theta +
               //                   good_line[best_line_idx[bl]+2].theta +
               //                   good_line[best_line_idx[bl]+3].theta)/4;
               //  
               //   for(int l=0; l<img_grad.rows; l++)
               //   	{
               //   		best_line[bl].y[l] = l; 
               //   		float r = best_line[bl].y[l]/sin(best_line[bl].theta);	
               //   		best_line[bl].x[l] = abs((int)(xincept + r * (cos(best_line[bl].theta))));
               //   	}
               //}
            }
            
            cout << "Best lines extracted successfully!" << endl;
                    
         }//END topIndex Iteration loop 
      total_v_frames++;
      }//END THE LOOP ELSE LOOP FROM SECOND FRAME OF THE VIDEO
     
     //Plotting 
   //  int x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0;  
     std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_g[NUM_GOOD_LINES];
     for (int i=0; i<NUM_GOOD_LINES; i++)
     {
        x1 = (int) (good_line[i].x[0]);
        y1 = (int) (good_line[i].y[0]);
        x2 = (int) (good_line[i].x[img_gray.rows-1]);
        y2 = (int) (good_line[i].y[img_gray.rows-1]);
        
        lines_g[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
        (std::pair<int, int>(x1,y1+ROI_START_Y), std::pair<int, int>(x2,y2+ROI_START_Y)));  
        
        std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_g;  
        for(it_g=lines_g[i].begin();it_g!=lines_g[i].end();it_g++)  
        {  
             cv::line(img_org_full, cv::Point(it_g->first.first, it_g->first.second), 
             cv::Point(it_g->second.first, it_g->second.second), cv::Scalar(255,0,0), 1, 8,0);  
        }
     }
     
    // std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines[NUM_BEST_LINES];
     std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines[NUM_BEST_LINES];
    // std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_tmp[NUM_BEST_LINES];

     for (int i=0; i<NUM_BEST_LINES; i++)
     {
        if(total_v_frames%15 == 0)
        {
        x1 = (int) (best_line[i].x[0]);
        y1 = (int) (best_line[i].y[0]);
        x2 = (int) (best_line[i].x[img_gray.rows-1]);
        y2 = (int) (best_line[i].y[img_gray.rows-1]);
          x3 = (int) (x1 + x2)/2;
          y3 = (int) (y1 + y2)/2;
          x4 = (int) (2*x1-x3);
          y4 = (int) (2*y1-y3);

          
          lines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
          (std::pair<int, int>(x1,y1+ROI_START_Y), std::pair<int, int>(x2,y2+ROI_START_Y)));  
          
          lines_tmp[i] = lines[i];
        }
          std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
          for(it=lines_tmp[i].begin();it!=lines_tmp[i].end();it++)  
          {  
               cv::line(img_org_full, cv::Point(it->first.first, it->first.second), 
               cv::Point(it->second.first, it->second.second), cv::Scalar(0,0,255), 2, 8,0);  
          }
     }
    //Region of interest   
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > roi_lines[5];
    x21 = (int) (0);
    y21 = (int) (ROI_START_Y);
    x22 = x21;
    y22 = (int) (img_org_full.rows-1);
    roi_lines[0].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = (int) (0);
    y21 = (int) (img_org_full.rows-1);
    x22 = (int) img_grad.cols;
    y22 = y21;
    roi_lines[1].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = (int) img_grad.cols;
    y21 = (int) (img_org_full.rows-1);
    x22 = x21;
    y22 = (int) (ROI_START_Y);
    roi_lines[2].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = (int) img_grad.cols;
    y21 = (int) (ROI_START_Y);
    x22 = (int) (0);
    y22 = (int) y21;
    roi_lines[3].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = (int) img_grad.cols/2;
    y21 = (int) (ROI_START_Y);
    x22 = x21;
    y22 = (int) (img_org_full.rows-1);
    roi_lines[4].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    	
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator roi_it;  
    for(int i=0; i<5;i++)
    {
       for(roi_it=roi_lines[i].begin();roi_it!=roi_lines[i].end();roi_it++)  
       {  
            cv::line(img_org_full, cv::Point(roi_it->first.first, roi_it->first.second), cv::Point(roi_it->second.first, roi_it->second.second), cv::Scalar(0,255,0), 2, 8,0);  
       }
    }  
  
    output_cap.write(img_org_full);
//    cv::imshow("Lane Detection", img_org_full);
//    cv::waitKey(1); 
   
    t2_clock = clock();
    float diff ((float)t2_clock-(float)t1_clock);
    cout<<"Total Running Time = " << diff/CLOCKS_PER_SEC << "\n"<< endl;
    cout << "________________________________________\n" << endl;
}//END VIDEO FRAMES for loop.
  
 capture.release();
 //output_cap.release();  
return 0;
}

//HELPER FUNCTIONS FOR CREATING LINES
bool CreateMemCreateLine(cl_context context,
   cl_mem memCreateLine[3], createLineParams cline_params,
   int *img_grad_array, lineStruct *line_new,
   int imgrows, int imgcols, int NUM_LINES)
{
   memCreateLine[0] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(createLineParams), &cline_params, NULL);

   memCreateLine[1] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   imgrows*imgcols*sizeof(int), img_grad_array, NULL);

   //memCreateLine[2] = clCreateBuffer(context,
   //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   //NUM_LINES*sizeof(float), xadd, NULL);

   memCreateLine[2] = clCreateBuffer(context,
   CL_MEM_READ_WRITE,
   NUM_LINES*sizeof(lineStruct), NULL, NULL);

   if (memCreateLine[0] == NULL || memCreateLine[1] == NULL
   || memCreateLine[2] == NULL )
      {
           cerr << "Error creating memory objects." <<  endl;
          return false;
      }

   return true;
}

void CleanupCreateLineKernel(cl_context context,
   cl_command_queue commandQueue,
   cl_program program, cl_kernel kernel,
   cl_mem memCreateLine[3])
{
   for(int i=0; i<3; i++)
   {
     if (memCreateLine[i] != 0)
            clReleaseMemObject(memCreateLine[i]);
   }
   
   if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

   if (kernel != 0)
       clReleaseKernel(kernel);

   if (program != 0)
       clReleaseProgram(program);

   if (context != 0)
       clReleaseContext(context);
}

//HELPER FUNCTIONS FOR CREATING LINES
bool CreateMemWt(cl_context context_wt, cl_mem mem_wt[7],
   lineStruct *good_line, lineStruct *best_line, int *img_grad_array, 
   noiseStruct *allnoises, wtKernParams *wtparams, 
   lineStruct *good_line_new, float *w_pf_new)
{
   mem_wt[0] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(lineStruct), good_line, NULL);

   mem_wt[1] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_BEST_LINES*sizeof(lineStruct), best_line, NULL);
   
   mem_wt[2] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   (wtparams->imgrows)*(wtparams->imgcols)*sizeof(int), 
   img_grad_array, NULL);

   //mem_wt[3] = clCreateBuffer(context_wt,
   //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   //wtparams->NUM_GOOD_LINES*sizeof(float),turnArray, NULL);
   //
   //mem_wt[4] = clCreateBuffer(context_wt,
   //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   //wtparams->NUM_GOOD_LINES*sizeof(float),distArray, NULL);

   //mem_wt[5] = clCreateBuffer(context_wt,
   //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   //wtparams->NUM_GOOD_LINES*sizeof(float),rngturnArray, NULL);

   //mem_wt[6] = clCreateBuffer(context_wt,
   //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   //wtparams->NUM_GOOD_LINES*sizeof(float),rngfwdArray, NULL);
   
   mem_wt[3] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(noiseStruct),allnoises, NULL);

   mem_wt[4] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(wtKernParams),wtparams, NULL);

   mem_wt[5] = clCreateBuffer(context_wt,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(lineStruct), NULL, NULL);

   mem_wt[6] = clCreateBuffer(context_wt,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(float), NULL, NULL);
   
   if (mem_wt[0] == NULL || mem_wt[1] == NULL
    || mem_wt[2] == NULL || mem_wt[3] == NULL
    || mem_wt[4] == NULL || mem_wt[5] == NULL
    || mem_wt[6] == NULL )
      {
          cerr << "Error creating mem_wt objects." <<  endl;
          return false;
      }
   
   return true;
}

void CleanupWtKernel(cl_context context_wt, 
   cl_command_queue commandQueue_wt, 
   cl_program program_wt, 
   cl_kernel kernel_wt, 
   cl_mem mem_wt[7])
{
   for(int i=0; i<7; i++)
   {
     if (mem_wt[i] != 0)
            clReleaseMemObject(mem_wt[i]);
   }
   
   if (commandQueue_wt != 0)
        clReleaseCommandQueue(commandQueue_wt);

   if (kernel_wt != 0)
       clReleaseKernel(kernel_wt);

   if (program_wt != 0)
       clReleaseProgram(program_wt);

   if (context_wt != 0)
       clReleaseContext(context_wt);
}
