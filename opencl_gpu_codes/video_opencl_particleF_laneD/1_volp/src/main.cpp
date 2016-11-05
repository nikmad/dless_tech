// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

// 11.09.2013: Modified the method for extracting good lines by removing
//       GOOD_LINE_DIST criteria. Now good lines can be spaced in anyway 
//       except that half the lines should be in 1st half and half in 2nd.
//       However, this assurance of half the lines in each half of the ROI
//       is only for "Image processing step" but there might still be 
//       skewing of all lines in a single half in particle filter step.


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
//using namespace cv;

#define CV_LOAD_IMAGE_GRAYSCALE 0
#define CV_LOAD_IMAGE_COLOR 1
#define CV_WINDOW_NORMAL 0
//#define PI 3.14159265358979323846 

//typedef struct{
//   float forwardNoise = 0.05;
//   float turnNoise = 0.05;
//   float senseNoise = 5.0;
//}noiseStruct;

float TURNMAX = 0.6;
float DISTMAX = 10.0;

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
   int *, float *, linestruct *, int, int, int);

void CleanupCreateLineKernel(cl_context, cl_command_queue,
   cl_program, cl_kernel, cl_mem *);

//bool CreateMemExtractGoodLine(cl_context, 
//   cl_mem *, linestruct *, float *, int *, 
//   goodParams *, int *, int *);
//
//void CleanupExtractGoodLineKernel(cl_context, cl_command_queue,
//   cl_program, cl_kernel, cl_mem *);

typedef unsigned long int ULONG;



//lines below give good results.
//allnoises->forwardNoise = 0.05;
//allnoises->turnNoise = 0.05;
//allnoises->senseNoise = 200.0;

bool CreateMemWt(cl_context, cl_mem *,
   linestruct *, linestruct *, int *, 
   float *, float *, float *, 
   float *, noiseStruct *, wtKernParams *, 
   linestruct *, float *);

void CleanupWtKernel(cl_context, 
   cl_command_queue, 
   cl_program, 
   cl_kernel, 
   cl_mem *);

int main( int argc, char** argv )
{
// 	system("rm lines_opencl.txt");
// 	system("rm goodcord.txt");
// 	system("rm bestcord.txt");
	//system("rm goodCord.txt");
	//system("rm bestCord.txt");
	//int cline_params.NUM_LINES = atoi(argv[1]);
	int BEST_LINE_DIST = atoi(argv[1]);
	int NUM_BEST_LINES = atoi(argv[2]);
  
   string input_video_file = "/home/nikhil/nik_workspace/git_space/dless_tech/opencl_gpu_codes/1_dependencies/video/clip5";
   //string input_video_file = "/home/madduri/git/nikmad/thesis_codes/video/clip5"; 
   //string input_video_file = "/home/nikhil/git/nikmad/opencv_codes/video/clip1"; 
   cv::VideoCapture capture(input_video_file);
   cv::Mat img_org_full;
   if( !capture.isOpened() )
       throw "Error when reading steam_avi";
   
   cv::namedWindow( "Lane Detection", CV_WINDOW_AUTOSIZE);
   cv::moveWindow( "Lane Detection", 200, 300);
 
   // double num_frames = cv::VideoCapture::get(CV_CAP_PROP_FRAME_COUNT);
   //double cv::VideoCapture::get num_frames;
   double num_frames;
   num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

   int total_v_frames = 0;
   int Iframes = 20; //Take accurate measurement after every Iframes.
   int NUM_GOOD_LINES = 250;
   int iterations = 1;
   int NUM_LINES = 3000;
   int NEIGHBORHOOD = 3;

   linestruct *good_line;
   good_line = new linestruct [NUM_GOOD_LINES];
   
   linestruct *best_line;
   best_line = new linestruct [NUM_BEST_LINES];
  
   int best_line_idx[NUM_BEST_LINES];
   float good_dist_vect[NUM_GOOD_LINES];
   
   int good_line_idx[NUM_GOOD_LINES];
   float dist_vect[NUM_LINES];
   int count =0;
   
   for(int framme=0; framme < num_frames; framme++)
   //for( ; ; )
   {
      clock_t t1_clock, t2_clock;
      t1_clock = clock();
    
   //   cout << "________________________________________\n" << endl;
      cout << "Frame Number of the Video = " << total_v_frames+1 << endl;
      capture >> img_org_full;
   
      cv::Rect rect;
      int ROI_START_X = (int)(0.20*img_org_full.cols);// % of image cropped in the right
      int ROI_START_Y = (int)(0.80*img_org_full.rows);// % of image cropped in the top
      rect = cv::Rect(0,ROI_START_Y,img_org_full.cols-ROI_START_X,img_org_full.rows-ROI_START_Y);
      
      cv::Mat img_gray_full;
      cv::cvtColor(img_org_full, img_gray_full, CV_RGB2GRAY, 1); 
      //cv::cvtColor(img_org_full, img_gray_full, CV_8U, 1); 
      cv::Mat img_gray = img_gray_full(rect);
 
	   cv::Mat img_grad(img_gray.rows, img_gray.cols-1, CV_8UC1); //gradient matrix
      int *img_grad_array;
      img_grad_array = new int [img_grad.rows*img_grad.cols]; 	
      
   
   	float GOOD_LINE_DIST = img_grad.cols/(4*NUM_GOOD_LINES);
   	//float GOOD_LINE_DIST = 4.0;
      
      createLineParams cline_params;
      cline_params.NEIGHBORHOOD = NEIGHBORHOOD;
      cline_params.NUM_LINES = NUM_LINES;
      cline_params.imgrows = img_grad.rows;
      cline_params.imgcols = img_grad.cols;
   	
   	for(int row=0; row < img_grad.rows; row++) {
   	for(int col=0; col < img_grad.cols; col++) {
   	img_grad.at<uchar>(row, col) =  (uchar)abs(
   					(int)img_gray.at<uchar>(row, col+1) - 
   			 	     	(int)img_gray.at<uchar>(row, col));
   	}
   	}
   	
   	double maxVal=0; 
   	double minVal=0;
   	cv::minMaxLoc(img_grad, &minVal, &maxVal, 0, 0);
   
   	cout << "MaxValue: " << maxVal << endl;
   	
   	//any gradient less than 30% of maxvalue is discarded. 
   	for(int row=0; row < img_grad.rows; row++) {
   	for(int col=0; col < img_grad.cols; col++) {
   	if((int)img_grad.at<uchar>(row, col) < 0.1*maxVal) 
   	img_grad.at<uchar>(row, col) = 0;
   	//else
   	//img_grad.at<uchar>(row, col) = 1;
   	}
   	}
   
      for(int l=0; l<img_grad.rows; l++)
      for(int m=0; m<img_grad.cols; m++)
         {
            img_grad_array[l*img_grad.cols+m] = (int)img_grad.at<uchar>(l,m);      
         }
   	
      //cout << img_grad << endl;
   	cout << img_grad.rows << endl;
   	cout << img_grad.cols << endl;
   	
      if(total_v_frames%Iframes == 0)
      {
         cout << "Im here in if" << endl;
         linestruct *line;
         line = new linestruct [NUM_LINES];
         
         linestruct *line_new;
         line_new = new linestruct [NUM_LINES];
         
         //OpenCL parameters
         cl_context context = 0;
         cl_command_queue commandQueue = 0;
         cl_program program = 0;
         cl_device_id device = 0;
         cl_kernel kernel = 0;
         cl_int errNum;
         cl_mem memCreateLine[4] = {0, 0, 0, 0};
         
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
         
         int c2;
         //float r;
         float *xadd;
         xadd = new float [NUM_LINES];
         
         for(int ln=0; ln<NUM_LINES; ln++)
         {
            srand(ln*10000+20*ln*ln*ln);
            c2 = rand()%10000;
            xadd[ln] = (float)(((float)c2/10000.0000) * img_grad.cols);
         }
         
        // for(int ln=NUM_LINES/2; ln<NUM_LINES; ln++)
        // {
        //    srand(ln*10000+ln*ln);
        //    c2 = rand()%10000;
        //    xadd[ln] = (float)(((float)c2/10000.0000)*(float)img_grad.cols/2 + (float)img_grad.cols/2);
        // }
         
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
            img_grad_array, xadd, line_new, img_grad.rows, 
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
         errNum = clEnqueueWriteBuffer( 
         		commandQueue, 
         		memCreateLine[2], 
         		CL_TRUE, 
         		0,
         		NUM_LINES*sizeof(float), 
         		(void *)xadd, 
         		0,
         		NULL, 
         		NULL);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error: enqueuing xadd buffer." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);          
             return 1;
         }
         //-----------
         //Setting the kernel arguments
         errNum  = clSetKernelArg(kernel, 0, 
             sizeof(cl_mem),(void *)&memCreateLine[0]);
         errNum |= clSetKernelArg(kernel, 1, 
             sizeof(cl_mem),(void *)&memCreateLine[1]);
         errNum |= clSetKernelArg(kernel, 2, 
             sizeof(cl_mem),(void *)&memCreateLine[2]);
         errNum |= clSetKernelArg(kernel, 3, 
             sizeof(cl_mem),(void *)&memCreateLine[3]);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error setting kernel arguments." <<  endl;
             CleanupCreateLineKernel(context, commandQueue, program, 
             	kernel, memCreateLine);    
             return 1;
         }
         //-----------
         // Queue the kernel up for execution across the array
         size_t globalWorkSize[1] = { NUM_LINES };
         size_t localWorkSize[1] = { 1 };
         
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
         // Read the output buffer back to the Host
         errNum = clEnqueueReadBuffer(commandQueue, memCreateLine[3], 
             	CL_TRUE, 0, NUM_LINES * sizeof(linestruct), 
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
                     if(fhalf <= NUM_GOOD_LINES/2 && shalf <= NUM_GOOD_LINES/2)
                     {
         	   	//DISTCRI//	   if(count == 0){
                  	   		good_line_idx[count] = ln;
                  	   		count++;

                              if(line[ln].x[0] <= img_grad.cols/2)
                              {
                                 fhalf++;
                              }
                              if(line[ln].x[0] > img_grad.cols/2)
                              {
                                 shalf++;
                              }
         	   	//DISTCRI//	   }
         	   	//DISTCRI//	   else 
         	   	//DISTCRI//	   	if (count > 0)
         	   	//DISTCRI//	   	{
         	   	//DISTCRI//	   		for(int kl=0; kl<count; kl++)
                  //DISTCRI//            {
         	   	//DISTCRI//	   	   	   if( abs(line[ln].x[img_grad.rows-1] - line[good_line_idx[kl]].x[img_grad.rows-1]) > 
                  //DISTCRI//                           ((float)GOOD_LINE_DIST/100.0)*(float)img_grad.cols 
                  //DISTCRI//                       // && 
                  //DISTCRI//                       // abs( line[ln].x[0] - line[good_line_idx[kl]].x[0] ) > 
                  //DISTCRI//                       //    ((float)(0.1*GOOD_LINE_DIST)/100.0)*(float)img_grad.cols 
                  //DISTCRI//                       )
         	   	//DISTCRI//	   	   	   	{
         	   	//DISTCRI//	   			         if( kl<count-1 ) continue;
         	   	//DISTCRI//	   			         else{	good_line_idx[count] = ln;
         	   	//DISTCRI//	   			         	count++;}
         	   	//DISTCRI//	   	         	}
                  //DISTCRI//                  if( abs(line[ln].x[img_grad.rows-1] - line[good_line_idx[kl]].x[img_grad.rows-1]) <
                  //DISTCRI//                           ((float)GOOD_LINE_DIST/100.0)*(float)img_grad.cols 
                  //DISTCRI//                          // || 
                  //DISTCRI//                          // abs( line[ln].x[0] - line[good_line_idx[kl]].x[0] ) < 
                  //DISTCRI//                          //    ((float)(0.1*GOOD_LINE_DIST)/100.0)*(float)img_grad.cols
                  //DISTCRI//                              ) 
                  //DISTCRI//                           break;
         	   	//DISTCRI//	   	   }//end for 'kl'
     		         //DISTCRI//         }//end if(count)
                     }//end if line[ln].x[0]
                 //    else continue;
                  
                     if(line[ln].x[0] <= img_grad.cols/2 && fhalf <= NUM_GOOD_LINES/2 && shalf >= NUM_GOOD_LINES/2)
                     //else if(line[ln].x[0] <= img_grad.cols/2 && fhalf <= NUM_GOOD_LINES/2)
                     {
                  	   		good_line_idx[count] = ln;
                  	   		count++;
                              fhalf++;
                     }
                     if(line[ln].x[0] > img_grad.cols/2 && shalf <= NUM_GOOD_LINES/2 && fhalf >= NUM_GOOD_LINES/2)
                     //else if(line[ln].x[0] > img_grad.cols/2 && shalf <= NUM_GOOD_LINES/2)
                     {
                  	   		good_line_idx[count] = ln;
                  	   		count++;
                              shalf++;
                     }
                  } //end if(distvect)

                 // if(dist_vect[ln] == *i && count > NUM_GOOD_LINES/2)
         	     // {
                 //   if(line[ln].x[0] > img_grad.cols/2)
                 //   {
                 //       cout << "im here 3" << endl;
         	     //   	   for(int kl=0; kl<count; kl++)
                 //       {
                 //          if( abs(line[ln].x[img_grad.rows-1] - line[good_line_idx[kl]].x[img_grad.rows-1]) > 
                 //                     ((float)GOOD_LINE_DIST/100.0)*(float)img_grad.cols 
                 //                 // && 
                 //                 // abs( line[ln].x[0] - line[good_line_idx[kl]].x[0] ) > 
                 //                 //    ((float)(0.1*GOOD_LINE_DIST)/100.0)*(float)img_grad.cols 
                 //                 )
         	     //   	      	{
         	     //   	            if( kl<count-1 ) continue;
         	     //   	            else{	good_line_idx[count] = ln;
         	     //   	            	count++;}
         	     //   	      	}
                 //            if( abs(line[ln].x[img_grad.rows-1] - line[good_line_idx[kl]].x[img_grad.rows-1]) <
                 //                     ((float)GOOD_LINE_DIST/100.0)*(float)img_grad.cols 
                 //                    // || 
                 //                    // abs( line[ln].x[0] - line[good_line_idx[kl]].x[0] ) < 
                 //                    //    ((float)(0.1*GOOD_LINE_DIST)/100.0)*(float)img_grad.cols
                 //                        ) 
                 //                     break;
                 //         //else break;
         	     //   	   }//end for 'kl'
                 //   }//end if line[ln].x[0]
                 //   else continue;
         	     // } //end if(distvectr)
               }//END for 'ln'
             }
             //else break;
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
                    //             for(int lrow=0; lrow < img_grad.rows; lrow++){
                    //                 if(good_line[ln].x[lrow] != good_line[best_line_idx[kl]].x[lrow])
                    //                    continue;
                    //                    else{ 
                    //                       intersect = 1;
                    //                       break;
                    //                       }
                    //             }

         		               	if(abs( good_line[ln].x[img_grad.rows-1] -
         		               		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
         		               		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                              //      &&
                              //      abs( good_line[ln].x[0] -
         		               //		good_line[best_line_idx[kl]].x[0] ) > 
         		               //		((float)(0.5*BEST_LINE_DIST)/100.0)*(float)img_grad.cols
                                   // && intersect == 0
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
                              //      ||
                              //      abs( good_line[ln].x[0] -
         		               //		good_line[best_line_idx[kl]].x[0] ) < 
         		               //		((float)(0.5*BEST_LINE_DIST)/100.0)*(float)img_grad.cols
                                   // || intersect == 1
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
      
      //else if(total_v_frames > 0)
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
         allnoises->senseNoise = 300.5;
         
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
            
            float *turnArray;
            turnArray = new float [NUM_GOOD_LINES];
            float *distArray;
            distArray = new float [NUM_GOOD_LINES];
            float *rngturnArray;
            rngturnArray = new float [NUM_GOOD_LINES];
            float *rngfwdArray;
            rngfwdArray = new float [NUM_GOOD_LINES];
            
            wtKernParams wtparams;
         
            linestruct *good_line_new;
            good_line_new = new linestruct [NUM_GOOD_LINES];
         
            float *w_pf_new;
            w_pf_new = new float [NUM_GOOD_LINES];
           
            int c2 =0; 
            //-----------
            for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            {
               c1 = rand()%10000;
               turnArray[gl] = ((float)c1/10000.0000)*TURNMAX - TURNMAX/2;
                     
               c2 = rand()%10000;
               distArray[gl] = ((float)c2/10000.0000)*DISTMAX - DISTMAX/2;
         
               RNG rng(10090329*gl + time(NULL)*gl*gl*gl + 2346*gl);
         	   rngturnArray[gl] = rng.gaussian(allnoises->turnNoise);
         	   rngfwdArray[gl] = rng.gaussian(allnoises->forwardNoise);
         	}
            
            wtparams.NEIGHBORHOOD = NEIGHBORHOOD;
            wtparams.NUM_GOOD_LINES = NUM_GOOD_LINES;
            wtparams.NUM_BEST_LINES = NUM_BEST_LINES;
            wtparams.imgrows = img_grad.rows;  
            wtparams.imgcols = img_grad.cols;
            //-----------
            cl_context context_wt = 0;
            cl_command_queue commandQueue_wt = 0;
            cl_program program_wt = 0;
            cl_device_id device_wt = 0;
            cl_kernel kernel_wt = 0;
            cl_int errNum_wt;
            cl_mem mem_wt[11] = {0,0,0,0,0,0,0,0,0,0,0};
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
                  img_grad_array, turnArray, distArray, rngturnArray, rngfwdArray, 
                  allnoises, &wtparams, good_line_new, w_pf_new))
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
            		NUM_GOOD_LINES*sizeof(linestruct), 
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
            		NUM_BEST_LINES*sizeof(linestruct), 
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
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[3], 
            		CL_TRUE, 
            		0,
                  NUM_GOOD_LINES*sizeof(float), 
         		   (void *)turnArray, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing turnArray buffer." <<  endl;
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
                  NUM_GOOD_LINES*sizeof(float), 
         		   (void *)distArray, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing distArray buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[5], 
            		CL_TRUE, 
            		0,
                  NUM_GOOD_LINES*sizeof(float), 
         		   (void *)rngturnArray, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing rngturnArray buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[6], 
            		CL_TRUE, 
            		0,
                  NUM_GOOD_LINES*sizeof(float), 
         		   (void *)rngfwdArray, 
            		0,
            		NULL, 
            		NULL);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error: enqueuing rngfwdArray buffer." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);          
                return 1;
            }
            //-----------
            errNum_wt = clEnqueueWriteBuffer( 
            		commandQueue_wt, 
            		mem_wt[7], 
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
            		mem_wt[8], 
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
            
            errNum_wt |= clSetKernelArg(kernel_wt, 3, 
                sizeof(cl_mem),(void *)&mem_wt[3]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 4, 
                sizeof(cl_mem),(void *)&mem_wt[4]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 5, 
                sizeof(cl_mem),(void *)&mem_wt[5]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 6, 
                sizeof(cl_mem),(void *)&mem_wt[6]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 7, 
                sizeof(cl_mem),(void *)&mem_wt[7]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 8, 
                sizeof(cl_mem),(void *)&mem_wt[8]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 9, 
                sizeof(cl_mem),(void *)&mem_wt[9]);
            
            errNum_wt |= clSetKernelArg(kernel_wt, 10, 
                sizeof(cl_mem),(void *)&mem_wt[10]);
            
            if (errNum_wt != CL_SUCCESS)
            {
                cerr << "Error setting kernel arguments." <<  endl;
                CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
                	kernel_wt, mem_wt);    
                return 1;
            }
            //-----------
            // Queue the kernel up for execution across the array
            size_t globalWorkSize_wt[1] = { NUM_GOOD_LINES };
            size_t localWorkSize_wt[1] = { 1 };
            
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
            // Read the output buffer back to the Host
            //OUTPUT 1
            errNum_wt = clEnqueueReadBuffer(
                           commandQueue_wt, 
                           mem_wt[9],
                           CL_TRUE, 0, 
                           NUM_GOOD_LINES * sizeof(linestruct), 
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
                           mem_wt[10],
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
            }
            //-----------
            //CLEANING UP
            CleanupWtKernel(context_wt, commandQueue_wt, program_wt, 
            	kernel_wt, mem_wt);
            //________________________________________________________________
            //
            //KERNEL3 END: "OpenCL-ling" EVALUATION OF PARTICLE WEIGHTS
            //________________________________________________________________
         
            //SELECTING THE LINES BASED ON WEIGHTS		
         	//Normalizing the weights.
         	float tWeight = 0.0;
         
         	for(int i=0; i<NUM_GOOD_LINES; i++)
         	{	tWeight = tWeight + w_pf[i]; 	}
         	
         	for(int i=0; i<NUM_GOOD_LINES; i++)
         	{	w_pf[i] = w_pf[i]/tWeight; }
         
         	srand(time(NULL));
         	int index = rand() % NUM_GOOD_LINES;
         	float beta1 = 0.0; 
         	float beta = 0.0;
         	float mw = 0.0; //max. value out of w[1...N]	
         	linestruct *good_line1;
         	good_line1 = new linestruct [NUM_GOOD_LINES];
         	
         	//Evaluating the maximum weight
         	for (int i=0; i<NUM_GOOD_LINES; i++)
         	{
         		if(w_pf[i] > mw)
         		{
         			mw = w_pf[i];
         		}	
         	}
         		cout << "mw = " << mw << endl;
         	
         	//RESAMPLING STEP
         	for (int i=0; i<NUM_GOOD_LINES; i++)
         	{
         		beta1 = rand()%NUM_GOOD_LINES;
         		beta += ((float)beta1/NUM_GOOD_LINES) * ( 2 * mw );
         		while (beta > w_pf[index])
         		{
         			beta -= w_pf[index];
         			index = (index + 1) % NUM_GOOD_LINES;
         		}
         		//particle1[i] = particle[index];
         		good_line1[i] = good_line[index];
         	}
         	
         	//Setting the noise and landmarks for the new particles
         	for (int i=0; i<NUM_GOOD_LINES; i++)
         	{
         		good_line[i] = good_line1[i];
         	}
          
         //   int best_line_idx[NUM_BEST_LINES];
         //   float good_dist_vect[NUM_GOOD_LINES];
            
            int best_line_idx_pf[NUM_BEST_LINES];
   
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
                          //    for(int lrow=0; lrow < img_grad.rows; lrow++){
                          //        if(good_line[ln].x[lrow] != good_line[best_line_idx[kl]].x[lrow])
                          //           continue;
                          //           else{ 
                          //              intersect = 1;
                          //              break;
                          //              }
                          //    }

         		            	if(abs( good_line[ln].x[img_grad.rows-1] -
         		            		good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
         		            		((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols
                              //   &&
                              //   abs( good_line[ln].x[0] -
         		            	//	good_line[best_line_idx[kl]].x[0] ) > 
         		            	//	((float)(0.5*BEST_LINE_DIST)/100.0)*(float)img_grad.cols
                                 //&& intersect == 0
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
                              //   ||
                              //   abs( good_line[ln].x[0] -
         		            	//	good_line[best_line_idx[kl]].x[0] ) < 
         		            	//	((float)(0.5*BEST_LINE_DIST)/100.0)*(float)img_grad.cols
                                 //|| intersect == 1
                                 )         break;
         		            }
            				}
            		}
            	}//END ln
            }
            
         //   linestruct *best_line;
         //   best_line = new linestruct [NUM_BEST_LINES];
            
            for(int bl=0; bl<NUM_BEST_LINES; bl++)
            {
            best_line[bl] = good_line[best_line_idx[bl]];
            }
            cout << "Best lines extracted successfully!" << endl;
                    
         }//END topIndex Iteration loop 
      total_v_frames++;
      }//END THE LOOP ELSE LOOP FROM SECOND FRAME OF THE VIDEO
     
     //Plotting 
    // std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_g[NUM_GOOD_LINES];
         int x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0;  
    // for (int i=0; i<NUM_GOOD_LINES; i++)
    // {
    //    x1 = (int) (good_line[i].x[0]);
    //    y1 = (int) (good_line[i].y[0]);
    //    x2 = (int) (good_line[i].x[img_gray.rows-1]);
    //    y2 = (int) (good_line[i].y[img_gray.rows-1]);
    //    
    //    lines_g[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
    //    (std::pair<int, int>(x1,y1+ROI_START_Y), std::pair<int, int>(x2,y2+ROI_START_Y)));  
    //    
    //    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_g;  
    //    for(it_g=lines_g[i].begin();it_g!=lines_g[i].end();it_g++)  
    //    {  
    //         cv::line(img_org_full, cv::Point(it_g->first.first, it_g->first.second), 
    //         cv::Point(it_g->second.first, it_g->second.second), cv::Scalar(255,0,0), 1, 8,0);  
    //    }
    // }
     
     std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines[NUM_BEST_LINES];
     for (int i=0; i<NUM_BEST_LINES; i++)
     {
        x1 = (int) (best_line[i].x[0]);
        y1 = (int) (best_line[i].y[0]);
        x2 = (int) (best_line[i].x[img_gray.rows-1]);
        y2 = (int) (best_line[i].y[img_gray.rows-1]);
       // x2 = (int) (r * cos(best_line[i].theta) + best_line[i].x[0]);
       // y2 = (int) (r * sin(best_line[i].theta) + best_line[i].y[0]);
        
        lines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
        (std::pair<int, int>(x1,y1+ROI_START_Y), std::pair<int, int>(x2,y2+ROI_START_Y)));  
        
        std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;  
        for(it=lines[i].begin();it!=lines[i].end();it++)  
        {  
             cv::line(img_org_full, cv::Point(it->first.first, it->first.second), 
             cv::Point(it->second.first, it->second.second), cv::Scalar(0,0,255), 2, 8,0);  
        }
     }
    //Region of interest   
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > roi_lines[5];
    x1 = (int) (0);
    y1 = (int) (ROI_START_Y);
    x2 = x1;
    y2 = (int) (img_org_full.rows-1);
    roi_lines[0].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
    x1 = (int) (0);
    y1 = (int) (img_org_full.rows-1);
   // x2 = (int) (img_org_full.cols-ROI_START_X);
    x2 = (int) img_grad.cols;
    y2 = y1;
    roi_lines[1].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
    //x1 = (int) (img_org_full.cols-ROI_START_X);
    x1 = (int) img_grad.cols;
    y1 = (int) (img_org_full.rows-1);
    x2 = x1;
    y2 = (int) (ROI_START_Y);
    roi_lines[2].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
    //x1 = (int) (img_org_full.cols-ROI_START_X);
    x1 = (int) img_grad.cols;
    y1 = (int) (ROI_START_Y);
    x2 = (int) (0);
    y2 = (int) y1;
    roi_lines[3].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
    x1 = (int) img_grad.cols/2;
    y1 = (int) (ROI_START_Y);
    x2 = x1;
    y2 = (int) (img_org_full.rows-1);
    roi_lines[4].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
    	
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator roi_it;  
    for(int i=0; i<5;i++)
    {
       for(roi_it=roi_lines[i].begin();roi_it!=roi_lines[i].end();roi_it++)  
       {  
            cv::line(img_org_full, cv::Point(roi_it->first.first, roi_it->first.second), cv::Point(roi_it->second.first, roi_it->second.second), cv::Scalar(0,255,0), 2, 8,0);  
       }
    }  
  
    cv::imshow("Lane Detection", img_org_full);
    cv::waitKey(1); 
   
    t2_clock = clock();
    float diff ((float)t2_clock-(float)t1_clock);
    cout<<"Total Running Time = " << diff/CLOCKS_PER_SEC << "\n"<< endl;
    cout << "________________________________________\n" << endl;
}//END VIDEO FRAMES for loop.
   
   //cout << "num_frames: " << num_frames << endl;
    
return 0;
}

//HELPER FUNCTIONS FOR CREATING LINES
bool CreateMemCreateLine(cl_context context,
   cl_mem memCreateLine[4], createLineParams cline_params,
   int *img_grad_array, float *xadd, linestruct *line_new,
   int imgrows, int imgcols, int NUM_LINES)
{
   memCreateLine[0] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(createLineParams), &cline_params, NULL);

   memCreateLine[1] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   imgrows*imgcols*sizeof(int), img_grad_array, NULL);

   memCreateLine[2] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   NUM_LINES*sizeof(float), xadd, NULL);

   memCreateLine[3] = clCreateBuffer(context,
   CL_MEM_READ_WRITE,
   NUM_LINES*sizeof(linestruct), NULL, NULL);

   if (memCreateLine[0] == NULL || memCreateLine[1] == NULL
   || memCreateLine[2] == NULL || memCreateLine[3] == NULL)
      {
           cerr << "Error creating memory objects." <<  endl;
          return false;
      }

   return true;
}

void CleanupCreateLineKernel(cl_context context,
   cl_command_queue commandQueue,
   cl_program program, cl_kernel kernel,
   cl_mem memCreateLine[4])
{
   for(int i=0; i<4; i++)
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
bool CreateMemWt(cl_context context_wt, cl_mem mem_wt[11],
   linestruct *good_line, linestruct *best_line, int *img_grad_array, 
   float *turnArray, float *distArray, float *rngturnArray, 
   float *rngfwdArray, noiseStruct *allnoises, wtKernParams *wtparams, 
   linestruct *good_line_new, float *w_pf_new)
{
   mem_wt[0] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(linestruct), good_line, NULL);

   mem_wt[1] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_BEST_LINES*sizeof(linestruct), best_line, NULL);
   
   mem_wt[2] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   (wtparams->imgrows)*(wtparams->imgcols)*sizeof(int), 
   img_grad_array, NULL);

   mem_wt[3] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(float),turnArray, NULL);
   
   mem_wt[4] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(float),distArray, NULL);

   mem_wt[5] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(float),rngturnArray, NULL);

   mem_wt[6] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(float),rngfwdArray, NULL);
   
   mem_wt[7] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(noiseStruct),allnoises, NULL);

   mem_wt[8] = clCreateBuffer(context_wt,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(wtKernParams),wtparams, NULL);

   mem_wt[9] = clCreateBuffer(context_wt,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(linestruct), NULL, NULL);

   mem_wt[10] = clCreateBuffer(context_wt,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(float), NULL, NULL);
   
   if (mem_wt[0] == NULL || mem_wt[1] == NULL
    || mem_wt[2] == NULL || mem_wt[3] == NULL
    || mem_wt[4] == NULL || mem_wt[5] == NULL
    || mem_wt[6] == NULL || mem_wt[7] == NULL
    || mem_wt[8] == NULL || mem_wt[9] == NULL
    || mem_wt[10] == NULL)
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
   cl_mem mem_wt[11])
{
   for(int i=0; i<11; i++)
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
