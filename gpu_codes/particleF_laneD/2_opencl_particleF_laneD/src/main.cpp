// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

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

float TURNMAX = 1.0;
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
	system("rm lines_opencl.txt");
	system("rm goodcord.txt");
	system("rm bestcord.txt");
	//system("rm goodCord.txt");
	//system("rm bestCord.txt");
	//int cline_params.NUM_LINES = atoi(argv[1]);
	int BEST_LINE_DIST = atoi(argv[1]);
	int NUM_BEST_LINES = atoi(argv[2]);
	cv::Mat img_org = cv::imread( argv[3], 
				CV_LOAD_IMAGE_COLOR);

	cv::Mat img_gray;
	img_gray = cv::imread( argv[3], 
				CV_LOAD_IMAGE_GRAYSCALE);
	 
	 cv::Mat img_grad(img_gray.rows, img_gray.cols-1, CV_8UC1); //gradient matrix
   int *img_grad_array;
   img_grad_array = new int [img_grad.rows*img_grad.cols]; 	
   

   int NUM_GOOD_LINES = 2000;
	float GOOD_LINE_DIST = img_grad.cols/(2*NUM_GOOD_LINES);
	//float GOOD_LINE_DIST = 4.0;
	int iterations = 20;
   int NUM_LINES = 10000;
   int NEIGHBORHOOD = 5;

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

	cout << "MaxValue: " << maxVal << "\n" << endl;
	
	//any gradient less than 30% of maxvalue is discarded. 
	for(int row=0; row < img_grad.rows; row++) {
	for(int col=0; col < img_grad.cols; col++) {
	if((int)img_grad.at<uchar>(row, col) < 0.3*maxVal) 
	img_grad.at<uchar>(row, col) = 0;
	else
	img_grad.at<uchar>(row, col) = 1;
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

float *xadd;
int tmp_coeff=0;

xadd = new float [NUM_LINES];

for(int ln=0; ln<NUM_LINES; ln++)
{
srand(ln*10000+20*ln*ln*ln);
tmp_coeff = rand()%10000;
xadd[ln] = (float)(((float)tmp_coeff/10000.0000) * img_grad.cols);
}

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
FILE* file_lines_ocl = fopen("lines_opencl.txt", "a"); 
char output_lines_ocl[255]; 

for (int i=0; i<NUM_LINES; i++)
{
sprintf(output_lines_ocl, "%10.9f    %10.9f\n", 
	line[i].dist, line[i].theta); 
fputs(output_lines_ocl, file_lines_ocl); 
}
fclose(file_lines_ocl);
cout << "Lines created succefully." << endl;
//________________________________________________________________
//
// KERNEL1 END: "OpenCL-ling" CREATING LINES
//________________________________________________________________

//************* GOOD LINES (C STYLE BELOW) ***************
int good_line_idx[NUM_GOOD_LINES];
float dist_vect[NUM_LINES];
int count =0;

for(int ln=0; ln<NUM_LINES; ln++)
{
dist_vect[ln] = line[ln].dist;
}

std::vector<float> myvector(dist_vect, dist_vect+NUM_LINES-1);
std::sort (myvector.begin(), myvector.end()); 

//KERNEL SUITABLE//int fastcount=0;
for( std::vector<float>::const_iterator i = myvector.end(); i != myvector.begin(); i--)
{
   if(*i != (float)0 ) 
	for(int ln=0; ln<NUM_LINES; ln++)
	{
	//KERNEL SUITABLE//	fastcount = 0;
      if(dist_vect[ln] == *i && count<NUM_GOOD_LINES)
		{
			if(count == 0){
      			good_line_idx[count] = ln;
      			count++;
			}
			else 
				if (count > 0)
				{
					for(int kl=0; kl<count; kl++)
               {
				   	if(abs( line[ln].x[img_grad.rows-1] -
				   		line[good_line_idx[kl]].x[img_grad.rows-1] ) > 
				   		((float)GOOD_LINE_DIST/100.0)*img_grad.cols)
				   		{
				      		//KERNEL SUITABLE//if( kl<count ) fastcount++;
				      		//KERNEL SUITABLE//if(fastcount == count){	good_line_idx[count] = ln;
				     			//KERNEL SUITABLE//count++;}
						      if( kl<count-1 ) continue;
						      else{	good_line_idx[count] = ln;
						      	count++;}
				      	}
					   if(abs( line[ln].x[img_grad.rows-1] -
					   line[good_line_idx[kl]].x[img_grad.rows-1] ) < 
					   ((float)GOOD_LINE_DIST/100.0)*img_grad.cols) break;
				   }
		      }
	   }
   }//END ln
}//end i

linestruct *good_line;
good_line = new linestruct [NUM_GOOD_LINES];

for(int gl=0; gl<NUM_GOOD_LINES; gl++)
{
good_line[gl] = line[good_line_idx[gl]];
}
cout << "Good lines extracted successfully!" << endl;
//************* GOOD LINES (OPENCL STYLE BELOW) *********************
//GOODLINE KERNEL //goodParams *good_params;
//GOODLINE KERNEL //good_params = new goodParams [1];
//GOODLINE KERNEL //
//GOODLINE KERNEL //good_params->GOOD_LINE_DIST = GOOD_LINE_DIST;
//GOODLINE KERNEL //good_params->NUM_GOOD_LINES = NUM_GOOD_LINES;
//GOODLINE KERNEL //good_params->NUM_LINES = NUM_LINES;
//GOODLINE KERNEL //good_params->count = 0;
//GOODLINE KERNEL //good_params->imgrows = img_grad.rows;
//GOODLINE KERNEL //good_params->imgcols = img_grad.cols;
//GOODLINE KERNEL //good_params->dist_val_thisLine = 0.0;
//GOODLINE KERNEL //
//GOODLINE KERNEL //cl_context context1 = 0;
//GOODLINE KERNEL //cl_command_queue commandQueue1 = 0;
//GOODLINE KERNEL //cl_program program1 = 0;
//GOODLINE KERNEL //cl_device_id device1 = 0;
//GOODLINE KERNEL //cl_kernel kernel1 = 0;
//GOODLINE KERNEL //cl_int errNum1;
//GOODLINE KERNEL //cl_mem memExtractGoodLine[6] = {0,0,0,0,0,0};
//GOODLINE KERNEL //
//GOODLINE KERNEL //   cout << "0 Testing" << endl;  
//GOODLINE KERNEL ////for( std::vector<float>::const_iterator i = myvector.end(); i != myvector.begin(); i--)
//GOODLINE KERNEL ////{
//GOODLINE KERNEL //for(int i=0; i<NUM_LINES-2; i++ )
//GOODLINE KERNEL //{
//GOODLINE KERNEL //   //std::vector<int>::const_iterator itt = myvector.end();
//GOODLINE KERNEL ////   int i=0;
//GOODLINE KERNEL //   good_params->dist_val_thisLine = dist_vect_sorted[i];
//GOODLINE KERNEL //   //good_params->iterator = *itt;
//GOODLINE KERNEL //	//if(*itt != (float)0 ) 
//GOODLINE KERNEL //	if(dist_vect_sorted[i] != 0 ) 
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //   //_________________________________________________________________
//GOODLINE KERNEL //   //
//GOODLINE KERNEL //   // KERNEL2 BEGIN: "OpenCL-ling" EXTRACTION OF GOOD LINES
//GOODLINE KERNEL //   //_________________________________________________________________
//GOODLINE KERNEL //
//GOODLINE KERNEL //   cout << "1 Testing" << endl;  
//GOODLINE KERNEL //   // Create an OpenCL context on first available platform
//GOODLINE KERNEL //   context1 = CreateContext();
//GOODLINE KERNEL //   if (context1 == NULL)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //        cerr << "Failed to create OpenCL context." <<  endl;
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //  
//GOODLINE KERNEL //   // Create OpenCL program from kernel source
//GOODLINE KERNEL //   program1 = CreateProgram(context1, device1, "extractGoodLineKernel.cl");
//GOODLINE KERNEL //   if (program1 == NULL)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //        cerr << "Failed: CreateProgram of extractGoodLineKernel" <<  endl;
//GOODLINE KERNEL //       CleanupCreateLineKernel(context1, commandQueue1, program1,
//GOODLINE KERNEL //       	 kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   // Create OpenCL kernel
//GOODLINE KERNEL //   kernel1 = clCreateKernel(program1, "extractGoodLineKernel", NULL);
//GOODLINE KERNEL //   if (kernel1 == NULL)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //        cerr << "Failed: clCreateKernel of extractGoodLineKernel" <<  endl;
//GOODLINE KERNEL //       CleanupCreateLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   cout << "2 Testing" << endl;  
//GOODLINE KERNEL //   //Creating Memory Objects
//GOODLINE KERNEL //   if (!CreateMemExtractGoodLine(context1, memExtractGoodLine, line,
//GOODLINE KERNEL //            dist_vect,good_line_idx, good_params,
//GOODLINE KERNEL //            count_new, good_line_idx_new))
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Failed: CreateMemExtractGoodLine" <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //         kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   //Enqueuing the created memory objects.
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   //8888888888888888888888888888   PAUSED ......................!!!!!!!!!
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   errNum1 = clEnqueueWriteBuffer( 
//GOODLINE KERNEL //   		commandQueue1, 
//GOODLINE KERNEL //   		memExtractGoodLine[0], 
//GOODLINE KERNEL //   		CL_TRUE, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NUM_LINES*sizeof(linestruct), 
//GOODLINE KERNEL //   		(void *)line, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NULL, 
//GOODLINE KERNEL //   		NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   cout << "3 Testing" << endl;  
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error: enqueuing 'line' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);          
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   errNum1 = clEnqueueWriteBuffer( 
//GOODLINE KERNEL //   		commandQueue1, 
//GOODLINE KERNEL //   		memExtractGoodLine[1], 
//GOODLINE KERNEL //   		CL_TRUE, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NUM_LINES*sizeof(float), 
//GOODLINE KERNEL //   		(void *)dist_vect, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NULL, 
//GOODLINE KERNEL //   		NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error: enqueuing 'dist_vect' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);          
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   errNum1 = clEnqueueWriteBuffer( 
//GOODLINE KERNEL //   		commandQueue1, 
//GOODLINE KERNEL //   		memExtractGoodLine[2], 
//GOODLINE KERNEL //   		CL_TRUE, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NUM_GOOD_LINES*sizeof(int), 
//GOODLINE KERNEL //   		(void *)good_line_idx, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NULL, 
//GOODLINE KERNEL //   		NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error: enqueuing 'good_line_idx' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);          
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   errNum1 = clEnqueueWriteBuffer( 
//GOODLINE KERNEL //   		commandQueue1, 
//GOODLINE KERNEL //   		memExtractGoodLine[3], 
//GOODLINE KERNEL //   		CL_TRUE, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		sizeof(goodParams), 
//GOODLINE KERNEL //   		(void *)good_params, 
//GOODLINE KERNEL //   		0,
//GOODLINE KERNEL //   		NULL, 
//GOODLINE KERNEL //   		NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error: enqueuing 'goodParams' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);          
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   //Setting the kernel arguments
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 0, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[0]);
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 1, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[1]);
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 2, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[2]);
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 3, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[3]);
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 4, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[4]);
//GOODLINE KERNEL //   errNum1  = clSetKernelArg(kernel1, 5, 
//GOODLINE KERNEL //       sizeof(cl_mem),(void *)&memExtractGoodLine[5]);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error setting kernel arguments." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);    
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   // Queue the kernel up for execution across the array
//GOODLINE KERNEL //   size_t globalWorkSize[1] = { NUM_LINES };
//GOODLINE KERNEL //   size_t localWorkSize[1] = { 1 };
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   errNum1 = clEnqueueNDRangeKernel(commandQueue1, kernel1, 1, NULL,
//GOODLINE KERNEL //                                   globalWorkSize, localWorkSize,
//GOODLINE KERNEL //                                   0, NULL, NULL);
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //        cerr << "Error queuing kernel for execution." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   // Read the output buffer back to the Host
//GOODLINE KERNEL //   errNum1 = clEnqueueReadBuffer(
//GOODLINE KERNEL //                  commandQueue1, 
//GOODLINE KERNEL //                  memExtractGoodLine[4], 
//GOODLINE KERNEL //                  CL_TRUE, 0, 
//GOODLINE KERNEL //                  sizeof(int), 
//GOODLINE KERNEL //                  count_new, 
//GOODLINE KERNEL //                  0, NULL, NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //        cerr << "Error reading 'count_new' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   // Read the output buffer back to the Host
//GOODLINE KERNEL //   errNum1 = clEnqueueReadBuffer(
//GOODLINE KERNEL //                  commandQueue1, 
//GOODLINE KERNEL //                  memExtractGoodLine[5], 
//GOODLINE KERNEL //                  CL_TRUE, 0, 
//GOODLINE KERNEL //                  NUM_GOOD_LINES * sizeof(int), 
//GOODLINE KERNEL //                  good_line_idx_new, 
//GOODLINE KERNEL //                  0, NULL, NULL);
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   if (errNum1 != CL_SUCCESS)
//GOODLINE KERNEL //   {
//GOODLINE KERNEL //       cerr << "Error reading 'good_line_idx_new' buffer." <<  endl;
//GOODLINE KERNEL //       CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //       	kernel1, memExtractGoodLine);
//GOODLINE KERNEL //       return 1;
//GOODLINE KERNEL //   }
//GOODLINE KERNEL //   //-----------
//GOODLINE KERNEL //   //________________________________________________________________
//GOODLINE KERNEL //   //
//GOODLINE KERNEL //   //END: "OpenCL-ling" EXTRACTION OF GOOD LINES
//GOODLINE KERNEL //   //________________________________________________________________
//GOODLINE KERNEL //
//GOODLINE KERNEL //   for(int gl=0; gl<NUM_GOOD_LINES; gl++)
//GOODLINE KERNEL //      good_line_idx[gl] = good_line_idx_new[gl];
//GOODLINE KERNEL //   
//GOODLINE KERNEL //   good_params->count = *count_new;
//GOODLINE KERNEL // //  if(good_params->count == NUM_GOOD_LINES)
//GOODLINE KERNEL // //     break;
//GOODLINE KERNEL //
//GOODLINE KERNEL //   CleanupExtractGoodLineKernel(context1, commandQueue1, program1, 
//GOODLINE KERNEL //   	kernel1, memExtractGoodLine);
//GOODLINE KERNEL //   }

//*********** BEST LINES (C STYLE) ********************
int best_line_idx[NUM_BEST_LINES];
float good_dist_vect[NUM_GOOD_LINES];

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
		if(good_dist_vect[ln] == *i && count<NUM_BEST_LINES)
		{
			if(count == 0){
      			best_line_idx[count] = ln;
      			count++;
			}
			else 
				if (count > 0)
				{
		//CONDITION BELOW IS TO CHECK IF THE "GOOD" LINES ARE NOT VERY CLOSE TO EACH OTHER SO 
		//AS TO AVOID ANY SINGLE ROAD LINE BEING REPRESENTED BY TOO MANY "GOOD" LINES
					for(int kl=0; kl<count; kl++){
					if(abs( good_line[ln].x[img_grad.rows-1] -
						good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
						((float)BEST_LINE_DIST/100.0)*img_grad.cols)
						//0.3*img_grad.cols)
						{
						if( kl<count-1 ) continue;
						else{	best_line_idx[count] = ln;
							count++;}
						}
					if(abs( good_line[ln].x[img_grad.rows-1] -
					good_line[best_line_idx[kl]].x[img_grad.rows-1] ) < 
					((float)BEST_LINE_DIST/100.0)*img_grad.cols) break;
					//0.3*img_grad.cols) break;
					}
				}
		}
	}//END ln
}

//Best line weights
//cout << "Best line weights" << endl;
//for(int i=0; i<NUM_BEST_LINES; i++)
//{
//cout << good_line[best_line_idx[i]].dist << endl; 
//}

linestruct *best_line;
best_line = new linestruct [NUM_BEST_LINES];

for(int bl=0; bl<NUM_BEST_LINES; bl++)
{
best_line[bl] = good_line[best_line_idx[bl]];
}
cout << "Best lines extracted successfully!" << endl;
//*********** PLOTTING *******************
//------GOOD LINES-------
std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > goodlines[NUM_GOOD_LINES];

int x1good=0, y1good=0, x2good=0, y2good=0;  
float r=0.0;

for (int i=0; i<NUM_GOOD_LINES; i++)
{
   r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
   x1good = (int) (good_line[i].x[0]);
   y1good = (int) (good_line[i].y[0]);
   x2good = (int) (r * cos(good_line[i].theta) + good_line[i].x[0]);
   y2good = (int) (r * sin(good_line[i].theta) + good_line[i].y[0]);
   
   goodlines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1good,y1good), std::pair<int, int>(x2good,y2good)));  
   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_good;  
   for(it_good = goodlines[i].begin();it_good!=goodlines[i].end();it_good++)  
   {  
      cv::line(img_org, cv::Point(it_good->first.first, it_good->first.second), 
   	cv::Point(it_good->second.first, it_good->second.second), cv::Scalar(0,255,0), 2,8,0);  
   }
} 
//------BEST LINES-------
std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > bestlines[NUM_BEST_LINES];
int x1best=0, y1best=0, x2best=0, y2best=0;  

for (int i=0; i<NUM_BEST_LINES; i++)
{
   r = sqrt(pow(img_gray.rows,2) + pow(img_gray.cols,2));
   x1best = (int) (best_line[i].x[0]);
   y1best = (int) (best_line[i].y[0]);
   x2best = (int) (r * cos(best_line[i].theta) + best_line[i].x[0]);
   y2best = (int) (r * sin(best_line[i].theta) + best_line[i].y[0]);
   
   bestlines[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1best,y1best), std::pair<int, int>(x2best,y2best)));  
   
   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_best;  
   for(it_best=bestlines[i].begin();it_best!=bestlines[i].end();it_best++)  
   {  
        cv::line(img_org, cv::Point(it_best->first.first, it_best->first.second), 
   	cv::Point(it_best->second.first, it_best->second.second), cv::Scalar(0,0,255), 2, 8,0);  
   }
}
	cv::namedWindow("Test Window",CV_WINDOW_NORMAL);	
	cv::imshow("Test Window", img_org);
	cv::waitKey( 0 );

//*****************************************
//___________________________________________________________________
//
//                          PARTICLE FILTER
//___________________________________________________________________

   clock_t t1_clock, t2_clock;
   t1_clock = clock();
//linestruct *bestlinepf;
//bestlinepf = new linestruct [NUM_BEST_LINES];
//
//linestruct *goodlinepf;
//goodlinepf = new linestruct [NUM_GOOD_LINES];

noiseStruct *allnoises;
allnoises = new noiseStruct [1];

allnoises->forwardNoise = 0.05;
allnoises->turnNoise = 0.05;
allnoises->senseNoise = 100.5;

float turn1 =0.0, dist1 = 0.0;

//Moving BEST LINES
for(int bl=0; bl<NUM_BEST_LINES; bl++) //'gl' for goodline
{	
	//bestlinepf[bl] = best_line[bl]; 
	move_line(&best_line[bl], allnoises, 0.5, 5.0, img_grad.rows, img_grad.cols);
	
	for(int blord=0;blord<img_grad.rows;blord++)//glord for goodline coordinate
		{
		if(best_line[bl].x[blord]>=NEIGHBORHOOD && best_line[bl].x[blord] <= img_grad.cols-NEIGHBORHOOD){  
		 	for(int m=best_line[bl].x[blord]-NEIGHBORHOOD; m<best_line[bl].x[blord]+NEIGHBORHOOD; m++)
		 	{
		 		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		 	}
		 }
		 else if(best_line[bl].x[blord] > img_grad.cols-NEIGHBORHOOD){  
		 	for(int m=best_line[bl].x[blord]-NEIGHBORHOOD; m<img_grad.cols; m++)
		 	{
		 		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		 	}
		 }
		 else if(best_line[bl].x[blord]<NEIGHBORHOOD){
		 	for(int m=0; m<best_line[bl].x[blord]+NEIGHBORHOOD; m++)
		 	{
		 		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		 	}
		 }
		} //end blord
} //end bl

//for(int ln=0; ln<NUM_GOOD_LINES; ln++)
//{goodlinepf[ln] = good_line[ln];}

float w_pf[NUM_GOOD_LINES];

for(int topIndex=0; topIndex<iterations; topIndex++) 
{
	//Moving BEST LINES
   for(int bl=0; bl<NUM_BEST_LINES; bl++) //'gl' for goodline
	{	
		move_line(&best_line[bl], allnoises, 0.5, 5.0, img_grad.rows, img_grad.cols);
		for(int blord=0;blord<img_grad.rows;blord++)//glord for goodline coordinate
		{
		   RNG rng(time(NULL));
		   if(best_line[bl].x[blord]>=NEIGHBORHOOD && best_line[bl].x[blord] <= img_grad.cols-NEIGHBORHOOD){  
		    	for(int m=best_line[bl].x[blord]-NEIGHBORHOOD; m<best_line[bl].x[blord]+NEIGHBORHOOD; m++)
		    	{
		    		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		    	}
		    }
		    else if(best_line[bl].x[blord] > img_grad.cols-NEIGHBORHOOD){  
		    	for(int m=best_line[bl].x[blord]-NEIGHBORHOOD; m<img_grad.cols; m++)
		    	{
		    		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		    	}
		    }
		    else if(best_line[bl].x[blord]<NEIGHBORHOOD){
		    	for(int m=0; m<best_line[bl].x[blord]+NEIGHBORHOOD; m++)
		    	{
		    		best_line[bl].dist += img_grad.at<uchar>(blord,m);
		    	}
		    }
		} //end blord
	}//end bl

	float weight_pf = 0.0;
	float prob_dist = 0.0;
	//int bl_points[NUM_BEST_LINES];
   //for(int gl=0; gl<NUM_BEST_LINES; gl++) bl_points[gl] = 0;
   	
   //************* PARTICLE WEIGHTS (C STYLE BELOW) *********************
   //C STYLE//for(int i=0; i<NUM_GOOD_LINES; i++)
   //C STYLE//{
   //C STYLE//    tmp_coeff = rand()%10000;
   //C STYLE//    turn1 = ((float)tmp_coeff/10000.0000)*TURNMAX;
   //C STYLE//    tmp_coeff = rand()%10000;
   //C STYLE//    dist1 = ((float)tmp_coeff/10000.0000)*DISTMAX;
	//C STYLE//
   //C STYLE//    move_line(&good_line[i], allnoises, turn1, dist1, 
   //C STYLE//      img_grad.rows, img_grad.cols);
	//C STYLE//    
	//C STYLE//    good_line[i].dist = 0.0;
	//C STYLE//    for(int l=0; l<img_grad.rows; l++)
	//C STYLE//    {	
	//C STYLE//    	RNG rng(time(NULL));
	//C STYLE//    	if(good_line[i].x[l]>=NEIGHBORHOOD && 
   //C STYLE//         good_line[i].x[l] <= img_grad.cols-NEIGHBORHOOD){  
	//C STYLE//        	for(int m = good_line[i].x[l]-NEIGHBORHOOD; 
   //C STYLE//            m < good_line[i].x[l]+NEIGHBORHOOD; m++)
	//C STYLE//        		{
	//C STYLE//        			good_line[i].dist += img_grad.at<uchar>(l,m);
	//C STYLE//    	      }
	//C STYLE//    	}
	//C STYLE//     	else if(good_line[i].x[l] > img_grad.cols-NEIGHBORHOOD){  
	//C STYLE//    		for(int m=good_line[i].x[l]-NEIGHBORHOOD; m<img_grad.cols; m++)
	//C STYLE//    		{
	//C STYLE//    			good_line[i].dist += img_grad.at<uchar>(l,m);
	//C STYLE//    		}
	//C STYLE//    	}
	//C STYLE//    	else if(good_line[i].x[l]<NEIGHBORHOOD){
	//C STYLE//    		for(int m=0; m<good_line[i].x[l]+NEIGHBORHOOD; m++)
	//C STYLE//    		{
	//C STYLE//    			good_line[i].dist += img_grad.at<uchar>(l,m);
	//C STYLE//    		}
	//C STYLE//    	}
	//C STYLE//    }//end l
	//C STYLE//    
	//C STYLE//    w_pf[i] = 0.0;	
	//C STYLE//    
   //C STYLE//    //Computing the weight of each particle
	//C STYLE//    for(int gl=0; gl<NUM_BEST_LINES; gl++)
   //C STYLE//    {	
	//C STYLE//       prob_dist = gaussian(good_line[i].theta, allnoises->turnNoise,
   //C STYLE//         best_line[gl].theta);	
	//C STYLE//       prob_dist *= gaussian(good_line[i].x[0], 
   //C STYLE//         allnoises->senseNoise, best_line[gl].x[0]);
	//C STYLE//       prob_dist *= gaussian(good_line[i].x[img_grad.rows-1], 
   //C STYLE//         allnoises->senseNoise, best_line[gl].x[img_grad.rows-1]);
	//C STYLE//       	
   //C STYLE//       if( prob_dist > w_pf[i])
	//C STYLE//       {
	//C STYLE//         w_pf[i] = prob_dist;
	//C STYLE//         //bl_points[gl]++;
	//C STYLE//       }
	//C STYLE//    } //CLOSING gl-LOOP.	
	//C STYLE//} //CLOSING i-LOOP.	
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
   
   //-----------
   for(int gl=0; gl<NUM_GOOD_LINES; gl++)
   {
      srand(10090329*gl + time(NULL)*gl*gl*gl + 2346*gl);
      tmp_coeff = rand()%10000;
      turnArray[gl] = ((float)tmp_coeff/10000.0000)*TURNMAX;
            
      tmp_coeff = rand()%10000;
      distArray[gl] = ((float)tmp_coeff/10000.0000)*DISTMAX;
      
      //RNG rng(time(NULL));
      RNG rng(10090329*gl + time(NULL)*gl*gl*gl + 2346*gl);
	   rngturnArray[gl] = rng.gaussian(allnoises->turnNoise);
	   rngfwdArray[gl] = rng.gaussian(allnoises->forwardNoise);
      
      //tmp_coeff = rand()%10000;
      //rngturnArray[gl] = ((float)tmp_coeff/10000.0000)*allnoises->turnNoise;
      //
      //tmp_coeff = rand()%10000;
      //rngfwdArray[gl] = ((float)tmp_coeff/10000.0000)*allnoises->forwardNoise;
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
   size_t localWorkSize_wt[1] = { 100 };
   
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

   //Writing OUTPUT FILES
   FILE* file_best_cord = fopen("bestcord.txt", "a"); 
   char output_best_cord[255]; 
   for (int i=0; i<NUM_BEST_LINES; i++)
   {
    	sprintf(output_best_cord, "%7.3f   %10.9f\n", best_line[i].x[0], best_line[i].theta); 
      fputs(output_best_cord, file_best_cord); 
   }
   fclose(file_best_cord); 
        	
   FILE* file_good_cord = fopen("goodcord.txt", "a"); 
   char output_good_cord[255]; 
   for (int i=0; i<NUM_GOOD_LINES; i++)
   {
    	sprintf(output_good_cord, "%7.3f   %10.9f\n", good_line[i].x[0], good_line[i].theta);
   	fputs(output_good_cord, file_good_cord); 
   }
   fclose(file_good_cord); 

}//END topIndex Iteration loop 

   t2_clock = clock();
   float diff ((float)t2_clock-(float)t1_clock);
   cout<<"Total Running Time = " << diff/CLOCKS_PER_SEC << "\n"<< endl;
 


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
