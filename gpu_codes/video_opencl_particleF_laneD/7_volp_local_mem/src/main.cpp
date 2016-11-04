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

#define CV_LOAD_IMAGE_GRAYSCALE 0
#define CV_LOAD_IMAGE_COLOR 1
#define CV_WINDOW_NORMAL 0

#define AOCL_ALIGNMENT 256
void * acl_aligned_malloc (size_t size) 
{
    void * result = NULL;
    posix_memalign (&result, AOCL_ALIGNMENT, size);
    return result;
}

void acl_aligned_free (void * ptr) 
{
    free (ptr);
}

float TURNMAX = 1.5;
float DISTMAX = 30.0;

bool createMemCreateLine(cl_context, cl_mem *, struct createLineParams,
   int *, struct lineStruct *, int, int, int);

void cleanupMemCreateLine(cl_mem *);

bool createMemGoodLineWeight(cl_context, cl_mem *,
   struct lineStruct *, struct lineStruct *, int *, 
   struct noiseStruct *, struct wtKernParams *, 
   struct lineStruct *, float *);

void cleanupMemGoodLineWeight( cl_mem *);

void cleanupGenContext(cl_context, 
cl_command_queue,
cl_program,
cl_kernel *,
cl_uint); 

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
 
   double num_frames;
   num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

   int total_v_frames = 0;
   int Iframes = 1000;//Take accurate measurement after every Iframes.
   int NUM_GOOD_LINES = 300;
//   int iterations = 1;
   int NUM_LINES = 3000;
   int NEIGHBORHOOD = 3;

   //POSIX_ALIGNMENT ------------------- 
   struct lineStruct *good_line;
   good_line = (struct lineStruct *) acl_aligned_malloc(sizeof(struct lineStruct) * NUM_GOOD_LINES);
   if(good_line == NULL)
   {
     printf("Allocation of the 'good_line' has failed\n");
     return -1;
   } 
   //-----------------------------------
   
   //POSIX_ALIGNMENT ------------------- 
   struct lineStruct *best_line;
   best_line = (struct lineStruct *) acl_aligned_malloc(sizeof(struct lineStruct) * NUM_BEST_LINES);
   if(best_line == NULL)
   {
     printf("Allocation of the 'best_line' has failed\n");
     return -1;
   } 
   //-----------------------------------
  
   int best_line_idx[NUM_BEST_LINES];
   float good_dist_vect[NUM_GOOD_LINES];
   
   int good_line_idx[NUM_GOOD_LINES];
   float dist_vect[NUM_LINES];
   int count =0;
   
   int x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0;  
   int x21=0, y21=0, x22=0, y22=0;  
   std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_tmp[NUM_BEST_LINES];
   
   //************************************************************
         //OpenCL parameters
         cl_context context = 0;
         cl_device_id device = 0;
         cl_command_queue commandQueue = 0;
         cl_program program = 0;
         cl_kernel *kernel = NULL;
         cl_uint numKernels = 0;
         cl_int errNum = 0;
         
         size_t lengths[1];
         unsigned char* binaries[1] ={NULL}; 
         cl_int status[1];
         const char options[] = "";
      
         // Create an OpenCL context on first available platform
         context = CreateContext();
         if (context == NULL)
         {
             cerr << "ERROR: Failed to create OpenCL context." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             return -1;
         }
         
         // Create a command-queue on the first device available
         // on the created context
         commandQueue = CreateCommandQueue(context, &device);
         if (commandQueue == NULL)
         {
             cerr << "ERROR: Failed to create commandQueue." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             return -1;
         }

         //Create the program and build
         //FILE *fp = fopen("pFlaneD.aocx","rb"); 
         //fseek(fp,0,SEEK_END); 
         //lengths[0] =ftell(fp); 
         //binaries[0]= (unsigned char*)malloc(sizeof(unsigned char)*lengths[0]); 
         //rewind(fp); 
         //fread(binaries[0],lengths[0],1,fp); 
         //fclose(fp);

         //program = clCreateProgramWithBinary(context,1,&device,lengths,
         //            (const unsigned char **)binaries,status,&errNum);
         
         program = CreateProgram(context, device, "createLineKernel.cl");
         
         if (program == NULL) 
         {
            cerr << "ERROR: Failed to create program." <<  endl;
            cleanupGenContext( context, commandQueue, program, kernel, numKernels );
            return -1;
         }
         
         clBuildProgram(program,1,&device,options,NULL,NULL);
         clUnloadCompiler(); 
         //-----------
         // Create OpenCL kernel
         errNum = clCreateKernelsInProgram(program, 0, 
                                           NULL, &numKernels);
         if (errNum != CL_SUCCESS)
         {
            cerr << "ERROR: Obtaining the number of kernels in pFlaneD.cl" <<  endl;
            cleanupGenContext( context, commandQueue, program, kernel, numKernels );
            return -1;
         }
         
         kernel = new cl_kernel[numKernels]; 
         errNum = clCreateKernelsInProgram(program, numKernels, kernel, 
                                           &numKernels);
         if (errNum != CL_SUCCESS)
         {
            cerr << "ERROR: Creating the kernels from pFlaneD.cl" <<  endl;
            cleanupGenContext( context, commandQueue, program, kernel, numKernels );
            return -1;
         }
   
   //************************************************************
   
   float total_clocks = 0.0; 
   
   for(int framme=0; framme < num_frames-4; framme++)
   {
      clock_t t_total_start, t_total_end;
      t_total_start = clock();
    
      cout << "Frame Number of the Video = " << total_v_frames+1 << endl;
      capture >> img_org_full;
   
      cv::Rect rect;
      //int ROI_START_X = (int)(0.20*img_org_full.cols);// % of image cropped in the right
      //int ROI_START_Y = (int)(0.80*img_org_full.rows);// % of image cropped in the top
      //rect = cv::Rect(0,ROI_START_Y,img_org_full.cols-ROI_START_X,img_org_full.rows-ROI_START_Y);
      
      //original
      //int ROI_START_X = (int)(0.10*img_org_full.cols);// % of image cropped in the right
      //int ROI_START_Y = (int)(0.70*img_org_full.rows);// % of image cropped in the top
      //rect = cv::Rect(ROI_START_X, ROI_START_Y, img_org_full.cols-(int)(0.30*img_org_full.cols), img_org_full.rows-ROI_START_Y);
      
      int ROI_START_X = (int)(0.10*img_org_full.cols);// % of image cropped in the right
      int ROI_START_Y = (int)(0.70*img_org_full.rows);// % of image cropped in the top
      int xROI = img_org_full.cols-(int)(0.30*img_org_full.cols);
      int yROI = img_org_full.rows-ROI_START_Y;
      rect = cv::Rect(ROI_START_X, ROI_START_Y, xROI, yROI);
      
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
      
      //POSIX_ALIGNMENT ------------------- 
      int *img_grad_array;
      img_grad_array = (int *) acl_aligned_malloc(sizeof(int) * img_grad.rows * img_grad.cols);
      if(img_grad_array == NULL)
      {
        printf("Allocation of the 'img_grad_array' has failed\n");
        return -1;
      } 
      //--------------------------------
   
           float GOOD_LINE_DIST = img_grad.cols/(4*NUM_GOOD_LINES);
     
      //POSIX_ALIGNMENT ------------------- 
      struct createLineParams *cline_params_tmp;
      struct createLineParams cline_params;
      cline_params_tmp = (struct createLineParams *) acl_aligned_malloc(sizeof(struct createLineParams));
      if(cline_params_tmp == NULL)
      {
        printf("Allocation of the 'cline_params_tmp' has failed\n");
        return -1;
      }

      cline_params = *cline_params_tmp;
      //--------------------------------

      cline_params.NEIGHBORHOOD = NEIGHBORHOOD;
      cline_params.NUM_LINES = NUM_LINES;
      cline_params.imgrows = img_grad.rows;
      cline_params.imgcols = img_grad.cols;
           
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
         
         clock_t t_context1_start, t_context1_end;
         t_context1_start = clock();

         //OpenCL parameters
         //cl_program program = 0;
         //cl_kernel kernels = 0;
         //cl_int errNum;
         //cl_context context = 0;
         //cl_command_queue commandQueue = 0;
         //cl_device_id device = 0;
         //
         //size_t lengths[1];
         //unsigned char* binaries[1] ={NULL}; 
         //cl_int status[1];
         //const char options[] = "";
      
         //// Create an OpenCL context on first available platform
         //context = CreateContext();
         //if (context == NULL)
         //{
         //     cerr << "Failed to create OpenCL context." <<  endl;
         //    return -1;
         //}
         //
         //// Create a command-queue on the first device available
         //// on the created context
         //commandQueue = CreateCommandQueue(context, &device);
         //if (commandQueue == NULL)
         //{
         //    cerr << "Failed to create commandQueue." <<  endl;
         //    CleanupCreateLineKernel(context, commandQueue, program, 
         //            kernels, memCreateLine);
         //    return -1;
         //}

         ////Create the program and build
         //FILE *fp = fopen("pFlaneD.aocx","rb"); 
         //fseek(fp,0,SEEK_END); 
         //lengths[0] =ftell(fp); 
         //binaries[0]= (unsigned char*)malloc(sizeof(unsigned char)*lengths[0]); 
         //rewind(fp); 
         //fread(binaries[0],lengths[0],1,fp); 
         //fclose(fp);

         //program = clCreateProgramWithBinary(context,1,&device,lengths,
         //            (const unsigned char **)binaries,status,&errNum);
         //
         //if (program == NULL) 
         //{
         //   CleanupCreateLineKernel(context, commandQueue, program, 
         //           kernels, memCreateLine);
         //   return -1;
         //}
         //
         //clBuildProgram(program,1,&device,options,NULL,NULL);
         //clUnloadCompiler(); 
         ////-----------
         //// Create OpenCL kernel
         //cl_uint numKernels;
         //errNum = clCreateKernelsInProgram(program, 0, 
         //                                  NULL, &numKernels);
         //
         //cl_kernel *kernel = new cl_kernel[numKernels]; 
         //errNum = clCreateKernelsInProgram(program, numKernels, kernel, 
         //                                  &numKernels);
         
         t_context1_end = clock();
         float diff_context1 ((float)t_context1_end-(float)t_context1_start);
         cout<<"Context 1 Time = " << diff_context1/CLOCKS_PER_SEC << "\n"<< endl;
    
         //-----------

         struct lineStruct *line;
         line = new struct lineStruct [NUM_LINES];
         
         //POSIX_ALIGNMENT ------------------- 
         struct lineStruct *line_new;
         line_new = (struct lineStruct *) acl_aligned_malloc(sizeof(struct lineStruct) * NUM_LINES);
         if(line_new == NULL)
         {
           printf("Allocation of the 'line_new' has failed\n");
           return -1;
         } 
         //_________________________________________________________________
         //
         // KERNEL1 BEGIN: "OpenCL-ling" CREATING LINES
         //_________________________________________________________________
         
         //int c2;
 
         ////POSIX_ALIGNMENT ------------------- 
         //float *xadd;
         //xadd = (float *) acl_aligned_malloc(sizeof(float) * NUM_LINES);
         //if(xadd == NULL)
         //{
         //  printf("Allocation of the 'xadd' has failed\n");
         //  return -1;
         //} 
         ////--------------------------------
         //for(int ln=0; ln<NUM_LINES; ln++)
         //{
         //   srand(ln*10000+20*ln*ln*ln);
         //   c2 = rand()%10000;
         //   xadd[ln] = (float)(((float)c2/10000.0000) * img_grad.cols);
         //}
         ////-----------
         
         clock_t t_kernel1_start, t_kernel1_end;
         t_kernel1_start = clock();
         
         //cl_uint numMemCreateLine = 3;
         cl_mem memCreateLine[3] = {0, 0, 0};
         
         //Creating Memory Objects for createLineKernel
         if (!createMemCreateLine(context, memCreateLine, cline_params,
            img_grad_array, line_new, img_grad.rows, 
            img_grad.cols, NUM_LINES))
         {
             cerr << "ERROR: CreateMemCreateLine" <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
         }
         //-----------
         //Enqueuing the created memory objects.
         errNum = clEnqueueWriteBuffer( 
                         commandQueue, 
                         memCreateLine[0], 
                         CL_TRUE, 
                         0,
                         sizeof(struct createLineParams), 
                         (void *)&cline_params, 
                         0,
                         NULL, 
                         NULL);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error: enqueuing cline_params buffer." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
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
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
         }
         //-----------
         //errNum = clEnqueueWriteBuffer( 
         //                commandQueue, 
         //                memCreateLine[2], 
         //                CL_TRUE, 
         //                0,
         //                NUM_LINES*sizeof(float), 
         //                (void *)xadd, 
         //                0,
         //                NULL, 
         //                NULL);
         //
         //if (errNum != CL_SUCCESS)
         //{
         //    cerr << "Error: enqueuing xadd buffer." <<  endl;
         //    CleanupCreateLineKernel(context, commandQueue, program, 
         //            kernel[0], memCreateLine);          
         //    return -1;
         //}
         //-----------
         
         //Setting the kernel arguments
         errNum  = clSetKernelArg(kernel[0], 0, 
             sizeof(cl_mem),(void *)&memCreateLine[0]);
         errNum |= clSetKernelArg(kernel[0], 1, 
             sizeof(cl_mem),(void *)&memCreateLine[1]);
         //errNum |= clSetKernelArg(kernel[0], 2, 
         //    sizeof(cl_mem),(void *)&memCreateLine[2]);
         errNum |= clSetKernelArg(kernel[0], 2, 
             sizeof(cl_mem),(void *)&memCreateLine[2]);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error setting kernel arguments." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
         }
         //-----------
         // Queue the kernel up for execution across the array
         
         //size_t globalWorkSize[1] = { NUM_LINES };
         //size_t localWorkSize[1] = { 250 };
         
         size_t localWorkSize[1] = { 250 };
         size_t lws = localWorkSize[0];
        	
        	size_t globalWorkSize[1] = { 0 };
        	
        	if((int)(NUM_LINES/lws)*lws < NUM_LINES)
        	{
        		globalWorkSize[0] = (lws * ((int)(NUM_LINES/lws) + 1));
        	}
        	else
        	{
        		globalWorkSize[0] = NUM_LINES;
        	}
        
         errNum = clEnqueueNDRangeKernel(commandQueue, kernel[0], 1, NULL,
                                         globalWorkSize, localWorkSize,
                                         0, NULL, NULL);
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error queuing kernel for execution." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
         }
         //-----------
         clFinish(commandQueue);
         //-----------
          
         // Read the output buffer back to the Host
         errNum = clEnqueueReadBuffer(commandQueue, memCreateLine[2], 
                     CL_TRUE, 0, NUM_LINES * sizeof(struct lineStruct), 
                     line_new, 0, NULL, NULL);
         
         if (errNum != CL_SUCCESS)
         {
             cerr << "Error reading result buffer." <<  endl;
             cleanupGenContext( context, commandQueue, program, kernel, numKernels );
             cleanupMemCreateLine( memCreateLine );
             return -1;
         }
         
         t_kernel1_end = clock();
         float diff_kernel1 ((float)t_kernel1_end-(float)t_kernel1_start);
         cout<<"Kernel 1 Time = " << diff_kernel1/CLOCKS_PER_SEC << "\n"<< endl;
        
         //-----------
         for (int i = 0; i < NUM_LINES; i++)
         {
             line[i] = line_new[i];        
         }
         
         //-----------
         clFinish(commandQueue);
         //-----------
         //cout << "line[0].x[0] = " << line[0].x[0] << endl;
         //cout << "line[1].x[0] = " << line[1].x[0] << endl;
         //cout << "line[100].x[0] = " << line[100].x[0] << endl;
         //cout << "line[947].x[0] = " << line[947].x[0] << endl;
         //cout << "line[1566].x[0] = " << line[1566].x[0] << endl;
         //cout << "line[2146].x[0] = " << line[2146].x[0] << endl;
         //cout << "line[2794].x[0] = " << line[2794].x[0] << endl;
         
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
                           if(line[ln].x[0] <= img_grad.cols/2 && fhalf <= NUM_GOOD_LINES/2)
                           {
                              good_line_idx[count] = ln;
                              count++;
                              fhalf++;
                           }
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
         
        cout << "fhalf = " << fhalf << endl;
        cout << "shalf = " << shalf << endl;
         
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
                     for(int kl=0; kl<count; kl++)
                     {
                        if(abs( good_line[ln].x[img_grad.rows-1] -
                           good_line[best_line_idx[kl]].x[img_grad.rows-1] ) > 
                           ((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols)
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
                            ((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols)         
                            break;
                     }
                  }
               }//end good_dist_vect[ln]
             }//END ln
         }//end i
         
         for(int bl=0; bl<NUM_BEST_LINES; bl++)
         {
         best_line[bl] = good_line[best_line_idx[bl]];
         }
         cout << "Best lines extracted successfully!" << endl;
   
         //cout << "best_line[0].x[0] = " << best_line[0].x[0] << endl;
         //cout << "best_line[1].x[0] = " << best_line[1].x[0] << endl;
         //cout << "good_line[1].x[0] = " << good_line[1].x[0] << endl;
         //cout << "good_line[50].x[0] = " << good_line[50].x[0] << endl;
         //cout << "good_line[150].x[0] = " << good_line[150].x[0] << endl;
         //cout << "good_line[250].x[0] = " << good_line[250].x[0] << endl;
         
         //clReleaseKernel(kernel[0]);
         //
         //CleanupCreateLineKernel(context, commandQueue, program, 
         //        kernel[1], memCreateLine);
  
          
         clFinish(commandQueue);
         
         total_v_frames++;
         
         acl_aligned_free(line_new);
         cleanupMemCreateLine( memCreateLine );
      
      } //END total_video_frames%Iframes =0 step.
      else
      { //IF VIDEO FRAME > 0
         cout << "Im here in else" << endl;
         
         clock_t t_context2_start, t_context2_end;
         t_context2_start = clock();
         
         //OpenCL parameters
         //cl_program program_wt = 0;
         //cl_kernel kernels1 = 0;
         //cl_int errNum = 0;
         //cl_context context_wt = 0;
         //cl_command_queue commandQueue = 0;
         //cl_device_id device_wt = 0;
         //cl_mem memGoodLineWeight[7] = {0,0,0,0,0,0,0};
         //
         //size_t lengths1[1];
         //unsigned char* binaries1[1] ={NULL}; 
         //cl_int status1[1];
         //const char options[] = "";

         //// Create an OpenCL context on first available platform
         //context_wt = CreateContext();
         //if (context_wt == NULL)
         //{
         //     cerr << "Failed to create OpenCL context." <<  endl;
         //    return -1;
         //}
         //
         //// Create a command-queue on the first device_wt available
         //// on the created context
         //commandQueue = CreateCommandQueue(context_wt, &device_wt);
         //if (commandQueue == NULL)
         //{
         //    cerr << "Failed to create commandQueue." <<  endl;
         //    CleanupWtKernel(context_wt, commandQueue, program_wt, 
         //            kernels1, memGoodLineWeight);
         //    return -1;
         //}

         //FILE *fp1 = fopen("pFlaneD.aocx","rb"); 
         //fseek(fp1,0,SEEK_END); 
         //lengths1[0] =ftell(fp1); 
         //binaries1[0]= (unsigned char*)malloc(sizeof(unsigned char)*lengths1[0]); 
         //rewind(fp1); 
         //fread(binaries1[0],lengths1[0],1,fp1); 
         //fclose(fp1);

         //program_wt = clCreateProgramWithBinary(context_wt,1,&device_wt,lengths1,
         //            (const unsigned char **)binaries1,status1,&errNum);
         //
         //if (program_wt == NULL) 
         //{
         //   CleanupWtKernel(context_wt, commandQueue, program_wt, 
         //           kernels1, memGoodLineWeight);
         //   return -1;
         //}
         //
         //clBuildProgram(program_wt,1,&device_wt,options,NULL,NULL);
         //clUnloadCompiler(); 
         ////-----------
         //// Create OpenCL kernel
         //cl_uint numKernels1;
         //errNum = clCreateKernelsInProgram(program_wt, 0, 
         //                                  NULL, &numKernels1);
         //
         //cl_kernel *kernel = new cl_kernel[numKernels1]; 
         //errNum = clCreateKernelsInProgram(program_wt, numKernels1, kernel, 
         //                                  &numKernels1);
         ////-----------
        
         t_context2_end = clock();
         float diff_context2 ((float)t_context2_end-(float)t_context2_start);
         cout<<"Context 2 Time = " << diff_context2/CLOCKS_PER_SEC << "\n"<< endl;
    
         //___________________________________________________________________
         //
         //                          PARTICLE FILTER
         //___________________________________________________________________
         
         //POSIX_ALIGNMENT ------------------- 
         struct noiseStruct *allnoises;
         allnoises = (struct noiseStruct *) acl_aligned_malloc(sizeof(struct noiseStruct));
         if(allnoises == NULL)
         {
           printf("Allocation of the 'allnoises' has failed\n");
           return -1;
         } 
         //-----------------------------------
         
         allnoises->forwardNoise = 0.05;
         allnoises->turnNoise = 0.05;
         allnoises->senseNoise = 100.5;
         
         float turn1 =0.0, dist1 = 0.0;
         int c1 = 0;
         
         float w_pf[NUM_GOOD_LINES];
         
         //for(int topIndex=0; topIndex<iterations; topIndex++) 
         //{
                 float weight_pf = 0.0;
                 float prob_dist = 0.0;
                    
            //_________________________________________________________________
            //
            // KERNEL3 BEGIN: "OpenCL-ling" EVALUATION OF PARTICLE WEIGHTS
            //_________________________________________________________________
            
            ////POSIX_ALIGNMENT ------------------- 
            //float *turnArray;
            //turnArray = (float *) acl_aligned_malloc(sizeof(float) * NUM_GOOD_LINES);
            //if(turnArray == NULL)
            //{
            //  printf("Allocation of the 'turnArray' has failed\n");
            //  return -1;
            //} 
            ////-----------------------------------
            ////POSIX_ALIGNMENT ------------------- 
            //float *distArray;
            //distArray = (float *) acl_aligned_malloc(sizeof(float) * NUM_GOOD_LINES);
            //if(distArray == NULL)
            //{
            //  printf("Allocation of the 'distArray' has failed\n");
            //  return -1;
            //} 
            ////-----------------------------------
            ////POSIX_ALIGNMENT ------------------- 
            //float *rngturnArray;
            //rngturnArray = (float *) acl_aligned_malloc(sizeof(float) * NUM_GOOD_LINES);
            //if(rngturnArray == NULL)
            //{
            //  printf("Allocation of the 'rngturnArray' has failed\n");
            //  return -1;
            //} 
            ////-----------------------------------
            ////POSIX_ALIGNMENT ------------------- 
            //float *rngfwdArray;
            //rngfwdArray = (float *) acl_aligned_malloc(sizeof(float) * NUM_GOOD_LINES);
            //if(rngfwdArray == NULL)
            //{
            //  printf("Allocation of the 'rngfwdArray' has failed\n");
            //  return -1;
            //} 
            ////-----------------------------------
            
            //POSIX_ALIGNMENT ------------------- 
            struct wtKernParams *wtparams_tmp;
            struct wtKernParams wtparams;
            wtparams_tmp = (struct wtKernParams *) acl_aligned_malloc(sizeof(struct wtKernParams));
            if(wtparams_tmp == NULL)
            {
              printf("Allocation of the 'wtparams_tmp' has failed\n");
              return -1;
            } 
            wtparams = *wtparams_tmp;
            //-----------------------------------
         
            //POSIX_ALIGNMENT ------------------- 
            struct lineStruct *good_line_new;
            good_line_new = (struct lineStruct *) acl_aligned_malloc(sizeof(struct lineStruct) * NUM_GOOD_LINES);
            if(good_line_new == NULL)
            {
              printf("Allocation of the 'good_line_new' has failed\n");
              return -1;
            } 
            //-----------------------------------
         
            //POSIX_ALIGNMENT ------------------- 
            float *w_pf_new;
            w_pf_new = (float *) acl_aligned_malloc(sizeof(float) * NUM_GOOD_LINES);
            if(w_pf_new == NULL)
            {
              printf("Allocation of the 'w_pf_new' has failed\n");
              return -1;
            } 
            //-----------------------------------
           
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
            //        rngturnArray[gl] = rng.gaussian(allnoises->turnNoise);
            //        rngfwdArray[gl] = rng.gaussian(allnoises->forwardNoise);
            // }
            
            wtparams.NEIGHBORHOOD = NEIGHBORHOOD;
            wtparams.NUM_GOOD_LINES = NUM_GOOD_LINES;
            wtparams.NUM_BEST_LINES = NUM_BEST_LINES;
            wtparams.imgrows = img_grad.rows;  
            wtparams.imgcols = img_grad.cols;
            
            srand((framme*13434976+24385*framme*framme*framme)*time(NULL));
            wtparams.framme_seed = rand()%100000000;
            //-----------
            //cout << "best_line[0].x[0] = " << best_line[0].x[0] << endl;
            //cout << "best_line[1].x[0] = " << best_line[1].x[0] << endl;
            //cout << "good_line[1].x[0] = " << good_line[1].x[0] << endl;
            //cout << "good_line[50].x[0] = " << good_line[50].x[0] << endl;
            //cout << "good_line[150].x[0] = " << good_line[150].x[0] << endl;
            //cout << "good_line[250].x[0] = " << good_line[250].x[0] << endl;
            
            clock_t t_kernel2_start, t_kernel2_end;
            t_kernel2_start = clock();
        
            //cl_uint numMemGoodLineWeight = 7; 
            cl_mem memGoodLineWeight[7] = {0,0,0,0,0,0,0};
         
            //Creating Memory Objects
            if (!createMemGoodLineWeight(context, memGoodLineWeight, good_line, best_line, 
                  img_grad_array, allnoises, &wtparams, good_line_new, w_pf_new))
            {
                cerr << "Failed: create" <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            //Enqueuing the created memory objects.
            errNum = clEnqueueWriteBuffer( 
                            commandQueue, 
                            memGoodLineWeight[0], 
                            CL_TRUE, 
                            0,
                            NUM_GOOD_LINES*sizeof(struct lineStruct), 
                            (void *)good_line, 
                            0,
                            NULL, 
                            NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error: enqueuing good_line buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            errNum = clEnqueueWriteBuffer( 
                            commandQueue, 
                            memGoodLineWeight[1], 
                            CL_TRUE, 
                            0,
                            NUM_BEST_LINES*sizeof(struct lineStruct), 
                            (void *)best_line, 
                            0,
                            NULL, 
                            NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error: enqueuing best_line buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            errNum = clEnqueueWriteBuffer( 
                            commandQueue, 
                            memGoodLineWeight[2], 
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
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
           // //-----------
           // errNum = clEnqueueWriteBuffer( 
           //                 commandQueue, 
           //                 memGoodLineWeight[3], 
           //                 CL_TRUE, 
           //                 0,
           //       NUM_GOOD_LINES*sizeof(float), 
           //                 (void *)turnArray, 
           //                 0,
           //                 NULL, 
           //                 NULL);
           // 
           // if (errNum != CL_SUCCESS)
           // {
           //     cerr << "Error: enqueuing turnArray buffer." <<  endl;
           //     CleanupWtKernel(context_wt, commandQueue, program_wt, 
           //             kernel[1], memGoodLineWeight);          
           //     return -1;
           // }
           // //-----------
           // errNum = clEnqueueWriteBuffer( 
           //                 commandQueue, 
           //                 memGoodLineWeight[4], 
           //                 CL_TRUE, 
           //                 0,
           //       NUM_GOOD_LINES*sizeof(float), 
           //                 (void *)distArray, 
           //                 0,
           //                 NULL, 
           //                 NULL);
           // 
           // if (errNum != CL_SUCCESS)
           // {
           //     cerr << "Error: enqueuing distArray buffer." <<  endl;
           //     CleanupWtKernel(context_wt, commandQueue, program_wt, 
           //             kernel[1], memGoodLineWeight);          
           //     return -1;
           // }
           // //-----------
           // errNum = clEnqueueWriteBuffer( 
           //                 commandQueue, 
           //                 memGoodLineWeight[5], 
           //                 CL_TRUE, 
           //                 0,
           //       NUM_GOOD_LINES*sizeof(float), 
           //                 (void *)rngturnArray, 
           //                 0,
           //                 NULL, 
           //                 NULL);
           // 
           // if (errNum != CL_SUCCESS)
           // {
           //     cerr << "Error: enqueuing rngturnArray buffer." <<  endl;
           //     CleanupWtKernel(context_wt, commandQueue, program_wt, 
           //             kernel[1], memGoodLineWeight);          
           //     return -1;
           // }
           // //-----------
           // errNum = clEnqueueWriteBuffer( 
           //                 commandQueue, 
           //                 memGoodLineWeight[6], 
           //                 CL_TRUE, 
           //                 0,
           //       NUM_GOOD_LINES*sizeof(float), 
           //                 (void *)rngfwdArray, 
           //                 0,
           //                 NULL, 
           //                 NULL);
           // 
           // if (errNum != CL_SUCCESS)
           // {
           //     cerr << "Error: enqueuing rngfwdArray buffer." <<  endl;
           //     CleanupWtKernel(context_wt, commandQueue, program_wt, 
           //             kernel[1], memGoodLineWeight);          
           //     return -1;
           // }
            //-----------
            errNum = clEnqueueWriteBuffer( 
                            commandQueue, 
                            memGoodLineWeight[3], 
                            CL_TRUE, 
                            0,
                  sizeof(struct noiseStruct), 
                            (void *)allnoises, 
                            0,
                            NULL, 
                            NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error: enqueuing allnoises buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            errNum = clEnqueueWriteBuffer( 
                            commandQueue, 
                            memGoodLineWeight[4], 
                            CL_TRUE, 
                            0,
                  sizeof(struct wtKernParams), 
                            (void *)&wtparams, 
                            0,
                            NULL, 
                            NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error: enqueuing wtparams buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            //Setting the kernel arguments
            errNum  = clSetKernelArg(kernel[1], 0, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[0]);
            
            errNum |= clSetKernelArg(kernel[1], 1, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[1]);
            
            errNum |= clSetKernelArg(kernel[1], 2, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[2]);
            
            //errNum |= clSetKernelArg(kernel[1], 3, 
            //    sizeof(cl_mem),(void *)&memGoodLineWeight[3]);
            //
            //errNum |= clSetKernelArg(kernel[1], 4, 
            //    sizeof(cl_mem),(void *)&memGoodLineWeight[4]);
            //
            //errNum |= clSetKernelArg(kernel[1], 5, 
            //    sizeof(cl_mem),(void *)&memGoodLineWeight[5]);
            //
            //errNum |= clSetKernelArg(kernel[1], 6, 
            //    sizeof(cl_mem),(void *)&memGoodLineWeight[6]);
            
            errNum |= clSetKernelArg(kernel[1], 3, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[3]);
            
            errNum |= clSetKernelArg(kernel[1], 4, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[4]);
            
            errNum |= clSetKernelArg(kernel[1], 5, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[5]);
            
            errNum |= clSetKernelArg(kernel[1], 6, 
                sizeof(cl_mem),(void *)&memGoodLineWeight[6]);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error setting kernel arguments." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            // Queue the kernel up for execution across the array
            //size_t globalWorkSize_wt[1] = { NUM_GOOD_LINES };
            //size_t localWorkSize_wt[1] = { 50 };
         	
            size_t localWorkSize[1] = { 50 };
            size_t lws = localWorkSize[0];
         	
            size_t globalWorkSize[1] = { 0 };
         	
         	if((int)(NUM_GOOD_LINES/lws)*lws < NUM_GOOD_LINES)
         	{
         		globalWorkSize[0] = (lws * ((int)(NUM_GOOD_LINES/lws) + 1));
         	}
         	else
         	{
         		globalWorkSize[0] = NUM_GOOD_LINES;
         	}
           
            errNum = clEnqueueNDRangeKernel(commandQueue, kernel[1], 1, NULL,
                                            globalWorkSize, localWorkSize,
                                            0, NULL, NULL);
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error queuing weight kernel for execution." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //-----------
            clFinish(commandQueue);
            //-----------
            // Read the output buffer back to the Host
            //OUTPUT 1
            errNum = clEnqueueReadBuffer(
                           commandQueue, 
                           memGoodLineWeight[5],
                           CL_TRUE, 0, 
                           NUM_GOOD_LINES * sizeof(struct lineStruct), 
                           good_line_new, 
                           0, NULL, NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error reading good_line_new buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            //OUTPUT 2
            errNum = clEnqueueReadBuffer(
                           commandQueue, 
                           memGoodLineWeight[6],
                           CL_TRUE, 0, 
                           NUM_GOOD_LINES * sizeof(float),
                           w_pf_new, 
                           0, NULL, NULL);
            
            if (errNum != CL_SUCCESS)
            {
                cerr << "Error reading w_pf_new buffer." <<  endl;
                cleanupGenContext( context, commandQueue, program, kernel, numKernels );
                cleanupMemGoodLineWeight( memGoodLineWeight );
                return -1;
            }
            
            t_kernel2_end = clock();
            float diff_kernel2 ((float)t_kernel2_end-(float)t_kernel2_start);
            cout<<"Kernel 2 Time = " << diff_kernel2/CLOCKS_PER_SEC << "\n"<< endl;
        
            //-----------
            //ASSIGNING THE OUTPUT VARIABLES TO THE LOCAL VARIABLES
            for (int gl = 0; gl < NUM_GOOD_LINES; gl++)
            {
                good_line[gl] = good_line_new[gl];        
                w_pf[gl] = w_pf_new[gl];
            }
           // cout << "After particle filter: " << endl;
           // cout << "best_line[0].x[0] = " << best_line[0].x[0] << endl;
           // cout << "best_line[1].x[0] = " << best_line[1].x[0] << endl;
           // cout << "good_line[1].x[0] = " << good_line[1].x[0] << endl;
           // cout << "good_line[50].x[0] = " << good_line[50].x[0] << endl;
           // cout << "good_line[150].x[0] = " << good_line[150].x[0] << endl;
           // cout << "good_line[250].x[0] = " << good_line[250].x[0] << endl;
            //-----------
            clFinish(commandQueue);
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
            
            cout << "fhalf 1 = " << fhalf << endl;
            cout << "shalf 1 = " << shalf << endl;

            float w_pf_fhalf[fhalf];
            float w_pf_shalf[shalf];

            struct lineStruct good_line_fhalf[fhalf];
            struct lineStruct good_line_shalf[shalf];

            for(int gl=0; gl<NUM_GOOD_LINES; gl++)
            {
               if(good_line[gl].x[0] <= img_grad.cols/2)
               {
                  w_pf_fhalf[fidx] = w_pf[gl];
                  good_line_fhalf[fidx] = good_line[gl];
                  fidx++;
               }
               else 
               {
                  w_pf_shalf[sidx] = w_pf[gl];
                  good_line_shalf[sidx] = good_line[gl];
                  sidx++;
               }
            }
            
            //SELECTING THE LINES BASED ON WEIGHTS                
            //Normalizing the weights.
            float tWeight_fhalf = 0.0;
         
            for(int i=0; i < fhalf; i++)
            {  tWeight_fhalf = tWeight_fhalf + w_pf_fhalf[i]; }        
            for(int i=0; i < fhalf; i++)
            {  w_pf_fhalf[i] = w_pf_fhalf[i]/tWeight_fhalf; }
            
            //FHALF    
            srand(time(NULL));
            int index = rand() % fhalf;
            float beta1 = 0.0; 
            float beta = 0.0;
            float mw = 0.0; //max. value out of w[1...N]        
            struct lineStruct *good_line_fhalf1;
            good_line_fhalf1 = new struct lineStruct [fhalf];
            
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
            {  tWeight_shalf = tWeight_shalf + w_pf_shalf[i]; }         
            for(int i=0; i < shalf; i++)
            {  w_pf_shalf[i] = w_pf_shalf[i]/tWeight_shalf; }
                 
            cout << "fhalf = " << fhalf << endl;
            cout << "shalf = " << shalf << endl;
           
           if(shalf != 0)
            {
               srand(time(NULL));
               index = rand() % shalf;
               beta1 = 0.0; 
               beta = 0.0;
               mw = 0.0; //max. value out of w[1...N]        
               struct lineStruct *good_line_shalf1;
               good_line_shalf1 = new struct lineStruct [shalf];
               
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
                             ((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols)
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
                              ((float)BEST_LINE_DIST/100.0)*(float)img_grad.cols)         
                               break;
                          }//for 'kl'
                      }//if(count>0)
                  }//if good_dist_vect
               }//for ln
            }
            
            for(int bl=0; bl<NUM_BEST_LINES; bl++)
            {
               best_line[bl] = good_line[best_line_idx[bl]];
            }
            
            cout << "Best lines extracted successfully!" << endl;
         
       //  }//END topIndex Iteration loop 
         //-----------
         //CLEANING UP
         //clReleaseKernel(kernel[0]);
         //CleanupWtKernel(context_wt, commandQueue, program_wt, 
         //        kernel[1], memGoodLineWeight);
         
         total_v_frames++;
         
         cleanupMemGoodLineWeight( memGoodLineWeight );
         
         acl_aligned_free(allnoises);
         acl_aligned_free(wtparams_tmp);
         acl_aligned_free(good_line_new);
         acl_aligned_free(w_pf_new);
      }//END THE LOOP ELSE LOOP FROM SECOND FRAME OF THE VIDEO
     
     ////Plotting 
     std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines_g[NUM_GOOD_LINES];
     for (int i=0; i<NUM_GOOD_LINES; i++)
     {
        x1 = (int) (good_line[i].x[0]);
        y1 = (int) (good_line[i].y[0]);
        x2 = (int) (good_line[i].x[img_gray.rows-1]);
        y2 = (int) (good_line[i].y[img_gray.rows-1]);
        
        lines_g[i].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
        (std::pair<int, int>(x1+ROI_START_X,y1+ROI_START_Y), std::pair<int, int>(x2+ROI_START_X,y2+ROI_START_Y)));  
        
        std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_g;  
        for(it_g=lines_g[i].begin();it_g!=lines_g[i].end();it_g++)  
        {  
             cv::line(img_org_full, cv::Point(it_g->first.first, it_g->first.second), 
             cv::Point(it_g->second.first, it_g->second.second), cv::Scalar(255,0,0), 1, 8,0);  
        }
     }
     
     std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines[NUM_BEST_LINES];
     for (int i=0; i<NUM_BEST_LINES; i++)
     {
        if(total_v_frames%1 == 0)
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
          (std::pair<int, int>(x1+ROI_START_X,y1+ROI_START_Y), std::pair<int, int>(x2+ROI_START_X,y2+ROI_START_Y)));  
          
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
    //OLD std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > roi_lines[5];
    //OLD x21 = (int) (0);
    //OLD y21 = (int) (ROI_START_Y);
    //OLD x22 = x21;
    //OLD y22 = (int) (img_org_full.rows-1);
    //OLD roi_lines[0].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    //OLD x21 = (int) (0);
    //OLD y21 = (int) (img_org_full.rows-1);
    //OLD x22 = (int) img_grad.cols;
    //OLD y22 = y21;
    //OLD roi_lines[1].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    //OLD x21 = (int) img_grad.cols;
    //OLD y21 = (int) (img_org_full.rows-1);
    //OLD x22 = x21;
    //OLD y22 = (int) (ROI_START_Y);
    //OLD roi_lines[2].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    //OLD x21 = (int) img_grad.cols;
    //OLD y21 = (int) (ROI_START_Y);
    //OLD x22 = (int) (0);
    //OLD y22 = (int) y21;
    //OLD roi_lines[3].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    //OLD x21 = (int) img_grad.cols/2;
    //OLD y21 = (int) (ROI_START_Y);
    //OLD x22 = x21;
    //OLD y22 = (int) (img_org_full.rows-1);
    //OLD roi_lines[4].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
            
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > roi_lines[5];
    x21 = (int) (ROI_START_X);
    y21 = (int) (ROI_START_Y);
    x22 = x21;
    y22 = y21 + yROI;
    roi_lines[0].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = x22;
    y21 = y22;
    x22 = x21 + xROI;
    y22 = y21;
    roi_lines[1].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = x22;
    y21 = y22;
    x22 = x21;
    y22 = (int) (ROI_START_Y);
    roi_lines[2].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = x22;
    y21 = y22;
    x22 = (int) (ROI_START_X);
    y22 = y21;
    roi_lines[3].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    x21 = x22 + (int) xROI/2;
    y21 = y22;
    x22 = x21;
    y22 = (int) (ROI_START_Y) + yROI;
    roi_lines[4].push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x21,y21), std::pair<int, int>(x22,y22)));  
    
    std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator roi_it;  
    for(int i=0; i<5;i++)
    {
       for(roi_it=roi_lines[i].begin();roi_it!=roi_lines[i].end();roi_it++)  
       {  
            cv::line(img_org_full, cv::Point(roi_it->first.first, roi_it->first.second), cv::Point(roi_it->second.first, roi_it->second.second), cv::Scalar(0,255,0), 2, 8,0);  
       }
    }  
  
    //*************START: NEW TESTING FOR BETTER LANE VISUALS*****************
      int NUM_HLINES = 10;
      int H_NEIGHBORHOOD_BASE = 5;
      //int color_hline[] = {0,102,255};//ORANGE
      int color_hline[] = {204,0,51};//BLUE
      for(int bl=0; bl<NUM_BEST_LINES; bl++)
      {
         int max_intensity_hline = 0;
         for(int hline=0; hline <= NUM_HLINES; hline++)
          {
             //int H_NEIGHBORHOOD = 2 * (int)((float)H_NEIGHBORHOOD_BASE * ((float)hline/(float)H_NEIGHBORHOOD_BASE)) + 2;
             int hlidx = hline * (img_gray.rows/NUM_HLINES); 
             int H_NEIGHBORHOOD = 20;
             if( y1 >= 0.30 )
             {
               H_NEIGHBORHOOD = 30;
             }
             for(int i=-1*(H_NEIGHBORHOOD); i<H_NEIGHBORHOOD; i++)
             {
                x1 = (int) ((best_line[bl].x[hlidx]) + i);
                y1 = (int) (best_line[bl].y[hlidx]);
                
                if((int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X ) > max_intensity_hline)
                {
                   max_intensity_hline = (int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X );
                }
             }
          }
         for(int hline=0; hline <= NUM_HLINES; hline++)
          {
             int hlidx = hline * (img_gray.rows/NUM_HLINES); 
             //int H_NEIGHBORHOOD = 2 * (int)((float)H_NEIGHBORHOOD_BASE * ((float)hline/(float)H_NEIGHBORHOOD_BASE)) + 2;
             int H_NEIGHBORHOOD = 20;
             if( y1 >= 0.30 )
             {
               H_NEIGHBORHOOD = 30;
             }
      
             //DRAW HLINES std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > hline_visual;
             //DRAW HLINES int xh1=0, yh1=0, xh2=0, yh2=0;  
             //DRAW HLINES 
             //DRAW HLINES xh1 =(int) (best_line[bl].x[hlidx]) - H_NEIGHBORHOOD; 
             //DRAW HLINES yh1 =(int) (best_line[bl].y[hlidx]);
             //DRAW HLINES xh2 =(int) (best_line[bl].x[hlidx]) + H_NEIGHBORHOOD; 
             //DRAW HLINES yh2 = yh1;
             //DRAW HLINES 
             //DRAW HLINES hline_visual.push_back(std::pair< std::pair<int, int>, std::pair<int, int> >
             //DRAW HLINES (std::pair<int, int>( xh1+ROI_START_X, yh1+ROI_START_Y ), std::pair<int, int>( xh2+ROI_START_X, yh2+ROI_START_Y )));  
      
             //DRAW HLINES std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it_hline;  
             //DRAW HLINES for(it_hline=hline_visual.begin();it_hline!=hline_visual.end();it_hline++)  
             //DRAW HLINES {  
             //DRAW HLINES      cv::line(img_org_full, cv::Point(it_hline->first.first, it_hline->first.second), 
             //DRAW HLINES      cv::Point(it_hline->second.first, it_hline->second.second), cv::Scalar(77,153,255), 1, 8,0);  
             //DRAW HLINES }
             
             for(int i=-1*(H_NEIGHBORHOOD); i<H_NEIGHBORHOOD; i++)
             {
                x1 = (int) ((best_line[bl].x[hlidx]) + i);
                y1 = (int) (best_line[bl].y[hlidx]);
                
                //cout << "value of "<< i << ": " << (int)img_gray_full.at<uchar>( y1+ROI_START_Y, x1 ) << endl;
                
                //if( y1 >= 0.30 * img_gray.rows && (int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X ) > 0.75 * max_intensity_hline)//day
                if( y1 >= 0.30 * img_gray.rows && (int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X ) > 0.70 * max_intensity_hline)
                {
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[0]   = color_hline[0]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[1]   = color_hline[1]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[2]   = color_hline[2]; 
                   
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[0] = color_hline[0]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[1] = color_hline[1]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[2] = color_hline[2]; 
                }
                //else if( y1 < 0.30 * img_gray.rows && (int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X ) > 0.75 * max_intensity_hline)//day
                else if( y1 < 0.30 * img_gray.rows && (int)img_gray_full.at<uchar>( y1+ROI_START_Y , x1+ROI_START_X ) > 0.40 * max_intensity_hline)
                {
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[0]   = color_hline[0]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[1]   = color_hline[1]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y , x1+ROI_START_X )[2]   = color_hline[2]; 
                   
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[0] = color_hline[0]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[1] = color_hline[1]; 
                   img_org_full.at<cv::Vec3b>( y1+ROI_START_Y+1 , x1+ROI_START_X )[2] = color_hline[2]; 
                }
             }
          }
      }
    //***************END: NEW TESTING FOR BETTER LANE VISUALS*****************
    output_cap.write(img_org_full);
    //cv::imshow("Lane Detection", img_org_full);
    //cv::waitKey(1); 
  
    acl_aligned_free(cline_params_tmp);
    acl_aligned_free(img_grad_array);
   
    t_total_end = clock();
    float diff_total = ((float)t_total_end-(float)t_total_start);
    
    cout<<"Total clocks = " << diff_total << "\n"<< endl;
    total_clocks += diff_total; 
    cout << "________________________________________\n" << endl;

   //FILE* file_rt = fopen("runningtime.txt", "a"); 
   //char output_rt[255]; 
   //
   //sprintf(output_rt, "%10.9f\n", diff_total);
   //fputs(output_rt, file_rt); 
   //fclose(file_rt); 

}//END VIDEO FRAMES for loop.
  
    cout<<"CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"<< endl;
    cout<<"Total_clocks = " << total_clocks << "\n"<< endl;
    acl_aligned_free(good_line);
    acl_aligned_free(best_line);
capture.release();
//output_cap.release();  
return 0;
}

//HELPER FUNCTIONS FOR CREATING LINES
bool createMemCreateLine(cl_context context,
   cl_mem memCreateLine[3], struct createLineParams cline_params,
   int *img_grad_array, struct lineStruct *line_new,
   int imgrows, int imgcols, int NUM_LINES)
{
   memCreateLine[0] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(struct createLineParams), &cline_params, NULL);

   memCreateLine[1] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   imgrows*imgcols*sizeof(int), img_grad_array, NULL);

//   memCreateLine[2] = clCreateBuffer(context,
//   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//   NUM_LINES*sizeof(float), xadd, NULL);

   memCreateLine[2] = clCreateBuffer(context,
   CL_MEM_READ_WRITE,
   NUM_LINES*sizeof(struct lineStruct), NULL, NULL);

   if (memCreateLine[0] == NULL || memCreateLine[1] == NULL
   || memCreateLine[2] == NULL )
      {
          cerr << "Error creating memory objects." <<  endl;
          return false;
      }

   return true;
}

void cleanupMemCreateLine(cl_mem memCreateLine[3] )
{
   for(int i=0; i<3; i++)
   {
     if (memCreateLine[i] != 0)
            clReleaseMemObject(memCreateLine[i]);
   }
}

//HELPER FUNCTIONS FOR EVALUATING LIKELIHOOD
bool createMemGoodLineWeight(cl_context context, cl_mem memGoodLineWeight[7],
   struct lineStruct *good_line, struct lineStruct *best_line, int *img_grad_array, 
   struct noiseStruct *allnoises, struct wtKernParams *wtparams, 
   struct lineStruct *good_line_new, float *w_pf_new)
{
   memGoodLineWeight[0] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_GOOD_LINES*sizeof(struct lineStruct), good_line, NULL);

   memGoodLineWeight[1] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   wtparams->NUM_BEST_LINES*sizeof(struct lineStruct), best_line, NULL);
   
   memGoodLineWeight[2] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   (wtparams->imgrows)*(wtparams->imgcols)*sizeof(int), 
   img_grad_array, NULL);

   memGoodLineWeight[3] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(struct noiseStruct),allnoises, NULL);

   memGoodLineWeight[4] = clCreateBuffer(context,
   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   sizeof(struct wtKernParams),wtparams, NULL);

   memGoodLineWeight[5] = clCreateBuffer(context,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(struct lineStruct), NULL, NULL);

   memGoodLineWeight[6] = clCreateBuffer(context,
   CL_MEM_READ_WRITE,
   wtparams->NUM_GOOD_LINES * sizeof(float), NULL, NULL);
   
   if (memGoodLineWeight[0] == NULL || memGoodLineWeight[1] == NULL
    || memGoodLineWeight[2] == NULL || memGoodLineWeight[3] == NULL
    || memGoodLineWeight[4] == NULL || memGoodLineWeight[5] == NULL
    || memGoodLineWeight[6] == NULL) 
    {
        cerr << "Error creating memGoodLineWeight objects." <<  endl;
        return false;
    }
   
   return true;
}

void cleanupMemGoodLineWeight( cl_mem memGoodLineWeight[7] )
{
   for(int i=0; i<7; i++)
   {
     if (memGoodLineWeight[i] != 0)
            clReleaseMemObject(memGoodLineWeight[i]);
   }
}

void cleanupGenContext(cl_context context,
   cl_command_queue commandQueue,
   cl_program program, cl_kernel *kernel, cl_uint numKernels) 
{

   for(size_t Nkern=0; Nkern<numKernels; Nkern++)
   {
      if( kernel[Nkern] != 0 )  
         clReleaseKernel( kernel[Nkern] ); 
   }

   if (program != 0)
       clReleaseProgram(program);
   
   if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

   if (context != 0)
       clReleaseContext(context);
   
   //NOTE: from cl.hpp, cl_device_id need not be 
   //released as there is no such thing defined!
}
