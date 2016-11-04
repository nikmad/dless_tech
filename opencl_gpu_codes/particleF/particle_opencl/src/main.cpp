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

// typedefs and global variables

#define PI 3.141592653589793

typedef unsigned long int ULONG;

float fNoise = 0.05;
float tNoise = 0.05;
float sNoise = 5.0;

//const int ARRAY_SIZE = 1000;
ULONG NUM_PARTICLES = 10000;

Matrix<float, 4, 2, RowMajor> landmarks;
float world_size = 100.0;
ULONG N = NUM_PARTICLES;
int iterations = 40;

typedef struct{
	float world_size;
	int landmarks_rows;
	int landmarks_cols; 
} MoveKernelParams; 


//Prototypes
float eval(MatrixXf, Vector3f, float, ULONG); 

bool CreateMemObjectsSenseKernel(cl_context, cl_mem *, Robot *);

bool CreateMemObjectsMoveKernel(cl_context, cl_mem *, Robot *, 
float *, float *, float *, float *, float *, float *, MoveKernelParams *);

void CleanupSenseKernel(cl_context, cl_command_queue, cl_program, 
cl_kernel, cl_mem memObjectsSenseKernel[2]);
void CleanupMoveKernel(cl_context, cl_command_queue, cl_program, 
cl_kernel, cl_mem memObjectsMoveKernel[10]);

//_____________________________________________________
//
//		MAIN
//_____________________________________________________

int main(int argc, char** argv)
{
clock_t t1_clock, t2_clock;
t1_clock = clock();
    
cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_device_id device = 0;
cl_kernel kernel = 0;
cl_mem memObjectsSenseKernel[2] = {0, 0};
cl_int errNum;

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
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);
    return 1;
}

//_____________________________________________________
//
//	KERNEL1: SET NOISE	
//_____________________________________________________

// Create OpenCL program from HelloWorld.cl kernel source
program = CreateProgram(context, device, "pfkernel.cl");
if (program == NULL)
{
    CleanupSenseKernel(context, commandQueue, program,
    	 kernel, memObjectsSenseKernel);
    return 1;
}

// Create OpenCL kernel
kernel = clCreateKernel(program, "pfkernel", NULL);
if (kernel == NULL)
{
     cerr << "Failed to create setnoise kernel" <<  endl;
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);
    return 1;
}
//*********************************************
landmarks << 20,20,
80,80,
20,80,
80,20;
float errorInitial;
float error[iterations];
Robot particle[N], particle_new[N];
Robot myrobot;
cout << sizeof(myrobot) << endl;
float srandSeed[N];
VectorXf measurement(landmarks.rows());

float *measurementBuffer = NULL;
measurementBuffer = new float [landmarks.rows()];

float w[N]; //Particle weights
MatrixXf particleCord(N,3);
Vector3f myrobotCord;

//Initializing myrobot:	
init_rob(&myrobot, world_size, time(NULL));
//move_rob(&myrobot, 0.1, 1.0, world_size);
measurement << sense_rob(&myrobot, landmarks);

for(int i=0; i<N; i++) {
srandSeed[i] = (float)i; 
init_rob(&particle[i], world_size, srandSeed[i]);
}

if (!CreateMemObjectsSenseKernel(
    	context, memObjectsSenseKernel, particle))
{
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);
    return 1;
}

errNum = clEnqueueWriteBuffer( 
		commandQueue, 
		memObjectsSenseKernel[1], 
		CL_TRUE, 
		0,
		NUM_PARTICLES * sizeof(Robot), 
		(void *)particle, 
		0,
		NULL, 
		NULL);

if (errNum != CL_SUCCESS)
{
    cerr << "Error writing 'particle' to buffer." <<  endl;
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);          
    return 1;
}

errNum  = clSetKernelArg(kernel, 0, 
    sizeof(cl_mem),(void *)&memObjectsSenseKernel[0]);
errNum |= clSetKernelArg(kernel, 1, 
    sizeof(cl_mem),(void *)&memObjectsSenseKernel[1]);
errNum |= clSetKernelArg(kernel, 2, 
    sizeof(cl_float), (void *)&fNoise);
errNum |= clSetKernelArg(kernel, 3, 
    sizeof(cl_float), (void *)&tNoise);
errNum |= clSetKernelArg(kernel, 4, 
    sizeof(cl_float), (void *)&sNoise);

if (errNum != CL_SUCCESS)
{
    cerr << "Error setting kernel arguments." <<  endl;
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);    
    return 1;
}

size_t globalWorkSize[1] = { NUM_PARTICLES };
size_t localWorkSize[1] = { 1 };

// Queue the kernel up for execution across the array
errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                globalWorkSize, localWorkSize,
                                0, NULL, NULL);
if (errNum != CL_SUCCESS)
{
     cerr << "Error queuing kernel for execution." <<  endl;
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);
    return 1;
}

// Read the output buffer back to the Host
errNum = clEnqueueReadBuffer(commandQueue, memObjectsSenseKernel[0], 
    	CL_TRUE, 0, NUM_PARTICLES * sizeof(Robot), 
    	particle_new, 0, NULL, NULL);

if (errNum != CL_SUCCESS)
{
     cerr << "Error reading result buffer." <<  endl;
    CleanupSenseKernel(context, commandQueue, program, 
    	kernel, memObjectsSenseKernel);
    return 1;
}

for (int i = 0; i < N; i++)
{
    particle[i] = particle_new[i];	
}
cout << "Particles Initialized and Noise set succesfully." <<  endl;

FILE* file = fopen("particleCord.txt", "a"); 
char output[255]; 

for (int i=0; i<N; i++)
{
sprintf(output, "Iteration=1:\n"); 
sprintf(output, "%10.9f %10.9f %10.9f\n", 
	particle[i].xCord, particle[i].yCord, particle[i].orientation); 
fputs(output, file); 
}

fclose(file); 
//_____________________________________________________
//
//	CALCULATE DISTANCE FROM EACH PARTICLE TO ROBOT
//_____________________________________________________

for (int i=0; i<N; i++)
{
	particleCord.row(i) << particle[i].xCord, 
	particle[i].yCord, particle[i].orientation;
}

myrobotCord << myrobot.xCord, myrobot.yCord, myrobot.orientation; 

errorInitial = eval(particleCord, myrobotCord, world_size, N);	

for(int topIndex=0; topIndex<iterations; topIndex++) 
{

	//Moving the robot and all the particles
	move_rob(&myrobot, 0.5, 5.0, world_size);
	measurement << sense_rob(&myrobot, landmarks);
	float turn1[N],dist1[N], rngturn[N], rngfwd[N];
	for(int i=0; i<N; i++)
	{
	int c1 = rand()%10000;
	turn1[i] = ((float)c1/10000.0000) * 0.5;
	int c2 = rand()%10000;
	dist1[i] = ((float)c2/10000.0000)*3;
	
	RNG rng(time(NULL));
	rngturn[i] = rng.gaussian(particle->turnNoise);
	rngfwd[i] = rng.gaussian(particle->forwardNoise);		
	}	
	
	//_____________________________________________________
	//
	//	KERNEL2: MOVE		
	//_____________________________________________________
	
	cl_mem memObjectsMoveKernel[10] = {0,0,0,0,0,0,0,0,0};
	MoveKernelParams move_params;

	move_params.world_size = world_size;
	move_params.landmarks_rows = landmarks.rows();
	move_params.landmarks_cols = landmarks.cols();
	
	program = CreateProgram(context, device, "movekernel.cl");
	if (program == NULL)
	{
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);
	    return 1;
	}
	
	// Create OpenCL kernel
	kernel = clCreateKernel(program, "movekernel", NULL);
	if (kernel == NULL)
	{
	     cerr << "Failed to create move kernel" <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);
	    return 1;
	}
	
	if (!CreateMemObjectsMoveKernel(context, memObjectsMoveKernel, 
	    	particle,turn1, dist1, rngturn, rngfwd, 
	    	measurement.data(), landmarks.data(), &move_params))
	{
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);
	    return 1;
	}
	
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[0], 
			CL_TRUE, 
			0,
			NUM_PARTICLES * sizeof(Robot), 
			(void *)particle, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'particle' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[1], 
			CL_TRUE, 
			0,
			NUM_PARTICLES * sizeof(float), 
			(void *)turn1, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'turn1' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[2], 
			CL_TRUE, 
			0,
			NUM_PARTICLES * sizeof(float), 
			(void *)dist1, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'dist1' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[3], 
			CL_TRUE, 
			0,
			NUM_PARTICLES * sizeof(float), 
			(void *)rngturn, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'rngturn' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[4], 
			CL_TRUE, 
			0,
			NUM_PARTICLES * sizeof(float), 
			(void *)rngfwd, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'rngfwd' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[5], 
			CL_TRUE, 
			0,
			landmarks.rows() * sizeof(float), 
			(void *)measurement.data(), 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'measurement' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[6], 
			CL_TRUE, 
			0,
			landmarks.size() * sizeof(float), 
			(void *)landmarks.data(), 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'landmarks' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}
	errNum = clEnqueueWriteBuffer( 
			commandQueue, 
			memObjectsMoveKernel[7], 
			CL_TRUE, 
			0,
			sizeof(MoveKernelParams), 
			(void *)&move_params, 
			0,
			NULL, 
			NULL);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error writing 'move_params' to buffer." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);          
	    return 1;
	}

	//int landrows, landcols;
	//landrows = landmarks.rows();
	//landcols = landmarks.cols();
	    
	errNum  = clSetKernelArg(kernel, 0, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[0]);
	errNum |= clSetKernelArg(kernel, 1, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[1]);
	errNum |= clSetKernelArg(kernel, 2, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[2]);
	errNum |= clSetKernelArg(kernel, 3, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[3]);
	errNum |= clSetKernelArg(kernel, 4, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[4]);
	errNum |= clSetKernelArg(kernel, 5, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[5]);
	errNum |= clSetKernelArg(kernel, 6, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[6]);
	errNum |= clSetKernelArg(kernel, 7, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[7]);
	errNum |= clSetKernelArg(kernel, 8, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[8]);
	errNum |= clSetKernelArg(kernel, 9, 
	    sizeof(cl_mem),(void *)&memObjectsMoveKernel[9]);

//	errNum |= clSetKernelArg(kernel, 9, 
//	    sizeof(cl_float),(void *)&world_size);
//	errNum |= clSetKernelArg(kernel, 10, 
//	    sizeof(cl_int),(void *)&landrows);
//	errNum |= clSetKernelArg(kernel, 11, 
//	    sizeof(cl_int),(void *)&landcols);
	
	if (errNum != CL_SUCCESS)
	{
	    cerr << "Error setting kernel arguments." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);    
	    return 1;
	}
	
	size_t globalWorkSize[1] = { NUM_PARTICLES };
	size_t localWorkSize[1] = { 250 };
	
	//enqueue the kernel for execution
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
	                                globalWorkSize, localWorkSize,
	                                0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
	      cerr << "Error queuing kernel for execution." <<  endl;
	        CleanupMoveKernel(context, commandQueue, program, 
			kernel, memObjectsMoveKernel);
	        return 1;
	}

	//read buffer from device to host
	errNum = clEnqueueReadBuffer(commandQueue, 
	    		memObjectsMoveKernel[8], 
			CL_TRUE, 0, 
			NUM_PARTICLES * sizeof(Robot), 
			particle_new, 
			0, NULL, NULL);
	
	if (errNum != CL_SUCCESS)
	{
	     cerr << "Error reading result buffer 'particle_new'." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);
	    return 1;
	}

	
	errNum = clEnqueueReadBuffer(commandQueue, 
	    		memObjectsMoveKernel[9], 
			CL_TRUE, 0, 
			NUM_PARTICLES * sizeof(float), 
			w, 
			0, NULL, NULL);
	
	if (errNum != CL_SUCCESS)
	{
	     cerr << "Error reading result buffer 'w'." <<  endl;
	    CleanupMoveKernel(context, commandQueue, program, 
	    	kernel, memObjectsMoveKernel);
	    return 1;
	}
	
	
	for (int i = 0; i < N; i++)
	{
	    particle[i] = particle_new[i];	
	}
	//Normalizing the weights.
	float tWeight = 0.0;
	for(int i=0; i<N; i++)
	{
		tWeight = tWeight + w[i];
	}
	
	for(int i=0; i<N; i++)
	{
		w[i] = w[i]/tWeight;
	}
	
	srand(time(NULL));
	int index = rand() % N;
	int beta1; 
	float beta = 0.0;
	float mw = 0.0; //max. value out of w[1...N]	
	Robot particle1[N];
	
	//Evaluating the maximum weight
	for (int i=0; i<N; i++)
	{
		if(w[i] > mw)
		{
			mw = w[i];
		}	
	}
	
	//RESAMPLING STEP
	for (int i=0; i<N; i++)
	{
		beta1 = rand()%N;
		beta += ((float)beta1/N) * ( 2 * mw );
		while (beta > w[index])
		{
			beta -= w[index];
			index = (index + 1) % N;
		}
		particle1[i] = particle[index];
	}
	
	//Setting the noise and landmarks for the new particles
	for (int i=0; i<N; i++)
	{
		particle[i] = particle1[i];
	//	setNoise_rob(&particle[i], 0.05, 0.05, 5.0);
	}
	FILE* file1 = fopen("particleCord.txt", "a"); 
	char output1[255]; 
	for (int i=0; i<N; i++)
	{
		particleCord.row(i) << particle[i].xCord, 
		particle[i].yCord, particle[i].orientation;
	 	sprintf(output1, "%10.9f %10.9f %10.9f\n", 
			particle[i].xCord, particle[i].yCord, particle[i].orientation); 
		fputs(output1, file1); 
	}
	
	fclose(file1); 
	
	myrobotCord << myrobot.xCord, myrobot.yCord, 
		myrobot.orientation;
	FILE* file2 = fopen("myrobotCord.txt", "a"); 
	char output2[255]; 
	sprintf(output2, "%10.9f %10.9f %10.9f\n", 
		myrobot.xCord, myrobot.yCord, myrobot.orientation);
	fputs(output2, file2); 
	fclose(file2); 
	
	//Comparing the coordinates of each particle to 
	//those of robot		
	error[topIndex] = eval(particleCord, myrobotCord, 
			world_size, N); 
}

//Printing the coordinates of particles and robot.
for (int i=0; i<10; i++)
{
	printf("xCord = %10.9f | yCord = %10.9f | orientation = %10.9f \n", 		particleCord(i,0), particleCord(i,1), particleCord(i,2));
}
cout << "My robot:" << endl;
printf("xCord = %10.9f | yCord = %10.9f | orientation = %10.9f \n",
	myrobotCord(0), myrobotCord(1), myrobotCord(2));

  //Printing the error at each iteration.
  cout << errorInitial << endl;
    for (int i=0; i<iterations; i++)
    {
    cout << error[i] << endl;
    }	

t2_clock = clock();
float diff ((float)t2_clock-(float)t1_clock);
    
    cout<<"Total Running Time = " << diff/CLOCKS_PER_SEC << "\n"<< endl;
   // system ("read");
    return 0;
}
 

//_____________________________________________________

bool CreateMemObjectsSenseKernel(cl_context context, 
		cl_mem memObjectsSenseKernel[2], Robot *particle)
{
    memObjectsSenseKernel[0] = clCreateBuffer(context, 
	CL_MEM_READ_WRITE,sizeof(Robot) * NUM_PARTICLES, NULL, NULL);

    memObjectsSenseKernel[1] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(Robot) * NUM_PARTICLES, particle, NULL);

    if (memObjectsSenseKernel[0] == NULL || 
		memObjectsSenseKernel[1] == NULL)
    {
         cerr << "Error creating memory objects." <<  endl;
        return false;
    }

    return true;
}

bool CreateMemObjectsMoveKernel(cl_context context, 
	cl_mem memObjectsMoveKernel[10], Robot *particle, 
	float *turn1, float *dist1, float *rngturn, 
	float *rngfwd, float *measurement, 
	float *landmarks_ptr, MoveKernelParams *move_params)
{
    memObjectsMoveKernel[0] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(Robot) * NUM_PARTICLES, particle, NULL);
    memObjectsMoveKernel[1] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * NUM_PARTICLES, turn1, NULL);
    memObjectsMoveKernel[2] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * NUM_PARTICLES, dist1, NULL);
    memObjectsMoveKernel[3] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * NUM_PARTICLES, rngturn, NULL);
    memObjectsMoveKernel[4] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * NUM_PARTICLES, rngfwd, NULL);
    memObjectsMoveKernel[5] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * landmarks.rows(), measurement, NULL);
    memObjectsMoveKernel[6] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(float) * landmarks.size(), landmarks_ptr, NULL);
    memObjectsMoveKernel[7] = clCreateBuffer(context, 
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	sizeof(MoveKernelParams), move_params, NULL);
    memObjectsMoveKernel[8] = clCreateBuffer(context, 
	CL_MEM_READ_WRITE,
	sizeof(Robot) * NUM_PARTICLES, NULL, NULL);
    memObjectsMoveKernel[9] = clCreateBuffer(context, 
	CL_MEM_READ_WRITE,
	sizeof(float) * NUM_PARTICLES, NULL, NULL);

    if (memObjectsMoveKernel[0] == NULL || memObjectsMoveKernel[1] == NULL
 || memObjectsMoveKernel[2] == NULL || memObjectsMoveKernel[3] == NULL 
 || memObjectsMoveKernel[4] == NULL || memObjectsMoveKernel[5] == NULL
 || memObjectsMoveKernel[6] == NULL || memObjectsMoveKernel[7] == NULL
 || memObjectsMoveKernel[8] == NULL || memObjectsMoveKernel[9] == NULL)
    {
         cerr << "Error creating memory objects." <<  endl;
        return false;
    }

    return true;
}

//  CleanupMoveKernel any created OpenCL resources

void CleanupSenseKernel(cl_context context, 
	cl_command_queue commandQueue,
        cl_program program, cl_kernel kernel, 
	cl_mem memObjectsSenseKernel[2])
{
    for (int i = 0; i < 2; i++)
    {
        if (memObjectsSenseKernel[i] != 0)
            clReleaseMemObject(memObjectsSenseKernel[i]);
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

void CleanupMoveKernel(cl_context context, 
	cl_command_queue commandQueue,
        cl_program program, cl_kernel kernel, 
	cl_mem memObjectsMoveKernel[10])
{
    for (int i = 0; i < 10; i++)
    {
        if (memObjectsMoveKernel[i] != 0)
            clReleaseMemObject(memObjectsMoveKernel[i]);
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

float eval(MatrixXf particleCord, Vector3f myrobotCord, 
		float world_size, ULONG N) 
{
float sum = 0.0, dx, dy, err;
for (int i=0; i<N; i++)
{
	dx = cyclicWorld(particleCord(i,0)-myrobotCord(0)+
		world_size/2.0, world_size) - world_size/2.0;
	
	dy = cyclicWorld(particleCord(i,1)-myrobotCord(1)+
		world_size/2.0, world_size) - world_size/2.0;
	
	err = sqrt(dx*dx + dy*dy);
	sum += err;
}
return sum/(float)N;
}
