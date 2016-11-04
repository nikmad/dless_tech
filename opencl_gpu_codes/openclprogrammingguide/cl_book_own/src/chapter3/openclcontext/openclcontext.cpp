#include <iostream>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <malloc.h> // needed for alloca
#endif // _WIN32

#if defined(linux) || defined(__APPLE__) || defined(__MACOSX)
# include <alloca.h>
#endif // linux

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void displayInfo(void)
{
cl_uint numPlatforms; 
cl_platform_id * platformIDs; 
cl_context context = NULL; 
size_t size;
//cl_int errNum;

//Obtain platform IDs
clGetPlatformIDs(0, NULL, &numPlatforms);
platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
clGetPlatformIDs(numPlatforms, platformIDs, NULL);
std::cout << numPlatforms << std::endl;

//Create a context using up all the available paltforms
cl_context_properties properties[] = {
CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[0], 0 }; 
context = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);

//Obtain the device IDs of all devices within the above created context 
clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
cl_device_id * devices = (cl_device_id*)alloca( sizeof(cl_device_id) * size);
clGetContextInfo(context,CL_CONTEXT_DEVICES, size, devices, NULL);

//Print the available devices
for(size_t i=0; i < sizeof(cl_device_id); i++)
{
	cl_device_type type;
clGetDeviceInfo(
devices[i],CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
switch (type) {
case CL_DEVICE_TYPE_GPU:
	std::cout << "CL_DEVICE_TYPE_GPU" << std::endl;
	break;
case CL_DEVICE_TYPE_ACCELERATOR:
	std::cout << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
	break;
case CL_DEVICE_TYPE_CPU:
	std::cout << "CL_DEVICE_TYPE_CPU" << std::endl;
	break;
}
}
}

int main(int argc, char** argv)
{
    cl_context context = 0;

	displayInfo();

    return 0;
}
