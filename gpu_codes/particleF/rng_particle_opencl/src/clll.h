#ifndef CLLL_H_INCLUDED
#define CLLL_H_INCLUDED
 
cl_context CreateContext();
cl_command_queue CreateCommandQueue(cl_context, cl_device_id *);
cl_program CreateProgram(cl_context, cl_device_id, const char*);

#endif
