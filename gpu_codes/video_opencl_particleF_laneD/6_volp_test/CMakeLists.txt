# (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

PROJECT(opencl_tests CXX C)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#cmake_policy(SET CMP0012 NEW)

SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../1_dependencies/cmake" )
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# this searches for NVIDIA or ATI OpenCL
# TODO: propably we have to change this later on and find out how to use several OpenCL implementations concurrently
FIND_PACKAGE( OpenCL REQUIRED )
FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )

#TODO: probably, we might want to have a mini helper library for useful things?
INCLUDE_DIRECTORIES( "${CMAKE_CURRENT_SOURCE_DIR}/src/ocl_helpers" )
INCLUDE_DIRECTORIES( "${CMAKE_SOURCE_DIR}/../../1_dependencies/khronos" )

MESSAGE(${CMAKE_CURRENT_SOURCE_DIR})

ADD_SUBDIRECTORY( src )

