# This is an example project to show and test the usage of the FindOpenCL
# script.

cmake_minimum_required( VERSION 2.6 )
project( STRUCTOCL )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../1_dependencies/cmake")

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )
include_directories( "${CMAKE_SOURCE_DIR}/../../1_dependencies/khronos" )

#ADD_DEFINITIONS(-std=c++0x)

SUBDIRS(src)

