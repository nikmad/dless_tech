cmake_minimum_required( VERSION 2.8 )
project( openclProgrammingGuide )

# for locating OpenCL package and including the directory
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )
find_package( OpenCL REQUIRED )
include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories( "${CMAKE_SOURCE_DIR}/khronos" )

SUBDIRS( src )




