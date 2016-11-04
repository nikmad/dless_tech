# (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>
cmake_minimum_required(VERSION 2.8)

INCLUDE(MacroApplication)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
MESSAGE( "${OPENCV_LIBS}" )

SET( OTHER_LIBS ${OPENCL_LIBRARIES} ${OpenCV_LIBS} )
APPLICATION( "${OTHER_LIBS}" )

file(COPY ${CMAKE_CURRENT_LIST_DIR}/createLineKernel.cl 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/particleplot.py 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/goodLineWeightKernel.cl 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/RNG.h 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

#ADD_SUBDIRECTORY(test)
