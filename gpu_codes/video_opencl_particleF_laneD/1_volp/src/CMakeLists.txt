# (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

cmake_minimum_required(VERSION 2.8)

INCLUDE(MacroApplication)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
MESSAGE( "${OPENCV_LIBS}" )

SET( OTHER_LIBS ${OPENCL_LIBRARIES} ${OpenCV_LIBS} )
APPLICATION( "${OTHER_LIBS}" )

#add_executable( acon laneD_particleF.cpp clll.cpp robot.cpp)
#target_link_libraries( acon ${OPENCL_LIBRARIES} ${OpenCV_LIBS})

#configure_file(createLineKernel.cl ${CMAKE_CURRENT_LIST_DIR}/createLineKernal.cl COPYONLY)
#configure_file(particleplot.py ${CMAKE_CURRENT_LIST_DIR}/particleplot.py COPYONLY)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/createLineKernel.cl 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/particleplot.py 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/goodLineWeightKernel.cl 
      DESTINATION ${PROJECT_BINARY_DIR}/bin)

#ADD_SUBDIRECTORY(test)
