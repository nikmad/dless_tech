cmake_minimum_required(VERSION 2.8)
add_executable( acon main.cpp clll.cpp robot.cpp)
target_link_libraries( acon ${OPENCL_LIBRARIES} )

configure_file(pfkernel.cl ${CMAKE_CURRENT_BINARY_DIR}/pfkernel.cl COPYONLY)
configure_file(movekernel.cl ${CMAKE_CURRENT_BINARY_DIR}/movekernel.cl COPYONLY)
configure_file(rng_own.h ${CMAKE_CURRENT_BINARY_DIR}/rng_own.h COPYONLY)
configure_file(particleplot.py ${CMAKE_CURRENT_BINARY_DIR}/particleplot.py COPYONLY)


