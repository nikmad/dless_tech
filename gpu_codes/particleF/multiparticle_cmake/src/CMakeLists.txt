add_executable(coordinates particlefilter.cpp)

add_custom_command(
OUTPUT blahblah 
COMMAND coordinates
)

add_custom_target(run ALL
DEPENDS blahblah)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/particleplot.py 
      DESTINATION ${PROJECT_BINARY_DIR}/src)

