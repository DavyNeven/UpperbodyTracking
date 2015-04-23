cmake_minimum_required(VERSION 2.8)

project( UpperBodyDetectionProject )

include(/usr/dlib-18.10/dlib/cmake)




find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(UpperBodyDetector UpperBodyDetector.cpp ObjectDetector.cpp LKT_Tracker.cpp)
target_link_libraries(UpperBodyDetector dlib ${OpenCV_LIBS} )
set_target_properties(UpperBodyDetector PROPERTIES COMPILE_FLAGS "-D_LINUX -D_REENTRANT -Wall  -O3 -march=nocona -msse3")
