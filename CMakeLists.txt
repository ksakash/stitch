cmake_minimum_required(VERSION 3.0.2)
project(stitch)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV 4 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(capture src/capture.cpp)
target_link_libraries(capture
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Eigen3_LIBS}
)

add_executable(stitch src/stitch.cpp)
target_link_libraries(stitch
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Eigen3_LIBS}
)

add_executable(stitch_no_gpu src/stitch_no_gpu.cpp)
target_link_libraries(stitch_no_gpu
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Eigen3_LIBS}
)
