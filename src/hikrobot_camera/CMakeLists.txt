cmake_minimum_required(VERSION 2.8.3)

project(hikrobot_camera)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
  camera_info_manager
)

catkin_package(
 
)

find_package(OpenCV 4.2.0 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
  /opt/MVS/include/
)

LINK_DIRECTORIES(
/opt/MVS/lib/64/
)

add_executable(hikrobot_camera src/hikrobot_camera.cpp)
target_link_libraries(
    hikrobot_camera
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
#    GCBase_gcc421_v3_0
   MvCameraControl
#    GenApi_gcc421_v3_0
   MVGigEVisionSDK
#    log4cpp_gcc421_v3_0
   MVRender
#    Log_gcc421_v3_0
   MvUsb3vTL
#    MathParser_gcc421_v3_0
#    NodeMapData_gcc421_v3_0
   MediaProcess
#    XmlParser_gcc421_v3_0
   X11
 )
add_executable(hikrobot_subcamera src/hikrobot_subcamera.cpp)
 target_link_libraries(
    hikrobot_subcamera
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
#    GCBase_gcc421_v3_0
   MvCameraControl
#    GenApi_gcc421_v3_0
   MVGigEVisionSDK
#    log4cpp_gcc421_v3_0
   MVRender
#    Log_gcc421_v3_0
   MvUsb3vTL
#    MathParser_gcc421_v3_0
#    NodeMapData_gcc421_v3_0
   MediaProcess
#    XmlParser_gcc421_v3_0
   X11
 )

add_executable(hikrobot_short src/hikrobot_short.cpp)
target_link_libraries(
hikrobot_short
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
#    GCBase_gcc421_v3_0
MvCameraControl
#    GenApi_gcc421_v3_0
MVGigEVisionSDK
#    log4cpp_gcc421_v3_0
MVRender
#    Log_gcc421_v3_0
MvUsb3vTL
#    MathParser_gcc421_v3_0
#    NodeMapData_gcc421_v3_0
MediaProcess
#    XmlParser_gcc421_v3_0
X11
)
