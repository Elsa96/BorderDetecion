cmake_minimum_required(VERSION 3.8)
project(borderDetection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

#寻找OpenCV库
find_package(OpenCV)
#OpenCV库的包含文件
include_directories(${OpenCV_INCLUDE_DIRS})
#设置源文件（主文件）
set(SOURCE_FILES main.cpp)
add_executable(borderDetection ${SOURCE_FILES})
#设置动态链接库
target_link_libraries(borderDetection ${OpenCV_LIBS})