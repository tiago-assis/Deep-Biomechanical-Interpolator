# Author: Laurent Chauvin

cmake_minimum_required(VERSION 3.0)

project(src_common)

file(GLOB SOURCE_FILES *.cpp)
add_library(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
