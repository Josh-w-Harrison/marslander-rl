cmake_minimum_required(VERSION 3.20)
project(lander_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(lander_bridge main.cpp lander.cpp)
target_include_directories(lander_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lander_bridge PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin")
