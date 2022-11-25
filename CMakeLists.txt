cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

project(ridge_demo)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_subdirectory("src")
add_subdirectory("examples")
