cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_executable(second_example second_example.cpp)
target_link_libraries(second_example PRIVATE CUDA::cublas CUDA::cusolver)
target_link_libraries(second_example PRIVATE ridge)
