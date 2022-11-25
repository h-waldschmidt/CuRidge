cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_library(ridge ridge.cu)

target_compile_features(ridge PUBLIC cxx_std_17)
set_target_properties(ridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
