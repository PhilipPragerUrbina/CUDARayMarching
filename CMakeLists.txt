cmake_minimum_required(VERSION 3.22)
project(RayMarcher CUDA)

set(CMAKE_CUDA_STANDARD 17)
include_directories(include)
add_executable(RayMarcher main.cu)

set_target_properties(RayMarcher PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
