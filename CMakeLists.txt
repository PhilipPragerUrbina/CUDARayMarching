#Create project
cmake_minimum_required(VERSION 3.22)
project(RayMarcher CUDA CXX) #CUDA for kernel pre-compilation, and C++ for host code
set(CMAKE_CUDA_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(SDL2_INCLUDE_DIR "include/SDL2")

# Support both 32 and 64 bit builds

set(SDL2_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_SOURCE_DIR}/lib/x64/SDL2main.lib")




#Get CUDA packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(SDL2 REQUIRED)

#find gui sources
file(GLOB IMGUI_SOURCE ${PROJECT_SOURCE_DIR}/include/imgui/*.cpp ${PROJECT_SOURCE_DIR}/include/imnodes/*.cpp)
#Create gui library
add_library("ImGui" STATIC ${IMGUI_SOURCE})



#Compile host code
include_directories(include include/imgui ${SDL2_INCLUDE_DIR} include/SDL2 include/imnodes)
add_executable(RayMarcher main.cu)
set_target_properties(RayMarcher PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#Add cuda driver api
target_link_libraries(RayMarcher PRIVATE CUDA::cudart CUDA::cuda_driver)
#add gui
target_link_libraries(RayMarcher PUBLIC ImGui)
target_link_libraries(RayMarcher PUBLIC ${SDL2_LIBRARY})

#Compile PTX code
cuda_compile_ptx(cuda_ptx_files GPU/Kernel.cu)
add_custom_target(ptx ALL DEPENDS ${cuda_ptx_files} GPU/Kernel.cu SOURCES GPU/Kernel.cu)
#Make sure both build when raymarcher is built
add_dependencies(RayMarcher ptx)
