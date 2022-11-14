//
// Created by Philip on 11/6/2022.
//

#pragma once
#include "GPUCompiler.cuh"
#include "../IO/Display.hpp"
#include "../Rays/Camera.cuh"


#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

/// The GPU shader for ray marching
class Shader {
private:
    Display* m_display; //output

    //Runtime compilation
    GPUCompiler m_compiler;
    CUmodule* m_module;

    Vector3* m_host_image_data;  //cpu image data
    CUdeviceptr m_device_image_data; //gpu image data

    dim3 m_threads; //how many threads in each block
    dim3 m_blocks; //how many blocks
    size_t m_image_size; //how large the output image is

    /// Allocate image data
    void allocate(){
        m_image_size = m_display->getHeight() * m_display->getWidth() * sizeof(Vector3); //get image size
        m_host_image_data = (Vector3*) malloc(m_image_size); //allocate host
        CUDA_SAFE_CALL(cuMemAlloc(&m_device_image_data, m_image_size)); //allocate device
    }
public:

    /// Create a new ray marching shader
    /// @param output The display to output to
   Shader(Display* output) : m_display(output){
        //compile at runtime
        //todo find file
        std::string  name = "cuda_compile_ptx_1_generated_Kernel.cu.ptx"; //filename
        m_compiler.loadPTXFile(name);
        m_module = m_compiler.compile();

       allocate(); //allocate image memory based on display
       m_threads = dim3(8, 8); //How many threads in each block
       m_blocks = dim3(m_display->getWidth() / m_threads.x, m_display->getHeight() / m_threads.y); //Calculate how many blocks needed based on threads
   }



   /// Run the kernel and render
   /// @param camera What viewpoint to use
   void run( Camera& camera){
        CUfunction k;
       CUDA_SAFE_CALL(cuModuleGetFunction(&k, *m_module, "_Z6kernelP7Vector36Camera"));

       void *args[] = { &m_device_image_data, &camera };

       CUDA_SAFE_CALL(cuLaunchKernel(k,m_blocks.x,m_blocks.y, m_blocks.z,m_threads.x, m_threads.y, m_threads.z, 0, NULL,args,0));



       cuCtxSynchronize();
       CUDA_SAFE_CALL(cuMemcpyDtoH(m_host_image_data, m_device_image_data, m_image_size));



   }

   ///save the data to the m_display
   void update(){
       for (int x = 0; x < m_display->getWidth(); x++){
           for (int y = 0; y < m_display->getHeight(); y++){
               int w = (y * m_display->getWidth() + x) ;
                m_display->setPixel(x, y, m_host_image_data[w]); //set the pixel
           }
       }
       m_display->update(); //update the m_display
    }

    ///Free memory
   ~Shader(){
        cuMemFree(m_device_image_data); //device
        free(m_host_image_data); //jost
   }
};



