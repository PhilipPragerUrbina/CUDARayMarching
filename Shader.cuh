//
// Created by Philip on 11/6/2022.
//

#pragma once

#include "IO/Display.hpp"
#include "Ray.cuh"
#include "Camera.cuh"

#include "SDF/MandleBulb.cuh"
#include "SDF//InfiniteSphere.cuh"

/// The kernel itself
/// @details Not in a class since it is a kernel
__global__ void kernel(Vector3* image, Camera camera){
    //get indexes
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     int w = (y * camera.getWidth() + x) ;

    Ray r(camera.getRayOrigin(),camera.getRayDirection(x,y));//create ray using camera
    SDF* sdf = new InfiniteSphere(); //create any sdf
    double dist = r.trace(sdf, 0.001, 100);   //trace
    delete sdf; //remove sdf
    image[w] = Vector3(dist*255); //save color value
}

/// The GPU shader for ray marching
class Shader {
private:
    Display* m_display; //output

    Vector3* m_host_image_data;  //cpu image data
    Vector3* m_device_image_data; //gpu image data

    dim3 m_threads; //how many threads in each block
    dim3 m_blocks; //how many blocks
    size_t m_image_size; //how large the output image is

    /// Allocate image data
    void allocate(){
        m_image_size = m_display->getHeight() * m_display->getWidth() * sizeof(Vector3); //get image size
        m_host_image_data = (Vector3*) malloc(m_image_size); //allocate host
        cudaMalloc(&m_device_image_data, m_image_size); //allocate device
    }
public:

    /// Create a new ray marching shader
    /// @param output The display to output to
   Shader(Display* output) : m_display(output){
       allocate(); //allocate image memory based on display
       m_threads = dim3(8, 8); //How many threads in each block
       m_blocks = dim3(m_display->getWidth() / m_threads.x, m_display->getHeight() / m_threads.y); //Calculate how many blocks needed based on threads
   }

   /// Run the kernel and render
   /// @param camera What viewpoint to use
   void run(const Camera& camera){
       kernel<<<m_blocks, m_threads>>>(m_device_image_data, camera);   //render the image on the gpu
       cudaDeviceSynchronize(); //wait to finish

       cudaMemcpy(m_host_image_data, m_device_image_data, m_image_size, cudaMemcpyDeviceToHost ); //save the resulting image to the cpu
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
        cudaFree(m_device_image_data); //device
        free(m_host_image_data); //jost
   }
};



