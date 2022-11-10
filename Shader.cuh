//
// Created by Philip on 11/6/2022.
//

#ifndef RAYMARCHER_SHADER_CUH
#define RAYMARCHER_SHADER_CUH


#include "IO/Display.hpp"
#include "Ray.cuh"
#include "Camera.cuh"

#include "SDF/MandleBulb.cuh"


__global__ void kernel(Vector3* image, Camera camera){
//get indexes
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = (y * camera.getWidth() + x) ;

    Ray r(camera.getRayOrigin(),camera.getRayDirection(x,y));
    SDF* sdf = new MandelBulb();
    double dist = r.trace(sdf);
    delete sdf;
    image[w] = Vector3(dist*255);
}

class Shader {
private:
    Display* display;
    Vector3* host_image_data;
    Vector3* device_image_data;

    void allocate(){
        size_t image_size = display->getHeight() * display->getWidth() * sizeof(Vector3);
        host_image_data = (Vector3*) malloc(image_size);
        cudaMalloc(&device_image_data, image_size);
    }
public:
   Shader(Display* output){
       display = output;
       allocate();
       std::cout << "Setup \n";
   }



   void run(int t){
        Camera camera(Vector3(0,(double)t * 0.04,-4), Vector3(0,0,1), 40,Vector3(0,1,0), display->getWidth(),display->getHeight()  );
        camera.setLookAt(Vector3());

       dim3 threadsperblock = dim3(8, 8);
       dim3 numblocks = dim3(display->getWidth() / threadsperblock.x, display->getHeight() / threadsperblock.y);
       std::cout << "Running \n";
       kernel<<<numblocks, threadsperblock>>>(device_image_data, camera);
       cudaDeviceSynchronize();
       size_t image_size = display->getHeight() * display->getWidth() * sizeof(Vector3);
       cudaMemcpy(host_image_data, device_image_data,image_size, cudaMemcpyDeviceToHost );
       std::cout << "Saving \n";
       update();
   }

   void update(){
       for (int x = 0; x < display->getWidth(); x++){

           for (int y = 0; y < display->getHeight(); y++){
               int w = (y * display->getWidth() + x) ;
                display->setPixel(x,y,host_image_data[w]);
           }
       }
       display->update();
    }



   ~Shader(){
        cudaFree(device_image_data);
        free(host_image_data);
   }
};


#endif //RAYMARCHER_SHADER_CUH
