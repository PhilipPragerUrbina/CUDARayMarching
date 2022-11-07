//
// Created by Philip on 11/6/2022.
//

#ifndef RAYMARCHER_SHADER_CUH
#define RAYMARCHER_SHADER_CUH


#include "IO/Display.hpp"

__device__ double mandleBulbSdf(const Vector3& pos) {
    Vector3 z = pos;
    double dr = 1.0;
    double r = 0.0;
    for (int i = 0; i < 20 ; i++) {
        r = z.length();
        if (r>100) break;
#define Power 2
        // convert to polar coordinates
        double theta = acos(z.z()/r);
        double phi = atan2(z.y(),z.x());
        dr =  pow( r, Power-1.0)*Power*dr + 1.0;

        // scale and rotate the point
        double zr = pow( r,Power);
        theta = theta*Power;
        phi = phi*Power;

        // convert back to cartesian coordinates
        z = Vector3 (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta))*zr;
        z+=pos;
    }
    return 0.5*log(r)*r/dr;
}

__device__ double trace(const Vector3& from, const Vector3& direction) {
    double totalDistance = 0.0;
    const int STEPS = 100;
    int steps;
    for (steps=0; steps < STEPS; steps++) {
        Vector3 p = from + direction * totalDistance;
        double distance = mandleBulbSdf(p);
        totalDistance += distance;
        if (distance < 0.000001) break;
    }
    if(steps == STEPS){
        return 0;
    }

    return 1.0-double (steps)/double(STEPS);
}


__global__ void kernel(Vector3* image, int width){
//get indexes
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int w = (y * width + x) ;

    Vector3 pos((double )x/width*4 - 2.0,(double)y/width*4.0 - 2.0,-4);
    Vector3 dir(0,0,1);
    double dist = trace(pos,dir);

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



   void run(){
       dim3 threadsperblock = dim3(8, 8);
       dim3 numblocks = dim3(display->getWidth() / threadsperblock.x, display->getHeight() / threadsperblock.y);
       std::cout << "Running \n";
       kernel<<<numblocks, threadsperblock>>>(device_image_data, display->getWidth());
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
