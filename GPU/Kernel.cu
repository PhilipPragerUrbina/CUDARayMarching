//
// Created by Philip on 11/12/2022.
//

#pragma once
#include "../Rays/Ray.cuh"
#include "../Rays/Camera.cuh"

#include "../SDF/MandleBulb.cuh"
#include "../SDF/InfiniteRepeat.cuh"
#include "../SDF/Sphere.cuh"

/// The kernel itself
/// @details Not in a class since it is a kernel
__global__ void kernel(Vector3* image, Camera camera){

    //get indexes
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = (y * camera.getWidth() + x) ;

    Ray r(camera.getRayOrigin(),camera.getRayDirection(x,y));//create ray using camera
    Sphere sphere = Sphere();
    InfiniteRepeat repeat(&sphere,Vector3(1));
    SDF* sdf = &repeat; //create any sdf
    double dist = r.trace(sdf, 0.001, 100);   //trace
    image[w] = Vector3(dist*255); //save color value
}