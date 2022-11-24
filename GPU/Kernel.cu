//
// Created by Philip on 11/12/2022.
//

#pragma once
#include "../Rays/Ray.cuh"
#include "../Rays/Camera.cuh"

extern "C" __device__ double getDistt(Ray r);

/// The kernel itself
/// @details Not in a class since it is a kernel
__global__ void kernel(Vector3* image, Camera camera){

    //get indexes
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = (y * camera.getWidth() + x) ;

    Ray r(camera.getRayOrigin(),camera.getRayDirection(x,y));//create ray using camera
    double dist = getDistt(r);   //trace
    image[w] = Vector3(dist* 255); //save color value
}