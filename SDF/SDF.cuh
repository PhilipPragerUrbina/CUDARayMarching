//
// Created by Philip on 11/7/2022.
//

#ifndef RAYMARCHER_SDF_CUH
#define RAYMARCHER_SDF_CUH

#include "../Math/Vector3.cuh"
class SDF {
public:
    __device__ virtual double getDist(Vector3 point) const{return -1;}


};


#endif //RAYMARCHER_SDF_CUH
