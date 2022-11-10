//
// Created by Philip on 11/7/2022.
//

#ifndef RAYMARCHER_INFINITESPHERE_CUH
#define RAYMARCHER_INFINITESPHERE_CUH

#include "SDF.cuh"
class InfiniteSphere : public SDF{
public:

    __device__ double sdSphere( const Vector3& p, const Vector3& b ) const
    {
        return sin(p.length()) - b.x()  ;
    }
    __device__ double inline opRep( const Vector3& p, const Vector3& c , double s ) const
    {
        Vector3 q =(p+0.5*c).mod(c)-0.5*c;
        return sdSphere(q,vec3(s,s,s));
    }


    __device__ double getDist(vec3 p) const override{
        return opRep(p,vec3(1,1,1),0.04);
    }





};


#endif //RAYMARCHER_INFINITESPHERE_CUH
