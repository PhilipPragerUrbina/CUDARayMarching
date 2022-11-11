//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "SDF.cuh"

/// A grid of infinite spheres
// todo separate repeat and sphere, as well as clean up
class InfiniteSphere : public SDF{
public:



    __device__ static double sdSphere( const Vector3& p, double s )
    {
        return p.length() - s  ;
    }
    __device__ double inline opRep( const Vector3& p, const Vector3& c  ) const
    {
        Vector3 q =(p+0.5*c).mod(c)-0.5*c;
        return sdSphere(q,0.04);
    }


    __device__ double getDist(const vec3& p) const override{
        return opRep(p,vec3(1,1,1));
    }





};



