//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "../SDF/SDF.cuh"

/// A sphere
class Sphere : public SDF{
private:
    const double m_radius;
public:

    /// Create a new sphere SDF
    /// @param radius How big it should be
    __device__ Sphere(const double radius = 0.04) : m_radius(radius){}


    __device__ double getDist(const Vector3& point) const override{
        return point.length() - m_radius;
    }





};



