//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "../Math/Vector3.cuh"

/// Interface for signed distance fields
class SDF {
public:
    /// Get the distance to a point using this SDF
    /// @param point The point to get the distance from
    /// @return The distance
    __device__ virtual double getDist(const Vector3& point) const{return -1;}
};
