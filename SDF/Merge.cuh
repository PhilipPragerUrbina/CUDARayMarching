//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "../SDF/SDF.cuh"

/// Union of sdf
class Merge : public SDF{
private:
    const SDF* m_function_a;
    const SDF* m_function_b;
public:


    /// Create a union of 2 2df
    /// @param function_a
    /// @param function_b
    __device__ Merge(const SDF *function_a, const SDF *function_b) : m_function_a(function_a), m_function_b(function_b){}

    __device__ double getDist(const Vector3& point) const override {
        return min(m_function_a->getDist(point), m_function_b->getDist(point));
    }
};



