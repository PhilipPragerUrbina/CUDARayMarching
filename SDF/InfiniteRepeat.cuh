//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "SDF.cuh"

/// Infinite repeat an SDF
class InfiniteRepeat : public SDF{
private:
    const SDF* m_function;
    const Vector3 m_spacing;
public:

    /// Create a sdf from another sdf that repeats infinitely
    /// @param function Function to repeat
    /// @param spacing How far apart to space the function
    __device__ InfiniteRepeat(const SDF *function, const Vector3& spacing) : m_spacing(spacing), m_function(function){}

    __device__ double getDist(const Vector3& point) const override {
        Vector3 q = (point + 0.5 * m_spacing).mod(m_spacing) - (0.5 * m_spacing);
        return m_function->getDist(q);
    }
};



