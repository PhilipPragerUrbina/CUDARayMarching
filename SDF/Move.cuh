//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "../SDF/SDF.cuh"

/// Move an SDF
class Move : public SDF{
private:
    const SDF* m_function;
    const Vector3 m_translate;
public:

    /// Create a sdf from another sdf that is moved
    /// @param function Function to repeat
    /// @param translate How to move it
    __device__ Move(const SDF *function, const Vector3& translate) : m_translate(translate), m_function(function){}

    __device__ double getDist(const Vector3& point) const override {
        return m_function->getDist(point - m_translate);
    }
};



