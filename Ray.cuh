//
// Created by Philip on 11/7/2022.
//

#pragma once
#include "Math/Vector3.cuh"

#include "SDF/SDF.cuh"

///A ray that can be marched against an SDF
class Ray {
private:
    Vector3 m_origin,m_direction;
public:

    /// Create a ray
    /// @param origin Where it starts
    /// @param direction The direction it goes(normalized)
    __device__ Ray(const Vector3& origin, const Vector3& direction): m_origin(origin), m_direction(direction){}

    /// March the ray against a SDF
    /// @param sdf The signed distance field
    /// @param min_dist The minimum distance before stopping
    /// @param max_steps The maximum number of steps before stopping
    /// @return The total distance of the ray
    __device__ double trace(const SDF* sdf, const double min_dist = 0.000001, const int max_steps = 100) const {
        double total_distance = 0.0;
        for (int steps=0; steps < max_steps; steps++) {
            Vector3 point = m_origin + m_direction * total_distance; //march ray
            double distance = sdf->getDist(point); //get distance
            total_distance += distance;
            if (distance < min_dist) {return 1.0-double (steps)/double(max_steps); };//hit
        }
        return 0; //No hit
        //todo return distance(as documented) rather than steps coloring
    }

};


