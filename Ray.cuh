//
// Created by Philip on 11/7/2022.
//

#ifndef RAYMARCHER_RAY_CUH
#define RAYMARCHER_RAY_CUH
#include "Math/Vector3.cuh"

#define STEPS 100
#define MINDIST 0.000001

#include "SDF/SDF.cuh"

//a march-able ray
class Ray {
private:
    Vector3 origin,direction;
public:
    __device__ Ray(const Vector3& origin, const Vector3& direction): origin(origin), direction(direction){}

    __device__ double trace(const SDF* sdf) const {
        double totalDistance = 0.0;

        int steps;
        for (steps=0; steps < STEPS; steps++) {
            Vector3 p = origin + direction * totalDistance;
            double distance = sdf->getDist(p);
            totalDistance += distance;
            if (distance < MINDIST) break;
        }
        if(steps == STEPS){
            return 0;
        }

        return 1.0-double (steps)/double(STEPS);
    }

};


#endif //RAYMARCHER_RAY_CUH
