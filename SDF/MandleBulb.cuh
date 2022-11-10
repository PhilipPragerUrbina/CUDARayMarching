//
// Created by Philip on 11/7/2022.
//

#ifndef RAYMARCHER_MANDLEBULB_CUH
#define RAYMARCHER_MANDLEBULB_CUH

#include "SDF.cuh"
class MandelBulb : public SDF{
public:
    __device__ double getDist(Vector3 point) const override{
            Vector3 z = point;
            double dr = 1.0;
            double r = 0.0;
            for (int i = 0; i < 20 ; i++) {
                r = z.length();
                if (r>100) break;
#define Power 2
                // convert to polar coordinates
                double theta = acos(z.z()/r);
                double phi = atan2(z.y(),z.x());
                dr =  pow( r, Power-1.0)*Power*dr + 1.0;

                // scale and rotate the point
                double zr = pow( r,Power);
                theta = theta*Power;
                phi = phi*Power;

                // convert back to cartesian coordinates
                z = Vector3 (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta))*zr;
                z+=point;
            }
            return 0.5*log(r)*r/dr;
    }



};


#endif //RAYMARCHER_MANDLEBULB_CUH
