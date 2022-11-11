//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "SDF.cuh"

/// SDF for a mandelbulb
class MandelBulb : public SDF{
private:
    const int m_power;
    const int m_iterations;
    const int m_bailout;
public:
    /// Create a new mandelbulb SDF
    /// @param power Creates different shapes of mandelbulbs
    /// @param iterations How much detail to compute
    /// @param bailout When a point is considered not part of the object. Larger means points are less likely to incorrectly escape, but increases render time.
    __device__ __host__ MandelBulb(const int power = 2, const int iterations = 20, const int bailout = 100) : m_power(power), m_iterations(iterations), m_bailout(bailout) {}


    /// Distance estimator for the fractal
    ///@reference http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
    __device__ double getDist(const Vector3& point) const override{
            Vector3 z = point;
            double dr = 1.0;
            double r = 0.0;
            for (int i = 0; i < m_iterations ; i++) {
                r = z.length();
                if (r>m_bailout ){break;};
                // convert to polar coordinates
                double theta = acos(z.z()/r);
                double phi = atan2(z.y(),z.x());
                dr =  pow( r, m_power-1.0)*m_power*dr + 1.0;
                // scale and rotate the point
                double zr = pow( r,m_power);
                theta = theta*m_power;
                phi = phi*m_power;
                // convert back to cartesian coordinates
                z = Vector3 (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta))*zr;
                z+=point;
            }
            return 0.5*log(r)*r/dr;
    }


};
