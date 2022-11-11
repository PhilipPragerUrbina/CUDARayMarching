//
// Created by Philip on 10/30/2022.
//

#pragma once

//use vector3 as rgb color
#include "../Math/Vector3.cuh"

/// Interface for displaying pixel m_data
/// @example Display m_data in gui, or save as image, or save as image sequence
class Display {
public:
    /// @return Width(X) of screen in pixels
    virtual int getWidth() const{return 0;};
    /// @return Height(Y) of screen in pixels
    virtual int getHeight() const{return 0;};

    /// Set a pixel color in the m_data
    /// @param x X coordinate
    /// @param y Y coordinate
    /// @param rgb  (0-255) range rgb
    virtual void setPixel(int x, int y,const Vector3& rgb){};

    /// Get a pixel color that was set previously
    /// @param x X coordinate
    /// @param y Y coordinate
    /// @return RGB color in range 0-255
    virtual Vector3 getPixel(int x,int y) const{return {};};

    /// update or save the representation of the m_data
    virtual void update(){};
};


