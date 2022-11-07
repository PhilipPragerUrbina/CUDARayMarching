//
// Created by Philip on 10/30/2022.
//

#ifndef RAYMARCHERCPU_DISPLAY_HPP
#define RAYMARCHERCPU_DISPLAY_HPP

//use vector3 as rgb color
#include "../Math/Vector3.cuh"

//interface for displaying pixel data
class Display {
public:
    //get dimensions of screen
    virtual int getWidth() const{return 0;};
    virtual int getHeight() const{return 0;};

    //set a pixel color (0-255) range
    virtual void setPixel(int x, int y,Vector3 rgb){};

    //get a pixel color
    virtual Vector3 getPixel(int x,int y) const{return Vector3();};

    //update or save the representation of the data
    virtual void update(){};
};


#endif //RAYMARCHERCPU_DISPLAY_HPP
