//
// Created by Philip on 11/7/2022.
//

#ifndef RAYMARCHER_CAMERA_CUH
#define RAYMARCHER_CAMERA_CUH
#include "Math/Vector3.cuh"

//generate rays in 3d space
class Camera {
private:
    Vector3 m_position; //camera position
    Vector3 m_direction; //camera direction
    double m_aspect_ratio; //relation width and height of camera
    Vector3 m_up; //direction of up for the world
    //camera settings
    double m_field_of_view_radians; //field of view in radians

    //ray tracing things(going from screen space to world space like in ray tracing)
    Vector3 m_lower_left_corner; //corner of screen to offset by
    Vector3 m_horizontal; //rotated m_vertical direction of camera
    Vector3 m_vertical; //rotated m_horizontal direction of camera

    int width;
    int height;

    //update ray tracing direction variables
    void update(){
        //calculate height and width of the viewport in the world
        //the height corresponds to the field of view
        double view_height = 2.0* tan(m_field_of_view_radians/2.0);
        //width corresponds to height based on aspect ratio
        double  view_width = m_aspect_ratio * view_height;
        //rotate and scale to get horizontal and vertical vectors
        Vector3 horizontal_direction=  m_up.cross(m_direction).normalized(); //rotate
        m_horizontal =horizontal_direction * view_width; //scale
        m_vertical = m_direction.cross(horizontal_direction) * view_height; //get perpendicular vector
        //get lowest corner
        m_lower_left_corner = m_position - m_horizontal/2.0 - m_vertical/2.0 - m_direction;
    }


public:
    //create a camera at a position looking in a direction, with a certain size in world units, and a fov in degrees. +Z is up by default
    Camera(Vector3 position, Vector3 direction, double degree_fov, Vector3 up , int wwidth,int wheight){
        width =wwidth;
        height = wheight;

        //set values
        m_position = position;
        m_direction = direction.normalized();
        m_aspect_ratio = (double)width/(double)height;
        m_up = up;
        //update
        setFOVDegrees(degree_fov);
        //no need to update, setFOV already has
    }

    //get where rays should originate in this camera
    __device__ Vector3 getRayOrigin() const{
        return m_position;
    }

    __device__  inline int getWidth() const{
        return width;
    }

    //get the proper ray direction at a given position in screen space
    __device__ Vector3  getRayDirection(int w_x,int w_y) const{
        double x = ((double)w_x/(double)width);
        double y = ((double)w_y/(double)height);

        return  m_lower_left_corner +  m_horizontal * x  +  m_vertical * y - m_position;
    }

    //change direction to look at coordinate and update
    void setLookAt(Vector3 look_at){
        m_direction = (m_position-look_at).normalized();
        update();
    }

    //set the camera fov in degrees and update
    void setFOVDegrees(double degrees){
        m_field_of_view_radians = ( degrees * 3.1415 ) / 180.0 ; //degrees to radians
        update();
    }
    //set the camera fov in radians and update
    void setFOVRadians(double radians){
        m_field_of_view_radians = radians;
        update();
    }

};


#endif //RAYMARCHER_CAMERA_CUH
