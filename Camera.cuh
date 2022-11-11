//
// Created by Philip on 11/7/2022.
//

#pragma once

#include "Math/Vector3.cuh"

/// A camera for generating rays in 3d space
class Camera {
private:
    Vector3 m_position,m_direction ; //camera in world space
    Vector3 m_up; //m_direction of up for the world
    double m_field_of_view_radians; //field of view in radians
    int m_width, m_height; //screen dimensions

    //Precomputed values
    Vector3 m_lower_left_corner; //corner of screen to offset by
    Vector3 m_horizontal; //rotated m_vertical m_direction of camera
    Vector3 m_vertical; //rotated m_horizontal m_direction of camera

    /// Update the precomputed values based on changed settings
    void update(){
        //calculate m_height and m_width of the viewport in the world
        //the m_height corresponds to the field of view
        double view_height = 2.0* tan(m_field_of_view_radians/2.0);
        //m_width corresponds to m_height based on aspect ratio
        double  view_width = (double)m_width / (double)m_height * view_height; //calculate aspect ratio
        //rotate and scale to get horizontal and vertical vectors
        Vector3 horizontal_direction=  m_up.cross(m_direction).normalized(); //rotate
        m_horizontal =horizontal_direction * view_width; //scale
        m_vertical = m_direction.cross(horizontal_direction) * view_height; //get perpendicular vector
        //get lowest corner
        m_lower_left_corner = m_position - m_horizontal/2.0 - m_vertical/2.0 - m_direction;
    }

public:
    /// Create a new camera, and calculate values
    /// @param position Where the camera is
    /// @param direction Where it is looking(normalized)
    /// @param degree_fov Field of view in degrees
    /// @param up Orientation
    /// @param width Width in pixels
    /// @param height Height in pixels
    Camera(const Vector3& position, const Vector3& direction,const Vector3& up, double degree_fov , int width,int height) : m_position(position), m_direction(direction), m_up(up), m_width(width), m_height(height){
        setFOVDegrees(degree_fov);
        //no need to update, setFOV already has
    }

    /// @return Where the ray should originate
    __device__ Vector3 getRayOrigin() const{
        return m_position;
    }

    /// Get the proper ray direction at a given position in screen space
    /// @param x X in pixels
    /// @param y Y in pixels
    /// @return Ray direction
    __device__ Vector3  getRayDirection(int x,int y) const{
        double w_x = ((double)x/(double)m_width); //convert to 0-1 values
        double w_y = ((double)y/(double)m_height);
        //calculate direction
        return  m_lower_left_corner +  m_horizontal * w_x  +  m_vertical * w_y - m_position;
    }

    /// @return The width of the camera in pixels
    __device__  inline int getWidth() const{
        return m_width;
    }
    /// @return The Height of the camera in pixels
    __device__  inline int getHeight() const{
        return m_height;
    }

    Vector3 getPosition() const {
        return m_position;
    }

    /// Set the direction and update
    /// @param direction Direction to look(Normalized)
    void setDirection(const Vector3& direction){
        m_direction = direction;
        update();
    }
    /// Change direction to look at a point and update
    /// @param look_at Where to look
    void setLookAt(const Vector3& look_at){
        m_direction = (m_position-look_at).normalized();
        update();
    }
    /// @param position
    /// @warning Call setLookAt again to keep looking at a point after moving
    void setPosition(const Vector3& position){
       m_position = position;
    }
    /// Set the FOV and update
    /// @param degrees FOV in degrees
    void setFOVDegrees(double degrees){
        const double PI = 3.1415; //close enough
        m_field_of_view_radians = ( degrees * PI ) / 180.0 ; //degrees to radians
        update();
    }

};


