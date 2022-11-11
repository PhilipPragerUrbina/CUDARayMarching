//
// Created by Philip on 10/30/2022.
//

#pragma once

#include "Display.hpp"

//uses the wonderful stb image write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

/// Save pixel m_data as an image, implements the m_display interface
class Image : public Display{
public:
    /// Enum for file types the class can write
    enum FileType{
        JPG,PNG,BMP,TGA, HDR
    };

    /// Create a new image
    /// @param width Width of image
    /// @param height Height of image
    /// @param filepath Where to overwrite or create image in(Directory must already exist)
    /// @param type JPG,PNG,TGA,HDR,BMP
    /// @param quality For JPG, the quality 0-100 to save the image as.
    Image(int width, int height, std::string filepath, FileType type = JPG, int quality = 100) : m_width(width), m_height(height), m_type(type), m_path(filepath) ,m_quality(quality){
        m_data = new double[width * height * NUM_CHANNELS];
    }

    /// Clean up the image m_data
    ~Image(){
        delete[] m_data;
    }


    /// Asserts bounds
    void setPixel(int x, int y, const Vector3& rgb) override{
            assert(inBounds(x,y)); //check if in bounds
            int index = NUM_CHANNELS * (x + y * m_width); //get index
            m_data[index + 0] = rgb.x(); //set values
            m_data[index + 1] = rgb.y();
            m_data[index + 2] = rgb.z();
    }

    /// Returns black if out of bounds
    Vector3 getPixel(int x, int y) const override {
        if(inBounds(x,y)) { //check the bounds
            int index = NUM_CHANNELS * (x + y * m_width); //get index
            return Vector3(m_data[index + 0],m_data[index + 1],m_data[index + 2]);
        }
        return Vector3(); //return black
    }


    int getWidth() const override{ return m_width;}
    int getHeight() const override { return m_height;}

    /// save the m_data to a file
    void update() override {
        //all except one conversion requires int m_data
        //8 bit = 256 color values
        uint8_t* int_data = new uint8_t [m_width * m_height * NUM_CHANNELS];
        //convert to int m_data
        for (int i = 0; i < m_width * m_height * NUM_CHANNELS; ++i) {
            int_data[i] = (int)m_data[i];
        }

        switch (m_type) {
            case JPG:
                stbi_write_jpg(m_path.c_str(),m_width,m_height,NUM_CHANNELS, int_data,m_quality);
                break;

            case PNG:
                stbi_write_png(m_path.c_str(),m_width,m_height,NUM_CHANNELS, int_data, m_width*NUM_CHANNELS);
                break;
            case TGA:
                stbi_write_tga(m_path.c_str(), m_width,m_height,NUM_CHANNELS, int_data);
                break;
            case BMP:
                stbi_write_bmp(m_path.c_str(), m_width,m_height,NUM_CHANNELS, int_data);
                break;
            case HDR:
                //convert m_data to float*
                float* float_data = new float [m_width * m_height * NUM_CHANNELS];
                for (int i = 0; i < m_width * m_height * NUM_CHANNELS; ++i) {
                    float_data[i] = (float)m_data[i];
                }
                stbi_write_hdr(m_path.c_str(),m_width,m_height,NUM_CHANNELS,float_data); //write hdr
                delete[] float_data;
                break;

        }
        delete[] int_data;

    }

    /// check if in image bounds
     inline bool inBounds(int x, int y) const{
        return x >= 0 && y >=0 && x < m_width && y < m_height;
    }


private:
    //settings
    int m_width;
    int m_height;
    std::string m_path;
    FileType m_type;
    int m_quality;

    //only supports 3 channels
    const int NUM_CHANNELS = 3;

    //Image m_data. Will be converted later to not lose any information.
    double* m_data;

};


