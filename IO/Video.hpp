//
// Created by Philip on 10/30/2022.
//

#pragma once

#include "Image.hpp"
#include <string>

/// Saves image sequence in m_folder
class Video : public Display{
private:
    Image* m_current_frame; //The image currently being written to
    int m_width, m_height; //dimensions
    std::string m_folder; //Folder to put the images in
    int m_frame_count= 0; //current m_frame_count count

    /// Initialize a new frame by creating the new image
    void newFrame(){
        m_current_frame = new Image(m_width, m_height , "./" + m_folder + "/" + m_folder + "_" + std::to_string(m_frame_count) + ".jpg", Image::JPG); //create jpg image in directory
        m_frame_count++;
    }

    /// Save the image and clean up
    void stopFrame(){
        m_current_frame->update();
        delete m_current_frame;
    }

public:

    /// Create a new image sequence
    /// @param folder A directory that already exists, to put images in
    /// @param width Width in pixels
    /// @param height Height in pixels
    /// @details Images are stored as a JPG, with the name (folder_name)_(frame count). Example: "render_2.jpg"
    /// @example How to convert sequence to video: $ ffmpeg -i test_video_%d.jpg -c:v libx264 -vf format=yuv420p output.mp4
    Video(const std::string& folder, int width, int height) : m_folder(folder) , m_width(width), m_height(height){
        newFrame(); //create first frame
    }

    /// Save the final frame
    ~Video(){
        stopFrame();
    }

    /// Proceed to next frame
    void update() override{
        stopFrame();
        newFrame();
    }

    //pass these methods to the frame
    void setPixel(int x, int y, const Vector3& rgb) override{
        m_current_frame->setPixel(x, y, rgb);
    }
    Vector3 getPixel(int x, int y) const override{
        return m_current_frame->getPixel(x, y);
    }

    int getWidth() const override{
        return m_width;
    }
    int getHeight() const override{
        return m_height;
    }

};

