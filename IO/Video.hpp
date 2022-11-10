//
// Created by Philip on 10/30/2022.
//

#ifndef RAYMARCHERCPU_VIDEO_HPP
#define RAYMARCHERCPU_VIDEO_HPP

#include "Image.hpp"
#include <string>
//saves image sequence in folder
class Video : public Display{
private:
    Image* current_frame;
    int width, height;
    std::string folder;
    int frame = 0;

    void newFrame(){
        current_frame = new Image( width, height ,"./" + folder + "/" + folder + "_"+ std::to_string(frame) + ".jpg", Image::JPG);
        frame++;
    }

    void stopFrame(){
        current_frame->update();
        delete current_frame;
    }

public:
    Video(std::string folder_name, int w, int h){
       folder = folder_name;
       width =w;
       height= h;
       newFrame();
    }

    void update() override{
        stopFrame();
        newFrame();
    }

    void setPixel(int x, int y, Vector3 rgb) override{
        current_frame->setPixel(x,y,rgb);
    }

    Vector3 getPixel(int x, int y) const override{
        return current_frame->getPixel(x,y);
    }

    int getWidth() const override{
        return width;
    }

    int getHeight() const override{
        return height;
    }


    ~Video(){
        stopFrame();
    }







};

#endif //RAYMARCHERCPU_VIDEO_HPP
