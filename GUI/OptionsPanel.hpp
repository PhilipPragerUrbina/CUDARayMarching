//
// Created by Philip on 11/21/2022.
//

#pragma once

#include "../GPU/Shader.cpp"
#include "../IO/Image.hpp"
#include "../IO/Video.hpp"
#include "Window.hpp"
#include <iostream>

class OptionsPanel : public Window{
private:
    NodeView* m_view;

public:

    OptionsPanel( NodeView* view) : m_view(view){

    }

    void render(){


        std::string compiled = m_view->compile();
        std::cout << compiled << "\n";


        //Display* d = new Image(2000,2000, "jes.jpg", Image::JPG);
        Display* d = new Video( "test_video",1000,1000); //image sequence

        Shader s(d, compiled ); //renderer

        Camera camera(Vector3(0,0,-4), Vector3(0,0,1), Vector3(0,1,0),40, d->getWidth(), d->getHeight()  ); //create camera

        for (int i = 0; i < 5; i++){
            //animate camera
            camera.setPosition(camera.getPosition() + Vector3(0.01, 0.04,0.01)); //move up
            camera.setLookAt(Vector3()); //update direction

            std::cout << i << " Frame \n"; //output frame count

            s.run(camera); //render
            s.update(); //save
        }
    }

    void init() override{

    }

    void draw() override{

        if(ImGui::Button("Go")){
            render();
        }


    }

    void cleanup() override{

    }

};
