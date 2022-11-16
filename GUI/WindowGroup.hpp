//
// Created by Philip on 11/15/2022.
//

#pragma once
#include <vector>
#include "Window.hpp"

///A window that renders multiple other window's within itself
class WindowGroup : public Window{
private:
    std::vector<Window*> m_contents; //Windows within
    std::string m_name; //name of group
public:

    /// Create a new group of windows
    /// @param contents The windows it contains. Will delete on destruction.
    /// @param name The name of the window
    WindowGroup(const std::vector<Window*>& contents, const std::string& name) : m_contents(contents), m_name(name){}

    ///Init the children
    void init() override{
        for (Window* window : m_contents){
            window->init();
        }
    }

    ///Draw the window and the children
    void draw() override{
        ImGui::Begin(m_name.c_str());
        //draw children
        for (Window* window : m_contents){
            window->draw();
        }
        ImGui::End();

    }

    /// Close children
    void cleanup() override{
        for (Window* window : m_contents){
            window->cleanup();
        }
    }

    /// Delete children
    ~WindowGroup(){
        for (Window* window : m_contents){
            delete window;
        }
    }

};
