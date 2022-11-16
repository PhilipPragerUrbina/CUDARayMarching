//
// Created by Philip on 11/15/2022.
//

#pragma once

///Interface for GUI windows for different purposes
class Window {
public:
    ///Do set up for the window
    virtual void init(){};

    ///draw a window frame
    virtual void draw(){};

    ///Do clean up for the window
    virtual void cleanup(){};

};
