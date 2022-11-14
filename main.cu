#include <iostream>
#include "GPU/Shader.cuh"
#include "IO/Image.hpp"
#include "IO/Video.hpp"

//https://github.com/thedmd/imgui-node-editor

#include <SDL2/SDL.h>
#include <SDL2/SDL_timer.h>
int main(int argc, char* argv[])
{

    // returns zero on success else non-zero
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        printf("error initializing SDL: %s\n", SDL_GetError());
    }
    SDL_Window* win = SDL_CreateWindow("GAME", // creates a window
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       1000, 1000, 0);

    // triggers the program that controls
    // your graphics hardware and sets flags
    Uint32 render_flags = SDL_RENDERER_ACCELERATED;

    // creates a renderer to render our images
    SDL_Renderer* rend = SDL_CreateRenderer(win, -1, render_flags);







    // let us control our image position
    // so that we can move it with our keyboard.
    SDL_Rect dest ;
    dest.x = 0;
    dest.y = 0;
    dest.w = 32;
    dest.h = 32;



    // controls animation loop
    int close = 0;

    // speed of box
    int speed = 300;

    // animation loop
    while (!close) {
        SDL_Event event;


        // Events management
        while (SDL_PollEvent(&event)) {
            switch (event.type) {

                case SDL_QUIT:
                    // handling of close button
                    close = 1;
                    break;

                case SDL_KEYDOWN:
                    // keyboard API for key pressed
                    switch (event.key.keysym.scancode) {
                        case SDL_SCANCODE_W:
                        case SDL_SCANCODE_UP:
                            dest.y -= speed / 30;
                            break;
                        case SDL_SCANCODE_A:
                        case SDL_SCANCODE_LEFT:
                            dest.x -= speed / 30;
                            break;
                        case SDL_SCANCODE_S:
                        case SDL_SCANCODE_DOWN:
                            dest.y += speed / 30;
                            break;
                        case SDL_SCANCODE_D:
                        case SDL_SCANCODE_RIGHT:
                            dest.x += speed / 30;
                            break;
                        default:
                            break;
                    }
            }
        }

        // right boundary
        if (dest.x + dest.w > 1000)
            dest.x = 1000 - dest.w;

        // left boundary
        if (dest.x < 0)
            dest.x = 0;

        // bottom boundary
        if (dest.y + dest.h > 1000)
            dest.y = 1000 - dest.h;

        // upper boundary
        if (dest.y < 0)
            dest.y = 0;
        SDL_SetRenderDrawColor(rend, 10,10,10,255);
        // clears the screen
        SDL_RenderClear(rend);
        SDL_SetRenderDrawColor(rend, 10,100,10,255);
        SDL_RenderDrawRect(rend,&dest);

        // triggers the double buffers
        // for multiple rendering
        SDL_RenderPresent(rend);

        // calculates to 60 fps
        SDL_Delay(1000 / 60);
    }



    // destroy renderer
    SDL_DestroyRenderer(rend);

    // destroy window
    SDL_DestroyWindow(win);

    // close SDL
    SDL_Quit();



    //Display* d = new Image(2000,2000, "jes.jpg", Image::JPG);
    Display* d = new Video( "test_video",1000,1000); //image sequence

    Shader s(d); //renderer

    Camera camera(Vector3(0,0,-4), Vector3(0,0,1), Vector3(0,1,0),40, d->getWidth(), d->getHeight()  ); //create camera

    for (int i = 0; i < 5; i++){
        //animate camera
        camera.setPosition(camera.getPosition() + Vector3(0.01, 0.04,0.01)); //move up
        camera.setLookAt(Vector3()); //update direction

        std::cout << i << " Frame \n"; //output frame count

        s.run(camera); //render
        s.update(); //save
    }
    return 0;
}