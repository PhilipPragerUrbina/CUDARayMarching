//
// Created by Philip on 11/15/2022.
//

#pragma once

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_sdlrenderer.h"
#include "imnodes.h"
#include <stdio.h>
#include <SDL.h>
#include "WindowGroup.hpp"
#include "NodeView.hpp"
///The main gui application
class Application {
private:
    std::string m_name; //the name of the application
    int m_width,m_height; //the dimensions of the window

    //SDL
    SDL_Window* m_window = nullptr;
    SDL_Renderer* m_renderer = nullptr;

    bool m_successful_setup; //Did everything initialize correctly

    bool m_done = false; //Should main loop be ran

    const ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f); //Background color

    WindowGroup* m_group = nullptr; //Actual window contents

    /// Set up SDL, create the SDL window and renderer
    /// @return true for success
    bool initSDL(){
        //initialize the SDL library
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0){
            std::cerr <<  SDL_GetError() << "\n"; //error
            return false;
        }
        //Create the window
        SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI); //Imgui window flags
        SDL_Window* window = SDL_CreateWindow(m_name.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_width, m_height, window_flags);
        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);         //create renderer
        if(renderer == NULL) {
            std::cerr <<  SDL_GetError() << "\n"; //error
            return false;
        }
        m_window = window; //success
        m_renderer = renderer;
        return true;
    }

    /// Set up ImGUI context
    /// @return true for success
    void initImGui(){
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForSDLRenderer(m_window, m_renderer); //depends on SDL
        ImGui_ImplSDLRenderer_Init(m_renderer);
    }

    ///Poll events
    void poll(){
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            //basic window controls
            if (event.type == SDL_QUIT){ m_done = true;}
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window)){ m_done = true;}
        }
    }

public:
    /// ///Initialize contexts
    /// @param name The name of the application window
    /// @param width Width of window in pixels
    /// @param height Height of window in pixels
    /// @param contents What the application should contain. Will delete on destruction.
    Application(const std::string& name, int width, int height, WindowGroup* contents) : m_name(name), m_height(height),m_width(width), m_group(contents){
        m_successful_setup = initSDL();//initialize SDL
        if(m_successful_setup){
            initImGui(); //Initialize ImGui if successful
            m_group->init();
        }
    }



    /// Run the application loop
    void run(){
        if(!m_successful_setup){
            return;//nothing to run
        }

        bool done = false;
        while (!done) //main loop
        {

            poll();
            ImGui_ImplSDLRenderer_NewFrame(); //new frame
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            //draw stuff
            m_group->draw();

            //render frame
            ImGui::Render();
            SDL_SetRenderDrawColor(m_renderer, (Uint8)(clear_color.x * 255), (Uint8)(clear_color.y * 255), (Uint8)(clear_color.z * 255), (Uint8)(clear_color.w * 255));
            SDL_RenderClear(m_renderer);
            ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
            SDL_RenderPresent(m_renderer);
        }

    }




    ///Clean up contexts and windows
    ~Application(){
        if(!m_successful_setup){
            return;//nothing to clean up
        }
        m_group->cleanup();
        delete m_group;
        ImGui_ImplSDLRenderer_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        SDL_DestroyRenderer(m_renderer);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
    }



};
