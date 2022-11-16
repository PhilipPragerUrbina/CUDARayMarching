//
// Created by Philip on 11/15/2022.
//

#pragma once
#include <vector>
#include "Window.hpp"
//https://github.com/Nelarius/imnodes
///Displays Nodes
class NodeView : public Window{

public:
    void init() override{
        ImNodes::CreateContext();
    }

    void draw() override{
        ImNodes::BeginNodeEditor();
        {
            ImNodes::BeginNode(0);
            ImNodes::BeginNodeTitleBar();
            ImGui::TextUnformatted("output node");
            ImNodes::EndNodeTitleBar();
            ImGui::Dummy(ImVec2(80.0f, 45.0f));

            const int output_attr_id = 2;
            ImNodes::BeginOutputAttribute(output_attr_id);
            float a = 0;
            ImGui::InputFloat("output", &a);
            ImNodes::EndOutputAttribute();
            ImNodes::EndNode();
        }

        {
            ImNodes::BeginNode(1);
            ImNodes::BeginNodeTitleBar();
            ImGui::TextUnformatted("output node");
            ImNodes::EndNodeTitleBar();
            ImGui::Dummy(ImVec2(80.0f, 45.0f));

            int output_attr_id = 1;



            ImNodes::BeginInputAttribute(output_attr_id);
            float a = 0;
            ImGui::InputFloat("output", &a);
            ImNodes::EndInputAttribute();
            ImNodes::EndNode();
        }

        ImNodes::Link(0, 1, 2);

        ImNodes::EndNodeEditor();
        int output_attr_id = 1;
        if(ImNodes::IsPinHovered(&output_attr_id)){

        }else{

        }

    }

    void cleanup() override{
        ImNodes::DestroyContext();
    }



};
