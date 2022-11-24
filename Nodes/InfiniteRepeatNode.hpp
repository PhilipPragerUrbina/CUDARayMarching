//
// Created by Philip on 11/15/2022.
//

#pragma once

#include "Node.hpp"

/// SDF node for infinite repetition
class InfiniteRepeatNode : public Node{
private:
    float m_scale[3] = {1,1,1}; //Scale for infinite repeat sdf
public:
    InfiniteRepeatNode() : Node(1,1,"Infinite Repeat"){
        setColor(255,0,0);
    }

    void drawNodeContents() override{
        ImGui::InputFloat3("Scale", m_scale);

        ImNodes::BeginInputAttribute(getInputPinId(0));

        ImGui::TextUnformatted("SDF In");

        ImNodes::EndInputAttribute();

        ImNodes::BeginOutputAttribute(getOutputPinId(0));
        ImGui::TextUnformatted("SDF Out");
        ImNodes::EndOutputAttribute();
    }

    void compile(NodeCompiler *compiler) override{
        Node* sdf_in = getInput(0).connected_from;
        if(isInValidInput(sdf_in)){ //check if input exists
            return;
        }
        compiler->newNode(getID(), "InfiniteRepeat");
        compiler->refParameter(sdf_in->getID());
        compiler->comma();
        compiler->vectorParameter(m_scale[0],m_scale[1],m_scale[2]);
        compiler->closeNode();
       sdf_in->compile(compiler);

    }





};
