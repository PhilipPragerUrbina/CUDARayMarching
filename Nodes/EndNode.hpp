//
// Created by Philip on 11/15/2022.
//

#pragma once
#include "Node.hpp"

/// SDF node for infinite repetition
class EndNode : public Node{
public:
    EndNode() : Node(1,0,"End"){

    }

    void drawNodeContents() override{
        ImNodes::BeginInputAttribute(getInputPinId(0));
        ImGui::TextUnformatted("SDF in");
        ImNodes::EndInputAttribute();
    }

    void compile(NodeCompiler *compiler) override{
        Node* sdf_in = getInput(0).connected_from;
        if(isInValidInput(sdf_in)){ //check if input exists
            return;
        }
        compiler->ret(sdf_in->getID());
        sdf_in->compile(compiler);
    }

};
