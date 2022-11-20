//
// Created by Philip on 11/15/2022.
//

#pragma once
#include "Node.hpp"

/// SDF node for infinite repetition
class MergeNode : public Node{
public:
    MergeNode() : Node(2,1,"Merge"){}

    void drawNodeContents() override{
        ImNodes::BeginInputAttribute(getInputPinId(0));
        ImGui::TextUnformatted("SDF In");
        ImNodes::EndInputAttribute();

        ImNodes::BeginInputAttribute(getInputPinId(1));
        ImGui::TextUnformatted("SDF In");
        ImNodes::EndInputAttribute();

        ImNodes::BeginOutputAttribute(getOutputPinId(0));
        ImGui::TextUnformatted("SDF Out");
        ImNodes::EndOutputAttribute();
    }

    void compile(NodeCompiler *compiler) override{
        Node* sdf_in = getInput(0).connected_from;
        Node* sdf_in_2 = getInput(1).connected_from;
        if(sdf_in == nullptr || !sdf_in->isActive()){ //check if input exists
            return;
        }
        if(sdf_in_2 == nullptr || !sdf_in_2->isActive()){ //check if input exists
            return;
        }
        compiler->newNode(getID(), "Merge");
        compiler->refParameter(sdf_in->getID());
        compiler->comma();
        compiler->refParameter(sdf_in_2->getID());
        compiler->closeNode();
        sdf_in->compile(compiler);
        sdf_in_2->compile(compiler);

    }
};
