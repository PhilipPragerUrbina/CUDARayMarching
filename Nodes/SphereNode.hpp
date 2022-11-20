//
// Created by Philip on 11/15/2022.
//

#pragma once
#include "Node.hpp"

/// SDF node for infinite repetition
class SphereNode : public Node{
private:
    float m_radius = 1;
public:
    SphereNode() : Node(0,1,"Sphere"){

    }

    void drawNodeContents() override{
        ImGui::InputFloat("Radius", &m_radius);
        ImNodes::BeginOutputAttribute(getOutputPinId(0));
        ImGui::TextUnformatted("SDF Out");
        ImNodes::EndOutputAttribute();
    }

    void compile(NodeCompiler *compiler) override{
        compiler->newNode(getID(), "Sphere");
        compiler->doubleParameter(m_radius);
        compiler->closeNode();
    }

};
