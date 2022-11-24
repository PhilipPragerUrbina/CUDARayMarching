//
// Created by Philip on 11/15/2022.
//

#pragma once
#include "Node.hpp"

/// Node for vector input
class VectorNode : public Node{
private:
    float m_vector[3] = {0,0,0};
public:
    VectorNode() : Node(0,1,"Vector"){

    }

    void drawNodeContents() override{
        ImGui::InputFloat3("Vec", m_vector);
        ImNodes::BeginOutputAttribute(getOutputPinId(0));
        ImGui::TextUnformatted("Vec");
        ImNodes::EndOutputAttribute();
    }

    void compile(NodeCompiler *compiler) override{
        compiler->newNode(getID(), "Vector3");
        compiler->doubleParameter(m_vector[0]);
        compiler->comma();
        compiler->doubleParameter(m_vector[1]);
        compiler->comma(); //todo auto comma
        compiler->doubleParameter(m_vector[2]);

        compiler->closeNode();
    }

    PinType getRelativeOutputPinType(int relative_id) const override {
        return VECTOR; //only one output
    }

};
