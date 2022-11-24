//
// Created by Philip on 11/15/2022.
//

#pragma once

#include "Node.hpp"

/// SDF node for translating
class MoveNode : public Node{
private:
    float m_translate[3] = {0,0,0}; //Scale for infinite repeat sdf
public:
    MoveNode() : Node(2,1,"Move"){
        setColor(255,0,0);
    }

    void drawNodeContents() override{


        ImNodes::BeginInputAttribute(getInputPinId(0));

        ImGui::TextUnformatted("SDF In");

        ImNodes::EndInputAttribute();

        //todo combine into one method for checking if something is connected for default values
        Node* vec_in = getInput(1).connected_from;
        if(isInValidInput(vec_in)) {
            ImGui::InputFloat3("Move", m_translate);
        }
        ImNodes::BeginInputAttribute(getInputPinId(1));

        ImGui::TextUnformatted("Move");

        ImNodes::EndInputAttribute();


        ImNodes::BeginOutputAttribute(getOutputPinId(0));
        ImGui::TextUnformatted("SDF Out");
        ImNodes::EndOutputAttribute();
    }

    PinType getRelativeInputPinType(int relative_id) const override{
        if(relative_id ==0){
            return SDF;
        }else{
            return VECTOR;
        }
    }

    void compile(NodeCompiler *compiler) override{
        Node* sdf_in = getInput(0).connected_from;
        if(isInValidInput(sdf_in)){ //check if input exists
            return;
        }


        compiler->newNode(getID(), "Move");
        compiler->refParameter(sdf_in->getID());
        compiler->comma();
        Node* vec_in = getInput(1).connected_from;
        if(isInValidInput(vec_in)){
            compiler->vectorParameter(m_translate[0],m_translate[1],m_translate[2]);
        }else{
            compiler->valParameter(vec_in->getID());
        }

        compiler->closeNode();
       sdf_in->compile(compiler);
        if(!isInValidInput(vec_in)){
            vec_in->compile(compiler);
        }

    }





};
