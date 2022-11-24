//
// Created by Philip on 11/15/2022.
//

#pragma once

#include "Nodes/SphereNode.hpp"
#include "Nodes/InfiniteRepeatNode.hpp"
#include "Nodes/MergeNode.hpp"
#include "Nodes/EndNode.hpp"
#include "Window.hpp"
#include "../Nodes/MoveNode.hpp"
#include "../Nodes/VectorNode.hpp"

//https://github.com/Nelarius/imnodes

///Displays and manages Nodes
class NodeView : public Window{
private:
    //Nodes and connections
    std::vector<Node*> m_nodes;
    std::vector<Node::Connection> m_connections;
    //assign unique connection ids
    int m_latest_connection;
    int m_end_node;

    /// Take inputs
    void update(){
        int start,end,start_node,end_node;
        if(ImNodes::IsLinkCreated(&start_node, &start,&end_node, &end)){ //Create new link
            Node* node_out = m_nodes[start_node];
            Node* node_in = m_nodes[end_node];
            createConnection(start, end, node_out, node_in);
        }

        int destroyed;
        if(ImNodes::IsLinkDestroyed(&destroyed)){ //destroy link
            Node::Connection empty_connection = m_connections[destroyed];
            empty_connection.connected_to = nullptr;
            empty_connection.connected_from = nullptr;
            m_connections[destroyed].connected_from->setOutput(empty_connection);
            m_connections[destroyed].connected_to->setInput(empty_connection);
            m_connections[destroyed] = empty_connection;
        }

        if(ImNodes::IsLinkDropped() ) { //Open new node menu
            ImGui::OpenPopup("New");
        }

        int id;
        if(ImNodes::IsNodeHovered(&id)){ //delete(hide) node
            if(ImGui::IsKeyDown(ImGuiKey_Delete)){
                m_nodes[id]->close();
            }
        }

        //New Node Menu
        if (ImGui::BeginPopupContextWindow("New")) {
            ImVec2 position = ImGui::GetMousePosOnOpeningCurrentPopup();
            if (ImGui::MenuItem("Infinite Repeat")) {
                addNode(new InfiniteRepeatNode,position);
            }
            if (ImGui::MenuItem("Sphere")) {
                addNode(new SphereNode,position);
            }
            if (ImGui::MenuItem("Merge")) {
                addNode(new MergeNode,position);
            }
            if (ImGui::MenuItem("Move")) {
                addNode(new MoveNode,position);
            }
            if (ImGui::MenuItem("Vector")) {
                addNode(new VectorNode,position);
            }

            ImGui::EndPopup();
        }
    }

    /// Create a connection between two nodes
    /// @param start_pin id of starting pin
    /// @param end_pin id of ending pin
    /// @param node_start id of starting node
    /// @param node_end id of ending node
    void createConnection(int start_pin, int end_pin, Node *node_start, Node *node_end) {
        //check type
        if(node_start->getOutputPinType(start_pin) != node_end->getInputPinType(end_pin)){
            return; //wrong types
        }

        //create connection
        Node::Connection connection;
        connection.this_pin_id = start_pin;
        connection.other_pin_id = end_pin;
        connection.connected_to = node_end;
        connection.id = m_connections.size();
        connection.connected_from = node_start;

        //set connection
        m_connections.push_back(connection);
        node_start->setOutput(connection);
        node_end->setInput(connection);
    }

    /// Add a node
    /// @param node The new node
    /// Nodes will be deleted on editor destruction
    void addNode(Node* node){
        node->setIds(m_nodes.size(), m_latest_connection);
        m_latest_connection+= node->getPinNumber();
        m_nodes.push_back(node);
    }

    /// Add a node at position
    /// @param node The new node
    /// @param pos The position of the m_nodes
    /// Nodes will be deleted on editor destruction
    void addNode(Node* node, ImVec2 pos){
        ImNodes::SetNodeScreenSpacePos(m_nodes.size(), pos); //set node position
        addNode(node);
    }
public:

    NodeView(){
        //editor must have an end node
        addNode(new EndNode);
        m_end_node = 0;

    }

    ///Create imnodes context
    void init() override{
        ImNodes::CreateContext();

    }

    /// Factory for adding nodes
    /// @param name The name of the node
    /// Will not add if a node with name not found
    void addNode(const std::string& name){
            if (name == "Infinite Repeat") {
                addNode(new InfiniteRepeatNode);
            }else if(name == "Sphere") {
                addNode(new SphereNode);
            }else if(name == "Merge"){
                addNode(new MergeNode);
            }
    }



    ///Draw the editor
    void draw() override{
        ImNodes::BeginNodeEditor();
        //settings
        ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
        ImGui::PushItemWidth(100);

        for (Node* node : m_nodes){ //draw m_nodes
          if(node->isActive()){
              node->drawNode();
              node->drawOutputs();
          }
        }

        ImNodes::PopAttributeFlag();
        ImGui::PopItemWidth();
        ImNodes::EndNodeEditor();

        update();   //Get input
    }

    std::string compile(){
        NodeCompiler compiler("getDistt");
        m_nodes[m_end_node]->compile(&compiler);


        compiler.finish();


        return compiler.getOutput();
    }

    ///Destroy imnodes context
    void cleanup() override{
        ImNodes::DestroyContext();
    }

    ///Delete m_nodes
    ~NodeView(){
        for (Node* node : m_nodes){
            delete node;
        }
    }

};
