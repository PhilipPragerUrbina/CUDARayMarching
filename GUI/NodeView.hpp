//
// Created by Philip on 11/15/2022.
//

#pragma once

#include "Nodes/SphereNode.hpp"
#include "Nodes/InfiniteRepeatNode.hpp"
#include "Window.hpp"

//https://github.com/Nelarius/imnodes

///Displays Nodes
class NodeView : public Window{
private:
    //Nodes and connections
    std::vector<Node*> m_nodes;
    std::vector<Node::Connection> m_connections;
    //assign unique connection ids
    int m_latest_connection;
    int m_end_node = 0;

    /// Take inputs
    void update(){
        int start,end,start_node,end_node;
        if(ImNodes::IsLinkCreated(&start_node, &start,&end_node, &end)){ //Create new link
            Node* node_out = m_nodes[start_node];
            Node* node_in = m_nodes[end_node];
            Node::Connection connection;
            connection.this_pin_id = start;
            connection.other_pin_id = end;
            connection.connected_to = node_in;
            connection.id = m_connections.size();
            connection.connected_from = node_out;
            m_connections.push_back(connection);
            node_out->setOutput(connection);
            node_in->setInput(connection);
        }

        int destroyed;
        if(ImNodes::IsLinkDestroyed(&destroyed)){ //destroy link
            m_connections[destroyed].connected_to = nullptr;
            m_connections[destroyed].connected_from->setOutput(m_connections[destroyed]);
        }

        if(ImNodes::IsLinkDropped() ) { //Open new node menu
            ImGui::OpenPopup("New");
        }

        int id;
        if(ImNodes::IsNodeHovered(&id)){ //delete(hide) node
            if(ImGui::IsKeyDown(ImGuiKey_Backspace)){
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
            ImGui::EndPopup();
        }
    }
public:

    NodeView(){
        addNode(new EndNode);
    }

    ///Create imnodes context
    void init() override{
        ImNodes::CreateContext();

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
        NodeCompiler compiler("foo");
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
