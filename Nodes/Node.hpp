//
// Created by Philip on 11/14/2022.
//

#pragma once
#include "imnodes.h"

/// A parent class for a node in the node editor
class Node {
public:
    /// Stores information about a connection from one node to another
    struct Connection{
        Node* connected_to = nullptr; //Other node
        Node* connected_from = nullptr; //this node
        int other_pin_id = 0; //Other pin
        int this_pin_id = 0;  //Which pin this connection is coming from
        int id; //Unique id of this connection
    };

    /// Set up the node parent class
    /// @param num_inputs
    /// @param num_outputs
    /// @param name
    Node(const int num_inputs, const int num_outputs, const std::string name) : NUM_INPUTS(num_inputs) , NUM_OUTPUTS(num_outputs) , NAME(name){ //set constants
        //create output array
        m_outputs = new Connection[NUM_OUTPUTS];
        m_inputs = new Connection[NUM_INPUTS];
    }
    /// Clean up the output array
    ~Node(){
        delete[] m_outputs;
        delete[] m_inputs;
    }

    /// Set an output to a connection
    /// @param connection The connection to set
    void setOutput(Connection connection){
        int relative_id = connection.this_pin_id - m_output_start; //get the location in array
        assert(relative_id < NUM_OUTPUTS); //Must be valid output
        m_outputs[relative_id] = connection;
    }

    /// Set an input to a connection
    /// @param connection The connection to set
    void setInput(Connection connection){
        int relative_id = connection.other_pin_id - m_input_start; //get the location in array
        assert(relative_id < NUM_INPUTS); //Must be valid output
        m_inputs[relative_id] = connection;
    }

    /// Get the number of pins the node has
    /// @return Total # of pins
    int getPinNumber() const {
        return NUM_OUTPUTS +NUM_INPUTS;
    }

    /// Set the m_nodes unique ids
    /// @param latest_node Latest available id for m_nodes
    /// @param latest_pin Latest available id for pins
    /// Will use the first pin id as a starting point, will increment for each following pin
    ///@warning Make sure to increment your latest pin by getPinNumber(), not by 1 after calling this
    void setIds(int latest_node, int latest_pin) {
        m_node_id = latest_node;
        m_input_start = latest_pin;
        m_output_start = latest_pin+NUM_INPUTS; //inputs after outputs
    }

    /// Set the node's title bar color
    /// @param r 0-255
    /// @param g 0-255
    /// @param b 0-255
    void setColor(int r, int g, int b){
        m_color = IM_COL32(r, g, b, 255);
    }

    /// Draw the node itself and call drawNodeContents
    void drawNode() {
        if(m_color !=0){ImNodes::PushColorStyle(ImNodesCol_TitleBar, m_color);} //set color

        ImNodes::BeginNode(m_node_id);

        ImNodes::BeginNodeTitleBar(); //title
        ImGui::TextUnformatted(NAME.c_str());
        ImNodes::EndNodeTitleBar();

        drawNodeContents();

        ImNodes::EndNode();

        if(m_color !=0){ImNodes::PopColorStyle();}


    }

    /// Draw the node's output links
    void drawOutputs() {
        for (int i = 0; i < NUM_OUTPUTS; i++){
            Connection connection = m_outputs[i];
            if(connection.connected_to != nullptr){ //is valid connection
                ImNodes::Link(connection.id ,connection.this_pin_id, connection.other_pin_id); //draw
            }
        }
    }

    /// Check if deleted
    /// @return False if deleted
    bool isActive() const{
        return !m_deleted;
    }

    /// Get node id
    /// @return Node id
    int getID(){
        return m_node_id;
    }

    ///Disable the node
    void close() {
        m_deleted = true;
        for (int i = 0; i < NUM_OUTPUTS; i++){
            m_outputs[i].connected_to = nullptr; //disable output links
        }
    }

    /// Compile the node into cuda code
    /// @param compiler The compiler to call to add instructions
   /// @return Is valid
    virtual void compile(NodeCompiler* compiler){

    }




private:
    int m_node_id = 0; //the id of the node itself, corresponds to its location in the node editor array
    int m_input_start = 0; //the id of it's first input pin
    int m_output_start = 0; //the id of it's first output pin
    bool m_deleted = false; //if the node is active or not
    Connection* m_outputs; //pointer to array of outgoing m_connections
    Connection* m_inputs; //pointer to array of outgoing m_connections
    const int NUM_OUTPUTS;
    const int NUM_INPUTS;
    const std::string NAME;
    int m_color = 0;

protected:

    /// Get the id to use for an output pin
    /// @param num_id Which output is it. For example, the first output is 0, the second is 1, etc.
    /// @return The id to use
    int getOutputPinId(int num_id){
        return m_output_start + num_id;
    }
    /// Get the id to use for an input pin
    /// @param num_id Which input is it.  For example, the first input is 0, the second is 1, etc.
    /// @return The id to use
    int getInputPinId(int num_id){
        return m_input_start + num_id;
    }

    /// Get the input connection
    /// @param i array index
    /// @return
    Connection getInput(int i) const{
        return m_inputs[i];
    }

    /// Draw node inputs, outputs, parameters, etc.
    virtual void drawNodeContents(){}


};

