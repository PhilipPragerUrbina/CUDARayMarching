//
// Created by Philip on 11/14/2022.
//

#pragma once

/// An interface for a node in the node editor
class Node {

    /// set the ids and increment accordingly
    virtual void setIds(int& latest_node_id, int& latest_connection_id){

    }

    virtual void drawNode(){};
    virtual void drawConnections(){};
    //todo manage connections
    //todo figure out how to specify what connection to add to
    //todo inputs and outputs
    //todo value interface, that updates it's value and draws itself
    virtual void addConnection(Node* other,int connection_id,  int connection_id_other){};

};
