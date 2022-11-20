//
// Created by Philip on 11/19/2022.
//

#pragma once
#include <map>
///Generate C++ code for nodes
class NodeCompiler {
private:
    std::string m_code = "";
    std::vector<std::string> m_lines;
    std::string m_latest_line = "";
    std::map<std::string,std::string> m_includes;
    std::string m_method_name;
public:
    /// Generate a c++ method
    /// @param method_name The name of the method
    NodeCompiler(std::string method_name) : m_method_name(method_name){

    }

    void newNode(int id, std::string type_name){
        m_includes[type_name] = "#include \" " + type_name + ".cuh\" \n";
        m_latest_line += type_name + " node_" + std::to_string(id) + "(";
    }

    void vectorParameter(float x,float y,float z){
        m_latest_line += "Vector3(" + std::to_string(x) +" , " + std::to_string(y) + " , " + std::to_string(z) + ")";
    }

    void comma(){
        m_latest_line+=" , ";
    }

    void doubleParameter(float x){
        m_latest_line += std::to_string(x);
    }

    void refParameter(int id){
        m_latest_line += "&node_" + std::to_string(id);
    }

    void closeNode(){
        m_latest_line += "); \n";
        m_lines.push_back(m_latest_line);
        m_latest_line = "";
    }

    void ret(int id){
        m_latest_line+= "return r.trace(&node_"+std::to_string(id)+");";
        m_lines.push_back(m_latest_line);
        m_latest_line = "";
    }

    ///Finalize the compilation
    void finish(){

        for (auto const& include : m_includes)
        {
           m_code += include.second;
        }
        m_code += "double " + m_method_name + "(Ray r) { \n";
        for (int i = m_lines.size()-1; i >= 0; i--){
            //reverse order
            m_code+= m_lines[i];
        }

        m_code += "\n}";
    }

    std::string getOutput(){
        return m_code;
    }

};
