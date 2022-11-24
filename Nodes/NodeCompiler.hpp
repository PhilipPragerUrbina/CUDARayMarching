//
// Created by Philip on 11/19/2022.
//

#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <regex>
///Generate C++ code for nodes
class NodeCompiler {
private:
    std::string m_code = "";
    std::vector<std::string> m_lines;
    std::string m_latest_line = "";
    std::map<std::string,std::string> m_includes;
    std::map<std::string,std::string> m_includes_2;

    std::string m_method_name;
    std::vector<std::string> m_header_names;
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
    void valParameter(int id){
        m_latest_line += "node_" + std::to_string(id);
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

    std::vector<std::string> getHeaderNames(){
        return m_header_names;
    }

    ///Finalize the compilation
    void finish(){
        m_code += open_file("../Rays/Ray.cuh");

        for (auto const& include : m_includes)
        {
         //  m_code += include.second;
           m_header_names.push_back(include.first + ".cuh");
            m_code += open_file("../SDF/"+include.first + ".cuh");

        }
        m_code += "extern \"C\" __device__ double " + m_method_name + "(Ray r) { \n";
        for (int i = m_lines.size()-1; i >= 0; i--){
            //reverse order
            m_code+= m_lines[i];
        }

        m_code += "\n}";
    }

    std::string open_file(std::string name){
        if(m_includes_2.find(name) != m_includes_2.end()){
            return "";
        }
        m_includes_2[name] = "a";

        std::ifstream t(name);
        std::stringstream buffer;
        buffer << t.rdbuf();

        std::string  s1 =  buffer.str();

        std::regex e(R"reg(\s*#\s*include\s*([<"])([^>"]+)([>"]))reg");
        std::regex ee("(#pragma once)|(#define FULL_COMPILATION)");


        std::sregex_iterator iter(s1.begin(), s1.end(), e);
        std::sregex_iterator end;

        while(iter != end)
        {


          //  std::cout << "expression match #" << 0 << ": " << (*iter)[0] << std::endl;
            for(unsigned i = 1; i < iter->size(); ++i)
            {
                if(i==2){


                    if(std::string((*iter)[i-1]) != "<"){
                        m_code += open_file( std::string((*iter)[i]));
                    }

                }
             //   std::cout << "capture submatch #" << i << ": " << (*iter)[i] << std::endl;
            }
            ++iter;
        }

        return  std::regex_replace(std::regex_replace(s1, ee, ""), e, "");;

    }

    std::string getOutput(){
        return m_code;
    }

};
