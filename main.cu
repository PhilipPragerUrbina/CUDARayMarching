#include <iostream>
#include "GPU/Shader.cuh"
#include "IO/Image.hpp"
#include "IO/Video.hpp"

#include "Nodes/NodeCompiler.hpp"
#include "Nodes/EndNode.hpp"
#include "Nodes/SphereNode.hpp"
#include "Nodes/InfiniteRepeatNode.hpp"
#include "Nodes/MergeNode.hpp"
#include "GUI/Application.hpp"


// Main code
int main(int, char**)
{

    NodeView* nodeview = new NodeView();


    WindowGroup* group = new WindowGroup({nodeview},"eee" );
    Application app( "w",1000,1000,group);
    app.run();


    std::cout << nodeview->compile() << "\n";


    //Display* d = new Image(2000,2000, "jes.jpg", Image::JPG);
    Display* d = new Video( "test_video",1000,1000); //image sequence

    Shader s(d); //renderer

    Camera camera(Vector3(0,0,-4), Vector3(0,0,1), Vector3(0,1,0),40, d->getWidth(), d->getHeight()  ); //create camera

    for (int i = 0; i < 5; i++){
        //animate camera
        camera.setPosition(camera.getPosition() + Vector3(0.01, 0.04,0.01)); //move up
        camera.setLookAt(Vector3()); //update direction

        std::cout << i << " Frame \n"; //output frame count

        s.run(camera); //render
        s.update(); //save
    }
    return 0;
}




