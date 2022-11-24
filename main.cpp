#include <iostream>



#include "GUI/Application.hpp"
#include "GUI/OptionsPanel.hpp"

// Main code
int main(int, char**)
{

    NodeView* nodeview = new NodeView();


    WindowGroup* group = new WindowGroup({nodeview, new OptionsPanel(nodeview)},"eee" );
    Application app( "wsgddgsgd",1000,1000,group);
    app.run();



    return 0;
}




