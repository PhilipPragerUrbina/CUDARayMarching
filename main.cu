#include <iostream>
#include "Math/Vector3.cuh"
#include "Shader.cuh"
#include "IO/Image.hpp"
#include "IO/Video.hpp"
int main()
{
    //Display* d = new Image(2000,2000, "jes.jpg", Image::JPG);
     Display* d = new Video( "test_video",1000,1000);
    Shader s(d);
    for (int i = 0; i < 100; i++){
        std::cout << i << " Frame \n";
        s.run(i);
    }
//$ ffmpeg -i test_video_%d.jpg -c:v libx264 -vf format=yuv420p output.mp4

}