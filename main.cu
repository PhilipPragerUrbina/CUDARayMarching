#include <iostream>
#include "Math/Vector3.cuh"
#include "Shader.cuh"
#include "IO/Image.hpp"

int main()
{
    Display* d = new Image(2000,2000, "jes.jpg", Image::JPG);
    Shader s(d);
    s.run();

}