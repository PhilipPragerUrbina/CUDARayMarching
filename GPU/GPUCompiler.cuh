//
// Created by Philip on 11/13/2022.
//

#pragma once
#include <vector>
#include "nvrtc.h"
#include "cuda.h"

class GPUCompiler {
private:
    std::vector<std::string> m_ptx; //gpu assembly code
    std::vector<std::string> m_files;
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    // todo destroy module on recompile, and allow updating of specific code
public:
    GPUCompiler(){}

    void addPTX(const std::string& ptx){
        m_ptx.push_back( ptx);
    }

    void loadPTXFile(const std::string& filename){
        m_files.push_back(filename);
    }

    //todo add saftey for missing ilfes and ushc
    //todo create sperate file for cuda safe call
    //todo document

    CUmodule* compile(){
        CUlinkState linker;
        //initialize device
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        //initialize context
        cuCtxCreate(&context, 0, cuDevice);
        //create linker
        cuLinkCreate(0,0,0,&linker);
        //link files
        for (std::string filename : m_files){
            cuLinkAddFile(linker,CU_JIT_INPUT_PTX,  filename.c_str(),0,0,0);
        }
        //link data
        for (std::string sub_ptx : m_ptx){
            //todo add name
            cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void *) sub_ptx.c_str(), sub_ptx.size(), "a", 0, 0, 0);
        }
        //pointers for output image
        void* cubin_image;
        size_t image_size;
        //create output image using linker
        cuLinkComplete(linker,&cubin_image,&image_size);


        //create module
        cuModuleLoadData(&module,cubin_image );

        //clean up linker(after module is loaded)
        cuLinkDestroy(linker);

        return &module;
    }


};
