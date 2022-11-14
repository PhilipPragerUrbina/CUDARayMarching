//
// Created by Philip on 11/12/2022.
//

#pragma once

#include "nvrtc.h"
#include "cuda.h"
///Use NVRTC to compile cuda code at run-time
class RuntimeMethod {
private:
    std::string m_source;
    std::string m_ptx;

public:
    RuntimeMethod(const std::string& source) : m_source(source){
        nvrtcProgram runtime_program;
        nvrtcCreateProgram(&runtime_program,
                           source.c_str(),"test.cu",0,NULL,NULL);
        const char *opts[] = {"--fmad=false"};
        nvrtcCompileProgram(runtime_program,1,opts);


        size_t logSize;
        nvrtcGetProgramLogSize(runtime_program, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(runtime_program, log);

// Obtain PTX from the program.
        size_t ptxSize;
        nvrtcGetPTXSize(runtime_program, &ptxSize);
        char *ptx = new char[ptxSize];
        nvrtcGetPTX(runtime_program, ptx);
        m_ptx = std::string(ptx,ptxSize);

        nvrtcDestroyProgram(&runtime_program);
    }

    void add(){
        CUdevice cuDevice;
        CUcontext context;
        CUmodule module;
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&context, 0, cuDevice);
        cuModuleLoadDataEx(&module, ptx, 0, 0, 0);


    }


};
