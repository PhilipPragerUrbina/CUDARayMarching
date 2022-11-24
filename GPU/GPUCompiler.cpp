//
// Created by Philip on 11/13/2022.
//

#pragma once
#include <vector>
#include "nvrtc.h"
#include "cuda.h"
#include "SafeCall.h"

/// Compile cuda code at runtime
class GPUCompiler {
private:
    std::vector<std::string> m_ptx; //gpu assembly code
    std::vector<std::string> m_files; //gpu assembly code stored in files
    //cuda stuff
    CUdevice m_device;
    CUcontext m_context;
    CUmodule m_module;
    //Can only compile once
    bool m_has_compiled = false;
public:

    /// Create CUDA driver context
    GPUCompiler(){
        //initialize device
        CUDA_CHECK(cuInit(0));
        CUDA_CHECK(cuDeviceGet(&m_device, 0));
        //initialize context
        CUDA_CHECK(cuCtxCreate(&m_context, 0, m_device));
    }

    /// Clean up context and device
    ~GPUCompiler(){
        if(m_has_compiled){
            CUDA_CHECK(cuModuleUnload(m_module));
        }
        CUDA_CHECK(cuCtxDestroy(m_context));
    }

    /// Add PTX code to project
    /// @param ptx GPU assembly
    void addPTX(const std::string& ptx){
        m_ptx.push_back( ptx);
    }

    /// Add PTX code stored in file
    /// @param filename The location of file
    void loadPTXFile(const std::string& filename){
        m_files.push_back(filename);
    }

    /// Load and compile cuda C++ code
    /// @param source The c++ 17 source code
    void loadCPP(std::string source){
        //create nvrtc program
        nvrtcProgram runtime_program;
        NVRTC_CHECK(nvrtcCreateProgram(&runtime_program,source.c_str(), "", 0, 0, 0));

        //compile
        const char *options[] = {"-std=c++17", "-rdc=true"}; //use c++ 17 and create linkable code
        nvrtcResult compile_result = nvrtcCompileProgram(runtime_program, 2, options);

        //Get log
        size_t log_size;
        NVRTC_CHECK(nvrtcGetProgramLogSize(runtime_program, &log_size));
        char *log = new char[log_size];
        NVRTC_CHECK(nvrtcGetProgramLog(runtime_program, log));
        std::cout << log << '\n';
        delete[] log;

        NVRTC_CHECK(compile_result);   //check for compile errors after log

        //get ptx bytecode
        size_t ptx_size;
        NVRTC_CHECK(nvrtcGetPTXSize(runtime_program, &ptx_size));
        char *ptx_chars = new char[ptx_size];
        NVRTC_CHECK(nvrtcGetPTX(runtime_program, ptx_chars));

        //convert to string
        std::string final_ptx = std::string(ptx_chars, ptx_size);
        delete[] ptx_chars;

        NVRTC_CHECK(nvrtcDestroyProgram(&runtime_program)); //clean up

        addPTX(final_ptx); //add to project
    }


    //todo add saftey for missing ilfes and ushc

    /// Compile the project
    /// @return The cuda module. lifetime is owned by this compiler
    CUmodule* compile(){
        if(m_has_compiled){
            return &m_module; //has already compiled
        }
        //create linker
        CUlinkState linker;
        CUDA_CHECK(cuLinkCreate(0,0,0,&linker));

        for (std::string filename : m_files){ //link files
            CUDA_CHECK(cuLinkAddFile(linker, CU_JIT_INPUT_PTX, filename.c_str(), 0, 0, 0));
        }
        for (std::string sub_ptx : m_ptx){ //link strings
            CUDA_CHECK(cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void *) sub_ptx.c_str(), sub_ptx.size(), "", 0, 0, 0));
        }

        //create output image using linker
        void* cubin_image; //gpu assembly
        size_t image_size;
        CUDA_CHECK(cuLinkComplete(linker, &cubin_image, &image_size));

        //create module
        CUDA_CHECK(cuModuleLoadData(&m_module, cubin_image ));
        m_has_compiled = true;

        CUDA_CHECK(cuLinkDestroy(linker));     //clean up linker(after module is loaded)

        return &m_module;
    }


};
