#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>

#include <fstream>

#pragma region Macro

//에러 처리 매크로
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

//Legacy Code

//cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//if (cudaStatus != cudaSuccess)
//{
//    fprintf(stderr, "cudaMalloc failed!");
//    goto Error;
//}

// CUDA 에러 체크 매크로 : CUDA API 호출 실패시 에러 코드와 발생 위치(파일/라인)을 출력하는 패턴이 유용
inline void checkCuda(cudaError_t result, const char* func, const char* file, int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
            << " (" << cudaGetErrorString(result) << ") at "
            << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        cudaGetErrorString(result);
        std::exit(99);
    }
}

#define CheckCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

#pragma endregion