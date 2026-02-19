#pragma once

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

#pragma endregion