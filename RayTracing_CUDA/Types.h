#pragma once

//CUDA 함수 한정자
#define GLOBAL __global__ //GPU에서 실행, CPU에서 호출
//return은 반드시 void
//Device code와 Host code의 경계

#define DEVICE __device__ //GPU에서 실행, GPU에서 호출 가능
#define HOST   __host__   //CPU에서 실행, CPU에서 호출 가능 (default) 
#define HOSTDEVICE __host__ __device__ //한 함수 정의: CPU/GPU 양쪽에서 호출 가능
