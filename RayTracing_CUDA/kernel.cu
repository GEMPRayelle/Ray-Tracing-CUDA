#include "device_launch_parameters.h"

#include "CUDA_memo.h"
#include "pch.h"

//CUDA 함수 한정자
#define GLOBAL __global__ //GPU에서 실행, CPU에서 호출
#define DEVICE __device__ //GPU에서 실행, GPU에서 호출
#define HOST   __host__   //CPU에서 실행, CPU에서 호출 (default) 
//return은 반드시 void
//Device code와 Host code의 경계

//GPU에서 벡터 덧셈을 수행하는 헬퍼 함수
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

//GPU에서 실행되는 커널 함수 - 각 스레드가 배열의 한 원소씩 덧셈 수행
GLOBAL void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x; //현재 스레드의 인덱스를 배열 인덱스로 사용

    printf("Thread %d: %d + %d = %d\n", i, a[i], b[i], a[i] + b[i]);

    //* threadIdx
    //현재 작업을 수행중인 스레드의 ID(고유 번호)
    //각 스레드는 자신의 ID(고유 번호)를 알고있음 이를 통해 담당할 데이터를 결정
    c[i] = a[i] + b[i];

    //threadIdx.x  // 블록 내 x축 스레드 인덱스 (0부터 시작)
    //threadIdx.y  // 블록 내 y축 스레드 인덱스
    //threadIdx.z  // 블록 내 z축 스레드 인덱스
    
    //blockIdx.x   // 그리드 내 x축 블록 인덱스
    //blockIdx.y   // 그리드 내 y축 블록 인덱스
    //blockIdx.z   // 그리드 내 z축 블록 인덱스
    
    //blockDim.x   // 블록의 x축 크기 (스레드 개수)
    //gridDim.x    // 그리드의 x축 크기 (블록 개수)
}

//렌더 커널(픽셀 병렬 처리)
GLOBAL void Render(float* frameBuffer, int maxX, int maxY)
{
    //각 스레드가 담당할 픽셀 좌표(i,j)를 계산
    int i = threadIdx.x + blockIdx.x * blockDim.x; //block내 스레드 위치 + block Offset
    int j = threadIdx.y + blockIdx.y * blockDim.y; //block내 스레드 위치 + block Offset
    //이미지 범위를 벗어나는 스레드는 즉시 종료
    if (i >= maxX || j >= maxY) return;

    int pixelIndex = j * maxX * 3 + i * 3;
    frameBuffer[pixelIndex + 0] = float(i) / float(maxX); // R
    frameBuffer[pixelIndex + 1] = float(j) / float(maxY); // G
    frameBuffer[pixelIndex + 2] = 0.2f;                   // B
}

int main()
{
    // 프로그램 전체 실행 흐름은
    // 할당 -> 복사 -> 연산 -> 회수 4단계로 이루어짐

#pragma region Device Info Check
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Grid Size: %d × %d × %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
#pragma endregion

#pragma region Ray Tracing in CUDA - CUDA Setup
    /*
    std::cout << "CUDA Vector Addition Example" << std::endl;

    //Host(CPU)쪽 입력 배열 a,b와 출력 결과인 c 선언
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // 두 벡터를 병렬로 덧셈
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    
    //GPU의 병렬 처리로
    //각 스레드 번호 i마다 c[i]에 해당하는 작업을 동시에 수행함
    //CPU일 경우엔 for문을 사용해 순서대로 n번의 연산이 필요하지만
    //GPU는 수 많은 스레드가 addWithCuda 함수를 동시에 실행함

    // Thread 0: c[0] = a[0] + b[0]
    // Thread 1: c[1] = a[1] + b[1]
    // Thread 2: c[2] = a[2] + b[2]
    // Thread 3: c[3] = a[3] + b[3]
    // Thread 4: c[4] = a[4] + b[4]

    // cudaDeviceReset : 프로그램 종료 전 반드시 호출, GPU 디바이스 리셋
    // Nsight, Visual Profiler 등 프로파일링 도구가 완전한 트레이스를 표시하려면 필요
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    */
#pragma endregion

#pragma region Ray Tracing in CUDA - FrameBuffer(Image) Rendering
    
    const int imageWidth = 1440;
    const int imageHeight = 720;

    const int blockWidth = 8;
    const int blockHeight = 8;

    dim3 threads(blockWidth, blockHeight);
    dim3 blocks(
        (imageWidth + blockWidth - 1) / blockWidth,
        (imageHeight + blockHeight - 1) / blockHeight
    );

    //Unified Memory(cudaMallocManaged)
    float* frameBuffer = nullptr;
    size_t frameBufferSize = 3 * size_t(imageWidth) * size_t(imageHeight) * sizeof(float);
    CheckCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    //cudaMalloc, cudaMemcpy : GPU 전용 할당, CPU<->GPU 복사를 직접 관리해야 함
    //cudaMallocManaged      : CPU/GPU가 같은 포인터로 접근. 데이터 이동을 런타임이 처리한다

    Render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight);

    //커널 실행 직후에는 비동기라서 에러가 늦게 나타나기에
    //두개의 에러 조합으로 실행 에러를 즉시 잡는게 좋음
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    std::ofstream outFile("output.ppm");
    outFile << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    //좌하단이 (0,0)인 좌표계로 가정
    //이미지 뷰어는 좌상단이 (0,0)처럼 보이는 경우가 많아 j를 역순으로 내려가며 기록함
    for (int j = imageHeight - 1; j >= 0; --j) {
        for (int i = 0; i < imageWidth; ++i) {
            size_t pixelIndex = size_t(j) * size_t(imageWidth) * 3 + size_t(i) * 3;
            int ir = int(255.99f * frameBuffer[pixelIndex + 0]);
            int ig = int(255.99f * frameBuffer[pixelIndex + 1]);
            int ib = int(255.99f * frameBuffer[pixelIndex + 2]);
            outFile << ir << " " << ig << " " << ib << "\n";
        }
    }

    CHECK_CUDA(cudaDeviceReset());
#pragma endregion


}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    // Device(GPU) 측 포인터 선언
    int* dev_a = 0; //GPU 메모리 주소를 저장할 포인터
    int* dev_b = 0; //GPU 메모리 주소를 저장할 포인터
    int* dev_c = 0; //GPU 메모리 주소를 저장할 포인터
   
    cudaError_t cudaStatus; //에러 처리

    // 사용할 GPU 선택 (멀티 GPU 시스템에서는 번호 변경 필요)
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    

    //1. GPU 메모리 할당 - 입력이 2개 (dev_a, dev_b) 출력은 1개 (dev_c) 
    CHECK_CUDA(cudaMalloc((void**)&dev_c, size * sizeof(int))); //size개의 int 공간을 생성
    // GPU는 CPU의 RAM을 직접 읽을 수 없어서, GPU 전용인 VRAM에 공간을 따로 확보함
    // [parameter]
    // void** devPtr : 할당된 GPU 메모리 주소를 받을 포인터
    // size_t size   : 할당된 바이트 크기
    // 할당 성공시 cudaSuccess를 return함

    CHECK_CUDA(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dev_b, size * sizeof(int)));


    // [복사 방향]
    //cudaMemcpyHostToHost      // CPU → CPU
    //cudaMemcpyHostToDevice    // CPU → GPU
    //cudaMemcpyDeviceToHost    // GPU → CPU
    //cudaMemcpyDeviceToDevice  // GPU → GPU
    //cudaMemcpyDefault         // 자동 감지 (Unified Memory)

    //2. 데이터 복사:  Host(CPU) 메모리 -> Device(GPU) 메모리로 입력 데이터 복사
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    //CPU에 있는 데이터를 GPU 메모리로 복사
    //PCIe 버스를 통한 데이터 전송 
    //PCIe : 메인보드와 GPU, SSD등 다양한 HW 구성 요소를 연결하는 PC의 핵심 인터페이스 표준
    //동기 작업 (완료될때까지 대기)

    // [parameter]
    // void* dst                 : 목적지 주소
    // const void* src           : 소스 주소
    // size_t count              : 복사할 바이트 크기
    // enum cudaMemcpyKind kind  : 복사 방향
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //3. 커널 실행 : <<<1, size>>> = 1개의 block, size개의 스레드 생성
    // 각각의 요소들이 하나의 쓰레드와 함께 GPU에서 커널 실행 (GPU에게 일을 시키는 명령어)
    // 각 스레드가 배열의 한 원소씩 병렬로 덧셈 수행
    // 커널 호출은 비동기 실행으로 
    // CPU는 커널 완료를 기다리지 않고 다음 코드 실행
    // GPU는 cudaDeviceSynchronize() 함수로 완료 대기
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    
    // 커널 실행시 에러 확인 (비동기 에러)
    cudaError_t err = cudaStatus = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernal launch failed: %s\n", cudaGetErrorString(err));
    }

    // 커널 실행 완료 대기 및 런타임 에러 확인 (동기화)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernal execution failed: %s\n", cudaGetErrorString(err));
    }

    //4. 결과 회수: Device(GPU) -> Host(CPU)로 결과를 복사
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    // GPU가 계산한 결과 dev_c는 GPU 메모리에서만 존재
    // 확인하기 위해서 CPU 메모리 c로 복사가 필요
    // 동기 작업 (복사 완료 보장)
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    //5. GPU 메모리 해제 
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
