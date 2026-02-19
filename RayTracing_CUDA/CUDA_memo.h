#pragma once

//CUDA (Compute Unified Device Architectrue)
//Nvidia가 개발한 GPU 병렬 컴퓨팅 플랫폼

//HOST vs DEVICE
//Host: CPU와 그에 연결된 메인 메모리
// - 순차적 처리에 최적화
// - 프로그램 제어 흐름 관리

//Device: GPU와 그에 연결된 비디오 메모리 (VRAM)
// - 병렬 처리에 최적화
// - 수천개의 연산을 동시에 수행

//CUDA Programming Model
//Thread Hierarchy architecture (스레드 계층 구조)
// - Thread : 실제 연산을 수행하는 프로그램 실행의 가장 작은 단위
// - Block  : 여러 스레드를 묶은 그룹
// - Grid	: 여러 블럭을 묶은 전체 작업 공간
// - Kernal : GPU에서 실행되는 함수

//메모리 복사가 필요한 이유
// 메모리 분리를 하는 이유
//1. GPU는 CPU의 RAM에 직접 접근이 불가능
//2. CPU는 GPU의 VRAM에 직접 접근이 불가능
//3. 데이터 교환을 위해 명시적 복사가 필요
//4. PCIe 버스를 통한 데이터 전송
// 병복 현상
//1. CPU - GPU 간 데이터 전송은 상대적으로 느리다
//2. 계산량이 많을수록 전송 오버헤드 상쇄
//3. 작은 데이터는 CPU가 더 빠를 수 있음


// [커널 실행 문법] 
// -> function<<<gridDim, blockDim, sharedMem, stream>>>(args...);
// gridDim   : 그리드 크기 (블록 개수)
// blockDim  : 블록 크기 (블록당 스레드 개수)
// sharedMem : 공유 메모리 크기 (바이트, 선택)
// stream    : CUDA 스트림 (선택)