#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define WA 10000	// Matrix A width
#define HA 10000	// Matrix A height
#define WB 10000	// Matrix B width
#define HB WA		// Matrix B height
#define WC WB		// Matrix C width 
#define HC HA		// Matrix C height

#define N 100
#define M 100
#define BLOCK_SIZE 16


// Allocates a matrix with random float entries
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


// Code running on GPU //

__global__ void
matrixMul_naive( float* C, float* A, float* B, int wA, int wB)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }
  C[ i * wB + j ] = accu;
}

int main(){
    cudaEvent_t start, stop;    
    float msecTotal;
   
    cudaEventCreate(&start);

    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    float* h_C = (float*) malloc(mem_size_C);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, d_B, mem_size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N)/dimBlock.x, (M)/dimBlock.y);

    cudaEventRecord(start, NULL); 


    // execute the kernel
    matrixMul_naive<<< dimGrid, dimBlock >>>(d_C, d_A, d_B,WA,WB);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
   
    printf("GPU Execution time: %f (ms) \n", msecTotal);

return 0;
    
}



