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

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

__global__ void
matrixMul_coalescing( float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        As(ty, tx) = A[a + wA * ty + tx];
        Bs(tx, ty) = B[b + wB * ty + tx];

        __syncthreads();

             for (int k = 0; k < BLOCK_SIZE; ++k)
	  Csub += AS(ty, k) * BS(tx, k);

                __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/* Codes running on GPU */

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

    float *transpB;
    transpB = (float *) malloc(mem_size_B);

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
   
    for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			*(transpB + i * M + j) = *(h_B + j * M + i);

    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, transpB, mem_size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N)/dimBlock.x, (M)/dimBlock.y);

    cudaEventRecord(start, NULL); 

    // execute the kernel
    matrixMul_coalescing<<< dimGrid, dimBlock >>>(d_C, d_A, d_B,WA,WB);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      
    printf("GPU Execution time: %f (ms) \n", msecTotal);

return 0;
    
}



