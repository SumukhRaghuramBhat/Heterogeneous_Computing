#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 1000 		//Job size = 1K, 10K, 100K, 1M and 10M

//add kernel
__global__ void add(int *a, int *b, int *c)  
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

//function to generate random numbers 
void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%N;
	}
}

int main(void)
{

	int *a, *b, *c; 	// host copies of a, b, c
	int *d_a, *d_b, *d_c;		// device copies of a, b, c
	int size = N * sizeof(int);
	
	//time start and stop
	cudaEvent_t start, stop; 
	float time;

	cudaEventCreate(&start); 
	cudaEventCreate(&stop);

	//Allocate device memory
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	//Allocate CPU memory 
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);

	cudaEventRecord( start, 0 );
	
	//Copy CPU memory to GPU memory
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	
	// Launch add() kernel on GPU with N blocks
	add<<<1,N>>>(d_a, d_b, d_c); //N Threads and 1 Thread Block 

	//Copy from device to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf("GPU Execution Time = %f\n",time);
	
	//Cleanup
	free(a); 
	free(b);
	free(c);
	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}



