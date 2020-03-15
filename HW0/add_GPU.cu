#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 1000000        //Job size = 1M
#define M 128 		// Varying Thread block size- 128, 256, 512, 1024

//add kernel
__global__ void add(int *a, int *b, int *c, int k)  
{
	int index = threadIdx.x+ blockIdx.x * blockDim.x;
	if (index<k)
		c[index] = a[index] + b[index];
}

//Random number generator function
void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%N;
	}
}

int main(void)
{

	int *a, *b, *c; 
	int *d_a, *d_b, *d_c;
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
	
	
	//Call the add kernel
	add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c,N); 
	
	printf("GPU Execution Time = %f\n",time);
	
	// Copy from device to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Execution Time = %f\n",time);
	
	//Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;

}



