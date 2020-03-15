#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define N   10000000 	//job size = 1K, 10K, 100K, 1M and 10M
#define M   128 	//Threads per block =128
#define R   16 		//radius = 2,4,8,16

// CUDA API error checking macro
static void handleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define cudaCheck( err ) (handleError( err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) 
{
	__shared__ int temp[M + 2 * R];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + R;
	temp[lindex] = in[gindex]; // Read input elements into shared memory
	
	if (threadIdx.x < R) 
	{
	temp[lindex - R] = in[gindex - R];
	temp[lindex + M] = in[gindex + M];
	}
	
	// Synchronize (ensure all the data is available)
	__syncthreads();
	int result = 0;
	
	// Apply the stencil
	for (int offset = -R ; offset <= R ; offset++)
	{
		result += temp[lindex + offset];
	}
	
	// Store the result
	out[gindex] = result;
}	

int main()
{
	unsigned int i;
	int h_in[N + 2 * R], h_out[N];
	int *d_in, *d_out;

	//time start and stop
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for( i = 0; i < (N + 2*R); ++i )
	h_in[i] = 1; 

	// Allocate device memory
	cudaCheck( cudaMalloc( &d_in, (N + 2*R) * sizeof(int)) );
	cudaCheck( cudaMalloc( &d_out, N * sizeof(int)) );

	//copy fro CPU to GPU memory
	cudaCheck( cudaMemcpy( d_in, h_in, (N + 2*R) * sizeof(int), cudaMemcpyHostToDevice) );
	cudaEventRecord( start, 0 );

	//Call stencil kernel
	stencil_1d<<< (N + M - 1)/M, M >>> (d_in, d_out);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("GPU Execution Time = %f\n",time);

	//copy from device to host
	cudaCheck( cudaMemcpy( h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost) );

	// Cleanup
	cudaFree(d_in);
	cudaFree(d_out);

  return 0;
}

