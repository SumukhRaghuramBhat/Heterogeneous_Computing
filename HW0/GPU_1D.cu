#include <stdio.h>
#include <time.h>

#define N  10000000	//Job Size = 1K, 10K, 100K, 1M and 10M
#define M  128 	         //Threads per block = 128
#define R  2 		//Radius = 2,4,8,16

// CUDA API error checking macro
static void handleError(cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line );
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) 
{

 	//index of a thread across all threads + Radius
	 int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + R;
    
	// Apply the stencil
    	int result = 0;
    	for (int offset = -R ; offset <= R ; offset++)
        result += in[gindex + offset];

    	// Store the result
    	out[gindex - R] = result;
}

int main()
{
	unsigned int i;

	//time start and stop
	cudaEvent_t start, stop; 
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//CPU array copies
	int h_in[N + 2 * R], h_out[N];

	// GPU array copies
	int *d_in, *d_out;

	for( i = 0; i < (N + 2*R); ++i )
	h_in[i] = 1; 

	// Allocate device memory
	cudaCheck( cudaMalloc( &d_in, (N + 2*R) * sizeof(int)) );
	cudaCheck( cudaMalloc( &d_out, N * sizeof(int)) );

	//copy fro CPU to GPU memory
	cudaCheck( cudaMemcpy( d_in, h_in, (N + 2*R) * sizeof(int), cudaMemcpyHostToDevice) );

	cudaEventRecord( start, 0 );
	// Call stencil kernel
	stencil_1d<<< (N + M - 1)/M, M >>> (d_in, d_out);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf(" GPU Execution Time = %f\n",time);

	// Copy results from device memory to host
	cudaCheck( cudaMemcpy( h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost) );

	// Cleanup
	cudaFree(d_in);
	cudaFree(d_out);

  return 0;
}

