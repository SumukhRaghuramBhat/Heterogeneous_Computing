#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>


#define N 1000 		//Job Size = 1K,10K,100K,1M,10M
#define R 2 		//Radius = 2,4,8,16

int main(void)
{
	int i; 
	int array[N + 4];
	int add[N];
	int offset;
	int num[100];
	int j;
	int k;
	
	for (i = 0; i<N+4; i++)
	{
		array[i] = rand()%N+4;
		//printf("array = %d\n",array[i]);
	}
	
	int index = 0;
	for (j = 0; j<N;j++)
	{
		add[j] = 0;		
	}
	clock_t begin = clock();
	for(k = R;k<N+R; k++)
	{
		// Apply the stencil
		for(offset = -R;offset <= R;offset++)
		{
			add[k-R] = add[k-R] + num[index + offset + R]; 	// Store the result
		}
	index++;
	}

	clock_t end = clock();
	double execution_time = (double)(end - begin)/ CLOCKS_PER_SEC;
	printf("CPU Execution Time = %f\n",execution_time);
	
	return 0;
}

