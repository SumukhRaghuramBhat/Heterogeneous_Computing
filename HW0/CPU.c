#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
#define N 10000	 //Job size = 1K, 10K, 100K, 1M, 10M

int main()
{
	static int a[N];
	static int b[N];
	static int c[N];
	int i;

	//double execution_time = 0.0;
	srand(time(NULL));
	clock_t begin = clock();

	for (i = 0; i<N; i++)
	{
		a[i] = rand()%N;
		b[i] = rand()%N;
		c[i] = a[i] + b[i];
	}
	clock_t end = clock();
	double execution_time = execution_time + (double)(end - begin)/CLOCKS_PER_SEC;
	printf("CPU Execution time = %f\n",execution_time);
	
	return 0;
}


	

