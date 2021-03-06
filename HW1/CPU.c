#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1000

typedef unsigned long int ulong;

ulong A[MATRIX_SIZE * MATRIX_SIZE];
ulong B[MATRIX_SIZE * MATRIX_SIZE];
ulong C[MATRIX_SIZE * MATRIX_SIZE];

int main(void) { 
  for (ulong row = 0; row < MATRIX_SIZE; row++) {
    for (ulong col = 0; col < MATRIX_SIZE; col++) {
      A[row * MATRIX_SIZE + col] = 1;
      B[row * MATRIX_SIZE + col] = 2;
      C[row * MATRIX_SIZE + col] = 0;
    }
  }

  clock_t begin = clock();
  
  for (int row = 0; row < MATRIX_SIZE; row++) {
    for (int col = 0; col < MATRIX_SIZE; col++) {
      ulong sum = 0;
      for (int k = 0; k < MATRIX_SIZE; k++) {
        sum += A[row * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + col];
      }
      C[row * MATRIX_SIZE + col] = sum;
    }
  }

  clock_t end = clock();
  double runtime = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("Runtime: %lf secs\n", runtime);

  return 0;
}
