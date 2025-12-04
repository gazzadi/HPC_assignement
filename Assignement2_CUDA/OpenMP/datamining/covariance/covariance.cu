#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"

#ifndef CHUNK_SIZE
  #define CHUNK_SIZE 8
#endif

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#ifdef CUSTOM
/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  int i, j;

  *float_n = 1.2;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}
#else
static
void init_array(int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{

  *float_n = 1.2;

  data[0][0] = 1;
  data[0][1] = 2;
  data[0][2] = 5;
  data[0][3] = 10;
  data[0][4] = 1;
  data[1][0] = 2;
  data[1][1] = 4;
  data[1][2] = 4;
  data[1][3] = 10;
  data[1][4] = 0;
  data[2][0] = 3;
  data[2][1] = 6;
  data[2][2] = 3;
  data[2][3] = 10;
  data[2][4] = 1;
  data[3][0] = 4;
  data[3][1] = 8;
  data[3][2] = 2;
  data[3][3] = 10;
  data[3][4] = 0;
  data[4][0] = 5;
  data[4][1] = 10;
  data[4][2] = 1;
  data[4][3] = 10;
  data[4][4] = 3;

}
#endif


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,N,m,n))

{
  int i, j;

  for (i = 0; i < m; i++){
    for (j = 0; j < m; j++) {
      printf (DATA_PRINTF_MODIFIER, symmat[i][j]);
      //if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
    printf ("\n");
  }
  printf ("\n");
}

#ifndef SEQ
  //Per parallelizzare con cuda, possiamo sicuramente parallelizzare ogni for in maniera distinta

  //Ogni for potrebbe essere una device function, mentre tutto il kernel è una __global__ function
__global__ void mean_calculation(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  mean[column] = 0.0;

  int row = blockIdx.y * blockDim.y * n + threadIdx.y;
  for (row; row < blockDim.y; row + n)
    mean[column] += data[row + column];
  mean[column] /= n;
}

/*
__global__ void distance_calc(float * __restrict__ mean, float * __restrict__ data)
{
  int i, j;
  for (i = 0; i < _PB_N; i++)
  {
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  }
}

__global__ void covariance_calc(float * __restrict__ symmat, float * __restrict__ data)
{
  int i, j1, j2;
  for (j1 = 0; j1 < _PB_M; j1++)
    for (j2 = j1; j2 < _PB_M; j2++)
    {
      symmat[j1][j2] = 0.0;

      for (i = 0; i < _PB_N; i++)
        symmat[j1][j2] += data[i][j1] * data[i][j2];

      symmat[j1][j2] /= _PB_N - 1;
      symmat[j2][j1] = symmat[j1][j2];
    }
}
*/


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance_cuda(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i=0, j=0, j1=0, j2=0;
  float *d_mean, *h_mean; 
  float *d_data, *h_data;

  gpuErrchk(cudaMalloc((void **)&d_mean, sizeof(float) * m));
  gpuErrchk(cudaMalloc((void **)&d_data, sizeof(float) * m * n));

  gpuErrchk(cudaMemcpy(d_data, data, sizeof(float) * m * n, cudaMemcpyHostToDevice));
  int BLOCK_SIZE = 16;
  mean_calculation<<<((n + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(d_mean, d_data, n, m);

  gpuErrchk(cudaMemcpy(h_mean, d_mean, sizeof(float) * m, cudaMemcpyDeviceToHost));

}

#endif

#ifdef SEQ
static
void kernel_covariance_seq(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i=0, j=0, j1=0, j2=0;
  
  for (j = 0; j < _PB_M; j++)
  {
    mean[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      mean[j] += data[i][j];
    mean[j] /= _PB_N;
  }
  
  for (i = 0; i < _PB_N; i++)
  {
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  }

  for (j1 = 0; j1 < _PB_M; j1++)
    for (j2 = j1; j2 < _PB_M; j2++)
    {
      symmat[j1][j2] = 0.0;

      for (i = 0; i < _PB_N; i++)
        symmat[j1][j2] += data[i][j1] * data[i][j2];

      symmat[j1][j2] /= _PB_N - 1;
      symmat[j2][j1] = symmat[j1][j2];
    }

}
#endif

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  
  /* Initialize array(s). */
  // non è per forza necessario accellerare la creazione di questo array (opzionale)
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  
  #ifdef SEQ
  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance_seq(m, n, float_n,
		     POLYBENCH_ARRAY(data), //polybench_array create a pointer to the variable passed
		     POLYBENCH_ARRAY(symmat),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  printf("\nseq time: ");
  polybench_print_instruments;
  #else
   /* Start timer. */
  polybench_start_instruments;
  

  /* Run kernel. */
  kernel_covariance_cuda(m, n, float_n,
		     POLYBENCH_ARRAY(data), //polybench_array create a pointer to the variable passed
		     POLYBENCH_ARRAY(symmat),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  printf("\ngpu time: ");
  polybench_print_instruments;
  #endif

  #ifdef PRINT
    printf("\n\ncovariance_matrix: \n");
    print_array(m, POLYBENCH_ARRAY(symmat));
  #endif

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}