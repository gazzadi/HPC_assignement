#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2f "

/* Include polybench common header. */
#include <polybench.h>
#include "covariance.h"

// --- DEFINIZIONI ---
#ifndef CHUNK_SIZE
  #define CHUNK_SIZE 8
#endif

// BLOCK_SIZE per i kernel 1D (Mean calculation)
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 256
#endif

#define TILE_SIZE 32 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifndef CUSTOM
/* Array inrow_thread_inizialization. */
static
void inrow_thread_id_array (int m, int n,
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
void inrow_thread_id_array(int m, int n,
     DATA_TYPE *float_n,
     DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  *float_n = 1.2;
  data[0][0] = 1; data[0][1] = 2; data[0][2] = 5; data[0][3] = 6; data[0][4] = 0; data[0][5] = 10;
  data[1][0] = 2; data[1][1] = 4; data[1][2] = 5; data[1][3] = 5; data[1][4] = 1; data[1][5] = 10;
  data[2][0] = 3; data[2][1] = 6; data[2][2] = 5; data[2][3] = 4; data[2][4] = 0; data[2][5] = 10;
  data[3][0] = 4; data[3][1] = 8; data[3][2] = 5; data[3][3] = 3; data[3][4] = 1; data[3][5] = 10;
  data[4][0] = 5; data[4][1] = 10; data[4][2] = 5; data[4][3] = 2; data[4][4] = 0; data[4][5] = 10;
  data[5][0] = 6; data[5][1] = 12; data[5][2] = 5; data[5][3] = 1; data[5][4] = 1; data[5][5] = 10;
}
#endif

/* DCE code. */
static
void print_array(int m,
     DATA_TYPE POLYBENCH_2D(symmat,M,N,m,n))
{
  int i, j;
  for (i = 0; i < m; i++){
    for (j = 0; j < m; j++) {
      printf (DATA_PRINTF_MODIFIER, symmat[i][j]);
    }
    printf ("\n");
  }
  printf ("\n");
}


#ifndef SEQ

// Kernel Mean calculation
__global__ void mean_calculation(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
    int block_dim_x = blockDim.x;
    int block_dim_y = blockDim.y;

    int row_block_id = blockIdx.y;
    int column_block_id = blockIdx.x;
    int column_thread_id = threadIdx.x;

    int global_col = column_block_id * block_dim_y + column_thread_id;
    
    // Bound Check
    if (global_col >= n) return;

    // Iteration on all the block rows
    for (int block_row = 0; block_row < block_dim_x; ++block_row)
    {
        int global_row = row_block_id * block_dim_x + block_row;
        
        // Bound Check
        if (global_row < m) {
            
            float val = data[global_row * n + global_col];
            // Using AtomicAdd prevent to consequential modification of the same memory cell
            atomicAdd(&mean[global_col], val);
        }
    }
}

// Kernel Deviation Matrix Calculation
__global__ void distance_calc(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
  int row_block_id = blockIdx.y;
  int column_block_id = blockIdx.x;
  int row_thread_id = threadIdx.y;
  int column_thread_id = threadIdx.x;

  int global_col = column_block_id * blockDim.y + column_thread_id;
  int global_row = row_block_id * blockDim.x + row_thread_id;

  if (global_col < n && global_row < m)
    data[global_row * n + global_col] -= mean[global_col];
}



// --- KERNEL OTTIMIZZATO CON TILING ---
__global__ void covariance_tiled(float * __restrict__ symmat, float * __restrict__ data, int n, int m)
{
  // Indexes inside the Tile
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Column Features
  int j1 = blockIdx.y * TILE_SIZE + ty; // row of covariance_matrix (Feature Y)
  int j2 = blockIdx.x * TILE_SIZE + tx; // column of covariance_matrix (Feature X)

  // We work only on the upper triangular matrix, because of symmetry
  if (blockIdx.x < blockIdx.y) return;

  // Two tiles
  // 1. Data of features block Y (tile_row)
  // 2. Data of features block X (tile_col)
  __shared__ float tile_row[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid Bank Conflicts
  __shared__ float tile_col[TILE_SIZE][TILE_SIZE + 1];

  float sum = 0.0f;

  //Iteration on all the samples wit tile size
  for (int k = 0; k < n; k += TILE_SIZE)
  {      
      int current_sample_idx = k + ty;
      
      int feature_y_idx = blockIdx.y * TILE_SIZE + tx;
      int feature_x_idx = blockIdx.x * TILE_SIZE + tx;

      if (current_sample_idx < n && feature_y_idx < m) {
          tile_row[ty][tx] = data[current_sample_idx * m + feature_y_idx];
      } else {
          tile_row[ty][tx] = 0.0f;
      }
      
      if (current_sample_idx < n && feature_x_idx < m) {
          tile_col[ty][tx] = data[current_sample_idx * m + feature_x_idx];
      } else {
          tile_col[ty][tx] = 0.0f;
      }

      __syncthreads();

      for (int s = 0; s < TILE_SIZE; ++s)
      {
          sum += tile_row[s][ty] * tile_col[s][tx];
      }

      // Synchronize before deleteing shared memory
      __syncthreads();
  }

  // Writing results
  if (j1 < m && j2 < m)
  {
      float val = sum / (float)(n - 1);
      symmat[j1 * m + j2] = val;
      if (j1 != j2) {
          symmat[j2 * m + j1] = val;
      }
  }
}

static void kernel_covariance_cuda(int m, int n,
           DATA_TYPE float_n,
           DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
           DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
           DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  float *d_mean; 
  float *d_data;

  gpuErrchk(cudaMalloc((void **)&d_mean, sizeof(float) * m)); 
  gpuErrchk(cudaMalloc((void **)&d_data, sizeof(float) * m * n));
  
  gpuErrchk(cudaMemset(d_mean, 0, sizeof(float) * m));
  gpuErrchk(cudaMemcpy(d_data, data, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  // --- 1. MEAN ---
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  mean_calculation<<<dimGrid, dimBlock>>>(d_mean, d_data, n, m);
  gpuErrchk(cudaDeviceSynchronize()); 
  
  // MEAN CALCULATION ON HOST
  float *h_mean = (float*)malloc(sizeof(float) * n); 
  gpuErrchk(cudaMemcpy(h_mean, d_mean, sizeof(float) * m, cudaMemcpyDeviceToHost));
  for (int i = 0; i < m; ++i) { 
    h_mean[i] /= (float)n; 
  }
  gpuErrchk(cudaMemcpy(d_mean, h_mean, sizeof(float) * m, cudaMemcpyHostToDevice));

  free(h_mean);


  // --- 2. DISTANCE ---
  dim3 dimBlockDist(32, 32); 
  dim3 dimGridDist((n + dimBlockDist.x - 1) / dimBlockDist.x, 
                   (m + dimBlockDist.y - 1) / dimBlockDist.y);

  distance_calc<<<dimGridDist, dimBlockDist>>>(d_mean, d_data, n, m);
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(d_mean);


  // --- 3. COVARIANCE ---
  float *d_symmat; 
  gpuErrchk(cudaMalloc((void **)&d_symmat, sizeof(float) * m * m)); 

  dim3 dimBlockCov(TILE_SIZE, TILE_SIZE); 
  dim3 dimGridCov((m + TILE_SIZE - 1) / TILE_SIZE, 
                  (m + TILE_SIZE - 1) / TILE_SIZE);

  covariance_tiled<<<dimGridCov, dimBlockCov>>>(d_symmat, d_data, n, m);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  
  gpuErrchk(cudaMemcpy(symmat, d_symmat, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

  cudaFree(d_data);
  cudaFree(d_symmat); 
  
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
  
  /* Inrow_thread_idialize array(s). */
  // non è per forza necessario accellerare la creazione di questo array (opzionale)
  inrow_thread_id_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  #ifdef PRINT
  printf("\nStarting matrix\n");
  print_array(m, POLYBENCH_ARRAY(data));
  #endif

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