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

// [FIX] NUOVO: TILE_SIZE per il kernel 2D (Covariance) per gestire la Shared Memory
// 32x32 floats = 4KB. 2 matrici = 8KB (Ben dentro il limite di 48KB)
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

//Mean calculation
__global__ void mean_calculation(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
  int row_block_id = blockIdx.y;
  int column_block_id = blockIdx.x;
  int column_thread_id = threadIdx.x;

  int global_col = column_block_id * BLOCK_SIZE + column_thread_id;
  if (global_col >= n) return;

  int block_start = row_block_id * n * BLOCK_SIZE + column_block_id * BLOCK_SIZE;

  for (int block_row = 0; block_row < BLOCK_SIZE; ++block_row)
  {
      int global_row = row_block_id * BLOCK_SIZE + block_row;
      if (global_row < m) {
          float val = data[block_start + block_row * n + column_thread_id];
          atomicAdd(&mean[global_col], val);
      }
  }
}

__global__ void distance_calc(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
  int row_block_id = blockIdx.y;
  int column_block_id = blockIdx.x;
  int row_thread_id = threadIdx.y;
  int column_thread_id = threadIdx.x;

  int global_col = column_block_id * 32 + column_thread_id; // Nota: qui dipende da come lanci il kernel (vedi sotto)
  int global_row = row_block_id * 32 + row_thread_id;

  if (global_col < n && global_row < m) // Corretto ordine indici: col < n, row < m
    data[global_row * n + global_col] -= mean[global_col];
}


// Kernel Covarianza Corretto: Calcola Cov(Col_i, Col_j)
// n = numero di colonne (Features) -> Dimensione Output NxN
// m = numero di righe (Osservazioni) -> Dimensione Riduzione Loop
__global__ void covariance_calc(DATA_TYPE * __restrict__ symmat, DATA_TYPE * __restrict__ data, int m, int n)
{
    // Indici delle Feature (Colonne)
    int j1 = blockIdx.y * blockDim.y + threadIdx.y; // Riga della matrice Covarianza (Feature 1)
    int j2 = blockIdx.x * blockDim.x + threadIdx.x; // Colonna della matrice Covarianza (Feature 2)

    // Bounds Check
    if (j1 >= n || j2 >= n) return;

    // Ottimizzazione Simmetria: Calcoliamo solo triangolare superiore + diagonale
    if (j2 < j1) return;

    DATA_TYPE sum = 0.0;

    // Loop di riduzione lungo le righe (Osservazioni - m)
    for (int i = 0; i < m; i++)
    {
        // Accediamo alla stessa riga 'i', ma colonne diverse 'j1' e 'j2'
        // data è linearizzato: index = riga * n + colonna
        sum += data[i * n + j1] * data[i * n + j2];
    }

    // Scrittura risultato normalizzato
    // Divisione per (m - 1) perché m è il numero di osservazioni
    DATA_TYPE val = sum / (DATA_TYPE)(m - 1);
    
    symmat[j1 * n + j2] = val;
    
    // Scrittura speculare per la triangolare inferiore
    if (j1 != j2) {
        symmat[j2 * n + j1] = val;
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
  //printf("\nCopia dati su GPU...");
  gpuErrchk(cudaMemcpy(d_data, data, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  // --- 1. MEDIA ---
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  //printf("\nLancio Kernel Media Grid(%d, %d)...", dimGrid.x, dimGrid.y);
  mean_calculation<<<dimGrid, dimBlock>>>(d_mean, d_data, n, m);
  gpuErrchk(cudaDeviceSynchronize()); 
  
  float *h_mean = (float*)malloc(sizeof(float) * n); 

  // Calcolo media su Host
  gpuErrchk(cudaMemcpy(h_mean, d_mean, sizeof(float) * m, cudaMemcpyDeviceToHost));
  for (int i = 0; i < m; ++i) { 
    h_mean[i] /= (float)n; 
  }
  gpuErrchk(cudaMemcpy(d_mean, h_mean, sizeof(float) * m, cudaMemcpyHostToDevice));

  free(h_mean);

  // --- 2. DISTANCE (DEVIAZIONE) ---
  dim3 dimBlockDist(32, 32); 
  dim3 dimGridDist((n + dimBlockDist.x - 1) / dimBlockDist.x, 
                   (m + dimBlockDist.y - 1) / dimBlockDist.y);

  //printf("\nLancio Kernel Distance Grid(%d, %d)...", dimGridDist.x, dimGridDist.y);
  distance_calc<<<dimGridDist, dimBlockDist>>>(d_mean, d_data, n, m);
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(d_mean);

  // --- 3. COVARIANZA ---
  
  float *d_symmat; // Puntatore device per la matrice risultato
  gpuErrchk(cudaMalloc((void **)&d_symmat, sizeof(float) * m * m)); 

  dim3 dimBlockCov(16, 16); 
  dim3 dimGridCov((n + dimBlockCov.x - 1) / dimBlockCov.x, 
                  (n + dimBlockCov.y - 1) / dimBlockCov.y);

  //printf("\nLancio Covariance Kernel Grid(%d, %d)...\n", dimGridCov.x, dimGridCov.y);
  
  covariance_calc<<<dimGridCov, dimBlockCov>>>(d_symmat, d_data, m, n);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  
  gpuErrchk(cudaMemcpy(symmat, d_symmat, sizeof(float) * n * n, cudaMemcpyDeviceToHost));

  cudaFree(d_data);
  cudaFree(d_symmat); 
  
}

#endif
// ... [Il resto del file main rimane uguale] ...

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
  
  printf("sono prima del kernel\n");
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