#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2f "

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"

#ifndef CHUNK_SIZE
  #define CHUNK_SIZE 8
#endif

#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 256
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

#ifndef CUSTOM
/* Array inrow_thread_idialization. */
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

  data[0][0] = 1;
  data[0][1] = 2;
  data[0][2] = 5;
  data[0][3] = 6;
  data[0][4] = 0;
  data[0][5] = 10;

  data[1][0] = 2;
  data[1][1] = 4;
  data[1][2] = 5;
  data[1][3] = 5;
  data[1][4] = 1;
  data[1][5] = 10;

  data[2][0] = 3;
  data[2][1] = 6;
  data[2][2] = 5;
  data[2][3] = 4;
  data[2][4] = 0;
  data[2][5] = 10;

  data[3][0] = 4;
  data[3][1] = 8;
  data[3][2] = 5;
  data[3][3] = 3;
  data[3][4] = 1;
  data[3][5] = 10;

  data[4][0] = 5;
  data[4][1] = 10;
  data[4][2] = 5;
  data[4][3] = 2;
  data[4][4] = 0;
  data[4][5] = 10;
  
  data[5][0] = 6;
  data[5][1] = 12;
  data[5][2] = 5;
  data[5][3] = 1;
  data[5][4] = 1;
  data[5][5] = 10;


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

__global__ void mean_calculation(float * __restrict__ mean, float * __restrict__ data, int n, int m)
{
  int row_block_id = blockIdx.y;
  int column_block_id = blockIdx.x;
  int column_thread_id = threadIdx.x;

  // Calcoliamo l'indice globale della colonna per evitare di scrivere fuori array
  int global_col = column_block_id * BLOCK_SIZE + column_thread_id;

  // Se siamo fuori dalle colonne valide, terminiamo il thread
  if (global_col >= n) return;

  int block_start = row_block_id * n * BLOCK_SIZE + column_block_id * BLOCK_SIZE;

  for (int block_row = 0; block_row < BLOCK_SIZE; ++block_row)
  {
      // Calcoliamo l'indice globale della riga
      int global_row = row_block_id * BLOCK_SIZE + block_row;

      // Controllo fondamentale: non leggere righe che non esistono!
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

  //indice globale della colonna
  int global_col = column_block_id * BLOCK_SIZE + column_thread_id;
  // Calcoliamo l'indice globale della riga
  int global_row = row_block_id * BLOCK_SIZE + row_thread_id;

  if (global_col < m && global_row < n)
    data[global_row * n + global_col] -= mean[global_col];
  
}

/*
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

static void kernel_covariance_cuda(int m, int n,
           DATA_TYPE float_n,
           DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
           DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
           DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  float *d_mean; 
  float *d_data;
  
  // Alloco memoria host per leggere il risultato
  float *h_mean = (float*)malloc(sizeof(float) * n); // Nota: Dimensione 'n' (colonne)

  gpuErrchk(cudaMalloc((void **)&d_mean, sizeof(float) * m)); // Dimensione 'n'
  gpuErrchk(cudaMalloc((void **)&d_data, sizeof(float) * m * n));

  // 1. IMPORTANTE: Azzera la memoria per atomicAdd
  gpuErrchk(cudaMemset(d_mean, 0, sizeof(float) * m));
  
  printf("\nCopia dati su GPU...");
  gpuErrchk(cudaMemcpy(d_data, data, sizeof(float) * m * n, cudaMemcpyHostToDevice));
  // 2. CONFIGURAZIONE DEL LANCIO 2D
  // Vogliamo blocchi di dimensione BLOCK_SIZE sull'asse X (colonne)
  // Sull'asse Y mettiamo 1, perché il loop 'for' interno al kernel gestisce le righe verticalmente
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  // La griglia deve coprire tutte le colonne (n) e tutte le righe (m)
  // Grid X: copre le colonne
  // Grid Y: copre le righe (che userai come row_block_id)
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               1);

  printf("\nLancio Kernel Grid(%d, %d) Block(%d, %d)...", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
  
  mean_calculation<<<dimGrid, dimBlock>>>(d_mean, d_data, n, m);

  gpuErrchk(cudaDeviceSynchronize()); 
  gpuErrchk(cudaPeekAtLastError()); // Catch errori di lancio

  // Copia risultati indietro
  gpuErrchk(cudaMemcpy(h_mean, d_mean, sizeof(float) * m, cudaMemcpyDeviceToHost));

  // 3. DIVISIONE FINALE (La media richiede somma / N_elementi)
  // Poiché atomicAdd ha fatto solo la somma, dobbiamo dividere per il numero di righe (m)
  // Lo facciamo sulla CPU per semplicità, visto che devi stampare
  printf("\nRisultati Media:\n");
  for (int i = 0; i < m; ++i) { // Iteriamo sulle colonne 'n'
    h_mean[i] /= (float)n; // <--- PASSAGGIO MANCANTE FONDAMENTALE
      
    // Stampiamo solo i primi 10 per non intasare il terminale
    if (i < 10) printf("Column[%d] Mean = %f \n", i, h_mean[i]);
  }

  // [NUOVO E FONDAMENTALE] 
  // Copiamo le MEDIE calcolate (non più le somme) indietro su d_mean nella GPU
  // affinché il prossimo kernel possa usarle.
  gpuErrchk(cudaMemcpy(d_mean, h_mean, sizeof(float) * m, cudaMemcpyHostToDevice));

  // Possiamo liberare h_mean qui se non serve più per stampe finali, 
  // ma tienilo se vuoi passarlo alla funzione di output PolyBench.
  // free(h_mean); // Scommenta se vuoi liberare subito


  // --- 4. KERNEL 2: CALCOLO DEVIAZIONE (Distance Calc) [NUOVO] ---
  
  // Configurazione 2D per coprire l'intera matrice
  // Nota: Data la natura di distance_calc, usiamo una griglia 2D classica
  dim3 dimBlockDist(32, 32); 
  dim3 dimGridDist((n + dimBlockDist.x - 1) / dimBlockDist.x, 
                   (m + dimBlockDist.y - 1) / dimBlockDist.y);

  printf("Lancio Kernel Distance Grid(%d, %d)...\n", dimGridDist.x, dimGridDist.y);
  
  // d_data viene modificato "in-place" (i valori vecchi vengono sovrascritti dai nuovi)
  distance_calc<<<dimGridDist, dimBlockDist>>>(d_mean, d_data, n, m);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());


  // --- 5. COPIA RISULTATI FINALI E CLEANUP ---
  
  // Se vuoi vedere il risultato finale di d_data (la matrice deviazione), 
  // devi copiarla indietro su 'data' (host)
  gpuErrchk(cudaMemcpy(data, d_data, sizeof(float) * m * n, cudaMemcpyDeviceToHost));



// --- INIZIO DEBUG: STAMPA MATRICE DEVIAZIONE ---
  printf("\n--- Matrice di Deviazione (Host View) ---\n");

  // 1. Alloco un buffer temporaneo su Host per leggere il risultato
  //    Usa DATA_TYPE se hai definito la macro, altrimenti float
  float *h_deviation = (float*)malloc(sizeof(float) * m * n);

  // 2. Copio i dati modificati dalla GPU alla CPU
  gpuErrchk(cudaMemcpy(h_deviation, d_data, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

  // 3. Stampa
  // Impostiamo un limite per evitare di intasare il terminale se la matrice è enorme
  int print_limit_rows = (m < 20) ? m : 20; 
  int print_limit_cols = (n < 20) ? n : 20;

  for (int i = 0; i < print_limit_rows; i++) {
      printf("Row %d: ", i);
      for (int j = 0; j < print_limit_cols; j++) {
          // Stampiamo il valore. Dovresti vedere valori negativi e positivi centrati sullo 0
          printf("%6.2f ", h_deviation[i * n + j]); 
      }
      printf("\n");
  }

  if (m > 20 || n > 20) {
      printf("... (stampa troncata per dimensioni eccessive) ...\n");
  }
  printf("-----------------------------------------\n");

  // 4. Pulizia memoria temporanea
  free(h_deviation);
  // --- FINE DEBUG ---







  // Solo ORA, alla fine di tutto, liberiamo la memoria GPU
  cudaFree(d_mean);
  cudaFree(d_data);
  
  // Libera h_mean se non l'hai fatto prima
  free(h_mean); 
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

  printf("\nStarting matrix\n");
  print_array(m, POLYBENCH_ARRAY(data));

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