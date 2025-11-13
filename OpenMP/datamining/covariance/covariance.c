#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"


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


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i=0, j=0, j1=0, j2=0;
  
  #pragma omp target enter data \
                          map(alloc: mean[0:_PB_M]) \
                          map(to: data[0:_PB_N][0:_PB_M]) \
                          map(to: float_n)
      
    /* Determine mean of column vectors of input data matrix 
    *   Calcola la media dei valori di ogni colonna (possiamo dire che ogni colonna rappresenta una feature)
    */

    //#pragma omp parallel for
    //#pragma omp target map(to: data[0:_PB_N][0:_PB_M]) map(from: mean[0:_PB_M])
    #pragma omp target teams distribute parallel for num_threads(128)//num_teams(_PB_M/) dist_schedule(static, 4)
      for (j = 0; j < _PB_M; j++)
      {
        mean[j] = 0.0;
        for (i = 0; i < _PB_N; i++)
          mean[j] += data[i][j];
        mean[j] /= float_n;
      }
        
      /* Center the column vectors. 
      *   Determina la distanza dei vari valori della matrice dalla media calcolata nel passo precedente, trovando di fatto la varianza dalla media della feature
      */
      //#pragma omp parallel for
      //#pragma omp target map(to: mean[0:_PB_M]) map(from: data[0:_PB_N][0:_PB_M])
      #pragma omp target teams distribute parallel for collapse(2) num_threads(128) //collapse2)
      for (i = 0; i < _PB_N; i++)
      {
        for (j = 0; j < _PB_M; j++)
          data[i][j] -= mean[j];

      }
        
      /* Calculate the m * m covariance matrix. 
      *   Calcola la matrice di covarianza, sulla diagonale è inserito la varianza di una singola feature (alta i valori della feature sono molto diversi)
      *   Fuori dalla diagonale c'è il rapporto tra le varie feature
      *     - Covarianza > 0 (Positiva): Le due features tendono a crescere (o calare) insieme.
      *     - Covarianza < 0 (Negativa): C'è una relazione inversa. All'aumentare dell'una, l'altra tende a diminuire.
      *     - Covarianza ≈ 0 (Quasi zero): Le due features sono incorrelate.
      */
      //#pragma omp parallel for
      //#pragma omp target map(to: data[0:_PB_N][0:_PB_M]) map(from: symmat[0:_PB_M][0:_PB_M])
      #pragma omp target teams distribute parallel for num_threads(128) //dist_schedule(static, 128)
      for (j1 = 0; j1 < _PB_M; j1++)
      {
        for (j2 = j1; j2 < _PB_M; j2++)
        {
          symmat[j1][j2] = 0.0;

          for (i = 0; i < _PB_N; i++)
            symmat[j1][j2] += data[i][j1] * data[i][j2];


          symmat[j2][j1] = symmat[j1][j2];
        }
      }

    #pragma omp target exit data \
                          map(release: mean[0:_PB_M]) \
                          map(from: symmat[0:_PB_M][0:_PB_M])
                          /*
                          map(release: data[0:_PB_N][0:_PB_M]) \ 
                          map(release: float_n)
*/

}

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
    mean[j] /= float_n;
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


      symmat[j2][j1] = symmat[j1][j2];
    }
}

static
void kernel_covariance_cpu(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i=0, j=0, j1=0, j2=0;
  /* Mean calculation avec réduction manuelle pour meilleure performance */
  #pragma omp parallel for private(i) schedule(static)
  for (j = 0; j < _PB_M; j++)
  {
    DATA_TYPE local_sum = 0.0;
    for (i = 0; i < _PB_N; i++)
      local_sum += data[i][j];
    mean[j] = local_sum / float_n;
  }
  
  /* Center the column vectors avec scheduling dynamique */
  #pragma omp parallel for private(j) schedule(static) collapse(2)
  for (i = 0; i < _PB_N; i++)
  {
    for (j = 0; j < _PB_M; j++)
    {
      data[i][j] -= mean[j];
    }
  }
  
  /* Covariance matrix avec optimisation de la localité des données */
  #pragma omp parallel for private(j2, i) schedule(dynamic, 8) 
  for (j1 = 0; j1 < _PB_M; j1++)
  {
    for (j2 = j1; j2 < _PB_M; j2++)
    {
      DATA_TYPE sum = 0.0;
      //#pragma omp for private(i) schedule(static) reduction(+: sum)
      for (i = 0; i < _PB_N; i++)
      {
        sum += data[i][j1] * data[i][j2];
      }
      symmat[j1][j2] = sum;
      symmat[j2][j1] = sum;
    }
  }
}

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
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  
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
  printf("seq: ");
  polybench_print_instruments;
  #endif



  #ifdef CPU
 /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance_cpu(m, n, float_n,
		     POLYBENCH_ARRAY(data), //polybench_array create a pointer to the variable passed
		     POLYBENCH_ARRAY(symmat),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  printf("cpu: ");
  polybench_print_instruments;
  #endif




  #ifdef GPU
   /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance(m, n, float_n,
		     POLYBENCH_ARRAY(data), //polybench_array create a pointer to the variable passed
		     POLYBENCH_ARRAY(symmat),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  printf("teams: ");
  polybench_print_instruments;
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
