# HPC_assignement

Repository where are stored the assignement of the group compose of:
- Chaima Ben Chouchene
- Yassine Nasri
- Amira Oueslati
- Davide Gazzadi

The team will work on the project datamining/covariance

To compile the project make sure to follow this steps:
1. cd ./OpenMP/datamining/covariance/
2. make EXT_CFLAGS="-DPOLYBENCH_TIME -DDATASET_MINI" clean all run

Possible options in EXT_CFLAGS:
- POLYBENCH_TIME : make the code print the time of execution of the program
- dataset specification: 
    - MINI_DATASET          32x32
    - SMALL_DATASET         500x500
    - STANDARD_DATASET      1000x1000 (Default)
    - LARGE_DATASET         2000x2000
    - EXTRALARGE_DATASET    4000x4000
    - CUSTOM                6x6 (Custom matrix with fixed values)
- SEQ : compile without using the kernels, but the sequential code
- PRINT : print the covariance matrix at the end of execution (use only with CUSTOM dataset)
- TILING : compile the code using the tiling method on covariance calculation, otherwise wil be use normal cuda parallelization

---

### Testing

To test the correctness of the calculation running the code with this code, change the type of execution to check all the results with the fixed dataset:
 make EXT_CFLAGS="-DPOLYBENCH_TIME -DGPU -DCUSTOM -DPRINT" clean all run

_The CUSTOM dataset run the code using this matrix:_

$$\mathbf{X} = \begin{pmatrix} 1 & 2 & 5 & 6 & 0 & 10 \\\ 2 & 4 & 5 & 5 & 1 & 10 \\\ 3 & 6 & 5 & 4 & 0 & 10 \\\ 4 & 8 & 5 & 3 & 1 & 10 \\\ 5 & 10 & 5 & 2 & 0 & 10 \\\ 6 & 12 & 5 & 1 & 1 & 10 \end{pmatrix}$$

---

## 1. Mean vector

Mean vector ($\boldsymbol{\mu}$) contains mean of each column :

$$\boldsymbol{\mu} = \begin{pmatrix} 3.5 & 7.0 & 5.0 & 3.5 & 0.5 & 10.0 \end{pmatrix}$$

---

## 2. Deviation Matrix

Deviation Matrix ($\mathbf{D}$) is obtained subtracting the corrispondent mean from each value in the matrix ($\mathbf{D} = \mathbf{X} - \boldsymbol{\mu}$):

$$\mathbf{D} = \begin{pmatrix} -2.5 & -5.0 & 0.0 & 2.5 & -0.5 & 0.0 \\\ -1.5 & -3.0 & 0.0 & 1.5 & 0.5 & 0.0 \\\ -0.5 & -1.0 & 0.0 & 0.5 & -0.5 & 0.0 \\\ 0.5 & 1.0 & 0.0 & -0.5 & 0.5 & 0.0 \\\ 1.5 & 3.0 & 0.0 & -1.5 & -0.5 & 0.0 \\\ 2.5 & 5.0 & 0.0 & -2.5 & 0.5 & 0.0 \end{pmatrix}$$

---

## 3. Covariance Matrix

Covariance matrix ($\mathbf{C}$) is calculated like $\mathbf{C} = \frac{1}{n-1} \mathbf{D}^T \mathbf{D}$, where $n=6$ (number of observation) and $n-1 = 5$.

$$\mathbf{C} = \begin{pmatrix} 3.5 & 7.0 & 0.0 & -3.5 & 0.3 & 0.0 \\\ 7.0 & 14.0 & 0.0 & -7.0 & 0.6 & 0.0 \\\ 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\ -3.5 & -7.0 & 0.0 & 3.5 & -0.3 & 0.0 \\\ 0.3 & 0.6 & 0.0 & -0.3 & 0.3 & 0.0 \\\ 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \end{pmatrix}$$

---

### Profiling

Using the make directives:
- profile : run profiling with gprof (tested and working)



 
