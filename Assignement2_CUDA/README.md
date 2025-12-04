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
    - CUSTOM                5x5 (Custom matrix with fixed values)
- type of execution:
    - SEQ : compile the code using the sequential kernel
    - CPU : compile the code using the cpu parallelization
    - GPU : compile the code using the gpu paralelization
- PRINT : print the covariance matrix at the end of execution (use only with CUSTOM dataset)


The makefile have two different version to compile with gcc o clang (there is commented code in ./OpenMP/utilities/common.mk)

---

### Testing

To test the correctness of the calculation running the code with this code, change the type of execution to check all the results with the fixed dataset:
 make EXT_CFLAGS="-DPOLYBENCH_TIME -DGPU -DCUSTOM -DPRINT" clean all run

_The CUSTOM dataset run the code using this matrix:_

         Original Matrix																		
        1	2	5	10	1														
        2	4	4	10	0														
        3	6	3	10	1														
        4	8	2	10	0														
        5	10	1	10	3														
 Mean	3	6	3	10	1														

_These are the steps for the calculation:_

    Deviation matrix transpose (D)				Deviation matrix transpose (D_t)			D_t  x  D 				
    -2	-4	2	0	0			                -2	-1	0	1	2			                10	20	-10	0	4
    -1	-2	1	0	-1			                -4	-2	0	2	4			                20	40	-20	0	8
    0	0	0	0	0			                2	1	0	-1	-2			                -10	-20	10	0	-4
    1	2	-1	0	-1			                0	0	0	0	0			                0	0	0	0	0
    2	4	-2	0	2			                0	-1	0	-1	2			                4	8	-4	0	6


_Results:_

    Covariance Matrix				
    2.5	    5	    -2.5	0	    1
    5	    10	    -5	    0	    2
    -2.5	-5	    2.5	    0	    -1
    0	    0	    0	    0	    0
    1	    2	    -1	    0	    1.5

---

### Profiling

Using the make directives:
- profile_val : run profiling with valgrind (takes a lot of time)
- profile_gprof : run profiling with gprof (tested and working)



 
