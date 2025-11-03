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

In this compile example it's specifiend DATASET_MINI as the dataset to use, but it's possible to leave it blank and it will use the Default one.
Using the mini dataset will be useful while doing the test, because it's faster.
The POLYBENCH_TIME definition is added to make the code print the time of execution of the program
