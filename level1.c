
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include "cluster.h"
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define maxIteration 3000
#define threshold 1e-4
char filename[200];

void error_message();
void readDataFromFile(int rank, data_struct *data, int sampleSize, int dimensionSize, int numprocs);
void initialClusters(data_struct *data, data_struct *cluster);
void clean(data_struct* data);

int main(int argc, char **argv) {
    int numprocs, rank;
    struct timeval first, second, lapsed;
    struct timezone tzp;
    int i, j;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    if(argc < 5 && rank == 0) {
        error_message();
        return 0;
    }
    
    int sampleSize = atoi(argv[1]); // the size of samples
    int dimensionSize = atoi(argv[2]); // the size of dimensions
    int clusterSize = atoi(argv[3]); // the size of clusters
    strcat(filename, argv[4]);
    
    /* Data is the original dataset, clusters is the initialised clusters*/
    data_struct data;
    data_struct clusters;
    
    /*Memory allocation*/
    data.dimension = dimensionSize;
    data.size = sampleSize;
    data.dataset = (double*)malloc(sampleSize * dimensionSize * sizeof(double));
    data.members = (unsigned int*)malloc(sampleSize * sizeof(unsigned int));
    
    clusters.dimension = dimensionSize;
    clusters.size = clusterSize;
    clusters.dataset = (double*)malloc(clusterSize * dimensionSize * sizeof(double));
    clusters.members = (unsigned int*)malloc(clusterSize * sizeof(unsigned int));
    
    /*Read data from file*/
    readDataFromFile(rank, &data, sampleSize, dimensionSize, numprocs);
    /*Initial clusters and broadcast to all processors*/
    if(rank == 0) {
        initialClusters(&data, &clusters);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(clusters.dataset, clusterSize * dimensionSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /*pData is used for each processor to store n/m data */
    data_struct pData;
    pData.dimension = dimensionSize;
    
    int local_count = sampleSize / numprocs + (rank < (sampleSize % numprocs));
    int start = rank * (sampleSize / numprocs) + (rank < (sampleSize % numprocs) ? rank : (sampleSize % numprocs));
    int end = start + local_count;
    pData.size = local_count;
    pData.dataset = (double*)malloc(local_count * dimensionSize * sizeof(double));
    pData.members = (unsigned int*)malloc(local_count * sizeof(unsigned int));
    for(i = 0; i < pData.size * pData.dimension; i++) {
        pData.dataset[i] = data.dataset[start * dimensionSize + i];
    }
    
    int iter;
    double* newCentroids; // the sum of data assigned to one cluster
    double* tempNewCentroids; //temperary sum of data assigned to one cluster
    int* tempClusterSize; //temperary number of data assigned to one cluster
    
    /*Memory allocation*/
    tempClusterSize = (int*)malloc(clusterSize * sizeof(int));
    newCentroids = (double*)malloc(dimensionSize * clusterSize * sizeof(double));
    tempNewCentroids = (double*)malloc(dimensionSize * clusterSize * sizeof(double));
    
    /*the number of samples in each centroid is set to be zero*/
    for(i = 0; i < clusterSize; i++) {
        tempClusterSize[i] = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
        printf("\n\nclustering is going to start!\n");
        printf("max iteration = %d \n", maxIteration);
    }
    
    /*get timestamp*/
    if(rank == 0) {
        gettimeofday(&first, &tzp);
    }
    for(iter = 0; iter < maxIteration; iter++) {
        int isContinue = clusterSize * dimensionSize; //Judging whether stops the clustering
        int i, j;
        clock_t start, finish, startCal1, finishCal1, startCal2, finishCal2, startCom, finishCom ;// Counting the elapsed time of one iteration
        double duration, durationCal1, durationCal2, durationCom;
        if(rank == 0) {
            if(iter == 0)
            start = clock();
        }
        
        /*The sum of data assigned to one cluster is set to zero*/
        for(i = 0; i < clusterSize * dimensionSize; i++) {
            newCentroids[i] = 0;
        }
        if(rank == 0) {
            if(iter == 0)
            startCal1 = clock();
        }
        /*Kmeans algorithm implementation*/
        kmeans(&pData, &clusters, newCentroids);
        if(rank == 0) {
            if(iter == 0) {
                finishCal1 = clock();
                durationCal1 = (double)(finishCal1 - startCal1) / CLOCKS_PER_SEC;
            }
        }
        if(rank == 0) {
            if(iter == 0) {
                startCom = clock();
            }
        }
        /*Synchronize n*/
        MPI_Allreduce(newCentroids, tempNewCentroids, clusterSize * dimensionSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(clusters.members, tempClusterSize, clusterSize, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0) {
            if(iter == 0) {
                finishCom = clock();
                durationCom = (double)(finishCom - startCom) / CLOCKS_PER_SEC;
                printf("Reduce time: %f seconds\n", durationCal1);
            }
        }
        for(i = 0; i < clusterSize; i++) {
            clusters.members[i] = tempClusterSize[i];
        }
        if(rank == 0) {
            if(iter == 0) {
                startCal2 = clock();
            }
        }
        /*Updating centroids*/
        for(i = 0; i < clusterSize; i++) {
            for(j = 0; j < dimensionSize; j++) {
                if(clusters.members[i] == 0) {
                    tempNewCentroids[i * dimensionSize + j] /= (double) (clusters.members[i] + 1);
                }
                else {
                    tempNewCentroids[i * dimensionSize + j] /= (double) clusters.members[i];
                }
            }
        }
        /*If the centroids do not change any more, the iteration stops*/
        for(i = 0; i < clusterSize; i++) {
            for(j = 0; j < dimensionSize; j++) {
                if(fabs(tempNewCentroids[i * dimensionSize + j] - clusters.dataset[i * dimensionSize + j]) == 0) {
                    isContinue--;
                }
            }
        }
        if(rank == 0) {
            if(iter == 0) {
                finishCal2 = clock();
                durationCal2 = (double)(finishCal2 - startCal2) / CLOCKS_PER_SEC;
            }
        }
        if(rank == 0) {
            if(iter == 0) {
                printf("Calculation time: %f seconds\n", durationCal2 + durationCal1);
            }
        }
        if(isContinue < 1) {
            break;
        }
        
        /*The new centroids are given to clusters for next iteration*/
        for(i = 0; i < clusterSize; i++) {
            for(j = 0; j < dimensionSize; j++) {
                clusters.dataset[i * dimensionSize + j] = (double)tempNewCentroids[i * dimensionSize + j];
            }
        }
        
        /*Counting time of one iteration*/
        if(rank == 0) {
            if(iter == 0) {
                finish = clock();
                duration = (double)(finish - start) / CLOCKS_PER_SEC;
                printf("Elapsed time of %d iteration: %f seconds\n", iter + 1, duration);
            }
        }
//        if(rank == 0)
//        printf("iter = %d\n", iter);
//        if(rank == 0)
//        for(i = 0; i < clusterSize; i++) {
//            printf("Cluster%d: %d\n", i, clusters.members[i]);
//        }
    }
    
    /*Free memory*/
    free(newCentroids);
    free(tempClusterSize);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /*Recording clustering results and elaspsed time of all iterations*/
    if(rank == 0) {
        /*
        printf("\n\nFinished after %d iterations\n", iter);
         */
        gettimeofday(&second, &tzp);
        
        if(first.tv_usec > second.tv_usec) {
            second.tv_usec += 1000000;
            second.tv_sec--;
        }
        lapsed.tv_usec = second.tv_usec - first.tv_usec;
        lapsed.tv_sec = second.tv_sec - first.tv_sec;
        
        printf("Time elapsed: %d.%06d seconds\n\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);
//        printf("Cluster sizes\n");
//        for(i = 0; i < clusterSize; i++) {
//            printf("Cluster%d: %d\n", i, clusters.members[i]);
//        }
    }
    clean(&pData);
    clean(&data);
    clean(&clusters);
    /*
    if(rank == 0) {
        printf("Program has finished!\n");
    }
     */
    MPI_Finalize();
}

void error_message() {
    char *help = "Four arguments are required\n"
    "First is number of samples\n"
    "Second is number of dimensions\n"
    "Third is number of clusters\n"
    "The last one is filename";
    printf("%s", help);
}

void readDataFromFile(int rank, data_struct *data, int sampleSize, int dimensionSize, int numprocs) {
    FILE* fread;
    if(NULL == (fread = fopen(filename, "r"))) {
        printf("open file(%s) error!\n", filename);
        exit(0);
    }
    double a = 0;
    int count;
    int row, dim;
    for(row = 0; row < sampleSize; row++) {
        for(dim = 0; dim < dimensionSize; dim++) {
            if(fscanf(fread, "%lf", &a) == 1) {
                if(row < sampleSize) {
                    data->dataset[row * dimensionSize + dim] = a;
                    data->members[row] = 0;
                }
            }
            else {
                printf("fscanf error: %d\n", row);
            }
        }
    }
}

/*Initial clusters from original dataset*/
void initialClusters(data_struct *data, data_struct *cluster) {
    int pick = 0;
    int n = cluster->dimension;
    int m = cluster->size;
    int samples = data->size;
    double *tempCentroids = cluster->dataset;
    double *tempDataset = data->dataset;
    unsigned int *tempSizes = data->members;
    int i, j;
    int step = samples / m;

    for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
            tempCentroids[i * n + j] = tempDataset[pick * n + j];
        }
        pick += step;
    }
}
void clean(data_struct* data) {
    free(data->dataset);
    free(data->members);
}
