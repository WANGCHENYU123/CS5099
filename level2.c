

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include "cluster.h"
#include <sys/time.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define maxIteration 3000
#define threshold 1e-4
#define mGroup 7
char filename[200];

void error_message();
void readDataFromFile(int rank, data_struct *data, int sampleSize, int dimensionSize, int numprocs);
void initialClusters(data_struct *data, data_struct *cluster);
void clean(data_struct* data);
void initGroup();

int main(int argc, char **argv) {
    int numprocs, rank;
    int i;
    struct timeval first, second, lapsed;
    struct timezone tzp;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Status status;
    MPI_Request request;
    
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
    
    /*pData is used for each processor to store n/m*mgroup data */
    data_struct pData;
    pData.dimension = dimensionSize;
    int dataCount;
    int dataStart;
    int dataEnd;
    int offset;
    offset = rank % (numprocs / mGroup);
    if(offset == 0) {
        dataCount = sampleSize / (numprocs / (numprocs / mGroup)) + ((rank / (numprocs / mGroup)) < (sampleSize % (numprocs / (numprocs / mGroup))));
        dataStart = rank / (numprocs / mGroup) * (sampleSize / (numprocs / (numprocs / mGroup))) + (rank / (numprocs / mGroup) < (sampleSize % (numprocs / (numprocs / mGroup))) ? (rank / (numprocs / mGroup)) : (sampleSize % (numprocs / (numprocs / mGroup))));
        dataEnd = dataStart + dataCount;
    }
    else {
        dataCount = sampleSize / (numprocs / (numprocs / mGroup)) + ((rank - offset) / (numprocs / mGroup) < (sampleSize % (numprocs / (numprocs / mGroup))));
        dataStart = (rank  - offset) / (numprocs / mGroup) * (sampleSize / (numprocs / (numprocs / mGroup))) + (((rank - offset) / (numprocs / mGroup)) < (sampleSize % (numprocs / (numprocs / mGroup))) ? ((rank - offset) / (numprocs / mGroup)) : (sampleSize % (numprocs / (numprocs / mGroup))));
        dataEnd = dataStart + dataCount;
    }
    pData.size = dataCount;
    pData.dataset = (double*)malloc(dataCount * dimensionSize * sizeof(double));
    pData.members = (unsigned int*)malloc(dataCount * sizeof(unsigned int));
    for(i = 0; i < pData.size * pData.dimension; i++) {
        pData.dataset[i] = data.dataset[dataStart * pData.dimension + i];
    }
    
    /*pCluster is used for each processor to store k/mgroup clusters*/
    data_struct pCluster;
    pCluster.dimension = dimensionSize;
    int clusterCount;
    int clusterStart;
    int clusterEnd;
    clusterCount = clusterSize / (numprocs / mGroup) + ((rank % (numprocs / mGroup)) < (clusterSize % (numprocs / mGroup)));
    clusterStart = (rank % (numprocs / mGroup)) * (clusterSize / (numprocs / mGroup)) + ((rank % (numprocs / mGroup)) < (clusterSize % (numprocs / mGroup)) ? (rank % (numprocs / mGroup)) : (clusterSize % (numprocs / mGroup)));
    clusterEnd = clusterStart + clusterCount;
    pCluster.size = clusterCount;
    pCluster.dataset = (double*)malloc(clusterCount * dimensionSize * sizeof(double));
    pCluster.members = (unsigned int*)malloc(clusterCount * sizeof(unsigned int));
    
    for(i = 0; i < pCluster.size * pCluster.dimension; i++) {
        pCluster.dataset[i] = clusters.dataset[clusterStart * pCluster.dimension + i];
    }
    
    int iter;
    double* newCentroidsClusPar;
    double* newCentroids;
    int* clusterSizeClusPar;
    double* disFromCenter;
    double* totalDis;
    
    disFromCenter = (double*)malloc(pData.size * pCluster.size * sizeof(double));
    totalDis = (double*)malloc(clusters.size * pData.size * sizeof(double));
    clusterSizeClusPar = (int*)malloc(clusterSize * sizeof(int));
    newCentroidsClusPar = (double*)malloc(dimensionSize * clusterSize * sizeof(double));
    newCentroids = (double*)malloc(dimensionSize * clusterSize * sizeof(double));
    
    /*The processors is partitioned into several groups by MPI_Group*/
    int n = 7;
    const int ranks[n] = {0, 4, 8, 12, 16, 20, 24};
    MPI_Group group1;
    MPI_Group_incl(world_group, n, ranks, &group1);
    MPI_Comm comm1;
    MPI_Comm_create(MPI_COMM_WORLD, group1, &comm1);

    if(rank == 0) {
        printf("\n\nclustering is going to start!\n");
    }

    /*get timestamp*/
    if(rank == 0) {
        gettimeofday(&first, &tzp);
    }
    for(iter = 0; iter < maxIteration; iter++) {
        int isContinue = clusterSize * dimensionSize; //Judging whether stops the clustering
        int i, j, k;
        clock_t start, finish, startCal1, finishCal1, startCal2, finishCal2, startCal3, finishCal3, startCom, finishCom, startCom1, finishCom1, startRecv, finishRecv, startBcast, finishBcast;  // Counting the elapsed time of one iteration
         double duration, durationCal1, durationCal2, durationCal3, durationCom, durationCom1, durationRecv;
        if(rank == 0) {
            if(iter == 0)
            start = clock();
        }
        
        /*The sum of data assigned to one cluster is set to zero*/
        for(i = 0; i < clusterSize * dimensionSize; i++) {
            newCentroidsClusPar[i] = 0;
        }
        if(rank == 0) {
            if(iter == 0)
            startCal1 = clock();
        }
        /*Calculate the distance from pData to pCluster stored in one processor*/
        calDis(&pData, &pCluster, disFromCenter);
        if(rank == 0) {
            if(iter == 0) {
                finishCal1 = clock();
                durationCal1 = (double)(finishCal1 - startCal1) / CLOCKS_PER_SEC;
                printf("step1.1 = %f\n", durationCal1);
            }
        }
        if(rank == 1) {
            if(iter == 0) {
                startCom = clock();
            }
        }
        /*Synchronize k*/
        if(rank >= 1 && rank <=3) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        }
        else if(rank >= 5 && rank <= 7) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 4, rank, MPI_COMM_WORLD);
        }
        else if(rank >= 9 && rank <= 11) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 8, rank, MPI_COMM_WORLD);
        }
        else if(rank >= 13 && rank <= 15) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 12, rank, MPI_COMM_WORLD);
        }
        else if(rank >=17 && rank <= 19) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 16, rank, MPI_COMM_WORLD);
        }
        else if(rank >=21 && rank <= 23) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 20, rank, MPI_COMM_WORLD);
        }
        else if(rank >= 25 && rank <= 27) {
            MPI_Send(&disFromCenter[0], pCluster.size * pData.size, MPI_DOUBLE, 24, rank, MPI_COMM_WORLD);
        }
        /*Finding the minimunm distance from pData to all clusters*/
        else if(rank % (numprocs/mGroup) == 0){
            for(i = 0; i < pCluster.size * pData.size; i++) {
                totalDis[i] = disFromCenter[i];
            }
            if(rank == 0) {
                if(iter == 0) {
                    startRecv = clock();
                }
            }
            for(i = 1; i < (numprocs/mGroup); i++)
                MPI_Recv(&totalDis[i * pCluster.size * pData.size], pCluster.size * pData.size, MPI_DOUBLE, i + rank, i + rank, MPI_COMM_WORLD, &status);
            if(rank == 0) {
                if(iter == 0) {
                    finishRecv = clock();
                    durationRecv = (double)(finishRecv - startRecv) / CLOCKS_PER_SEC;
                    printf("\nElapsed time of recv of %d iteration: %f seconds\n", iter + 1, durationRecv);
                }
            }
            int minIndex = 0;
            double minValue = 0;
            unsigned int *index = pData.members;
            
            /*the number of samples in each centroid is set to be zero*/
            for(i = 0; i < clusterSize; i++) {
                clusterSizeClusPar[i] = 0;
            }
            if(rank == 0) {
                if(iter == 0) {
                    startCal2 = clock();
                }
            }
            for(i = 0; i < pData.size; i++) {
                minValue = totalDis[i * pCluster.size];
                minIndex = 0;
                for(j = 0; j < numprocs / mGroup; j++) {
                    for(k = 0; k < pCluster.size; k++) {
                        if(totalDis[i * pCluster.size + pData.size * j * pCluster.size + k] < minValue) {
                            minValue = totalDis[i * pCluster.size + pData.size * j * pCluster.size + k];
                            minIndex = j * pCluster.size + k;
                        }
                    }
                }
                index[i] = minIndex;
                clusterSizeClusPar[minIndex]++;
                for(j = 0; j < pData.dimension; j++) {
                    newCentroidsClusPar[minIndex * clusters.dimension + j] += pData.dataset[i * pData.dimension + j];
                }
            }
            if(rank == 0) {
                if(iter == 0) {
                    finishCal2 = clock();
                    durationCal2 = (double)(finishCal2 - startCal2) / CLOCKS_PER_SEC;
                    printf("step1.2 = %f\n", durationCal2 );
                }
            }
        }
        if(rank == 1) {
            if(iter == 0) {
                finishCom = clock();
                durationCom = (double)(finishCom - startCom) / CLOCKS_PER_SEC;
                printf("\nElapsed time of send of %d iteration: %f seconds\n", iter + 1, durationCom);
            }
        }
        /*The first processor in each group reduces the sum of data assigned to one cluster and the number of data assigned to one cluster, then updating centroids*/
        if(MPI_COMM_NULL!=comm1) {
            if(rank == 0) {
                if(iter == 0) {
                    startCom1 = clock();
                }
            }
            MPI_Allreduce(newCentroidsClusPar, newCentroids, clusterSize * dimensionSize, MPI_DOUBLE, MPI_SUM, comm1);
            MPI_Allreduce(clusterSizeClusPar, clusters.members, clusterSize, MPI_UNSIGNED, MPI_SUM, comm1);
            if(rank == 0) {
                if(iter == 0) {
                    finishCom1 = clock();
                    durationCom1 = (double)(finishCom1 - startCom1) / CLOCKS_PER_SEC;
                    printf("\nElapsed time of reduce of %d iteration: %f seconds\n", iter + 1, durationCom1);
                }
            }
            if(rank == 0) {
                if(iter == 0)
                    startCal3 = clock();
            }
            for(i = 0; i < clusterSize; i++) {
                for(j = 0; j < dimensionSize; j++) {
                    if(clusters.members[i] == 0) {
                        newCentroids[i * dimensionSize + j] /= (double)(clusters.members[i] + 1);
                    }
                    else {
                        newCentroids[i * dimensionSize + j] /= (double)clusters.members[i];
                    }
                }
            }
            /*If the centroids do not change any more, the iteration stops*/
            for(i = 0; i < clusterSize; i++) {
                for(j = 0; j < dimensionSize; j++) {
                    if(fabs(newCentroids[i * dimensionSize + j] - clusters.dataset[i * dimensionSize + j]) == 0) {
                        isContinue--;
                    }
                }
            }
            if(rank == 0) {
                if(iter == 0) {
                    finishCal3 = clock();
                    durationCal3 = (double)(finishCal3 - startCal3) / CLOCKS_PER_SEC;
                    printf("step 2 = %f\n", durationCal3);
                }
            }
            if(isContinue < 1) {
                if(rank == 0)
                     MPI_Bcast(&isContinue, 1, MPI_INT, 0, MPI_COMM_WORLD);
                break;
            }
            
            /*Broadcast isContinue and send new partitioned centroids to other processors in one group*/
            else {
                for(i = 0; i < clusterSize; i++) {
                    for(j = 0; j < dimensionSize; j++) {
                        clusters.dataset[i * dimensionSize + j] = (double)newCentroids[i * dimensionSize + j];
                    }
                }
               
                MPI_Bcast(&isContinue, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Send(&clusters.dataset[pCluster.size * dimensionSize], pCluster.size * dimensionSize, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD);
                MPI_Send(&clusters.dataset[2 * pCluster.size * dimensionSize], pCluster.size * dimensionSize, MPI_DOUBLE, rank + 2, rank, MPI_COMM_WORLD);
                MPI_Send(&clusters.dataset[3 * pCluster.size * dimensionSize], pCluster.size * dimensionSize, MPI_DOUBLE, rank + 3, rank, MPI_COMM_WORLD);
              
                for(i = 0; i < pCluster.size; i++) {
                    for(j = 0; j < dimensionSize; j++) {
                        pCluster.dataset[i * dimensionSize + j] = clusters.dataset[i * dimensionSize + j];
                    }
                }
            }
        }
        else if(rank % (numprocs/mGroup) == 1) {
            if(isContinue < 1) {
                break;
            }
            else {
                MPI_Recv(&pCluster.dataset[0], pCluster.size * dimensionSize, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &status);
            }
        }
        else if(rank % (numprocs/mGroup) == 2) {
            if(isContinue < 1) {
                break;
            }
            else {
                MPI_Recv(&pCluster.dataset[0], pCluster.size * dimensionSize, MPI_DOUBLE, rank - 2, rank - 2, MPI_COMM_WORLD, &status);
            }
        }
        else if(rank % (numprocs/mGroup) == 3) {
            if(isContinue < 1) {
                break;
            }
            else {
                MPI_Recv(&pCluster.dataset[0], pCluster.size * dimensionSize, MPI_DOUBLE, rank - 3, rank - 3, MPI_COMM_WORLD, &status);
            }
        }
        if(rank == 0) {
            if(iter == 0) {
                printf("\nElapsed time of calculation of %d iteration: %f seconds", iter + 1, durationCal2 + durationCal1 + durationCal3);
                printf("\nElapsed time of communication of %d iteration: %f seconds", iter + 1, durationCom + durationCom1 + durationRecv);
            }
        }
        /*Counting time of one iteration*/
        if(rank == 0) {
            if(iter == 0) {
                finish = clock();
                duration = (double)(finish - start) / CLOCKS_PER_SEC;
                printf("\nelapsed time of %d iteration: %f seconds", iter + 1, duration);
            }
        }
    }
    
    /*Free memory*/
    free(newCentroidsClusPar);
    free(newCentroids);
    free(disFromCenter);
    free(totalDis);
    free(clusterSizeClusPar);
    
    MPI_Barrier(comm1);
    
    /*Recording clustering results and elaspsed time of all iterations*/
    if(rank == 0) {
        printf("\n\n finished after %d iterations \n", iter);
        gettimeofday(&second, &tzp);
        
        if(first.tv_usec > second.tv_usec) {
            second.tv_usec += 1000000;
            second.tv_sec--;
        }
        lapsed.tv_usec = second.tv_usec - first.tv_usec;
        lapsed.tv_sec = second.tv_sec - first.tv_sec;
        
        printf("\ntime elapsed: %d.%06dsec\n\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);
        printf("cluster sizes\n");
        for(int i = 0; i < clusterSize; i++) {
            printf("cluster%d: %d\n", i, clusters.members[i]);
        }
        printf("\n");
    }
    clean(&pData);
    clean(&data);
    clean(&pCluster);
    clean(&clusters);
    if(rank == 0) {
        printf("program has finieshed! \n");
    }
    MPI_Finalize();
    exit(0);
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





