#include "kmeans.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double euclideanDis(double *p1, double *p2, int length) {
    double dis = 0;
    for(int i = 0; i < length; i++) {
        dis += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return(dis);
}

void kmeans(data_struct *data, data_struct *clusters, double *newCentroids) {
    double tmpDis = 0;
    int tmpIndex = 0;
    double *dataset = data->dataset;
    double *centroids = clusters->dataset;
    unsigned int *index = data->members;
    unsigned int *clusterSize = clusters->members;
    double minDis;
    
    for(int i = 0; i < clusters->size; i++) {
        clusterSize[i] = 0;
    }
    
    for(int i = 0; i < data->size; i++) {
        tmpDis = 0;
        tmpIndex = 0;
        minDis = 10000;
        for(int k = 0; k < clusters->size; k++) {
            tmpDis = euclideanDis(dataset + i * data->dimension, centroids + k * clusters->dimension, data->dimension);
            if(tmpDis < minDis) {
                minDis = tmpDis;
                tmpIndex = k;
            }
        }
        index[i] = tmpIndex;
        clusterSize[tmpIndex]++;
        for(int j = 0; j < data->dimension; j++) {
            newCentroids[tmpIndex * clusters->dimension + j] += dataset[i * data->dimension + j];
        }
    }
}

void calDis(data_struct *data, data_struct *clusters, double *disFromCenter) {
    double tmpDis = 0;
    double *dataset = data->dataset;
    double *centroids = clusters->dataset;
    for(int i = 0; i < data->size; i++) {
        tmpDis = 0;
        for(int k = 0; k < clusters->size; k++) {
            tmpDis = euclideanDis(dataset + i * data->dimension, centroids + k * clusters->dimension, data->dimension);
            disFromCenter[clusters->size * i + k] = tmpDis;
        }
    }
}


