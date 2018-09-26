/*
LICENCE
*/
//common.h
//file to save common parameters
#ifndef COMMON_H
#define COMMON_H

#include "typedef.h"

///debug
#define DEBUG

///dimention
#define DIM 3

///omp configuration
#define CPU_OMP
#define OMP_THREADS 8
#define STATIC_CHUNK 500
#define STATIC_CHUNK_S 100
#define DYNAMIC_CHUNK 100
#define DYNAMIC_CHUNK_S 50

///gpu configuration
#define GPU_CUDA
//#define GPU_CUDA_
#define NUM_MP 4
#define NUM_THREADS 1024

#define A(i,j,ld) A[i*ld+j]
/*
#define NUM_BLOCKS 4
#define NUM_THREADS 192
*/

namespace mytype
{

static const char LOG_NAME[256] = "./out/LOG.txt"; // place for output

static const real PI = 3.14159265358979f;
static const real OVERPI = 0.31830988618379f;
static const real3 G = {0.0f , 0.0f , -9.18f};

static const real EPS = 1e-10;
static const real RMAX = 1e100;//maximum of residual

static const real EPS_BY_EPS = EPS * EPS;

}
#endif //COMMON_H
