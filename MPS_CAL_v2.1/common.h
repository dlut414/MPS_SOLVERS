/*
LICENCE
*/
//common.h
//file to save common parameters
#ifndef COMMON_H
#define COMMON_H

#include "typedef.h"

#define CPU_OMP
#define OMP_THREADS 8
#define STATIC_CHUNK 500
#define DYNAMIC_CHUNK 100

static const char LOG_NAME[256] = "./out/LOG.txt"; // place for output

static const real PI = 3.14159265358979;
static const real OVERPI = 0.31830988618379;
static const real3 G = {0.0 , 0.0 , -9.81};

static const real EPS = 1e-10;
static const real RMAX = 1e100;//maximum of residual

static const real EPS_BY_EPS = EPS * EPS;
#endif //COMMON_H
