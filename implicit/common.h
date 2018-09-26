/*
LICENCE
*/
//common.h
//file to save common parameters
#ifndef COMMON_H
#define COMMON_H

#include "typedef.h"

#define OMP_THREADS 4

static const double PI = 3.14159265358979;
static const double OVERPI = 0.31830988618379;
static const double3 G = {0.0 , 0.0 , -9.81};

static const double EPS = 1e-10;
static const double RMAX = 1e100;//maximum of residual

static const double EPS_BY_EPS = EPS * EPS;
#endif //COMMON_H
