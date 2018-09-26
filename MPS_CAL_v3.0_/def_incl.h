#ifndef DEFINE_H_INCLUDED
#define DEFINE_H_INCLUDED

//#define GPU

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <cstring>
#include <vector>
#include <numeric>
#include <sys/timeb.h>

#include "typedef.h"
#include "common.h"
#include "MOTION.h"

#ifdef CPU_OMP
    #include <omp.h>
#endif

#endif // DEFINE_H_INCLUDED
