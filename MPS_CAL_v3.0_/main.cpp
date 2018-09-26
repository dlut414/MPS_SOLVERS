/*
LICENCE
*/
//main.cpp
//main function
///main function
#include "def_incl.h"
#include "MPS_CPU_EXPL.h"
#include "MPS_CPU_IMPL.h"
using namespace std;

int main()
{
    system("mkdir out");
#ifdef CPU_OMP
    omp_set_num_threads(OMP_THREADS);
#endif

#ifndef GPU

    MPS_CPU_EXPL mps_expl_test;
    mps_expl_test.mps_cal();
/*
    MPS_CPU_IMPL mps_impl_test;
    mps_impl_test.mps_cal();
*/
#else

#endif
    return 0;
}
