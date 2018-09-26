/*
LICENCE
*/
//main.cpp
//main function
///main function
#include "def_incl.h"
#include "MPS_GPU_EXPL.h"

using namespace std;
using namespace mytype;

int main()
{
    system("mkdir out");
#ifdef CPU_OMP
    omp_set_num_threads(OMP_THREADS);
#endif

    MPS_GPU_EXPL mps_expl_test;
    mps_expl_test.mps_cal();

    return 0;
}
