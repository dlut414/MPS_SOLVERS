/*
LICENCE
*/
//main.cpp
//main function
///main function
#include "def_incl.h"
#include "MPS_GPU.h"
#include "DISPLAY.h"

using namespace std;
using namespace mytype;

int main(int argc, char** argv)
{
    system("mkdir out");
#ifdef CPU_OMP
    omp_set_num_threads(OMP_THREADS);
#endif
    DISPLAY::myInit(argc, argv);
    DISPLAY::myMainLoop();

    return 0;
}









