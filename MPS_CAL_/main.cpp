/*
LICENCE
*/
//main.cpp
//main function
///main function
#include <cuda_runtime.h>
#include "def_incl.h"
#include "MPS_GPU.h"
#include "Renderer.h"

using namespace std;
using namespace mytype;

int main(int argc, char** argv)
{
    system("mkdir out");
#ifdef CPU_OMP
    omp_set_num_threads(OMP_THREADS);
#endif

#ifdef GPU_CUDA
    cudaDeviceReset();
#endif
    Renderer::InitGL     (argc, argv);
    Renderer::MainLoop   ();
    Renderer::Final      ();
#ifdef GPU_CUDA
    cudaDeviceReset();
#endif
    return 0;
}









