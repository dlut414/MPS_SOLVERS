/*
LICENCE
*/
//cuda_ker.h
///defination of cuda kernel functions
#ifndef CUDA_KER_H
#define CUDA_KER_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "typedef.h"

namespace cudaker
{

void VecCpy(mytype::real* const des,
      const mytype::real* const src,
      const mytype::integer n);

void Axpy(mytype::real* const z,
    const mytype::real* const x,
    const mytype::real* const y,
    const mytype::real a, const mytype::integer n);

void MatVec(mytype::real* const des,
      const mytype::real* const mat,
      const mytype::real* const vec,
      const mytype::integer n);

void VecVec(mytype::real& des,
      const mytype::real* const vec1,
      const mytype::real* const vec2,
      const mytype::integer n);

void CG(const mytype::real* const A, mytype::real* const x, const mytype::real* const b, const mytype::integer n);

}

#endif // CUDA_KER_H
