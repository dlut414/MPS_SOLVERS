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

void dev_sort_i(mytype::integer* const des,
          const mytype::integer* const dev_p,
          const mytype::integer* const dev_i_index,
          const mytype::integer n);

void dev_sort_d(mytype::real* const des,
          const mytype::real* const dev_p,
          const mytype::integer* const dev_i_index,
          const mytype::integer n);

void dev_sort_i3(mytype::int3* const des,
           const mytype::int3* const dev_p,
           const mytype::integer* const dev_i_index,
           const mytype::integer n);

void dev_sort_d3(mytype::real3* const des,
           const mytype::real3* const dev_p,
           const mytype::integer* const dev_i_index,
           const mytype::integer n);

void dev_sort_normal(mytype::integer* const dev_p,
               const mytype::integer* const dev_i_index,
               const mytype::integer n);

}

#endif // CUDA_KER_H
