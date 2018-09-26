/*
LICENCE
*/
//renderer_gpu_cudaker.h
///defination of cuda kernel functions
#ifndef RENDERER_GPU_CUDAKER_H_INCLUDED
#define RENDERER_GPU_CUDAKER_H_INCLUDED

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "../typedef.h"

namespace cudaker
{

void dev_calVertex_n( mytype::real* const dev_r_vertex_n,
                const mytype::integer* const dev_i_markVox,
                const mytype::integer* const dev_i_voxToCell,
                const mytype::real3* const dev_r3_verList,
                const mytype::real3* const dev_d3_pos,
                const mytype::integer* const dev_i_type,
                const mytype::integer* const dev_i_link_cell,
                const mytype::integer* const dev_i_cell_start,
                const mytype::integer* const dev_i_cell_end,
                const mytype::real d_rzero,
                const mytype::integer i_nVertex,
                const mytype::integer i_marked,
                const mytype::GEOMETRY geo );

void dev_calNorm     ( mytype::real3* const dev_r3_vertex_norm,
                 const mytype::integer* const dev_i_markVox,
                 const mytype::real* const dev_r_vertex_n,
                 const mytype::int3 i3_dim,
                 const mytype::integer i_nVertex,
                 const mytype::integer i_marked );

void dev_zeroVert   (mytype::real* const dev_r_vertex_n, mytype::real3* const dev_r3_norm,
                     const mytype::integer* const dev_i_cellInFluid,
                     const mytype::integer* const dev_i_voxToCell,
                     const mytype::integer i_nVert);

}

#endif // RENDERER_GPU_CUDAKER_H_INCLUDED
