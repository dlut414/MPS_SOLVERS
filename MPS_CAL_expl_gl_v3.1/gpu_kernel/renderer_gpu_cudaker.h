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
                const mytype::real3* const dev_r3_verList,
                const mytype::real3* const dev_d3_pos,
                const mytype::integer* const dev_i_type,
                const mytype::integer* const dev_i_link_cell,
                const mytype::integer* const dev_i_cell_start,
                const mytype::integer* const dev_i_cell_end,
                const mytype::real d_rzero,
                const mytype::integer i_nVertex,
                const mytype::GEOMETRY geo );

void dev_calTriangle ( mytype::real3* const dev_r3_triangle,
                       mytype::real* const dev_r_alpha,
                 const mytype::real3* const dev_r3_verList,
                 const mytype::real* const dev_r_vertex_n,
                 const mytype::integer* const dev_i_voxList,
                 const mytype::real r_iso,
                 const mytype::integer i_nVoxel,
                 const uint* const dev_u_numVerTable,
                 const uint* const dev_u_triTable );

void dev_calNorm_1   ( mytype::real3* const dev_r3_vertex_norm,
                 const mytype::real* const dev_r_vertex_n,
                 const mytype::int3 i3_dim,
                 const mytype::integer i_nVertex );

void dev_calNorm_2   ( mytype::real3* const dev_r3_norm,
                 const mytype::real3* const dev_r3_triangle,
                 const mytype::real3* const dev_r3_verList,
                 const mytype::real3* const dev_r3_vertex_norm,
                 const mytype::real* const dev_r_vertex_n,
                 const mytype::integer* const dev_i_voxList,
                 const mytype::real r_iso,
                 const mytype::integer i_nVoxel,
                 const uint* const dev_u_numVerTable,
                 const uint* const dev_u_triTable );

void dev_calNorm_legacy( mytype::real3* const dev_r3_norm,
                 const mytype::real3* const dev_r3_triangle,
                 const mytype::real* const dev_r_alpha,
                 const mytype::real3* const dev_d3_pos,
                 const mytype::integer* const dev_i_type,
                 const mytype::integer* const dev_i_link_cell,
                 const mytype::integer* const dev_i_cell_start,
                 const mytype::integer* const dev_i_cell_end,
                 const mytype::real d_rzero,
                 const mytype::integer i_nMaxEdge,
                 const mytype::GEOMETRY geo );

}

#endif // RENDERER_GPU_CUDAKER_H_INCLUDED
