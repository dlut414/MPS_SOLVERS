/*
LICENCE
*/
//mps_gpu_cudaker.h
///defination of cuda kernel functions
#ifndef MPS_GPU_CUDAKER_H
#define MPS_GPU_CUDAKER_H

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

void dev_cal_n(mytype::real* const d_n,
         const mytype::real3* const d3_pos,
         const mytype::integer* const i_cell_list,
         const mytype::integer* const i_link_cell,
         const mytype::integer* const i_cell_start,
         const mytype::integer* const i_cell_end,
         const mytype::real d_rzero,
         const mytype::integer i_num_cells,
         const mytype::integer i_np);

void dev_calDash(mytype::real3* const d3_vel, mytype::real3* const d3_pos, mytype::real3* const d3_tmp,
           const mytype::real* const d_press, const mytype::integer* const i_type,
           const mytype::integer* const i_cell_list, const mytype::integer* const i_link_cell,
           const mytype::integer* const i_cell_start, const mytype::integer* const i_cell_end,
           const mytype::real d_dt,
           const mytype::real d_one_over_rho,
           const mytype::real d_one_over_nzero,
           const mytype::real d_rzero,
           const mytype::integer i_dim,
           const mytype::integer i_num_cells,
           const mytype::integer i_np);

void dev_calPres_expl(mytype::real* const d_press,
                const mytype::real* const d_n,
                const mytype::integer* const i_type,
                const mytype::integer* const i_normal,
                const mytype::real d_one_over_alpha,
                const mytype::real d_nzero,
                const mytype::real d_one_over_nzero,
                const mytype::integer i_np);

void dev_calVisc_expl(mytype::real3* const d3_vel, mytype::real3* const d3_pos, mytype::real3* const d3_tmp,
                const mytype::real* const d_press,
                const mytype::integer* const i_type,
                const mytype::integer* const i_cell_list,
                const mytype::integer* const i_link_cell,
                const mytype::integer* const i_cell_start,
                const mytype::integer* const i_cell_end,
                const mytype::real d_dt,
                const mytype::real d_2bydim_over_nzerobylambda,
                const mytype::real d_rlap,
                const mytype::real d_niu,
                const mytype::integer i_num_cells,
                const mytype::integer i_np);

void dev_calCol(mytype::real3* const d3_vel, mytype::real3* const d3_pos, mytype::real3* const d3_tmp,
          const mytype::integer* const i_type,
          const mytype::integer* const i_cell_list,
          const mytype::integer* const i_link_cell,
          const mytype::integer* const i_cell_start,
          const mytype::integer* const i_cell_end,
          const mytype::real d_dt,
          const mytype::real d_col_dis,
          const mytype::real d_col_rate,
          const mytype::integer i_num_cells,
          const mytype::integer i_np);

}

#endif // MPS_GPU_CUDAKER_H
