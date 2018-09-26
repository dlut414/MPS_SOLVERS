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
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../typedef.h"
#include "../matrix_COO.h"

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

void dev_sort_all(mytype::integer* const dev_i_id_tmp,
                  mytype::integer* const dev_i_type_tmp,
                  mytype::integer* const dev_i_cell_list_tmp,
                  mytype::real* const dev_r_press_tmp,
                  mytype::real* const dev_r_n_tmp,
                  mytype::real3* const dev_r3_pos_tmp,
                  mytype::real3* const dev_r3_vel_tmp,
                  mytype::integer* const dev_i_id,
                  mytype::integer* const dev_i_type,
                  mytype::integer* const dev_i_cell_list,
                  mytype::real* const dev_r_press,
                  mytype::real* const dev_r_n,
                  mytype::real3* const dev_r3_pos,
                  mytype::real3* const dev_r3_vel,
            const mytype::integer* const dev_i_index,
            const mytype::integer i_np);

void dev_cal_n( mytype::real* const dev_r_n,
          const mytype::real3* const dev_r3_pos,
          const mytype::integer* const dev_i_type,
          const mytype::integer* const dev_i_cell_list,
          const mytype::integer* const dev_i_link_cell,
          const mytype::integer* const dev_i_cell_start,
          const mytype::real r_rzero,
          const mytype::integer i_num_cells,
          const mytype::integer i_np );

void dev_calDash( mytype::real3* const dev_r3_vel,
                  mytype::real3* const dev_r3_pos,
            const mytype::real* const dev_r_press,
            const mytype::integer* const dev_i_type,
            const mytype::integer* const dev_i_cell_list,
            const mytype::integer* const dev_i_link_cell,
            const mytype::integer* const dev_i_cell_start,
            const mytype::real r_dt,
            const mytype::real r_one_over_rho,
            const mytype::real r_one_over_nzero,
            const mytype::real r_rzero,
            const mytype::integer i_dim,
            const mytype::integer i_num_cells,
            const mytype::integer i_np );

void dev_calPres_expl( mytype::real* const dev_r_press,
                 const mytype::real* const dev_r_n,
                 const mytype::integer* const dev_i_type,
                 const mytype::real r_one_over_alpha,
                 const mytype::real r_nzero,
                 const mytype::real r_one_over_nzero,
                 const mytype::integer i_np );

void dev_calVisc_expl( mytype::real3* const dev_r3_vel,
                       mytype::real3* const dev_r3_pos,
                 const mytype::real* const dev_r_press,
                 const mytype::integer* const dev_i_type,
                 const mytype::integer* const dev_i_cell_list,
                 const mytype::integer* const dev_i_link_cell,
                 const mytype::integer* const dev_i_cell_start,
                 const mytype::real r_dt,
                 const mytype::real r_2bydim_over_nzerobylambda,
                 const mytype::real r_rlap,
                 const mytype::real r_niu,
                 const mytype::integer i_num_cells,
                 const mytype::integer i_np );

void dev_calCol( mytype::real3* const dev_r3_vel,
                 mytype::real3* const dev_r3_pos,
           const mytype::integer* const dev_i_type,
           const mytype::integer* const dev_i_cell_list,
           const mytype::integer* const dev_i_link_cell,
           const mytype::integer* const dev_i_cell_start,
           const mytype::real r_dt,
           const mytype::real r_col_dis,
           const mytype::real r_col_rate,
           const mytype::integer i_num_cells,
           const mytype::integer i_np );

void dev_buildPoisson(    mytype::matrix_COO<mytype::integer, mytype::real>* const A,
                          mytype::real* const x,
                          mytype::real* const r0,
                          mytype::real* const r,
                          mytype::real* const p,
                          mytype::real* const s,
                          mytype::real* const As,
                    const mytype::real* const n,
                    const mytype::real3* const pos,
                    const mytype::integer* const type,
                    const mytype::integer* const cell_list,
                    const mytype::integer* const link_cell,
                    const mytype::integer* const cell_start,
                    const mytype::real rzero,
                    const mytype::real nzero,
                    const mytype::real beta,
                    const mytype::real b_tmp,
                    const mytype::real Aii_,
                    const mytype::GEOMETRY geo  );

void dev_solvePoisson(    mytype::real* const x,
                          mytype::real* const r0,
                          mytype::real* const r,
                          mytype::real* const p,
                          mytype::real* const s,
                          mytype::real* const As,
                    const mytype::matrix_COO<mytype::integer, mytype::real>* const A  );

void dev_solvePoisson2(mytype::real* const x, const mytype::matrix_COO<mytype::integer, mytype::real>* const A);

} //namespace

#endif // MPS_GPU_CUDAKER_H
