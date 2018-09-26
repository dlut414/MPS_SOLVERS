/*
LICENCE
*/
//mps_gpu_cudaker.h
///implementation of cuda kernel functions

#include <cmath>
#include <cstdio>
#include <cassert>

#include "mps_gpu_cudaker.h"
#include "typedef.h"
#include "common.h"
#include "marchingCube_define.h"

namespace cudaker
{
    inline cudaError_t checkCuda(cudaError_t result)
    {
    #ifdef DEBUG
        if(result != cudaSuccess)
        {
            fprintf(stderr, "CUDA Runtime Error: %s\n",
                    cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
    #endif
        return result;
    }

    inline cublasStatus_t checkCublas(cublasStatus_t result, char* msg)
    {
    #ifdef DEBUG
        if(result != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "cublas Runtime Error: %s\n", msg);
            assert(result == CUBLAS_STATUS_SUCCESS);
        }
    #endif
        return result;
    }

    __global__ void kerVecCpy(mytype::real* const des,
                        const mytype::real* const src,
                        const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            des[i] = src[i];
        }
    }

    __global__ void kerAxpy(mytype::real* const z,
                      const mytype::real* const x,
                      const mytype::real* const y,
                      const mytype::real a, const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            z[i] = a * x[i] + y[i];
        }
    }

    __global__ void kerMatVec(mytype::real* const des,
                        const mytype::real* const mat,
                        const mytype::real* const vec,
                        const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        mytype::real _tmp = 0.0;

        if(i < n)
        {
            for(mytype::integer j=0;j<n;j++)
            {
                _tmp += mat[i*n+j] * vec[j];
            }
            des[i] = _tmp;
        }
    }

    __global__ void kerVecVec(mytype::real& des,
                        const mytype::real* const vec1,
                        const mytype::real* const vec2,
                        const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        mytype::real _tmp = 0.0;

        if(i == 0)
        {
            for(mytype::integer j=0;j<n;j++)
            {
                _tmp += vec1[j] * vec2[j];
            }

            des = _tmp;
        }

    }

    void VecCpy(mytype::real* const des,
          const mytype::real* const src,
          const mytype::integer n)
    {
        kerVecCpy<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, src, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    void Axpy(mytype::real* const z,
        const mytype::real* const x,
        const mytype::real* const y,
        const mytype::real a, const mytype::integer n)
    {
        kerAxpy<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(z, x, y, a, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("Axpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("Axpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    void MatVec(mytype::real* const des,
          const mytype::real* const mat,
          const mytype::real* const vec,
          const mytype::integer n)
    {
        kerMatVec<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, mat, vec, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("MV -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("MV -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    void VecVec(mytype::real& des,
          const mytype::real* const vec1,
          const mytype::real* const vec2,
          const mytype::integer n)
    {

        kerVecVec<<<1, BLOCK_DIM_X_L>>>(des, vec1, vec2, n);
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("VV -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("VV -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif

    }

    void CG(const mytype::real* const A, mytype::real* const x, const mytype::real* const b, const mytype::integer n)
    {
        const mytype::real _ZERO = 0.0;
        const mytype::real _P_ONE = 1.0;
        const mytype::real _N_ONE = -1.0;

        int _num;

        mytype::real _rrold;
        mytype::real _rrnew;
        mytype::real _alpha;
        mytype::real _rn_over_ro;

        /*-----device memory-----*/
        mytype::real* dev_A;
        mytype::real* dev_x;
        mytype::real* dev_b;

        mytype::real* dev_Ap;
        mytype::real* dev_p;
        mytype::real* dev_r;
#ifdef DEBUG
        float time;
        cudaEvent_t startEvent, stopEvent;
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );

        checkCuda( cudaEventRecord(startEvent, 0) );
#endif
        checkCuda( cudaMalloc(&dev_A, n*n*sizeof(mytype::real)) );
        checkCuda( cudaMalloc(&dev_x, n*sizeof(mytype::real)) );
        checkCuda( cudaMalloc(&dev_b, n*sizeof(mytype::real)) );

        checkCuda( cudaMalloc(&dev_Ap, n*sizeof(mytype::real)) );
        checkCuda( cudaMalloc(&dev_p, n*sizeof(mytype::real)) );
        checkCuda( cudaMalloc(&dev_r, n*sizeof(mytype::real)) );
#ifdef DEBUG
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
        printf("time for MemAlloc: %f ms\n",time);
#endif
#ifdef DEBUG
        checkCuda( cudaEventRecord(startEvent, 0) );
#endif
        checkCuda( cudaMemcpy(dev_A, A, n*n*sizeof(mytype::real), cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(dev_x, x, n*sizeof(mytype::real), cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(dev_b, b, n*sizeof(mytype::real), cudaMemcpyHostToDevice) );
#ifdef DEBUG
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
        printf("time for Memcpy: %f ms\n",time);
#endif
        /*-----------------------*/

        /*-----CG by using cublas-----*/
        cublasHandle_t handle;

        checkCublas( cublasCreate(&handle), "create" );

        ///r = b - A*x
        checkCublas( cublasScopy(handle, n, dev_b, 1, dev_r, 1), "Dcopy1" );
        checkCublas( cublasSgemv(handle, CUBLAS_OP_N, n, n, &_N_ONE, dev_A, n, dev_x, 1, &_P_ONE, dev_r, 1), "Dgemv1" );
        ///p = r
        checkCublas( cublasScopy(handle, n, dev_r, 1, dev_p, 1), "Dcopy2" );
        ///_rrold = r*r
        checkCublas( cublasSdot(handle, n, dev_r, 1, dev_r, 1, &_rrold), "Ddot1" );

        _num = 0;
        while( _rrold > mytype::EPS_BY_EPS )
        {
            ///Ap = A*p
            checkCublas( cublasSgemv(handle, CUBLAS_OP_N, n, n, &_P_ONE, dev_A, n, dev_p, 1, &_ZERO, dev_Ap, 1), "Dgemv2" );
            ///_alpha = _rrold / Ap*p
            checkCublas( cublasSdot(handle, n, dev_Ap, 1, dev_p, 1, &_alpha), "Ddot2" );
            _alpha = _rrold / _alpha;

            ///x = x + _alpha*p
            checkCublas( cublasSaxpy(handle, n, &_alpha, dev_p, 1, dev_x, 1 ), "Daxpy1" );
            ///r = r - _alpha*Ap
            _alpha = -_alpha;
            checkCublas( cublasSaxpy(handle, n, &_alpha, dev_Ap, 1, dev_r, 1 ), "Daxpy2" );
            ///_rrnew = r*r
            checkCublas( cublasSdot(handle, n, dev_r, 1, dev_r, 1, &_rrnew), "Ddot2" );
            ///_rn_over_ro = _rrnew / _rrold
            _rn_over_ro = _rrnew / _rrold;
            ///p = _rn_over_ro*p + r
            checkCublas( cublasSscal(handle, n, &_rn_over_ro, dev_p, 1), "Dscal1" );
            checkCublas( cublasSaxpy(handle, n, &_P_ONE, dev_r, 1, dev_p, 1 ), "Daxpy3" );

            ///_rrold = _rrnew
            _rrold = _rrnew;

            _num++;
            //printf("CONVERGENCE -> RESIDUAL: %.2e\n",_rrnew);
        }

        checkCuda( cudaMemcpy(x, dev_x, n*sizeof(mytype::real), cudaMemcpyDeviceToHost) );

        checkCublas( cublasDestroy(handle), "destroy");
        /*----------------------------*/

        /*-----device memory-----*/
#ifdef DEBUG
        checkCuda( cudaEventRecord(startEvent, 0) );
#endif
        checkCuda( cudaFree(dev_A) );
        checkCuda( cudaFree(dev_x) );
        checkCuda( cudaFree(dev_b) );

        checkCuda( cudaFree(dev_Ap) );
        checkCuda( cudaFree(dev_p) );
        checkCuda( cudaFree(dev_r) );
#ifdef DEBUG
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
        printf("time for freeMem: %f ms\n",time);
#endif
        /*-----------------------*/
#ifdef DEBUG
        checkCuda( cudaEventDestroy(startEvent) );
        checkCuda( cudaEventDestroy(stopEvent) );
#endif
        printf("    CG -> times: %d \n", _num);
    }

    __global__ void kerSort_i(mytype::integer* const des,
                        const mytype::integer* const dev_p,
                        const mytype::integer* const dev_i_index,
                        const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            des[dev_i_index[i]] = dev_p[i];
        }
    }

    void dev_sort_i(mytype::integer* const des,
              const mytype::integer* const dev_p,
              const mytype::integer* const dev_i_index,
              const mytype::integer n)
    {
        kerSort_i<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, dev_p, dev_i_index, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerSort_d(mytype::real* const des,
                        const mytype::real* const dev_p,
                        const mytype::integer* const dev_i_index,
                        const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            des[dev_i_index[i]] = dev_p[i];
        }
    }

    void dev_sort_d(mytype::real* const des,
              const mytype::real* const dev_p,
              const mytype::integer* const dev_i_index,
              const mytype::integer n)
    {
        kerSort_d<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, dev_p, dev_i_index, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerSort_i3(mytype::int3* const des,
                         const mytype::int3* const dev_p,
                         const mytype::integer* const dev_i_index,
                         const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            des[dev_i_index[i]] = dev_p[i];
        }
    }

    void dev_sort_i3(mytype::int3* const des,
               const mytype::int3* const dev_p,
               const mytype::integer* const dev_i_index,
               const mytype::integer n)
    {
        kerSort_i3<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, dev_p, dev_i_index, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerSort_d3(mytype::real3* const des,
                         const mytype::real3* const dev_p,
                         const mytype::integer* const dev_i_index,
                         const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            des[dev_i_index[i]] = dev_p[i];
        }
    }

    void dev_sort_d3(mytype::real3* const des,
               const mytype::real3* const dev_p,
               const mytype::integer* const dev_i_index,
               const mytype::integer n)
    {
        kerSort_d3<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(des, dev_p, dev_i_index, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerSort_normal(mytype::integer* const dev_p,
                             const mytype::integer* const dev_i_index,
                             const mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n)
        {
            dev_p[i] = dev_i_index[dev_p[i]];
        }
    }

    void dev_sort_normal( mytype::integer* const dev_p,
                    const mytype::integer* const dev_i_index,
                    const mytype::integer n )
    {
        kerSort_normal<<<(n+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(dev_p, dev_i_index, n);

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("cpy -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("cpy -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerSort_all_tmp( mytype::integer* const dev_i_id_tmp,
                                     mytype::integer* const dev_i_type_tmp,
                                     mytype::integer* const dev_i_cell_list_tmp,
                                     mytype::integer* const dev_i_normal_tmp,
                                     mytype::real* const dev_d_press_tmp,
                                     mytype::real* const dev_d_n_tmp,
                                     mytype::real3* const dev_d3_pos_tmp,
                                     mytype::real3* const dev_d3_vel_tmp,

                                     mytype::integer* const dev_i_id,
                                     mytype::integer* const dev_i_type,
                                     mytype::integer* const dev_i_cell_list,
                                     mytype::integer* const dev_i_normal,
                                     mytype::real* const dev_d_press,
                                     mytype::real* const dev_d_n,
                                     mytype::real3* const dev_d3_pos,
                                     mytype::real3* const dev_d3_vel,
                               const mytype::integer* const dev_i_index,
                               const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np)
        {
            const mytype::integer _index = dev_i_index[i];

            dev_i_id_tmp        [_index]     = dev_i_id         [i];
            dev_i_type_tmp      [_index]     = dev_i_type       [i];
            dev_i_cell_list_tmp [_index]     = dev_i_cell_list  [i];
            dev_i_normal_tmp    [_index]     = dev_i_normal     [i];

            dev_d_press_tmp     [_index]     = dev_d_press      [i];
            dev_d_n_tmp         [_index]     = dev_d_n          [i];

            dev_d3_pos_tmp      [_index]     = dev_d3_pos       [i];
            dev_d3_vel_tmp      [_index]     = dev_d3_vel       [i];
        }
    }

    __global__ void kerSort_all( mytype::integer* const dev_i_id_tmp,
                                 mytype::integer* const dev_i_type_tmp,
                                 mytype::integer* const dev_i_cell_list_tmp,
                                 mytype::integer* const dev_i_normal_tmp,
                                 mytype::real* const dev_d_press_tmp,
                                 mytype::real* const dev_d_n_tmp,
                                 mytype::real3* const dev_d3_pos_tmp,
                                 mytype::real3* const dev_d3_vel_tmp,

                                 mytype::integer* const dev_i_id,
                                 mytype::integer* const dev_i_type,
                                 mytype::integer* const dev_i_cell_list,
                                 mytype::integer* const dev_i_normal,

                                 mytype::real* const dev_d_press,
                                 mytype::real* const dev_d_n,
                                 mytype::real3* const dev_d3_pos,
                                 mytype::real3* const dev_d3_vel,

                           const mytype::integer* const dev_i_index,
                           const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np)
        {
            dev_i_id        [i]      = dev_i_id_tmp         [i];
            dev_i_type      [i]      = dev_i_type_tmp       [i];
            dev_i_cell_list [i]      = dev_i_cell_list_tmp  [i];

            dev_d_press     [i]      = dev_d_press_tmp      [i];
            dev_d_n         [i]      = dev_d_n_tmp          [i];

            dev_d3_pos      [i]      = dev_d3_pos_tmp       [i];
            dev_d3_vel      [i]      = dev_d3_vel_tmp       [i];

            dev_i_normal    [i]      = dev_i_index[dev_i_normal_tmp[i]];
        }
    }

    void dev_sort_all( mytype::integer* const dev_i_id_tmp,
                       mytype::integer* const dev_i_type_tmp,
                       mytype::integer* const dev_i_cell_list_tmp,
                       mytype::integer* const dev_i_normal_tmp,
                       mytype::real* const dev_d_press_tmp,
                       mytype::real* const dev_d_n_tmp,
                       mytype::real3* const dev_d3_pos_tmp,
                       mytype::real3* const dev_d3_vel_tmp,

                       mytype::integer* const dev_i_id,
                       mytype::integer* const dev_i_type,
                       mytype::integer* const dev_i_cell_list,
                       mytype::integer* const dev_i_normal,

                       mytype::real* const dev_d_press,
                       mytype::real* const dev_d_n,
                       mytype::real3* const dev_d3_pos,
                       mytype::real3* const dev_d3_vel,

                 const mytype::integer* const dev_i_index,
                 const mytype::integer i_np )
    {
        ///call routines
        kerSort_all_tmp<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_i_id_tmp,
                                                                            dev_i_type_tmp,
                                                                            dev_i_cell_list_tmp,
                                                                            dev_i_normal_tmp,
                                                                            dev_d_press_tmp,
                                                                            dev_d_n_tmp,
                                                                            dev_d3_pos_tmp,
                                                                            dev_d3_vel_tmp,

                                                                            dev_i_id,
                                                                            dev_i_type,
                                                                            dev_i_cell_list,
                                                                            dev_i_normal,

                                                                            dev_d_press,
                                                                            dev_d_n,
                                                                            dev_d3_pos,
                                                                            dev_d3_vel,

                                                                            dev_i_index,
                                                                            i_np );

        kerSort_all    <<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_i_id_tmp,
                                                                            dev_i_type_tmp,
                                                                            dev_i_cell_list_tmp,
                                                                            dev_i_normal_tmp,
                                                                            dev_d_press_tmp,
                                                                            dev_d_n_tmp,
                                                                            dev_d3_pos_tmp,
                                                                            dev_d3_vel_tmp,

                                                                            dev_i_id,
                                                                            dev_i_type,
                                                                            dev_i_cell_list,
                                                                            dev_i_normal,

                                                                            dev_d_press,
                                                                            dev_d_n,
                                                                            dev_d3_pos,
                                                                            dev_d3_vel,

                                                                            dev_i_index,
                                                                            i_np );
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_sort_all -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_sort_all -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __device__ inline mytype::real dev_d_weight( const mytype::real _r0,
                                                 const mytype::real _r )
    {
        //danger when _r == 0
        if(_r >= _r0) return 0.0;
        else          return (_r0 / _r - 1.0);
    }

    __global__ void kerCal_n( mytype::real* const dev_d_n,
                        const mytype::real3* const dev_d3_pos,
                        const mytype::integer* const dev_i_type,
                        const mytype::integer* const dev_i_cell_list,
                        const mytype::integer* const dev_i_link_cell,
                        const mytype::integer* const dev_i_cell_start,
                        const mytype::integer* const dev_i_cell_end,
                        const mytype::real d_rzero,
                        const mytype::integer i_num_cells,
                        const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np && dev_i_type[i] != 2)
        {
            mytype::real  _n     = 0.0f;
            mytype::real3 _pos_i = dev_d3_pos[i];

            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            const mytype::integer __offset = 28 * dev_i_cell_list[i];
            const mytype::integer __num    =      dev_i_link_cell[__offset];

            for(mytype::integer dir=1;dir<=__num;dir++)
            {
                mytype::integer __cell = dev_i_link_cell[__offset + dir];

                if(__cell < i_num_cells)
                {
                    mytype::integer __start = dev_i_cell_start[__cell];
                    mytype::integer __end   = dev_i_cell_end[__cell];

                    for(mytype::integer j=__start;j<__end;j++)
                    {
                        if(j != i)
                        {
                            mytype::real __rr = (dev_d3_pos[j].x - _pos_i.x) * (dev_d3_pos[j].x - _pos_i.x)
                                              + (dev_d3_pos[j].y - _pos_i.y) * (dev_d3_pos[j].y - _pos_i.y)
                                              + (dev_d3_pos[j].z - _pos_i.z) * (dev_d3_pos[j].z - _pos_i.z);

                            _n += dev_d_weight( d_rzero, sqrt(__rr) );
                        }
                    }
                }
            }

            dev_d_n[i] = _n;
        }

    }

    void dev_cal_n( mytype::real* const dev_d_n,
              const mytype::real3* const dev_d3_pos,
              const mytype::integer* const dev_i_type,
              const mytype::integer* const dev_i_cell_list,
              const mytype::integer* const dev_i_link_cell,
              const mytype::integer* const dev_i_cell_start,
              const mytype::integer* const dev_i_cell_end,
              const mytype::real d_rzero,
              const mytype::integer i_num_cells,
              const mytype::integer i_np )
    {
        ///call routines
        kerCal_n<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_d_n,
                                                                     dev_d3_pos,
                                                                     dev_i_type,
                                                                     dev_i_cell_list,
                                                                     dev_i_link_cell,
                                                                     dev_i_cell_start,
                                                                     dev_i_cell_end,
                                                                     d_rzero,
                                                                     i_num_cells,
                                                                     i_np );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_cal_n -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_cal_n -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalDash_tmp ( mytype::real3* const dev_d3_vel,
                                     mytype::real3* const dev_d3_pos,
                               const mytype::real* const dev_d_press,
                               const mytype::integer* const dev_i_type,
                               const mytype::integer* const dev_i_cell_list,
                               const mytype::integer* const dev_i_link_cell,
                               const mytype::integer* const dev_i_cell_start,
                               const mytype::integer* const dev_i_cell_end,
                               const mytype::real d_dt,
                               const mytype::real d_one_over_rho,
                               const mytype::real d_one_over_nzero,
                               const mytype::real d_rzero,
                               const mytype::integer i_dim,
                               const mytype::integer i_num_cells,
                               const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np && dev_i_type[i] == 0)
        {
            /*----------pressure gradient part----------*/
            mytype::real3 _ret            =      {0,0,0};
            mytype::real3 _pos_i          =      dev_d3_pos[i];
            mytype::real  _hat_p          =      dev_d_press[i];
            const mytype::integer _offset = 28 * dev_i_cell_list[i];
            const mytype::integer _num    =      dev_i_link_cell[_offset];

            //searching _hat_p (minimum of p in 27 cells)
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            for(mytype::integer dir=1;dir<=_num;dir++)
            {
                mytype::integer __cell = dev_i_link_cell[_offset + dir];

                if(__cell < i_num_cells)
                {
                    mytype::integer __start = dev_i_cell_start[__cell];
                    mytype::integer __end   = dev_i_cell_end[__cell];

                    for(mytype::integer j=__start;j<__end;j++)
                    {
                        //ignore type 2 particles
                        //if(dev_i_type[j] != 2)
                        {
                            mytype::real __rr = (dev_d3_pos[j].x - _pos_i.x) * (dev_d3_pos[j].x - _pos_i.x)
                                              + (dev_d3_pos[j].y - _pos_i.y) * (dev_d3_pos[j].y - _pos_i.y)
                                              + (dev_d3_pos[j].z - _pos_i.z) * (dev_d3_pos[j].z - _pos_i.z);

                            if( dev_d_press[j] < _hat_p && __rr <= (d_rzero*d_rzero) )
                            {
                                _hat_p = dev_d_press[j];
                            }
                        }
                    }
                }
            }

            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            for(mytype::integer dir=1;dir<=_num;dir++)
            {
                mytype::integer __cell = dev_i_link_cell[_offset+dir];

                if(__cell < i_num_cells)
                {
                    mytype::integer __start = dev_i_cell_start[__cell];
                    mytype::integer __end   = dev_i_cell_end[__cell];

                    for(mytype::integer j=__start;j<__end;j++)
                    {
                        if(j != i)
                        {
                            mytype::real3 __dr;

                            __dr.x = dev_d3_pos[j].x - _pos_i.x;
                            __dr.y = dev_d3_pos[j].y - _pos_i.y;
                            __dr.z = dev_d3_pos[j].z - _pos_i.z;

                            mytype::real __rr   = __dr.x * __dr.x +  __dr.y * __dr.y + __dr.z * __dr.z;

                            mytype::real __coef = (dev_d_press[j] - _hat_p) / __rr * dev_d_weight(d_rzero, sqrt(__rr));

                            _ret.x += __coef * __dr.x;
                            _ret.y += __coef * __dr.y;
                            _ret.z += __coef * __dr.z;
                        }
                    }
                }
            }

            mytype::real _coef = - d_dt * d_one_over_rho * i_dim * d_one_over_nzero;

            _ret.x *= _coef;
            _ret.y *= _coef;
            _ret.z *= _coef;
            /*-----------------------------------------*/

            /*----------cal tmp part----------*/
            //only apply to fluid particles
            dev_d3_vel[i].x += _ret.x;
            dev_d3_vel[i].y += _ret.y;
            dev_d3_vel[i].z += _ret.z;

            dev_d3_pos[i].x += d_dt * _ret.x;
            dev_d3_pos[i].y += d_dt * _ret.y;
            dev_d3_pos[i].z += d_dt * _ret.z;
            /*--------------------------------*/
        }
    }

    void dev_calDash( mytype::real3* const dev_d3_vel,
                      mytype::real3* const dev_d3_pos,
                const mytype::real* const dev_d_press,
                const mytype::integer* const dev_i_type,
                const mytype::integer* const dev_i_cell_list,
                const mytype::integer* const dev_i_link_cell,
                const mytype::integer* const dev_i_cell_start,
                const mytype::integer* const dev_i_cell_end,
                const mytype::real d_dt,
                const mytype::real d_one_over_rho,
                const mytype::real d_one_over_nzero,
                const mytype::real d_rzero,
                const mytype::integer i_dim,
                const mytype::integer i_num_cells,
                const mytype::integer i_np )
    {
        ///call routines
        kerCalDash_tmp<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_d3_vel,
                                                                           dev_d3_pos,
                                                                           dev_d_press,
                                                                           dev_i_type,
                                                                           dev_i_cell_list,
                                                                           dev_i_link_cell,
                                                                           dev_i_cell_start,
                                                                           dev_i_cell_end,
                                                                           d_dt,
                                                                           d_one_over_rho,
                                                                           d_one_over_nzero,
                                                                           d_rzero,
                                                                           i_dim,
                                                                           i_num_cells,
                                                                           i_np );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calDash -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calDash -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalPres_fluid_expl( mytype::real* const dev_d_press,
                                     const mytype::real* const dev_d_n,
                                     const mytype::integer* const dev_i_type,
                                     const mytype::real d_one_over_alpha,
                                     const mytype::real d_nzero,
                                     const mytype::real d_one_over_nzero,
                                     const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np && dev_i_type[i] != 2)
        {
            mytype::real _tmp = d_one_over_alpha * (dev_d_n[i] - d_nzero) * d_one_over_nzero;

            dev_d_press[i] = (_tmp > 0.0 ? _tmp : 0.0);
        }
    }

    __global__ void kerCalPres_bd2_expl( mytype::real* const dev_d_press,
                                   const mytype::integer* const dev_i_type,
                                   const mytype::integer* const dev_i_normal,
                                   const mytype::integer i_np)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(dev_i_type[i] == 2)
        {
            dev_d_press[i] = dev_d_press[dev_i_normal[i]];
        }
    }

    void dev_calPres_expl( mytype::real* const dev_d_press,
                     const mytype::real* const dev_d_n,
                     const mytype::integer* const dev_i_type,
                     const mytype::integer* const dev_i_normal,
                     const mytype::real d_one_over_alpha,
                     const mytype::real d_nzero,
                     const mytype::real d_one_over_nzero,
                     const mytype::integer i_np )
    {
        ///call routines
        kerCalPres_fluid_expl<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_d_press,
                                                                                  dev_d_n,
                                                                                  dev_i_type,
                                                                                  d_one_over_alpha,
                                                                                  d_nzero,
                                                                                  d_one_over_nzero,
                                                                                  i_np );
        kerCalPres_bd2_expl<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_d_press,
                                                                                dev_i_type,
                                                                                dev_i_normal,
                                                                                i_np );
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calPres_expl -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calPres_expl -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalVisc_tmp( mytype::real3* const dev_d3_vel,
                                    mytype::real3* const dev_d3_pos,
                              const mytype::real* const dev_d_press,
                              const mytype::integer* const dev_i_type,
                              const mytype::integer* const dev_i_cell_list,
                              const mytype::integer* const dev_i_link_cell,
                              const mytype::integer* const dev_i_cell_start,
                              const mytype::integer* const dev_i_cell_end,
                              const mytype::real3 G,
                              const mytype::real d_dt,
                              const mytype::real d_2bydim_over_nzerobylambda,
                              const mytype::real d_rlap,
                              const mytype::real d_niu,
                              const mytype::integer i_num_cells,
                              const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np)
        {
            if(dev_i_type[i] == 0)
            {
                mytype::real3 _ret          =      {0.0, 0.0, 0.0};
                mytype::integer _offset     = 28 * dev_i_cell_list[i];
                mytype::integer _num        =      dev_i_link_cell[_offset];

                //searching neighbors
                //loop: surrounding cells including itself (totally 27 cells)
                //loop: from bottom to top, from back to front, from left to right
                for(mytype::integer dir=1;dir<=_num;dir++)
                {
                    mytype::integer __cell = dev_i_link_cell[_offset+dir];

                    if(__cell < i_num_cells)
                    {
                        mytype::integer __start = dev_i_cell_start[__cell];
                        mytype::integer __end   = dev_i_cell_end[__cell];

                        for(mytype::integer j=__start;j<__end;j++)
                        {
                            if(j != i)
                            {
                                mytype::real3 __dr;

                                __dr.x = dev_d3_pos[j].x - dev_d3_pos[i].x;
                                __dr.y = dev_d3_pos[j].y - dev_d3_pos[i].y;
                                __dr.z = dev_d3_pos[j].z - dev_d3_pos[i].z;

                                mytype::real3 __du;

                                __du.x = dev_d3_vel[j].x - dev_d3_vel[i].x;
                                __du.y = dev_d3_vel[j].y - dev_d3_vel[i].y;
                                __du.z = dev_d3_vel[j].z - dev_d3_vel[i].z;

                                mytype::real __tmp = dev_d_weight(d_rlap , sqrt( __dr.x*__dr.x + __dr.y*__dr.y + __dr.z*__dr.z ));

                                _ret.x += __tmp * __du.x;
                                _ret.y += __tmp * __du.y;
                                _ret.z += __tmp * __du.z;
                            }
                        }
                    }
                }

                mytype::real __coef = d_niu * d_2bydim_over_nzerobylambda;

                dev_d3_vel[i].x += d_dt * (__coef * _ret.x + G.x);
                dev_d3_vel[i].y += d_dt * (__coef * _ret.y + G.y);
                dev_d3_vel[i].z += d_dt * (__coef * _ret.z + G.z);

                dev_d3_pos[i].x += d_dt * dev_d3_vel[i].x;
                dev_d3_pos[i].y += d_dt * dev_d3_vel[i].y;
                dev_d3_pos[i].z += d_dt * dev_d3_vel[i].z;
            }
        }
    }

    void dev_calVisc_expl( mytype::real3* const dev_d3_vel,
                           mytype::real3* const dev_d3_pos,
                     const mytype::real* const dev_d_press,
                     const mytype::integer* const dev_i_type,
                     const mytype::integer* const dev_i_cell_list,
                     const mytype::integer* const dev_i_link_cell,
                     const mytype::integer* const dev_i_cell_start,
                     const mytype::integer* const dev_i_cell_end,
                     const mytype::real d_dt,
                     const mytype::real d_2bydim_over_nzerobylambda,
                     const mytype::real d_rlap,
                     const mytype::real d_niu,
                     const mytype::integer i_num_cells,
                     const mytype::integer i_np )
    {
        ///call routines
        kerCalVisc_tmp<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_d3_vel,
                                                                           dev_d3_pos,
                                                                           dev_d_press,
                                                                           dev_i_type,
                                                                           dev_i_cell_list,
                                                                           dev_i_link_cell,
                                                                           dev_i_cell_start,
                                                                           dev_i_cell_end,
                                                                           mytype::G,
                                                                           d_dt,
                                                                           d_2bydim_over_nzerobylambda,
                                                                           d_rlap,
                                                                           d_niu,
                                                                           i_num_cells,
                                                                           i_np );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calVisc_tmp -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calVisc_tmp -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalCol_tmp( mytype::real3* const dev_d3_vel,
                                   mytype::real3* const dev_d3_pos,
                             const mytype::integer* const dev_i_type,
                             const mytype::integer* const dev_i_cell_list,
                             const mytype::integer* const dev_i_link_cell,
                             const mytype::integer* const dev_i_cell_start,
                             const mytype::integer* const dev_i_cell_end,
                             const mytype::real d_dt,
                             const mytype::real d_col_dis,
                             const mytype::real d_col_rate,
                             const mytype::integer i_num_cells,
                             const mytype::integer i_np )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_np)
        {
            mytype::real3 _crt = {0.0, 0.0, 0.0};

            if(dev_i_type[i] == 0)
            {
                //searching neighbors
                //loop: surrounding cells including itself (totally 27 cells)
                //loop: from bottom to top, from back to front, from left to right
                const mytype::integer __offset = 28 * dev_i_cell_list[i];
                const mytype::integer __num = dev_i_link_cell[__offset];

                for(mytype::integer dir=1;dir<=__num;dir++)
                {
                    mytype::integer __cell = dev_i_link_cell[__offset+dir];

                    if(__cell < i_num_cells)
                    {
                        mytype::integer __start = dev_i_cell_start[__cell];
                        mytype::integer __end = dev_i_cell_end[__cell];

                        for(mytype::integer j=__start;j<__end;j++)
                        {
                            if(j != i)
                            {
                                mytype::real3 __dr;
                                __dr.x = dev_d3_pos[j].x - dev_d3_pos[i].x;
                                __dr.y = dev_d3_pos[j].y - dev_d3_pos[i].y;
                                __dr.z = dev_d3_pos[j].z - dev_d3_pos[i].z;

                                mytype::real3 __du;
                                __du.x = dev_d3_vel[j].x - dev_d3_vel[i].x;
                                __du.y = dev_d3_vel[j].y - dev_d3_vel[i].y;
                                __du.z = dev_d3_vel[j].z - dev_d3_vel[i].z;

                                mytype::real __ds = sqrt(__dr.x*__dr.x + __dr.y*__dr.y + __dr.z*__dr.z);
                                mytype::real __one_over_ds = 1.0f / __ds;
                                mytype::real __vabs = 0.5f * __one_over_ds * (__du.x*__dr.x + __du.y*__dr.y + __du.z*__dr.z);

                                if( (__ds <= d_col_dis) && (__vabs <= 0.0) )
                                {
                                    _crt.x += d_col_rate * __vabs * __one_over_ds * __dr.x;
                                    _crt.y += d_col_rate * __vabs * __one_over_ds * __dr.y;
                                    _crt.z += d_col_rate * __vabs * __one_over_ds * __dr.z;
                                }
                            }
                        }
                    }
                }
            }

            dev_d3_vel[i].x += _crt.x;
            dev_d3_vel[i].y += _crt.y;
            dev_d3_vel[i].z += _crt.z;
            dev_d3_pos[i].x += d_dt * _crt.x;
            dev_d3_pos[i].y += d_dt * _crt.y;
            dev_d3_pos[i].z += d_dt * _crt.z;
        }
    }

    void dev_calCol( mytype::real3* const dev_d3_vel,
                     mytype::real3* const dev_d3_pos,
               const mytype::integer* const dev_i_type,
               const mytype::integer* const dev_i_cell_list,
               const mytype::integer* const dev_i_link_cell,
               const mytype::integer* const dev_i_cell_start,
               const mytype::integer* const dev_i_cell_end,
               const mytype::real d_dt,
               const mytype::real d_col_dis,
               const mytype::real d_col_rate,
               const mytype::integer i_num_cells,
               const mytype::integer i_np )
    {
        ///call routines
        kerCalCol_tmp<<<(i_np+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(  dev_d3_vel,
                                                                           dev_d3_pos,
                                                                           dev_i_type,
                                                                           dev_i_cell_list,
                                                                           dev_i_link_cell,
                                                                           dev_i_cell_start,
                                                                           dev_i_cell_end,
                                                                           d_dt,
                                                                           d_col_dis,
                                                                           d_col_rate,
                                                                           i_num_cells,
                                                                           i_np );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calCol -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calCol -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalVertex_n( mytype::real* const dev_r_vertex_n,
                              const mytype::real3* const dev_r3_verList,
                              const mytype::real3* const dev_d3_pos,
                              const mytype::integer* const dev_i_type,
                              const mytype::integer* const dev_i_link_cell,
                              const mytype::integer* const dev_i_cell_start,
                              const mytype::integer* const dev_i_cell_end,
                              const mytype::real d_rzero,
                              const mytype::integer i_nVertex,
                              const mytype::GEOMETRY geo )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_nVertex)
        {
            mytype::real  _n     = 0.0f;
            mytype::real3 _pos_i = dev_r3_verList[i];

            mytype::integer __cellx = mytype::integer( (_pos_i.x - geo.d_cell_left)   / geo.d_cell_size );
            mytype::integer __celly = mytype::integer( (_pos_i.y - geo.d_cell_back)   / geo.d_cell_size );
            mytype::integer __cellz = mytype::integer( (_pos_i.z - geo.d_cell_bottom) / geo.d_cell_size );
            mytype::integer __cellId = __cellz * geo.i_cell_sheet + __celly * geo.i_cell_dx + __cellx;

            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            const mytype::integer __offset = 28 * __cellId;
            const mytype::integer __num    =      dev_i_link_cell[__offset];

            for(mytype::integer dir=1;dir<=__num;dir++)
            {
                mytype::integer __cell = dev_i_link_cell[__offset + dir];

                if(__cell < geo.i_num_cells)
                {
                    mytype::integer __start = dev_i_cell_start[__cell];
                    mytype::integer __end   = dev_i_cell_end[__cell];

                    for(mytype::integer j=__start;j<__end;j++)
                    {
                        if(dev_i_type[j] == 0)
                        {
                            mytype::real __rr = (dev_d3_pos[j].x - _pos_i.x) * (dev_d3_pos[j].x - _pos_i.x)
                                              + (dev_d3_pos[j].y - _pos_i.y) * (dev_d3_pos[j].y - _pos_i.y)
                                              + (dev_d3_pos[j].z - _pos_i.z) * (dev_d3_pos[j].z - _pos_i.z);

                            _n += dev_d_weight( d_rzero, sqrt(__rr) );
                        }
                    }
                }
            }

            dev_r_vertex_n[i] = _n;
        }
    }

    void dev_calVertex_n( mytype::real* const dev_r_vertex_n,
                    const mytype::real3* const dev_r3_verList,
                    const mytype::real3* const dev_d3_pos,
                    const mytype::integer* const dev_i_type,
                    const mytype::integer* const dev_i_link_cell,
                    const mytype::integer* const dev_i_cell_start,
                    const mytype::integer* const dev_i_cell_end,
                    const mytype::real d_rzero,
                    const mytype::integer i_nVertex,
                    const mytype::GEOMETRY geo )
    {
        ///call routines
        kerCalVertex_n<<<(i_nVertex+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>( dev_r_vertex_n,
                                                                                dev_r3_verList,
                                                                                dev_d3_pos,
                                                                                dev_i_type,
                                                                                dev_i_link_cell,
                                                                                dev_i_cell_start,
                                                                                dev_i_cell_end,
                                                                                d_rzero,
                                                                                i_nVertex,
                                                                                geo );
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calVoxel_n -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calVoxel_n -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __device__ void vertexInterp(mytype::real3& edge,
                           const mytype::real3 v1,
                           const mytype::real3 v2,
                           const mytype::real  n1,
                           const mytype::real  n2)
    {
        mytype::real _sumInv = 1.0f / (n1 + n2);

        edge.x = (v1.x * n1 + v2.x * n2 ) * _sumInv;
        edge.y = (v1.y * n1 + v2.y * n2 ) * _sumInv;
        edge.z = (v1.z * n1 + v2.z * n2 ) * _sumInv;
    }

    __global__ void kerCalTriangle ( mytype::real3* const dev_r3_triangle,
                                     mytype::real* const dev_r_alpha,
                               const mytype::real3* const dev_r3_verList,
                               const mytype::real* const dev_r_vertex_n,
                               const mytype::integer* const dev_i_voxList,
                               const mytype::real d_beta,
                               const mytype::integer i_nMaxEdge,
                               const mytype::integer i_nVoxel,
                               const uint* const dev_u_numVerTable,
                               const uint* const dev_u_triTable )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_nVoxel)
        {
            for(mytype::integer j=0;j<12;j++)
            {
                dev_r_alpha[12*i+j] = 0.0f;
            }

            mytype::integer _offset = 8 * i;

            mytype::integer _num[8];
            _num[0] = dev_i_voxList[_offset  ];
            _num[1] = dev_i_voxList[_offset+1];
            _num[2] = dev_i_voxList[_offset+2];
            _num[3] = dev_i_voxList[_offset+3];
            _num[4] = dev_i_voxList[_offset+4];
            _num[5] = dev_i_voxList[_offset+5];
            _num[6] = dev_i_voxList[_offset+6];
            _num[7] = dev_i_voxList[_offset+7];

            uint _index;
            _index  = uint(dev_r_vertex_n[_num[0]] > d_beta);
            _index += uint(dev_r_vertex_n[_num[1]] > d_beta) << 1;
            _index += uint(dev_r_vertex_n[_num[2]] > d_beta) << 2;
            _index += uint(dev_r_vertex_n[_num[3]] > d_beta) << 3;
            _index += uint(dev_r_vertex_n[_num[4]] > d_beta) << 4;
            _index += uint(dev_r_vertex_n[_num[5]] > d_beta) << 5;
            _index += uint(dev_r_vertex_n[_num[6]] > d_beta) << 6;
            _index += uint(dev_r_vertex_n[_num[7]] > d_beta) << 7;

            //if(_index == 0x00 || _index == 0xff) return;

#if USE_SHARED_MEM

            extern __shared__ mytype::real3 _edge[];

            vertexInterp(_edge[              threadIdx.x], dev_r3_verList[_num[0]], dev_r3_verList[_num[1]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[1]]);
            vertexInterp(_edge[   blockDim.x+threadIdx.x], dev_r3_verList[_num[1]], dev_r3_verList[_num[2]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[2]]);
            vertexInterp(_edge[2* blockDim.x+threadIdx.x], dev_r3_verList[_num[2]], dev_r3_verList[_num[3]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[3]]);
            vertexInterp(_edge[3* blockDim.x+threadIdx.x], dev_r3_verList[_num[3]], dev_r3_verList[_num[0]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[0]]);

            vertexInterp(_edge[4* blockDim.x+threadIdx.x], dev_r3_verList[_num[4]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[4]], dev_r_vertex_n[_num[5]]);
            vertexInterp(_edge[5* blockDim.x+threadIdx.x], dev_r3_verList[_num[5]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[5]], dev_r_vertex_n[_num[6]]);
            vertexInterp(_edge[6* blockDim.x+threadIdx.x], dev_r3_verList[_num[6]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[6]], dev_r_vertex_n[_num[7]]);
            vertexInterp(_edge[7* blockDim.x+threadIdx.x], dev_r3_verList[_num[7]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[7]], dev_r_vertex_n[_num[4]]);

            vertexInterp(_edge[8* blockDim.x+threadIdx.x], dev_r3_verList[_num[0]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[4]]);
            vertexInterp(_edge[9* blockDim.x+threadIdx.x], dev_r3_verList[_num[1]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[5]]);
            vertexInterp(_edge[10*blockDim.x+threadIdx.x], dev_r3_verList[_num[2]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[6]]);
            vertexInterp(_edge[11*blockDim.x+threadIdx.x], dev_r3_verList[_num[3]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[7]]);

            uint _numVer = dev_u_numVerTable[_index];

            for(uint j=0; j<_numVer; j++)
            {
                uint __p = _index * 16 + j;

                dev_r3_triangle[12*i+j] = _edge[dev_u_triTable[__p] * blockDim.x + threadIdx.x];
                dev_r_alpha[12*i+j] = 1.0f;
            }
#else

            mytype::real3 _edge[12];

            vertexInterp(_edge[0], dev_r3_verList[_num[0]], dev_r3_verList[_num[1]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[1]]);
            vertexInterp(_edge[1], dev_r3_verList[_num[1]], dev_r3_verList[_num[2]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[2]]);
            vertexInterp(_edge[2], dev_r3_verList[_num[2]], dev_r3_verList[_num[3]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[3]]);
            vertexInterp(_edge[3], dev_r3_verList[_num[3]], dev_r3_verList[_num[0]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[0]]);

            vertexInterp(_edge[4], dev_r3_verList[_num[4]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[4]], dev_r_vertex_n[_num[5]]);
            vertexInterp(_edge[5], dev_r3_verList[_num[5]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[5]], dev_r_vertex_n[_num[6]]);
            vertexInterp(_edge[6], dev_r3_verList[_num[6]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[6]], dev_r_vertex_n[_num[7]]);
            vertexInterp(_edge[7], dev_r3_verList[_num[7]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[7]], dev_r_vertex_n[_num[4]]);

            vertexInterp(_edge[8], dev_r3_verList[_num[0]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[4]]);
            vertexInterp(_edge[9], dev_r3_verList[_num[1]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[5]]);
            vertexInterp(_edge[10], dev_r3_verList[_num[2]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[6]]);
            vertexInterp(_edge[11], dev_r3_verList[_num[3]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[7]]);

            uint _numVer = dev_u_numVerTable[_index];

            for(uint j=0; j<_numVer; j++)
            {
                uint __p = _index * 16 + j;

                dev_r3_triangle[12*i+j] = _edge[dev_u_triTable[__p]];
                dev_r_alpha[12*i+j] = 1.0f;
            }
#endif
        }
    }

    void dev_calTriangle ( mytype::real3* const dev_r3_triangle,
                           mytype::real* const dev_r_alpha,
                     const mytype::real3* const dev_r3_verList,
                     const mytype::real* const dev_r_vertex_n,
                     const mytype::integer* const dev_i_voxList,
                     const mytype::real d_beta,
                     const mytype::integer i_nMaxEdge,
                     const mytype::integer i_nVoxel,
                     const uint* const dev_u_numVerTable,
                     const uint* const dev_u_triTable )
    {
#if USE_SHARED_MEM

        ///call routines
        kerCalTriangle<<<(i_nVoxel+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L, sizeof(mytype::real3)*12*BLOCK_DIM_X_L>>> ( dev_r3_triangle,
                                                                                dev_r_alpha,
                                                                                dev_r3_verList,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                d_beta,
                                                                                i_nMaxEdge,
                                                                                i_nVoxel,
                                                                                dev_u_numVerTable,
                                                                                dev_u_triTable );
#else

        ///call routines
        kerCalTriangle<<<(i_nVoxel+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>> ( dev_r3_triangle,
                                                                                dev_r_alpha,
                                                                                dev_r3_verList,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                d_beta,
                                                                                i_nMaxEdge,
                                                                                i_nVoxel,
                                                                                dev_u_numVerTable,
                                                                                dev_u_triTable );
#endif

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calTriangle -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calTriangle -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerCalNorm( mytype::real3* const dev_r3_norm,
                          const mytype::real3* const dev_r3_triangle,
                          const mytype::real* const dev_r_alpha,
                          const mytype::real3* const dev_d3_pos,
                          const mytype::integer* const dev_i_type,
                          const mytype::integer* const dev_i_link_cell,
                          const mytype::integer* const dev_i_cell_start,
                          const mytype::integer* const dev_i_cell_end,
                          const mytype::real d_rzero,
                          const mytype::integer i_nMaxEdge,
                          const mytype::GEOMETRY geo )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_nMaxEdge)
        {
            mytype::real3 _pos_i[6];
            _pos_i[0].x = dev_r3_triangle[i].x + DIFF;
            _pos_i[0].y = dev_r3_triangle[i].y;
            _pos_i[0].z = dev_r3_triangle[i].z;

            _pos_i[1].x = dev_r3_triangle[i].x - DIFF;
            _pos_i[1].y = dev_r3_triangle[i].y;
            _pos_i[1].z = dev_r3_triangle[i].z;

            _pos_i[2].x = dev_r3_triangle[i].x;
            _pos_i[2].y = dev_r3_triangle[i].y + DIFF;
            _pos_i[2].z = dev_r3_triangle[i].z;

            _pos_i[3].x = dev_r3_triangle[i].x;
            _pos_i[3].y = dev_r3_triangle[i].y - DIFF;
            _pos_i[3].z = dev_r3_triangle[i].z;

            _pos_i[4].x = dev_r3_triangle[i].x;
            _pos_i[4].y = dev_r3_triangle[i].y;
            _pos_i[4].z = dev_r3_triangle[i].z + DIFF;

            _pos_i[5].x = dev_r3_triangle[i].x;
            _pos_i[5].y = dev_r3_triangle[i].y;
            _pos_i[5].z = dev_r3_triangle[i].z - DIFF;

            mytype::real _n[6];
            for(mytype::integer p=0; p<6; p++)
            {
                if(dev_r_alpha[i] < 0.01f) break; ///for performance

                mytype::integer __cellx = mytype::integer( (_pos_i[p].x - geo.d_cell_left)   / geo.d_cell_size );
                mytype::integer __celly = mytype::integer( (_pos_i[p].y - geo.d_cell_back)   / geo.d_cell_size );
                mytype::integer __cellz = mytype::integer( (_pos_i[p].z - geo.d_cell_bottom) / geo.d_cell_size );
                mytype::integer __cellId = __cellz * geo.i_cell_sheet + __celly * geo.i_cell_dx + __cellx;

                //searching neighbors
                //loop: surrounding cells including itself (totally 27 cells)
                //loop: from bottom to top, from back to front, from left to right
                const mytype::integer __offset = 28 * __cellId;
                const mytype::integer __num    =      dev_i_link_cell[__offset];

                _n[p] = 0.0f;
                for(mytype::integer dir=1; dir<=__num; dir++)
                {
                    mytype::integer __cell = dev_i_link_cell[__offset + dir];

                    if(__cell < geo.i_num_cells)
                    {
                        mytype::integer __start = dev_i_cell_start[__cell];
                        mytype::integer __end   = dev_i_cell_end[__cell];

                        for(mytype::integer j=__start; j<__end; j++)
                        {
                            if(dev_i_type[j] == 0)
                            {
                                mytype::real __rr = (dev_d3_pos[j].x - _pos_i[p].x) * (dev_d3_pos[j].x - _pos_i[p].x)
                                                  + (dev_d3_pos[j].y - _pos_i[p].y) * (dev_d3_pos[j].y - _pos_i[p].y)
                                                  + (dev_d3_pos[j].z - _pos_i[p].z) * (dev_d3_pos[j].z - _pos_i[p].z);

                                _n[p] += dev_d_weight( d_rzero, sqrt(__rr) );
                            }
                        }
                    }
                }
            }

            dev_r3_norm[i].x = _n[1] - _n[0];
            dev_r3_norm[i].y = _n[3] - _n[2];
            dev_r3_norm[i].z = _n[5] - _n[4];
        }
    }

    void dev_calNorm     ( mytype::real3* const dev_r3_norm,
                     const mytype::real3* const dev_r3_triangle,
                     const mytype::real* const dev_r_alpha,
                     const mytype::real3* const dev_d3_pos,
                     const mytype::integer* const dev_i_type,
                     const mytype::integer* const dev_i_link_cell,
                     const mytype::integer* const dev_i_cell_start,
                     const mytype::integer* const dev_i_cell_end,
                     const mytype::real d_rzero,
                     const mytype::integer i_nMaxEdge,
                     const mytype::GEOMETRY geo )
    {
        kerCalNorm<<<(i_nMaxEdge+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>( dev_r3_norm,
                                                                                   dev_r3_triangle,
                                                                                   dev_r_alpha,
                                                                                   dev_d3_pos,
                                                                                   dev_i_type,
                                                                                   dev_i_link_cell,
                                                                                   dev_i_cell_start,
                                                                                   dev_i_cell_end,
                                                                                   d_rzero,
                                                                                   i_nMaxEdge,
                                                                                   geo );
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calNorm -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calNorm -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

}//namespace
