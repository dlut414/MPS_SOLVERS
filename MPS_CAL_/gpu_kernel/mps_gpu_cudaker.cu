/*
LICENCE
*/
//mps_gpu_cudaker.cu
///implementation of cuda kernel functions

#include <cmath>
#include <cstdio>
#include <cassert>

#include "mps_gpu_cudaker.h"
#include "../typedef.h"
#include "../common.h"

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
        return (_r < _r0) ? (_r0 / _r - 1.0) : (0.0);
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
            mytype::real3 _ret            =      {0.f, 0.f, 0.f};
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
                mytype::real3 _ret          =      {0.f, 0.f, 0.f};
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
            mytype::real3 _crt = {0.f, 0.f, 0.f};

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

}//namespace
