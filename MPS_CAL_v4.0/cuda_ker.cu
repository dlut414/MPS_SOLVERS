/*
LICENCE
*/
//cuda_ker.h
///implementation of cuda kernel functions

#include <cstdio>
#include <cassert>
#include <cublas_v2.h>

#include "cuda_ker.h"
#include "typedef.h"
#include "common.h"

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
        kerVecCpy<<<(n+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(des, src, n);

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
        kerAxpy<<<(n+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(z, x, y, a, n);

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
        kerMatVec<<<(n+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(des, mat, vec, n);

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

        kerVecVec<<<1, NUM_THREADS>>>(des, vec1, vec2, n);
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

        mytype::integer _num;

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
        checkCublas( cublasDcopy(handle, n, dev_b, 1, dev_r, 1), "Dcopy1" );
        checkCublas( cublasDgemv(handle, CUBLAS_OP_N, n, n, &_N_ONE, dev_A, n, dev_x, 1, &_P_ONE, dev_r, 1), "Dgemv1" );
        ///p = r
        checkCublas( cublasDcopy(handle, n, dev_r, 1, dev_p, 1), "Dcopy2" );
        ///_rrold = r*r
        checkCublas( cublasDdot(handle, n, dev_r, 1, dev_r, 1, &_rrold), "Ddot1" );

        _num = 0;
        while( _rrold > mytype::EPS_BY_EPS )
        {
            ///Ap = A*p
            checkCublas( cublasDgemv(handle, CUBLAS_OP_N, n, n, &_P_ONE, dev_A, n, dev_p, 1, &_ZERO, dev_Ap, 1), "Dgemv2" );
            ///_alpha = _rrold / Ap*p
            checkCublas( cublasDdot(handle, n, dev_Ap, 1, dev_p, 1, &_alpha), "Ddot2" );
            _alpha = _rrold / _alpha;

            ///x = x + _alpha*p
            checkCublas( cublasDaxpy(handle, n, &_alpha, dev_p, 1, dev_x, 1 ), "Daxpy1" );
            ///r = r - _alpha*Ap
            _alpha = -_alpha;
            checkCublas( cublasDaxpy(handle, n, &_alpha, dev_Ap, 1, dev_r, 1 ), "Daxpy2" );
            ///_rrnew = r*r
            checkCublas( cublasDdot(handle, n, dev_r, 1, dev_r, 1, &_rrnew), "Ddot2" );
            ///_rn_over_ro = _rrnew / _rrold
            _rn_over_ro = _rrnew / _rrold;
            ///p = _rn_over_ro*p + r
            checkCublas( cublasDscal(handle, n, &_rn_over_ro, dev_p, 1), "Dscal1" );
            checkCublas( cublasDaxpy(handle, n, &_P_ONE, dev_r, 1, dev_p, 1 ), "Daxpy3" );

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

/*
    __global__ void DoCG(mytype::real* dev_A, mytype::real* dev_x, mytype::real* dev_b,
                         mytype::real* dev_Ap, mytype::real* dev_p, mytype::real* dev_r,
                         mytype::integer n)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;
        const mytype::integer in = i * n;

        mytype::integer _num;

        __shared__ mytype::real _alpha;
        __shared__ mytype::real _rrold;
        __shared__ mytype::real _rrnew;

        _num = 0;

            mytype::real __tmp = 0.0;

            ///matVec & axpy
            if(i < n)
            {
                for(mytype::integer j=0;j<n;j++)
                {
                    __tmp += dev_A[in+j] * dev_x[j];
                }
                dev_p[i] = dev_r[i] = dev_b[i] - __tmp;
            }
            __syncthreads();

            ///vecVec
            if(threadIdx.x == 0)
            {
                __tmp = 0.0;
                for(mytype::integer j=0;j<n;j++)
                {
                    __tmp += dev_r[j] * dev_r[j];
                }
                _rrold = __tmp;
            }
            __syncthreads();

            //repeat
            while(_rrold > mytype::EPS_BY_EPS)
            {
                ///matVec
                __tmp = 0.0;
                if(i < n)
                {
                    for(mytype::integer j=0;j<n;j++)
                    {
                        __tmp += dev_A[in+j] * dev_p[j];
                    }
                    dev_Ap[i] = __tmp;
                }
                __syncthreads();

                ///vecVec
                __tmp = 0.0;
                if(threadIdx.x == 0)
                {
                    for(mytype::integer j=0;j<n;j++)
                    {
                        __tmp += dev_Ap[j] * dev_p[j];
                    }
                    _alpha = _rrold / __tmp;
                }
                __syncthreads();

                ///axpy
                if(i < n)
                {
                    dev_r[i] = dev_r[i] - _alpha * dev_Ap[i];
                    dev_x[i] = dev_x[i] + _alpha * dev_p[i];
                }
                __syncthreads();

                ///vecVec
                if(threadIdx.x == 0)
                {
                    __tmp = 0.0;
                    for(mytype::integer j=0;j<n;j++)
                    {
                        __tmp += dev_r[j] * dev_r[j];
                    }
                    _rrnew = __tmp;
                }
                __syncthreads();

                ///axpy
                if(i < n)
                {
                    dev_p[i] = _rrnew / _rrold * dev_p[i] + dev_r[i];
                }
                _rrold = _rrnew;
                //printf("CONVERGENCE -> RESIDUAL: %.2e\n",__rrnew);
                _num++;

                __syncthreads();
            }
    }
*/
}
