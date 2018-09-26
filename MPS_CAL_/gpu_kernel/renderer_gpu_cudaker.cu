/*
LICENCE
*/
//renderer_gpu_cudaker.h
///implementation of cuda kernel functions

#include <cmath>
#include <cstdio>
#include <cassert>

#include "mps_gpu_cudaker.h"
#include "../typedef.h"
#include "../common.h"
#include "../marchingCube_define.h"

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///====================================================================RENDERER===========================================================================///
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    ///particle number density function
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    __device__ inline mytype::real dev_d_weight( const mytype::real& _r0,
                                                 const mytype::real& _r )
    {
        //danger when _r == 0
        return (_r < _r0) ? (_r0 / _r - 1.f) : (0.f);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///cubic spline weight function
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __device__ inline mytype::real dev_r_cubic_weight(
                                                        const mytype::real& _r0,
                                                        const mytype::real& _r
                                                     )
    {
        mytype::real __q = _r / _r0;

        if(__q <= 0.5f) return 8.0f * mytype::OVERPI * (1 - 6*__q*__q + 6*__q*__q*__q);
        else if(__q <= 1.0f) return 16.0f * mytype::OVERPI * (1.0f-__q) * (1.0f-__q) * (1.0f-__q);
        else return 0.0f;
    }

    __device__ inline void vertexInterp(mytype::real3& edge,
                                  const mytype::real3& v1,
                                  const mytype::real3& v2,
                                  const mytype::real&  n1,
                                  const mytype::real&  n2,
                                  const mytype::real&  iso)
    {

        mytype::real _dnInv = 1.f / (n2 - n1);

        //v == (iso*(v2-v1) + n2v1 - n1v2) / (n2 - n1)
        edge.x = (iso * (v2.x - v1.x) + v1.x * n2 - v2.x * n1 ) * _dnInv;
        edge.y = (iso * (v2.y - v1.y) + v1.y * n2 - v2.y * n1 ) * _dnInv;
        edge.z = (iso * (v2.z - v1.z) + v1.z * n2 - v2.z * n1 ) * _dnInv;

    }
    __device__ inline void vertexInterp0(mytype::real3& edge,
                                   const mytype::real3& v1,
                                   const mytype::real3& v2,
                                   const mytype::real&  n1,
                                   const mytype::real&  n2)
    {

        mytype::real _dnInv = 1.f / (n2 - n1);

        //v(n=0) == (n2v1 - n1v2) / (n2 - n1)
        edge.x = ( v1.x * n2 - v2.x * n1 ) * _dnInv;
        edge.y = ( v1.y * n2 - v2.y * n1 ) * _dnInv;
        edge.z = ( v1.z * n2 - v2.z * n1 ) * _dnInv;

    }
    __device__ inline void vertexInterp2(mytype::real3& edge,
                                   const mytype::real3& v1,
                                   const mytype::real3& v2)
    {
        edge.x = 0.5f * (v1.x + v2.x);
        edge.y = 0.5f * (v1.y + v2.y);
        edge.z = 0.5f * (v1.z + v2.z);
    }

    __device__ inline void normInterp(mytype::real3& norm,
                                const mytype::real& v01,
                                const mytype::real& v02,
                                const mytype::real3& n1,
                                const mytype::real3& n2)
    {
        mytype::real __tmp01 = abs(v01), __tmp02 = abs(v02);
        norm.x = __tmp02 * n1.x + __tmp01 * n2.x;
        norm.y = __tmp02 * n1.y + __tmp01 * n2.y;
        norm.z = __tmp02 * n1.z + __tmp01 * n2.z;
    }

    __device__ inline void normInterp2(mytype::real3& norm,
                                 const mytype::real3& n1,
                                 const mytype::real3& n2)
    {
        norm.x = n1.x + n2.x;
        norm.y = n1.y + n2.y;
        norm.z = n1.z + n2.z;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///particle density at vertex
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalVertex_n( mytype::real* const dev_r_vertex_n,
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
                              const mytype::GEOMETRY geo )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;
/*
        if(i < i_marked)
        {
            const mytype::integer cellid = dev_i_markCell[i];

            const mytype::integer ratio = round(1 / VOXEL_SIZE);
            const mytype::integer ratio3 = ratio * ratio * ratio;

            for(mytype::integer vox=0; vox<ratio3; vox++)
            {
                        mytype::real  _n     = 0.0f;

                        const mytype::integer voxid = dev_i_cellToVox[ratio3 * cellid + vox];
                        const mytype::real3  _pos_i = dev_r3_verList[voxid];
                        const mytype::integer __offset = 28 * cellid;

                        for(mytype::integer dir=1;dir<=27;dir++)
                        {
                            const mytype::integer __cell = dev_i_link_cell[__offset + dir];

                            if(__cell < geo.i_num_cells)
                            {
                                const mytype::integer __start = dev_i_cell_start[__cell];
                                const mytype::integer __end   = dev_i_cell_end[__cell];

                                for(mytype::integer j=__start; j<__end; j++)
                                {
                                    if(dev_i_type[j] == 0)
                                    {
                                        mytype::real __rrx = (dev_d3_pos[j].x - _pos_i.x);
                                        mytype::real __rry = (dev_d3_pos[j].y - _pos_i.y);
                                        mytype::real __rrz = (dev_d3_pos[j].z - _pos_i.z);

                                        _n += dev_r_cubic_weight( d_rzero, sqrt(__rrx*__rrx + __rry*__rry + __rrz*__rrz) );
                                    }
                                }
                            }
                        }
                        dev_r_vertex_n[voxid] = _n;
            }
        }
*/

        if(i < i_marked)
        {
                        mytype::real  _n     = 0.0f;

                        const mytype::integer voxid = dev_i_markVox[i];
                        const mytype::real3  pos_i = dev_r3_verList[voxid];
                        const mytype::integer __offset = 28 * dev_i_voxToCell[voxid];

                        for(mytype::integer dir=1;dir<=27;dir++)
                        {
                            const mytype::integer __cell = dev_i_link_cell[__offset + dir];

                            if(__cell < geo.i_num_cells)
                            {
                                const mytype::integer __start = dev_i_cell_start[__cell];
                                const mytype::integer __end   = dev_i_cell_end[__cell];

                                for(mytype::integer j=__start; j<__end; j++)
                                {
                                    if(dev_i_type[j] == 0)
                                    {
                                        mytype::real __rrx = (dev_d3_pos[j].x - pos_i.x);
                                        mytype::real __rry = (dev_d3_pos[j].y - pos_i.y);
                                        mytype::real __rrz = (dev_d3_pos[j].z - pos_i.z);

                                        _n += dev_r_cubic_weight( d_rzero, sqrt(__rrx*__rrx + __rry*__rry + __rrz*__rrz) );
                                    }
                                }
                            }
                        }
                        dev_r_vertex_n[voxid] = _n;
        }
    }

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
                    const mytype::GEOMETRY geo )
    {
        ///call routines
        kerCalVertex_n<<<(i_marked+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>( dev_r_vertex_n,
                                                                                dev_i_markVox,
                                                                                dev_i_voxToCell,
                                                                                dev_r3_verList,
                                                                                dev_d3_pos,
                                                                                dev_i_type,
                                                                                dev_i_link_cell,
                                                                                dev_i_cell_start,
                                                                                dev_i_cell_end,
                                                                                d_rzero,
                                                                                i_nVertex,
                                                                                i_marked,
                                                                                geo );
    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calVertex_n -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calVertex_n -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    //////////////////////////////////////////////////////////////////////////////////
    ///norm calculator
    //////////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalNorm  (mytype::real3* const dev_r3_vertex_norm,
                           const mytype::integer* const dev_i_markVox,
                           const mytype::real* const dev_r_vertex_n,
                           const mytype::int3 i3_dim,
                           const mytype::integer i_nVertex,
                           const mytype::integer i_marked)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        ///problem in dealing with vertex at boundary
        /*
        if( i+(i3_dim.x)*(i3_dim.y) < i_nVertex && i >= (i3_dim.x)*(i3_dim.y) )
        {
            mytype::real _n[6];

            _n[0] = dev_r_vertex_n[i-1];
            _n[1] = dev_r_vertex_n[i+1];
            _n[2] = dev_r_vertex_n[i-i3_dim.x];
            _n[3] = dev_r_vertex_n[i+i3_dim.x];
            _n[4] = dev_r_vertex_n[i-i3_dim.x*i3_dim.y];
            _n[5] = dev_r_vertex_n[i+i3_dim.x*i3_dim.y];

            dev_r3_vertex_norm[i].x = _n[0] - _n[1];
            dev_r3_vertex_norm[i].y = _n[2] - _n[3];
            dev_r3_vertex_norm[i].z = _n[4] - _n[5];
        }
        */
        /*
        if(i < i_marked)
        {
            const mytype::integer voxid = dev_i_markVox[i];

            if(voxid+i3_dim.x*i3_dim.y >= i_nVertex || voxid-i3_dim.x*i3_dim.y >= i_nVertex) return;

            mytype::real _n[7];

            _n[0] = dev_r_vertex_n[voxid];
            _n[1] = dev_r_vertex_n[voxid+1];
            _n[2] = dev_r_vertex_n[voxid+i3_dim.x];
            _n[3] = dev_r_vertex_n[voxid+i3_dim.x*i3_dim.y];
            _n[4] = dev_r_vertex_n[voxid-1];
            _n[5] = dev_r_vertex_n[voxid-i3_dim.x];
            _n[6] = dev_r_vertex_n[voxid-i3_dim.x*i3_dim.y];

            if(_n[1] != 0x0fffffff) dev_r3_vertex_norm[voxid].x = _n[0] - _n[1];
            else dev_r3_vertex_norm[voxid].x = _n[4] - _n[0];
            if(_n[2] != 0x0fffffff) dev_r3_vertex_norm[voxid].y = _n[0] - _n[2];
            else dev_r3_vertex_norm[voxid].y = _n[5] - _n[0];
            if(_n[3] != 0x0fffffff) dev_r3_vertex_norm[voxid].z = _n[0] - _n[3];
            else dev_r3_vertex_norm[voxid].z = _n[6] - _n[0];
        }
        */
        if( i < i_marked )
        {
            const mytype::integer voxid = dev_i_markVox[i];

            if(voxid+i3_dim.x*i3_dim.y >= i_nVertex || voxid-i3_dim.x*i3_dim.y >= i_nVertex) return;

            mytype::real _n[6];

            _n[0] = dev_r_vertex_n[voxid-1];
            _n[1] = dev_r_vertex_n[voxid+1];
            _n[2] = dev_r_vertex_n[voxid-i3_dim.x];
            _n[3] = dev_r_vertex_n[voxid+i3_dim.x];
            _n[4] = dev_r_vertex_n[voxid-i3_dim.x*i3_dim.y];
            _n[5] = dev_r_vertex_n[voxid+i3_dim.x*i3_dim.y];

            dev_r3_vertex_norm[voxid].x = _n[0] - _n[1];
            dev_r3_vertex_norm[voxid].y = _n[2] - _n[3];
            dev_r3_vertex_norm[voxid].z = _n[4] - _n[5];
        }
    }

    void dev_calNorm     ( mytype::real3* const dev_r3_vertex_norm,
                     const mytype::integer* const dev_i_markVox,
                     const mytype::real* const dev_r_vertex_n,
                     const mytype::int3 i3_dim,
                     const mytype::integer i_nVertex,
                     const mytype::integer i_marked )
    {
        kerCalNorm  <<<(i_marked+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>( dev_r3_vertex_norm,
                                                                                    dev_i_markVox,
                                                                                    dev_r_vertex_n,
                                                                                    i3_dim,
                                                                                    i_nVertex,
                                                                                    i_marked );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calNorm_1 -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calNorm_1 -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    __global__ void kerZeroVert(mytype::real* const dev_r_vertex_n, mytype::real3* const dev_r3_norm,
                          const mytype::integer* const dev_i_cellInFluid,
                          const mytype::integer* const dev_i_voxToCell,
                          const mytype::integer i_nVert)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_nVert)
        {
            mytype::integer cell = dev_i_voxToCell[i];
            if( dev_i_cellInFluid[cell] > 0.f )
            {
                dev_r_vertex_n[i] = 0x0fffffff;
            }
            else
            {
                dev_r_vertex_n[i] = 0.f;
            }
            dev_r3_norm[i].x = dev_r3_norm[i].y = dev_r3_norm[i].z = 0.f;
        }
    }
    void dev_zeroVert   (mytype::real* const dev_r_vertex_n, mytype::real3* const dev_r3_norm,
                   const mytype::integer* const dev_i_cellInFluid,
                   const mytype::integer* const dev_i_voxToCell,
                   const mytype::integer i_nVert)
    {
        kerZeroVert<<<(i_nVert+BLOCK_DIM_X_L-1)/BLOCK_DIM_X_L, BLOCK_DIM_X_L>>>(dev_r_vertex_n, dev_r3_norm, dev_i_cellInFluid, dev_i_voxToCell, i_nVert);
    }

}//namespace
