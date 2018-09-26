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
    __device__ inline mytype::real dev_d_weight( const mytype::real _r0,
                                                 const mytype::real _r )
    {
        //danger when _r == 0
        return (_r < _r0) ? (_r0 / _r - 1.0) : (0.0);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///cubic spline weight function
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __device__ inline mytype::real dev_r_cubic_weight(
                                                        const mytype::real _r0,
                                                        const mytype::real _r
                                                     )
    {
        mytype::real __q = _r / _r0;

        if(__q <= 0.5f) return 8.0f * mytype::OVERPI * (1 - 6*__q*__q + 6*__q*__q*__q);
        else if(__q <= 1.0f) return 16.0f * mytype::OVERPI * (1.0f-__q) * (1.0f-__q) * (1.0f-__q);
        else return 0.0f;
    }

    __device__ inline void vertexInterp(mytype::real3& edge,
                                  const mytype::real3 v1,
                                  const mytype::real3 v2,
                                  const mytype::real  n1,
                                  const mytype::real  n2,
                                  const mytype::real  iso)
    {

        mytype::real _dnInv = 1.f / (n2 - n1);

        //v == (iso*(v2-v1) + n2v1 - n1v2) / (n2 - n1)
        edge.x = (iso * (v2.x - v1.x) + v1.x * n2 - v2.x * n1 ) * _dnInv;
        edge.y = (iso * (v2.y - v1.y) + v1.y * n2 - v2.y * n1 ) * _dnInv;
        edge.z = (iso * (v2.z - v1.z) + v1.z * n2 - v2.z * n1 ) * _dnInv;

    }
    __device__ inline void vertexInterp0(mytype::real3& edge,
                                   const mytype::real3 v1,
                                   const mytype::real3 v2,
                                   const mytype::real  n1,
                                   const mytype::real  n2)
    {

        mytype::real _dnInv = 1.f / (n2 - n1);

        //v(n=0) == (n2v1 - n1v2) / (n2 - n1)
        edge.x = ( v1.x * n2 - v2.x * n1 ) * _dnInv;
        edge.y = ( v1.y * n2 - v2.y * n1 ) * _dnInv;
        edge.z = ( v1.z * n2 - v2.z * n1 ) * _dnInv;

    }
    __device__ inline void vertexInterp2(mytype::real3& edge,
                                   const mytype::real3 v1,
                                   const mytype::real3 v2)
    {
        edge.x = 0.5f * (v1.x + v2.x);
        edge.y = 0.5f * (v1.y + v2.y);
        edge.z = 0.5f * (v1.z + v2.z);
    }

    __device__ inline void normInterp(mytype::real3& norm,
                                const mytype::real v01,
                                const mytype::real v02,
                                const mytype::real3 n1,
                                const mytype::real3 n2)
    {
        norm.x = v02 * n1.x + v01 * n2.x;
        norm.y = v02 * n1.y + v01 * n2.y;
        norm.z = v02 * n1.z + v01 * n2.z;
    }

    __device__ inline void normInterp2(mytype::real3& norm,
                                 const mytype::real3 n1,
                                 const mytype::real3 n2)
    {
        norm.x = n1.x + n2.x;
        norm.y = n1.y + n2.y;
        norm.z = n1.z + n2.z;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///particle density at vertex
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalVertex_n( mytype::real* const dev_r_vertex_n,
                              const mytype::real3* const dev_r3_verList,
                              const mytype::real3* const dev_d3_pos,
                              const mytype::integer* const dev_i_type,
                              const mytype::integer* const dev_i_link_cell,
                              const mytype::integer* const dev_i_cell_start,
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
                    mytype::integer __end   = dev_i_cell_start[__cell+1];

                    for(mytype::integer j=__start;j<__end;j++)
                    {
                        if(dev_i_type[j] == 0)
                        {
                            mytype::real __rr = (dev_d3_pos[j].x - _pos_i.x) * (dev_d3_pos[j].x - _pos_i.x)
                                              + (dev_d3_pos[j].y - _pos_i.y) * (dev_d3_pos[j].y - _pos_i.y)
                                              + (dev_d3_pos[j].z - _pos_i.z) * (dev_d3_pos[j].z - _pos_i.z);

                            _n += dev_r_cubic_weight( d_rzero, sqrt(__rr) );
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

    /////////////////////////////////////////////////////////////////////////////////////////
    ///triangle calculator
    /////////////////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalTriangle ( mytype::real3* const dev_r3_triangle,
                                     mytype::real* const dev_r_alpha,
                               const mytype::real3* const dev_r3_verList,
                               const mytype::real* const dev_r_vertex_n,
                               const mytype::integer* const dev_i_voxList,
                               const mytype::real r_iso,
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
            _index  = uint(dev_r_vertex_n[_num[0]] > r_iso);
            _index += uint(dev_r_vertex_n[_num[1]] > r_iso) << 1;
            _index += uint(dev_r_vertex_n[_num[2]] > r_iso) << 2;
            _index += uint(dev_r_vertex_n[_num[3]] > r_iso) << 3;
            _index += uint(dev_r_vertex_n[_num[4]] > r_iso) << 4;
            _index += uint(dev_r_vertex_n[_num[5]] > r_iso) << 5;
            _index += uint(dev_r_vertex_n[_num[6]] > r_iso) << 6;
            _index += uint(dev_r_vertex_n[_num[7]] > r_iso) << 7;

            if(_index == 0x00 || _index == 0xff) return;

#if USE_SHARED_MEM

            extern __shared__ mytype::real3 _edge[];

            vertexInterp(_edge[              threadIdx.x], dev_r3_verList[_num[0]], dev_r3_verList[_num[1]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[1]], r_iso);
            vertexInterp(_edge[   blockDim.x+threadIdx.x], dev_r3_verList[_num[1]], dev_r3_verList[_num[2]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[2]], r_iso);
            vertexInterp(_edge[2* blockDim.x+threadIdx.x], dev_r3_verList[_num[2]], dev_r3_verList[_num[3]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[3]], r_iso);
            vertexInterp(_edge[3* blockDim.x+threadIdx.x], dev_r3_verList[_num[3]], dev_r3_verList[_num[0]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[0]], r_iso);

            vertexInterp(_edge[4* blockDim.x+threadIdx.x], dev_r3_verList[_num[4]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[4]], dev_r_vertex_n[_num[5]], r_iso);
            vertexInterp(_edge[5* blockDim.x+threadIdx.x], dev_r3_verList[_num[5]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[5]], dev_r_vertex_n[_num[6]], r_iso);
            vertexInterp(_edge[6* blockDim.x+threadIdx.x], dev_r3_verList[_num[6]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[6]], dev_r_vertex_n[_num[7]], r_iso);
            vertexInterp(_edge[7* blockDim.x+threadIdx.x], dev_r3_verList[_num[7]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[7]], dev_r_vertex_n[_num[4]], r_iso);

            vertexInterp(_edge[8* blockDim.x+threadIdx.x], dev_r3_verList[_num[0]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[4]], r_iso);
            vertexInterp(_edge[9* blockDim.x+threadIdx.x], dev_r3_verList[_num[1]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[5]], r_iso);
            vertexInterp(_edge[10*blockDim.x+threadIdx.x], dev_r3_verList[_num[2]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[6]], r_iso);
            vertexInterp(_edge[11*blockDim.x+threadIdx.x], dev_r3_verList[_num[3]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[7]], r_iso);

#else

            mytype::real3 _edge[12];
/*
            vertexInterp(_edge[0], dev_r3_verList[_num[0]], dev_r3_verList[_num[1]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[1]], r_iso);
            vertexInterp(_edge[1], dev_r3_verList[_num[1]], dev_r3_verList[_num[2]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[2]], r_iso);
            vertexInterp(_edge[2], dev_r3_verList[_num[2]], dev_r3_verList[_num[3]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[3]], r_iso);
            vertexInterp(_edge[3], dev_r3_verList[_num[3]], dev_r3_verList[_num[0]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[0]], r_iso);

            vertexInterp(_edge[4], dev_r3_verList[_num[4]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[4]], dev_r_vertex_n[_num[5]], r_iso);
            vertexInterp(_edge[5], dev_r3_verList[_num[5]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[5]], dev_r_vertex_n[_num[6]], r_iso);
            vertexInterp(_edge[6], dev_r3_verList[_num[6]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[6]], dev_r_vertex_n[_num[7]], r_iso);
            vertexInterp(_edge[7], dev_r3_verList[_num[7]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[7]], dev_r_vertex_n[_num[4]], r_iso);

            vertexInterp(_edge[8], dev_r3_verList[_num[0]], dev_r3_verList[_num[4]], dev_r_vertex_n[_num[0]], dev_r_vertex_n[_num[4]], r_iso);
            vertexInterp(_edge[9], dev_r3_verList[_num[1]], dev_r3_verList[_num[5]], dev_r_vertex_n[_num[1]], dev_r_vertex_n[_num[5]], r_iso);
            vertexInterp(_edge[10], dev_r3_verList[_num[2]], dev_r3_verList[_num[6]], dev_r_vertex_n[_num[2]], dev_r_vertex_n[_num[6]], r_iso);
            vertexInterp(_edge[11], dev_r3_verList[_num[3]], dev_r3_verList[_num[7]], dev_r_vertex_n[_num[3]], dev_r_vertex_n[_num[7]], r_iso);
*/
            vertexInterp2(_edge[0], dev_r3_verList[_num[0]], dev_r3_verList[_num[1]]);
            vertexInterp2(_edge[1], dev_r3_verList[_num[1]], dev_r3_verList[_num[2]]);
            vertexInterp2(_edge[2], dev_r3_verList[_num[2]], dev_r3_verList[_num[3]]);
            vertexInterp2(_edge[3], dev_r3_verList[_num[3]], dev_r3_verList[_num[0]]);

            vertexInterp2(_edge[4], dev_r3_verList[_num[4]], dev_r3_verList[_num[5]]);
            vertexInterp2(_edge[5], dev_r3_verList[_num[5]], dev_r3_verList[_num[6]]);
            vertexInterp2(_edge[6], dev_r3_verList[_num[6]], dev_r3_verList[_num[7]]);
            vertexInterp2(_edge[7], dev_r3_verList[_num[7]], dev_r3_verList[_num[4]]);

            vertexInterp2(_edge[8], dev_r3_verList[_num[0]], dev_r3_verList[_num[4]]);
            vertexInterp2(_edge[9], dev_r3_verList[_num[1]], dev_r3_verList[_num[5]]);
            vertexInterp2(_edge[10], dev_r3_verList[_num[2]], dev_r3_verList[_num[6]]);
            vertexInterp2(_edge[11], dev_r3_verList[_num[3]], dev_r3_verList[_num[7]]);

            uint _numVer = dev_u_numVerTable[_index];

            for(uint j=0; j<_numVer; j++)
            {
                uint __p = _index * 16 + j;

                dev_r3_triangle[12*i+j] = _edge[dev_u_triTable[__p]];
                //dev_r3_norm[12*i+j]     = _norm[dev_u_triTable[__p]];
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
                     const mytype::real r_iso,
                     const mytype::integer i_nVoxel,
                     const uint* const dev_u_numVerTable,
                     const uint* const dev_u_triTable )
    {
#if USE_SHARED_MEM

        ///call routines
        kerCalTriangle<<<(i_nVoxel+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S, sizeof(mytype::real3)*12*BLOCK_DIM_X_S>>>
                                                                              ( dev_r3_triangle,
                                                                                dev_r_alpha,
                                                                                dev_r3_verList,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                r_iso,
                                                                                i_nVoxel,
                                                                                dev_u_numVerTable,
                                                                                dev_u_triTable );
#else

        ///call routines
        kerCalTriangle<<<(i_nVoxel+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>
                                                                              ( dev_r3_triangle,
                                                                                dev_r_alpha,
                                                                                dev_r3_verList,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                r_iso,
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

    //////////////////////////////////////////////////////////////////////////////////
    ///norm calculator
    //////////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalNorm_1(mytype::real3* const dev_r3_vertex_norm,
                           const mytype::real* const dev_r_vertex_n,
                           const mytype::int3 i3_dim,
                           const mytype::integer i_nVertex)
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        ///problem in dealing with vertex at boundary
        if( i+(i3_dim.x)*(i3_dim.y) < i_nVertex && i >= (i3_dim.x)*(i3_dim.y) )
        {
            mytype::real _n[6];
            _n[0] = dev_r_vertex_n[i-1];
            _n[1] = dev_r_vertex_n[i+1];
            _n[2] = dev_r_vertex_n[i-i3_dim.x];
            _n[3] = dev_r_vertex_n[i+i3_dim.x];
            _n[4] = dev_r_vertex_n[i-i3_dim.x*i3_dim.y];
            _n[5] = dev_r_vertex_n[i+i3_dim.x*i3_dim.y];

            dev_r3_vertex_norm[i].x = _n[1] - _n[0];
            dev_r3_vertex_norm[i].y = _n[3] - _n[2];
            dev_r3_vertex_norm[i].z = _n[5] - _n[4];
        }
    }

    void dev_calNorm_1   ( mytype::real3* const dev_r3_vertex_norm,
                     const mytype::real* const dev_r_vertex_n,
                     const mytype::int3 i3_dim,
                     const mytype::integer i_nVertex )
    {
        kerCalNorm_1<<<(i_nVertex+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>( dev_r3_vertex_norm,
                                                                                    dev_r_vertex_n,
                                                                                    i3_dim,
                                                                                    i_nVertex );

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calNorm_1 -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calNorm_1 -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }


    __global__ void kerCalNorm_2 ( mytype::real3* const dev_r3_norm,
                             const mytype::real3* const dev_r3_triangle,
                             const mytype::real3* const dev_r3_verList,
                             const mytype::real3* const dev_r3_vertex_norm,
                             const mytype::real* const dev_r_vertex_n,
                             const mytype::integer* const dev_i_voxList,
                             const mytype::real r_iso,
                             const mytype::integer i_nVoxel,
                             const uint* const dev_u_numVerTable,
                             const uint* const dev_u_triTable )
    {
        const mytype::integer i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < i_nVoxel)
        {
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
            _index  = uint(dev_r_vertex_n[_num[0]] > r_iso);
            _index += uint(dev_r_vertex_n[_num[1]] > r_iso) << 1;
            _index += uint(dev_r_vertex_n[_num[2]] > r_iso) << 2;
            _index += uint(dev_r_vertex_n[_num[3]] > r_iso) << 3;
            _index += uint(dev_r_vertex_n[_num[4]] > r_iso) << 4;
            _index += uint(dev_r_vertex_n[_num[5]] > r_iso) << 5;
            _index += uint(dev_r_vertex_n[_num[6]] > r_iso) << 6;
            _index += uint(dev_r_vertex_n[_num[7]] > r_iso) << 7;

            if(_index == 0x00 || _index == 0xff) return;

#if USE_SHARED_MEM

            extern __shared__ mytype::real3 _norm[];

            normInterp(_norm[              threadIdx.x], dev_r3_triangle[12*i  ].x-dev_r3_verList[_num[0]].x, dev_r3_verList[_num[1]].x-dev_r3_triangle[12*i  ].x, dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[1]]);
            normInterp(_norm[   blockDim.x+threadIdx.x], dev_r3_triangle[12*i+1].y-dev_r3_verList[_num[1]].y, dev_r3_verList[_num[2]].y-dev_r3_triangle[12*i+1].y, dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[2]]);
            normInterp(_norm[2* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+2].x-dev_r3_verList[_num[2]].x, dev_r3_verList[_num[3]].x-dev_r3_triangle[12*i+2].x, dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[3]]);
            normInterp(_norm[3* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+3].y-dev_r3_verList[_num[3]].y, dev_r3_verList[_num[0]].y-dev_r3_triangle[12*i+3].y, dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[0]]);

            normInterp(_norm[4* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+4].x-dev_r3_verList[_num[4]].x, dev_r3_verList[_num[5]].x-dev_r3_triangle[12*i+4].x, dev_r3_vertex_norm[_num[4]], dev_r3_vertex_norm[_num[5]]);
            normInterp(_norm[5* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+5].y-dev_r3_verList[_num[5]].y, dev_r3_verList[_num[6]].y-dev_r3_triangle[12*i+5].y, dev_r3_vertex_norm[_num[5]], dev_r3_vertex_norm[_num[6]]);
            normInterp(_norm[6* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+6].x-dev_r3_verList[_num[6]].x, dev_r3_verList[_num[7]].x-dev_r3_triangle[12*i+6].x, dev_r3_vertex_norm[_num[6]], dev_r3_vertex_norm[_num[7]]);
            normInterp(_norm[7* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+7].y-dev_r3_verList[_num[7]].y, dev_r3_verList[_num[4]].y-dev_r3_triangle[12*i+7].y, dev_r3_vertex_norm[_num[7]], dev_r3_vertex_norm[_num[4]]);

            normInterp(_norm[8* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+8].z-dev_r3_verList[_num[0]].z, dev_r3_verList[_num[4]].z-dev_r3_triangle[12*i+8].z, dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[4]]);
            normInterp(_norm[9* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+9].z-dev_r3_verList[_num[1]].z, dev_r3_verList[_num[5]].z-dev_r3_triangle[12*i+9].z, dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[5]]);
            normInterp(_norm[10* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+10].z-dev_r3_verList[_num[2]].z, dev_r3_verList[_num[6]].z-dev_r3_triangle[12*i+10].z, dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[6]]);
            normInterp(_norm[11* blockDim.x+threadIdx.x], dev_r3_triangle[12*i+11].z-dev_r3_verList[_num[3]].z, dev_r3_verList[_num[7]].z-dev_r3_triangle[12*i+11].z, dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[7]]);

            uint _numVer = dev_u_numVerTable[_index];

            for(uint j=0; j<_numVer; j++)
            {
                uint __p = _index * 16 + j;

                dev_r3_norm[12*i+j]     = _norm[dev_u_triTable[__p] * blockDim.x + threadIdx.x];
            }
#else

            mytype::real3 _norm[12];
/*
            ///reducing # of registers
            normInterp(_norm[0], dev_r3_triangle[12*i  ].x-dev_r3_verList[_num[0]].x, dev_r3_verList[_num[1]].x-dev_r3_triangle[12*i  ].x, dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[1]]);
            normInterp(_norm[1], dev_r3_triangle[12*i+1].y-dev_r3_verList[_num[1]].y, dev_r3_verList[_num[2]].y-dev_r3_triangle[12*i+1].y, dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[2]]);
            normInterp(_norm[2], dev_r3_triangle[12*i+2].x-dev_r3_verList[_num[2]].x, dev_r3_verList[_num[3]].x-dev_r3_triangle[12*i+2].x, dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[3]]);
            normInterp(_norm[3], dev_r3_triangle[12*i+3].y-dev_r3_verList[_num[3]].y, dev_r3_verList[_num[0]].y-dev_r3_triangle[12*i+3].y, dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[0]]);

            normInterp(_norm[4], dev_r3_triangle[12*i+4].x-dev_r3_verList[_num[4]].x, dev_r3_verList[_num[5]].x-dev_r3_triangle[12*i+4].x, dev_r3_vertex_norm[_num[4]], dev_r3_vertex_norm[_num[5]]);
            normInterp(_norm[5], dev_r3_triangle[12*i+5].y-dev_r3_verList[_num[5]].y, dev_r3_verList[_num[6]].y-dev_r3_triangle[12*i+5].y, dev_r3_vertex_norm[_num[5]], dev_r3_vertex_norm[_num[6]]);
            normInterp(_norm[6], dev_r3_triangle[12*i+6].x-dev_r3_verList[_num[6]].x, dev_r3_verList[_num[7]].x-dev_r3_triangle[12*i+6].x, dev_r3_vertex_norm[_num[6]], dev_r3_vertex_norm[_num[7]]);
            normInterp(_norm[7], dev_r3_triangle[12*i+7].y-dev_r3_verList[_num[7]].y, dev_r3_verList[_num[4]].y-dev_r3_triangle[12*i+7].y, dev_r3_vertex_norm[_num[7]], dev_r3_vertex_norm[_num[4]]);

            normInterp(_norm[8], dev_r3_triangle[12*i+8].z-dev_r3_verList[_num[0]].z, dev_r3_verList[_num[4]].z-dev_r3_triangle[12*i+8].z, dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[4]]);
            normInterp(_norm[9], dev_r3_triangle[12*i+9].z-dev_r3_verList[_num[1]].z, dev_r3_verList[_num[5]].z-dev_r3_triangle[12*i+9].z, dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[5]]);
            normInterp(_norm[10], dev_r3_triangle[12*i+10].z-dev_r3_verList[_num[2]].z, dev_r3_verList[_num[6]].z-dev_r3_triangle[12*i+10].z, dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[6]]);
            normInterp(_norm[11], dev_r3_triangle[12*i+11].z-dev_r3_verList[_num[3]].z, dev_r3_verList[_num[7]].z-dev_r3_triangle[12*i+11].z, dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[7]]);
*/
            ///normInterp2
            normInterp2(_norm[0], dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[1]]);
            normInterp2(_norm[1], dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[2]]);
            normInterp2(_norm[2], dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[3]]);
            normInterp2(_norm[3], dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[0]]);

            normInterp2(_norm[4], dev_r3_vertex_norm[_num[4]], dev_r3_vertex_norm[_num[5]]);
            normInterp2(_norm[5], dev_r3_vertex_norm[_num[5]], dev_r3_vertex_norm[_num[6]]);
            normInterp2(_norm[6], dev_r3_vertex_norm[_num[6]], dev_r3_vertex_norm[_num[7]]);
            normInterp2(_norm[7], dev_r3_vertex_norm[_num[7]], dev_r3_vertex_norm[_num[4]]);

            normInterp2(_norm[8], dev_r3_vertex_norm[_num[0]], dev_r3_vertex_norm[_num[4]]);
            normInterp2(_norm[9], dev_r3_vertex_norm[_num[1]], dev_r3_vertex_norm[_num[5]]);
            normInterp2(_norm[10], dev_r3_vertex_norm[_num[2]], dev_r3_vertex_norm[_num[6]]);
            normInterp2(_norm[11], dev_r3_vertex_norm[_num[3]], dev_r3_vertex_norm[_num[7]]);

            uint _numVer = dev_u_numVerTable[_index];

            for(uint j=0; j<_numVer; j++)
            {
                uint __p = _index * 16 + j;

                dev_r3_norm[12*i+j]     = _norm[dev_u_triTable[__p]];
            }
#endif
        }
    }

    void dev_calNorm_2  ( mytype::real3* const dev_r3_norm,
                    const mytype::real3* const dev_r3_triangle,
                    const mytype::real3* const dev_r3_verList,
                    const mytype::real3* const dev_r3_vertex_norm,
                    const mytype::real* const dev_r_vertex_n,
                    const mytype::integer* const dev_i_voxList,
                    const mytype::real r_iso,
                    const mytype::integer i_nVoxel,
                    const uint* const dev_u_numVerTable,
                    const uint* const dev_u_triTable )
    {
#if USE_SHARED_MEM

        ///call routines
        kerCalNorm_2<<<(i_nVoxel+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S, sizeof(mytype::real3)*12*BLOCK_DIM_X_S>>>
                                                                              ( dev_r3_norm,
                                                                                dev_r3_triangle,
                                                                                dev_r3_verList,
                                                                                dev_r3_vertex_norm,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                r_iso,
                                                                                i_nVoxel,
                                                                                dev_u_numVerTable,
                                                                                dev_u_triTable );
#else

        ///call routines
        kerCalNorm_2<<<(i_nVoxel+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>
                                                                              ( dev_r3_norm,
                                                                                dev_r3_triangle,
                                                                                dev_r3_verList,
                                                                                dev_r3_vertex_norm,
                                                                                dev_r_vertex_n,
                                                                                dev_i_voxList,
                                                                                r_iso,
                                                                                i_nVoxel,
                                                                                dev_u_numVerTable,
                                                                                dev_u_triTable );
#endif

    #ifdef DEBUG
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();

        if(errSync != cudaSuccess) printf("dev_calNorm_2 -> Sync kernnel error: %s\n", cudaGetErrorString(errSync));
        if(errAsync != cudaSuccess) printf("dev_calNorm_2 -> Async kernnel error: %s\n", cudaGetErrorString(errAsync));
    #endif
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///legacy norm calculator
    ///////////////////////////////////////////////////////////////////////////////
    __global__ void kerCalNorm_legacy( mytype::real3* const dev_r3_norm,
                          const mytype::real3* const dev_r3_triangle,
                          const mytype::real* const dev_r_alpha,
                          const mytype::real3* const dev_d3_pos,
                          const mytype::integer* const dev_i_type,
                          const mytype::integer* const dev_i_link_cell,
                          const mytype::integer* const dev_i_cell_start,
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
                        mytype::integer __end   = dev_i_cell_start[__cell+1];

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

    void dev_calNorm_legacy     ( mytype::real3* const dev_r3_norm,
                     const mytype::real3* const dev_r3_triangle,
                     const mytype::real* const dev_r_alpha,
                     const mytype::real3* const dev_d3_pos,
                     const mytype::integer* const dev_i_type,
                     const mytype::integer* const dev_i_link_cell,
                     const mytype::integer* const dev_i_cell_start,
                     const mytype::real d_rzero,
                     const mytype::integer i_nMaxEdge,
                     const mytype::GEOMETRY geo )
    {
        kerCalNorm_legacy<<<(i_nMaxEdge+BLOCK_DIM_X_S-1)/BLOCK_DIM_X_S, BLOCK_DIM_X_S>>>( dev_r3_norm,
                                                                                   dev_r3_triangle,
                                                                                   dev_r_alpha,
                                                                                   dev_d3_pos,
                                                                                   dev_i_type,
                                                                                   dev_i_link_cell,
                                                                                   dev_i_cell_start,
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
