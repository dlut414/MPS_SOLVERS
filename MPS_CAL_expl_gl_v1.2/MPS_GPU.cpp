/*
LICENCE
*/
//MPS_GPU.cu
//implementation of class MPS_GPU
///receive data from class MPS and functions of main loop
#include <cassert>

#include "def_incl.h"
#include "MPS_GPU.h"
#include "mps_gpu_cudaker.h"

using namespace mytype;

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

inline real MPS_GPU::d_weight(const real _r0, const real _r) const
{
    //danger when _r == 0
    if(_r >= _r0)   return 0.0f;
    else            return (_r0 / _r - 1.0f);
}


MPS_GPU::MPS_GPU()
{
    fid_log = NULL;
    fid_out = NULL;

    i_tmp   = NULL;
    d_tmp   = NULL;
    i3_tmp  = NULL;
    d3_tmp  = NULL;
/*
    dev_i_tmp   = NULL;
    dev_i3_tmp  = NULL;
    dev_d_tmp   = NULL;
    dev_d3_tmp  = NULL;
*/
    i_ini_time  = 0;
    d_cal_time  = 0;
    t_loop_s    = 0;
    t_loop_e    = 0;

    dev_i_type          = NULL;
    dev_i_cell_list     = NULL;//cell # of each particle
    dev_i_link_cell     = NULL;//neighbor cells of cell i (27 in 3d)
    dev_i_part_in_cell  = NULL;//# of particles in each cell
    dev_i_cell_start    = NULL;//starting index of each cell
    dev_i_cell_end      = NULL;//end index of each cell
    dev_i_normal        = NULL;//corresponding 1 type particle of 2 type
    dev_d_press         = NULL;//pre: pressure list of particles
    dev_d_n             = NULL;//n: particle number density of particles
    dev_d3_pos          = NULL;//pos: position list of particles
    dev_d3_vel          = NULL;//vel: velocity list of particles
}

MPS_GPU::~MPS_GPU()
{
    fid_log = NULL;
    fid_out = NULL;

    delete[] i_tmp;
    delete[] d_tmp;
    delete[] i3_tmp;
    delete[] d3_tmp;

#ifdef GPU_CUDA
    //devFree();
#endif

}

///////////////////////////////
///initialization
///////////////////////////////
void MPS_GPU::Initial()
{
    char str[256];

    /*-----time of initialization-----*/
    t_loop_s = time(NULL);
    /*--------------------------------*/

    /*-----initialization of parent class-----*/
    loadCase();//must at first
    createGeo();
    /*----------------------------------------*/

    /*-----initialization of variables in MPS_CPU-----*/
    strcpy(c_name , "0000.out");
    i_step = 0;
    d_time = 0.0;

    i_ini_time = 0;
    d_cal_time = 0;

    i_tmp  = new integer[i_np];
    d_tmp  = new real[i_np];
    i3_tmp = new int3[i_np];
    d3_tmp = new real3[i_np];

    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(real),    i_np);
    memAdd(sizeof(int3),    i_np);
    memAdd(sizeof(real3),   i_np);
    /*------------------------------------------------*/

    /*-----initialize variables-----*/
    ///-----make the cell-list-----
    allocateCellList();//calculate num of cells and allocate memory
    divideCell();
#ifdef GPU_CUDA
    sort_all();
#endif
    makeLink();
    ///----------------------------
    calOnce();
#ifdef GPU_CUDA
    devInit();
#endif
    /*------------------------------*/

    /*-----end of initialization-----*/
    t_loop_e    = time(NULL);
    i_ini_time  = difftime(t_loop_e , t_loop_s);
    /*-------------------------------*/

    sprintf(str, "successfully Initialized!\n");
    throwScn(str);
    //throwLog(fid_log, str);
    sprintf(str, "memory usage: %.1f M byte.\n", d_mem / 1024);
    throwScn(str);
    //throwLog(fid_log, str);
}

/////////////////////////////////////
///calculate constant parameters
/////////////////////////////////////
void MPS_GPU::calOnce()
{
    char str[256];
    FILE*   _fid_log;
    integer _i_tmp;
    real    _r  = 0.0f;

    d_nzero     = 0.0f;
    d_lambda    = 0.0f;

    //nzero
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        real __tmp = 0.0f;
        ///-----note: using result of son class !!
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        integer __offset    = 28 * i_cell_list[i];
        integer __num       =      i_link_cell[__offset];

        for(integer dir=1;dir<=__num;dir++)
        {
            integer __cell = i_link_cell[__offset+dir];

            if(__cell >= 0 && __cell < i_num_cells)
            {
                integer __start = i_cell_start[__cell];
                integer __end   = i_cell_end[__cell];

                for(integer j=__start;j<__end;j++)
                {
                    if(j != i)
                    {
                        real3 __dr = d3_pos[j] - d3_pos[i];

                        __tmp += d_weight(d_rzero, sqrt( __dr * __dr ));
                    }
                }
            }
        }

        d_n[i] = __tmp;
        ///-------------------------------------------

        if(__tmp > d_nzero)
        {
            #ifdef CPU_OMP
                #pragma omp critical
            #endif
            {
                d_nzero = __tmp;
                _i_tmp = i;
            }
        }
    }

    //lambda
    for(integer i=0;i<i_np;i++)
    {
        if(i != _i_tmp)
        {
            _r = (d3_pos[i].x - d3_pos[_i_tmp].x) * (d3_pos[i].x - d3_pos[_i_tmp].x)

               + (d3_pos[i].y - d3_pos[_i_tmp].y) * (d3_pos[i].y - d3_pos[_i_tmp].y)

               + (d3_pos[i].z - d3_pos[_i_tmp].z) * (d3_pos[i].z - d3_pos[_i_tmp].z);

            _r = sqrt(_r);

            d_lambda += _r * _r * d_weight(d_rzero, _r);
        }
    }
    d_lambda = d_lambda / d_nzero;

    //alpha
    d_alpha = 1.0 / (d_rho * d_cs * d_cs);

    //maximum of dt
    d_dt = d_dt_max;

    //1.0 / d_*
    d_one_over_alpha            = d_rho * d_cs * d_cs;
    d_one_over_rho              = 1.0 / d_rho;
    d_one_over_nzero            = 1.0 / d_nzero;
    d_one_over_dim              = 1.0 / real(i_dim);
    d_one_over_dt               = 1.0 / d_dt;
    d_2bydim_over_nzerobylambda = (2.0 * i_dim) / (d_nzero * d_lambda);

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    //calculate normal
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] == 2)
        {
            real __dis = 2 * d_rzero * d_rzero;

            for(integer j=0;j<i_np;j++)
            {
                if(i_type[j] == 1)
                {
                    real __rr = (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]);

                    if(__rr < __dis)
                    {
                        __dis = __rr;
                        i_normal[i] = j;
                    }
                }
            }
        }
    }

    _fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Calculated const parameters!\n");
    throwScn(str);
    throwLog(_fid_log, str);
    fclose(_fid_log);

}

///////////////////////////////
///device init
///////////////////////////////
void MPS_GPU::devInit()
{
    checkCuda( cudaMalloc(&dev_i_id_tmp,           i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_type_tmp,         i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_cell_list_tmp,    i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_normal_tmp,       i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_d_press_tmp,        i_np*sizeof(real)) );
    checkCuda( cudaMalloc(&dev_d_n_tmp,            i_np*sizeof(real)) );
    checkCuda( cudaMalloc(&dev_d3_pos_tmp,         i_np*sizeof(real3)) );
    checkCuda( cudaMalloc(&dev_d3_vel_tmp,         i_np*sizeof(real3)) );

    checkCuda( cudaMalloc(&dev_i_id,               i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_index,            i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_type,             i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_normal,           i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_cell_list,        i_np*sizeof(integer)) );

    checkCuda( cudaMalloc(&dev_i_link_cell,     28*i_num_cells*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_part_in_cell,     i_num_cells*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_cell_start,       i_num_cells*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_i_cell_end,         i_num_cells*sizeof(integer)) );

    checkCuda( cudaMalloc(&dev_d_n,                i_np*sizeof(real)) );
    checkCuda( cudaMalloc(&dev_d_press,            i_np*sizeof(real)) );
    checkCuda( cudaMalloc(&dev_d3_pos,             i_np*sizeof(real3)) );
    checkCuda( cudaMalloc(&dev_d3_vel,             i_np*sizeof(real3)) );

    checkCuda( cudaMemcpy(dev_i_id,             i_id,               i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_index,          i_index,            i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_type,           i_type,             i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_normal,         i_normal,           i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_list,      i_cell_list,        i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_link_cell,      i_link_cell,     28*i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_part_in_cell,   i_part_in_cell,     i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_start,     i_cell_start,       i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_end,       i_cell_end,         i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );

    checkCuda( cudaMemcpy(dev_d_n,          d_n,            i_np*sizeof(real),              cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_d_press,      d_press,        i_np*sizeof(real),              cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_d3_pos,       d3_pos,         i_np*sizeof(real3),             cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_d3_vel,       d3_vel,         i_np*sizeof(real3),             cudaMemcpyHostToDevice) );
}

///////////////////////////////
///device free
///////////////////////////////
void MPS_GPU::devFree()
{
    checkCuda( cudaFree(dev_i_id_tmp) );
    checkCuda( cudaFree(dev_i_type_tmp) );
    checkCuda( cudaFree(dev_i_cell_list_tmp) );
    checkCuda( cudaFree(dev_i_normal_tmp) );
    checkCuda( cudaFree(dev_d_press_tmp) );
    checkCuda( cudaFree(dev_d_n_tmp) );
    checkCuda( cudaFree(dev_d3_pos_tmp) );
    checkCuda( cudaFree(dev_d3_vel_tmp) );

    checkCuda( cudaFree(dev_i_id) );
    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_i_type) );
    checkCuda( cudaFree(dev_i_cell_list) );
    checkCuda( cudaFree(dev_i_link_cell) );
    checkCuda( cudaFree(dev_i_part_in_cell) );
    checkCuda( cudaFree(dev_i_cell_start) );
    checkCuda( cudaFree(dev_i_cell_end) );
    checkCuda( cudaFree(dev_i_normal) );

    checkCuda( cudaFree(dev_d_press) );
    checkCuda( cudaFree(dev_d_n) );
    checkCuda( cudaFree(dev_d3_pos) );
    checkCuda( cudaFree(dev_d3_vel) );
}

//////////////////////////////////////
///communication from host to device
//////////////////////////////////////
void MPS_GPU::commHostToDevice()
{
    checkCuda( cudaMemcpy(dev_i_index,          i_index,            i_np*sizeof(integer),               cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_list,      i_cell_list,        i_np*sizeof(integer),               cudaMemcpyHostToDevice) );

    checkCuda( cudaMemcpy(dev_i_part_in_cell,   i_part_in_cell,     i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_start,     i_cell_start,       i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_i_cell_end,       i_cell_end,         i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) );

}

//////////////////////////////////////
///communication from device to host
//////////////////////////////////////
void MPS_GPU::commDeviceToHost()
{
    checkCuda( cudaMemcpy(d3_pos,       dev_d3_pos,         i_np*sizeof(real3),             cudaMemcpyDeviceToHost) );

}

/////////////////////////////////////////
///main loop
/////////////////////////////////////////
void MPS_GPU::step()
{
    long _start, _end;
    real d_part1;
    real d_part2;
    real d_part3;

    /*-----boundary motion-----*/
    //motion();
    //divide_cell();
    /*-------------------------*/

    /*-----devide cells-----*/
    _start = getSystemTime();
    divideCell();//cpu
    _end   = getSystemTime();
    d_part1 = _end - _start;
#ifdef GPU_CUDA
    _start = getSystemTime();
    commHostToDevice();
    _end   = getSystemTime();
    d_part2 = _end - _start;
#endif
    /*----------------------*/

    _start = getSystemTime();

#ifdef GPU_CUDA
    cudaker::dev_sort_all( dev_i_id_tmp,
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
#endif

    calVisc_expl();//gpu
    collision   ();//gpu
    cal_n       ();//gpu
    calPres_expl();//gpu
    calDash     ();//gpu

    _end = getSystemTime();
    d_part3 = _end - _start;
#ifdef GPU_CUDA
    _start = getSystemTime();
    commDeviceToHost();
    _end   = getSystemTime();
    d_part2 += _end - _start;
#endif
    update_dt();
    i_step++;

    printf("            divi time ->   %.3f \n", d_part1);
    printf("            comm time ->   %.3f \n", d_part2);
    printf("            calc time ->   %.3f \n", d_part3);
}

////////////////////////////////////
///set all values in p to zero
////////////////////////////////////
template<typename T>
void MPS_GPU::Zero(T* p, integer n)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<n;i++) p[i] = 0;
}

////////////////////////////////////
///make link list
////////////////////////////////////
void MPS_GPU::makeLink()
{
    //make link_cell list
    if(i_dim == 2)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(integer i=0;i<i_num_cells;i++)
        {
            integer __offset = 28 * i;

            i_link_cell[__offset + 0] = 9;

            i_link_cell[__offset + 1] = i - i_cell_dx - 1;  //-1,-1
            i_link_cell[__offset + 2] = i - i_cell_dx;      //-1,0
            i_link_cell[__offset + 3] = i - i_cell_dx + 1;  //-1,1

            i_link_cell[__offset + 4] = i - 1;              //0,-1
            i_link_cell[__offset + 5] = i;                  //0,0
            i_link_cell[__offset + 6] = i + 1;              //0,1

            i_link_cell[__offset + 7] = i + i_cell_dx - 1;  //1,-1
            i_link_cell[__offset + 8] = i + i_cell_dx;      //1,0
            i_link_cell[__offset + 9] = i + i_cell_dx + 1;  //1,1
        }
    }
    else if(i_dim == 3)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(integer i=0;i<i_num_cells;i++)
        {
            integer __offset = 28 * i;

            i_link_cell[__offset + 0] = 27;

            i_link_cell[__offset + 1] = i - i_cell_sheet - i_cell_dx - 1;   //-1,-1,-1
            i_link_cell[__offset + 2] = i - i_cell_sheet - i_cell_dx;       //-1,-1,0
            i_link_cell[__offset + 3] = i - i_cell_sheet - i_cell_dx + 1;   //-1,-1,1

            i_link_cell[__offset + 4] = i - i_cell_sheet - 1;               //-1,0,-1
            i_link_cell[__offset + 5] = i - i_cell_sheet;                   //-1,0,0
            i_link_cell[__offset + 6] = i - i_cell_sheet + 1;               //-1,0,1

            i_link_cell[__offset + 7] = i - i_cell_sheet + i_cell_dx - 1;   //-1,1,-1
            i_link_cell[__offset + 8] = i - i_cell_sheet + i_cell_dx;       //-1,1,0
            i_link_cell[__offset + 9] = i - i_cell_sheet + i_cell_dx + 1;   //-1,1,1

            i_link_cell[__offset + 10] = i - i_cell_dx - 1;                 //0,-1,-1
            i_link_cell[__offset + 11] = i - i_cell_dx;                     //0,-1,0
            i_link_cell[__offset + 12] = i - i_cell_dx + 1;                 //0,-1,1

            i_link_cell[__offset + 13] = i - 1;                             //0,0,-1
            i_link_cell[__offset + 14] = i;                                 //0,0,0
            i_link_cell[__offset + 15] = i + 1;                             //0,0,1

            i_link_cell[__offset + 16] = i + i_cell_dx - 1;                 //0,1,-1
            i_link_cell[__offset + 17] = i + i_cell_dx;                     //0,1,0
            i_link_cell[__offset + 18] = i + i_cell_dx + 1;                 //0,1,1

            i_link_cell[__offset + 19] = i + i_cell_sheet - i_cell_dx - 1;  //1,-1,-1
            i_link_cell[__offset + 20] = i + i_cell_sheet - i_cell_dx;      //1,-1,0
            i_link_cell[__offset + 21] = i + i_cell_sheet - i_cell_dx + 1;  //1,-1,1

            i_link_cell[__offset + 22] = i + i_cell_sheet - 1;              //1,0,-1
            i_link_cell[__offset + 23] = i + i_cell_sheet;                  //1,0,0
            i_link_cell[__offset + 24] = i + i_cell_sheet + 1;              //1,0,1

            i_link_cell[__offset + 25] = i + i_cell_sheet + i_cell_dx - 1;  //1,1,-1
            i_link_cell[__offset + 26] = i + i_cell_sheet + i_cell_dx;      //1,1,0
            i_link_cell[__offset + 27] = i + i_cell_sheet + i_cell_dx + 1;  //1,1,1
        }
    }
}

////////////////////////////////
///divide the domain into cells
////////////////////////////////
void MPS_GPU::divideCell()
{
    integer _excl = 0;

    Zero(i_part_in_cell, i_num_cells);
    Zero(i_cell_start, i_num_cells);

    //divide particles into cells
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] != 3)
        {
            integer __cellx = integer( (d3_pos[i].x - d_cell_left)   / d_cell_size );
            integer __celly = integer( (d3_pos[i].y - d_cell_back)   / d_cell_size );
            integer __cellz = integer( (d3_pos[i].z - d_cell_bottom) / d_cell_size );
            integer __num   = __cellz * i_cell_sheet + __celly * i_cell_dx + __cellx;

            if(__cellx < 0 || __celly < 0 || __cellz < 0 ||
               __cellx >= i_cell_dx || __celly >= i_cell_dy || __cellz >= i_cell_dz)
            {
                char str[256];
                sprintf(str,"particle exceeding cell -> id: %d, cellx: %d, celly: %d, cellz: %d exc. total: %d\n",
                            i_id[i], __cellx, __celly, __cellz, ++i_nexcl);
                throwScn(str);
                //throwLog(fid_log, str);
                i_type[i] = 3;
                d3_vel[i].x = d3_vel[i].y = d3_vel[i].z = 0.0;
                _excl++;
                continue;
                /*exit(4);*/
            }

            #ifdef CPU_OMP
                #pragma omp atomic
            #endif
            i_part_in_cell[__num]++;
            i_cell_list[i] = __num;
        }
    }

    //start in each cell
    //initialize i_cell_end
    //can not be paralleled !
    i_cell_start[0] = i_cell_end[0] = 0;
    for(integer i=1;i<i_num_cells;i++)
    {
        i_cell_end[i] = i_cell_start[i] = i_cell_start[i-1] + i_part_in_cell[i-1];
    }

    //put particles into cells
    //can not be paralleled !
    //*1).i_index -> new , i -> old; 2).i_index -> old , i -> new, which is better?
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] != 3)
        {
            integer __cell_index = i_cell_list[i];

            i_index[i] = i_cell_end[__cell_index];
            i_cell_end[__cell_index]++;
        }
        else
        {
            i_index[i] = i_np - 1;
        }
    }

#ifndef GPU_CUDA
    sort_all();
#endif

    i_np -= _excl;//delete excluded particles

}

////////////////////////////////
///sort particle by cell
////////////////////////////////
void MPS_GPU::sort_all()
{
    sort_i(i_id);
    sort_i(i_type);
    sort_i(i_cell_list);
    sort_d(d_press);
    sort_d(d_n);
    sort_d3(d3_pos);
    sort_d3(d3_vel);

    sort_normal();

}

///////////////////////////////////////////////////////////
///sorting functing of normals
///////////////////////////////////////////////////////////
void MPS_GPU::sort_normal()
{
    //pretend content is unchanged
    sort_i(i_normal);

#ifdef GPU_CUDA_
    int _byte = i_np*sizeof(integer);

    integer* dev_i_index;
    integer* dev_p;

    checkCuda( cudaMalloc(&dev_i_index, i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_p, _byte) );

    checkCuda( cudaMemcpy(dev_i_index, i_index, i_np*sizeof(integer), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_p, i_normal, _byte, cudaMemcpyHostToDevice) );

    cudaker::dev_sort_normal(dev_p, dev_i_index, i_np);

    checkCuda( cudaMemcpy(i_normal, dev_p, _byte, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_p) );
#else
    //change content
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        i_normal[i] = i_index[i_normal[i]];
    }
#endif

}

///////////////////////////////////////////////////////////
///sorting function putting partcles into new index, int
///////////////////////////////////////////////////////////
void MPS_GPU::sort_i(integer* const __p)
{
#ifdef GPU_CUDA_
    int _byte = i_np*sizeof(integer);

    integer* dev_i_index;
    integer* dev_p;
    integer* dev_tmp;

    checkCuda( cudaMalloc(&dev_tmp, _byte) );
    checkCuda( cudaMalloc(&dev_i_index, i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_p, _byte) );

    checkCuda( cudaMemcpy(dev_i_index, i_index, i_np*sizeof(integer), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_p, __p, _byte, cudaMemcpyHostToDevice) );

    cudaker::dev_sort_i(dev_tmp, dev_p, dev_i_index, i_np);

    checkCuda( cudaMemcpy(__p, dev_tmp, _byte, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(dev_tmp) );
    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_p) );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        i_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        __p[i] = i_tmp[i];
    }

#endif

}

///////////////////////////////////////////////////////////
///sorting function putting partcles into new index, double
///////////////////////////////////////////////////////////
void MPS_GPU::sort_d(real* const __p)
{
#ifdef GPU_CUDA_
    int _byte = i_np*sizeof(real);

    integer* dev_i_index;
    real* dev_p;
    real* dev_tmp;

    checkCuda( cudaMalloc(&dev_tmp, _byte) );
    checkCuda( cudaMalloc(&dev_i_index, i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_p, _byte) );

    checkCuda( cudaMemcpy(dev_i_index, i_index, i_np*sizeof(integer), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_p, __p, _byte, cudaMemcpyHostToDevice) );

    cudaker::dev_sort_d(dev_tmp, dev_p, dev_i_index, i_np);

    checkCuda( cudaMemcpy(__p, dev_tmp, _byte, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(dev_tmp) );
    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_p) );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        d_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        __p[i] = d_tmp[i];
    }

#endif

}

///////////////////////////////////////////////////////////
///sorting function putting partcles into new index, int3
///////////////////////////////////////////////////////////
void MPS_GPU::sort_i3(int3* const __p)
{
#ifdef GPU_CUDA_
    int _byte = i_np*sizeof(int3);

    integer* dev_i_index;
    int3* dev_p;
    int3* dev_tmp;

    checkCuda( cudaMalloc(&dev_tmp, _byte) );
    checkCuda( cudaMalloc(&dev_i_index, i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_p, _byte) );

    checkCuda( cudaMemcpy(dev_i_index, i_index, i_np*sizeof(integer), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_p, __p, _byte, cudaMemcpyHostToDevice) );

    cudaker::dev_sort_i3(dev_tmp, dev_p, dev_i_index, i_np);

    checkCuda( cudaMemcpy(__p, dev_tmp, _byte, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(dev_tmp) );
    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_p) );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        i3_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        __p[i] = i3_tmp[i];
    }

#endif

}

///////////////////////////////////////////////////////////
///sorting function putting partcles into new index, double3
///////////////////////////////////////////////////////////
void MPS_GPU::sort_d3(real3* const __p)
{
#ifdef GPU_CUDA_
    int _byte = i_np*sizeof(real3);

    integer* dev_i_index;
    real3* dev_p;
    real3* dev_tmp;

    checkCuda( cudaMalloc(&dev_tmp, _byte) );
    checkCuda( cudaMalloc(&dev_i_index, i_np*sizeof(integer)) );
    checkCuda( cudaMalloc(&dev_p, _byte) );

    checkCuda( cudaMemcpy(dev_i_index, i_index, i_np*sizeof(integer), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_p, __p, _byte, cudaMemcpyHostToDevice) );

    cudaker::dev_sort_d3(dev_tmp, dev_p, dev_i_index, i_np);

    checkCuda( cudaMemcpy(__p, dev_tmp, _byte, cudaMemcpyDeviceToHost) );

    checkCuda( cudaFree(dev_tmp) );
    checkCuda( cudaFree(dev_i_index) );
    checkCuda( cudaFree(dev_p) );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        d3_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        __p[i] = d3_tmp[i];
    }

#endif

}

/////////////////////////////
///gradient of press at i
/////////////////////////////
real3 MPS_GPU::d3_GradPres(const integer& i)
{
    real3 _ret = {0,0,0};
    real _hat_p = d_press[i];
    const integer _offset = 28 * i_cell_list[i];
    const integer _num = i_link_cell[_offset];

    //searching _hat_p (minimum of p in 27 cells)
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_offset + dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                //ignore type 2 particles
                if(i_type[j] != 2)
                {
                    real __rr = (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]);

                    if( d_press[j] < _hat_p && __rr <= (d_rzero*d_rzero) )
                    {
                        _hat_p = d_press[j];
                    }
                }
            }
        }
    }

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_offset+dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                //ignore type 2 and i itself
                //if(i_type[j] != 2 && j != i)
                //{
                if(j != i)
                {
                    real3 __dr = d3_pos[j] - d3_pos[i];
                    real __rr = __dr * __dr;

                    _ret = _ret + (d_press[j] - _hat_p) / __rr * d_weight(d_rzero,sqrt(__rr)) * __dr;
                }
                //}
            }
        }
    }

    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}

//////////////////////////////////////////
///divergence of velocity at i (unused)
//////////////////////////////////////////
real MPS_GPU::d_DivVel(const integer& i)
{
    real _ret = 0.0;
    integer _offset = 28 * i_cell_list[i];
    integer _num = i_link_cell[_offset];

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_offset+dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                if(j != i)
                {
                    real3 __dr = d3_pos[j] - d3_pos[i];
                    real3 __du = d3_vel[j] - d3_vel[i];
                    real __rr = __dr * __dr;

                    _ret = _ret + d_weight(d_rzero,sqrt(__rr)) / __rr * (__du * __dr);
                }
            }
        }
    }

    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}

/////////////////////////////////////////
///Laplacian of velocity at i
/////////////////////////////////////////
real3 MPS_GPU::d3_LapVel(const integer& i)
{
    integer _offset = 28 * i_cell_list[i];
    integer _num = i_link_cell[_offset];
    real3 _ret = {0.0, 0.0, 0.0};

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_offset+dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                if(j != i)
                {
                    real3 __dr = d3_pos[j] - d3_pos[i];
                    real3 __du = d3_vel[j] - d3_vel[i];

                    _ret = _ret + d_weight(d_rlap , sqrt( __dr * __dr )) * __du;
                }
            }
        }
    }

    _ret = (d_2bydim_over_nzerobylambda) * _ret;

    return _ret;
}

/////////////////////////////////////////
///laplacian of pressure at i
/////////////////////////////////////////
real MPS_GPU::d_LapPres(const integer& i)
{
    integer _offset = 28 * i_cell_list[i];
    integer _num = i_link_cell[_offset];
    real _ret = 0.0;

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_offset+dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                if(i_type[j] != 2 && j != i)
                {
                    real  __dp = d_press[j] - d_press[i];
                    real3 __dr = d3_pos[j] - d3_pos[i];

                    _ret = _ret + __dp * d_weight(d_rlap , sqrt( __dr * __dr ));
                }
            }
        }
    }

    _ret = (d_2bydim_over_nzerobylambda) * _ret;

    return _ret;
}

//////////////////////////////
///update dt
//////////////////////////////
void MPS_GPU::update_dt()
{
    d_dt = d_dt_max;
    for(integer i = 0; i < i_np; i++)
    {
        real __dt_tmp = d_CFL * d_dp / sqrt(d3_vel[i] * d3_vel[i]); //note: d3_vel != 0
        if(__dt_tmp < d_dt) d_dt = __dt_tmp;
    }

    if(d_dt < d_dt_min)
    {
        sprintf(c_log, "error: dt -> %e, too small \n", d_dt);
        throwScn(c_log);
        //throwLog(fid_log,c_log);
        exit(1);
    }

    printf("        dt:     %.4f \n", d_dt);
}

//////////////////////////////
///calculate dash part
//////////////////////////////
void MPS_GPU::calDash()
{
#ifdef GPU_CUDA

    ///combine two steps into one
    cudaker::dev_calDash( dev_d3_vel,
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

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        //only apply to fluid particles
        if(i_type[i] == 0)
        {
            real3 __tmp = - d_dt * d_one_over_rho * d3_GradPres(i);

            d3_vel[i] +=        __tmp;
            d3_pos[i] += d_dt * __tmp;
        }
    }

#endif

}

//////////////////////////////
///particle number density
//////////////////////////////
void MPS_GPU::cal_n()
{
#ifdef GPU_CUDA

    cudaker::dev_cal_n( dev_d_n,
                        dev_d3_pos,
                        dev_i_cell_list,
                        dev_i_link_cell,
                        dev_i_cell_start,
                        dev_i_cell_end,
                        d_rzero,
                        i_num_cells,
                        i_np );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i = 0; i < i_np; i++)//to be optimized
    {
        if(i_type[i] != 2)
        {
            real __n = 0.0;

            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            integer __offset = 28 * i_cell_list[i];
            integer __num    =      i_link_cell[__offset];

            for(integer dir=1;dir<=__num;dir++)
            {
                integer __cell = i_link_cell[__offset+dir];

                if(__cell >= 0 && __cell < i_num_cells)
                {
                    integer __start = i_cell_start[__cell];
                    integer __end = i_cell_end[__cell];

                    for(integer j=__start;j<__end;j++)
                    {
                        if(j != i)
                        {
                            real3 __dr = d3_pos[j] - d3_pos[i];

                            __n += d_weight(d_rzero, sqrt( __dr * __dr ));
                        }
                    }
                }
            }

            d_n[i] = __n;
        }
    }

#endif

}

//////////////////////////////////
///collision model
//////////////////////////////////
void MPS_GPU::collision()
{
    /*-----description of collision model-----
    vg = (rhoi*vi + rhoj*vj) / (2 * rhoij_average);
    mvr = rhoi * (vi - vg);
    vabs = (mvr * rji) / abs(rij);
    if vabs < 0, then do nothing
    else then,
        mvm = vrat * vabs * rji / abs(rij);
        vi -= mvm / rhoi;
        xi -= dt * mvm / rhoi;

        vj += mvm / rhoj;
        xj += dt * mvm / rhoj;
    ----------------------------------------*/
    int _ncol = 0;

#ifdef GPU_CUDA

    cudaker::dev_calCol( dev_d3_vel,
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

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        real3 _crt = {0.0, 0.0, 0.0};

        if(i_type[i] == 0)
        {
            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            integer __offset = 28 * i_cell_list[i];
            integer __num = i_link_cell[__offset];

            for(integer dir=1;dir<=__num;dir++)
            {
                integer __cell = i_link_cell[__offset+dir];

                if( (__cell >= 0) && (__cell < i_num_cells) )
                {
                    integer __start = i_cell_start[__cell];
                    integer __end = i_cell_end[__cell];

                    for(integer j=__start;j<__end;j++)
                    {
                        if(j != i)
                        {
                            real3 __dr = d3_pos[j] - d3_pos[i];
                            real3 __du = d3_vel[j] - d3_vel[i];
                            real __ds = sqrt(__dr * __dr);
                            real __one_over_ds = 1.0 / __ds;
                            real __vabs = 0.5f * __du * __dr * __one_over_ds;

                            if( (__ds <= d_col_dis) && (__vabs <= 0.0) )
                            {
                                _crt = _crt + d_col_rate * __vabs * __one_over_ds * __dr;

                                _ncol++;
                            }
                        }
                    }
                }
            }
        }

        d3_vel[i] +=        _crt;
        d3_pos[i] += d_dt * _crt;
    }

#endif

    sprintf(c_log, "        collision count: %4d \n", _ncol);
    //throwLog(fid_log, c_log);
    throwScn(c_log);
}

/////////////////////////////
///add motion of boundary
/////////////////////////////
void MPS_GPU::motion()
{
    M_motion.doMotion(d3_pos, d3_vel, i_np);
}

/////////////////////////////////////
///calculate vis and g explicitly
/////////////////////////////////////
void MPS_GPU::calVisc_expl()
{
#ifdef GPU_CUDA

    cudaker::dev_calVisc_expl( dev_d3_vel,
                               dev_d3_pos,
                               dev_d_press,
                               dev_i_type,
                               dev_i_cell_list,
                               dev_i_link_cell,
                               dev_i_cell_start,
                               dev_i_cell_end,
                               d_dt,
                               d_2bydim_over_nzerobylambda,
                               d_rlap,
                               d_niu,
                               i_num_cells,
                               i_np );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        //ignore boundary
        if(i_type[i] == 0)
        {
            d3_vel[i] += d_dt * (d_niu * d3_LapVel(i) + G);
            d3_pos[i] += d_dt * d3_vel[i];
        }
    }

#endif

}

/////////////////////////////////////
///calculate pressure explicitly
/////////////////////////////////////
void MPS_GPU::calPres_expl()
{
#ifdef GPU_CUDA

    cudaker::dev_calPres_expl( dev_d_press,
                               dev_d_n,
                               dev_i_type,
                               dev_i_normal,
                               d_one_over_alpha,
                               d_nzero,
                               d_one_over_nzero,
                               i_np );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        if(i_type[i] != 2)
        {
            real __tmp = d_one_over_alpha * (d_n[i] - d_nzero) * d_one_over_nzero;

            d_press[i] = __tmp > 0.0 ? __tmp : 0.0; //pressure before i is already changed
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        if(i_type[i] == 2)
        {
            d_press[i] = d_press[i_normal[i]];
        }
    }

#endif

}

#ifdef DEBUG
///////////////////////
///cell debugging code
///////////////////////
void MPS_GPU::check()
{
    for(integer i=0;i<i_np;i++)
    {
        integer _offset = 28 * i_cell_list[i];
        //integer _num    =      i_link_cell[_offset];
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        /*
        for(integer dir=1;dir<=_num;dir++)
        {
            integer __cell = i_link_cell[_offset+dir];

            if(__cell >= 0 && __cell < i_num_cells)
            {
                integer __start = i_cell_start[__cell];
                integer __end   = i_cell_end[__cell];

                for(integer j=__start;j<__end;j++)
                {
                    real3 __dr  = d3_pos[j] - d3_pos[i];
                    real  __ds  = sqrt(__dr * __dr);

                    if(__ds > 3.464102 * d_cell_size)
                    {
                        printf("error found !!! \n");
                        printf("%f,%f,%f -> ",d3_pos[i].x,d3_pos[i].y,d3_pos[i].z);
                        printf("%f,%f,%f\n",d3_pos[j].x,d3_pos[j].y,d3_pos[j].z);

                        printf("i cell: %d ; j cell: %d\n", i_cell_list[i], i_cell_list[j]);
                        for(integer k=1;k<=27;k++)
                        {
                            printf("%d,",i_link_cell[_offset + k]);
                        }
                        printf("\n");
                        printf("i_cell_d* : %d, %d, %d\n",i_cell_dx,i_cell_dy,i_cell_dz);
                        printf("i_cell_sheet : %d\n",i_cell_sheet);

                        exit(100);
                    }
                }
            }
        }
        */
        for(integer j=0;j<i_np;j++)
        {
            bool b_error = true;

            real3 __dr  = d3_pos[j] - d3_pos[i];
            real  __ds  = sqrt(__dr * __dr);

            if(__ds < d_rzero)
            {
                for(integer k=1;k<=27;k++)
                {
                    if(i_cell_list[j] == i_link_cell[_offset+k])
                    {
                        b_error = false;
                        break;
                    }
                }
            }
            else
            {
                b_error = false;
            }

            if(b_error)
            {
                printf("error found !!! \n");
                printf("%f,%f,%f -> ", d3_pos[i].x, d3_pos[i].y, d3_pos[i].z);
                printf("%f,%f,%f  \n", d3_pos[j].x, d3_pos[j].y, d3_pos[j].z);

                printf("i cell: %d ; j cell: %d\n", i_cell_list[i], i_cell_list[j]);
                for(integer k=1;k<=27;k++)
                {
                    printf("%d,",i_link_cell[_offset + k]);
                }
                printf("\n");
                printf("i_cell_d* : %d, %d, %d\n",i_cell_dx, i_cell_dy, i_cell_dz);
                printf("i_cell_sheet : %d\n",i_cell_sheet);

                exit(200);
            }
        }

    }
}

#endif
