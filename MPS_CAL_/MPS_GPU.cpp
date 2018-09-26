/*
LICENCE
*/
//MPS_GPU.cu
//implementation of class MPS_GPU
///receive data from class MPS and functions of main loop
#include <cassert>

#include "def_incl.h"
#include "MPS_GPU.h"
#include "./gpu_kernel/mps_gpu_cudaker.h"
#include "./gpu_kernel/renderer_gpu_cudaker.h"
#include "marchingCube_table.h"
#include "marchingCube_define.h"

using namespace mytype;

    inline void dev_addMem(real& mem, const int& size, const integer& n)
    {
        mem += real(size * n) / 1024;
    }

    inline cudaError_t checkCuda(cudaError_t result, const char* fun)
    {
    #ifdef DEBUG
        if(result != cudaSuccess)
        {
            fprintf(stderr, "CUDA Runtime Error: %s -> in %s \n",
                    cudaGetErrorString(result), fun);
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
    return (_r < _r0) ? (_r0 / _r - 1.0f) : (0.0f);
}

inline real MPS_GPU::r_cubic_weight(const mytype::real _r0, const mytype::real _r) const
{
    mytype::real __q = _r / _r0;

    if(__q <= 0.5f) return 8.0f * mytype::OVERPI * (1 - 6*__q*__q + 6*__q*__q*__q);
    else if(__q <= 1.0f) return 16.0f * mytype::OVERPI * (1.0f-__q) * (1.0f-__q) * (1.0f-__q);
    else return 0.0f;
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
    dev_i_cell_list     = NULL;
    dev_i_link_cell     = NULL;
    dev_i_part_in_cell  = NULL;
    dev_i_cell_start    = NULL;
    dev_i_cell_end      = NULL;
    dev_i_normal        = NULL;
    dev_d_press         = NULL;
    dev_d_n             = NULL;
    dev_r_vertex_n      = NULL;
    dev_r3_verList      = NULL;
    dev_d3_pos          = NULL;
    dev_d3_vel          = NULL;
    //dev_r3_triangle     = NULL;
    dev_r3_vertex_norm  = NULL;
    //dev_r3_norm         = NULL;
    //dev_r_alpha         = NULL;
    dev_i_voxList       = NULL;
    dev_u_numVerTable   = NULL;
    dev_u_triTable      = NULL;
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
    devFree();
#endif

}

///////////////////////////////
///initialization
///////////////////////////////
void MPS_GPU::Initial()
{
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
#ifdef RENDER
    ///voxel
    real vox_size = geo.d_cell_size * VOXEL_SIZE;
    integer ratio = round(1 / VOXEL_SIZE);
    int3 dim = {ratio*geo.i_cell_dx, ratio*geo.i_cell_dy, ratio*geo.i_cell_dz};

    vox_mc.setGeo(dim, vox_size, geo.d_cell_left, geo.d_cell_right, geo.d_cell_back, geo.d_cell_front, geo.d_cell_bottom, geo.d_cell_top);

    memAdd(sizeof(vox_mc),       1);
    memAdd(sizeof(real),         vox_mc.i_nVertex);
    memAdd(sizeof(real3),        vox_mc.i_nVertex);
#endif

    calOnce();

#ifdef GPU_CUDA
    devInit();
#endif
    /*------------------------------*/

    /*-----end of initialization-----*/
    t_loop_e    = time(NULL);
    i_ini_time  = difftime(t_loop_e , t_loop_s);
    /*-------------------------------*/

    FILE* _fid_log = fopen(LOG_NAME, "at");
    char  _str[256];

    sprintf(_str, "successfully Initialized!\n");
    throwScn(_str);
    throwLog(_fid_log, _str);
    sprintf(_str, "host memory usage:        %.1f M. \n", d_mem / 1024);
    throwScn(_str);
    throwLog(_fid_log, _str);

    fclose(_fid_log);
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
    r_iso       = 0.0f;

    //nzero
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        real __tmp1 = 0.0f;
        real __tmp2 = 0.0f;
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        integer __offset    = 28 * i_cell_list[i];
        integer __num       =      i_link_cell[__offset];

        for(integer dir=1;dir<=__num;dir++)
        {
            integer __cell = i_link_cell[__offset+dir];

            if(__cell >= 0 && __cell < geo.i_num_cells)
            {
                integer __start = i_cell_start[__cell];
                integer __end   = i_cell_end[__cell];

                for(integer j=__start;j<__end;j++)
                {
                    if(j != i)
                    {
                        real3 __dr = d3_pos[j] - d3_pos[i];

                        __tmp1 += d_weight(d_rzero, sqrt(__dr * __dr));
                        __tmp2 += r_cubic_weight(d_rzero, sqrt(__dr * __dr));
                    }
                }
            }
        }

        d_n[i] = __tmp1;
        ///-------------------------------------------

        if(__tmp1 > d_nzero)
        {
            #ifdef CPU_OMP
                #pragma omp critical
            #endif
            {
                d_nzero = __tmp1;
                _i_tmp = i;
            }
        }

        if(__tmp2 > r_iso)
        {
            #ifdef CPU_OMP
                #pragma omp critical
            #endif
            {
                r_iso = __tmp2;
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
    d_beta *= d_nzero;
    r_iso  *= ISO;

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

    /*-----new-----*/
    tmp1 = tmp2 = 0.f;
    cellBox.clear();
    for(integer i=0; i<geo.i_num_cells; i++)
    {
            integer iz = i / geo.i_cell_sheet;
            integer iy = i % geo.i_cell_sheet / geo.i_cell_dx;
            integer ix = i % geo.i_cell_sheet % geo.i_cell_dx;
            float size = geo.d_cell_size;

            Point vox0 = Point(ix * size,        iy * size,        iz * size);
            Point vox1 = Point(ix * size + size, iy * size,        iz * size);
            Point vox2 = Point(ix * size + size, iy * size + size, iz * size);
            Point vox3 = Point(ix * size,        iy * size + size, iz * size);
            Point vox4 = Point(ix * size,        iy * size,        iz * size + size);
            Point vox5 = Point(ix * size + size, iy * size,        iz * size + size);
            Point vox6 = Point(ix * size + size, iy * size + size, iz * size + size);
            Point vox7 = Point(ix * size,        iy * size + size, iz * size + size);

            cellBox.push_back(vox0); cellBox.push_back(vox1); cellBox.push_back(vox2); cellBox.push_back(vox3);
            cellBox.push_back(vox4); cellBox.push_back(vox5); cellBox.push_back(vox6); cellBox.push_back(vox7);
    }

    integer ratio = round(1 / VOXEL_SIZE);
    int3 dim = vox_mc.i3_dim;
    int sheet = (dim.x * dim.y);

    i_cellToVox.clear();
    for(integer i=0; i<geo.i_num_cells; i++)
    {
            integer cellz = i / geo.i_cell_sheet;
            integer celly = i % geo.i_cell_sheet / geo.i_cell_dx;
            integer cellx = i % geo.i_cell_sheet % geo.i_cell_dx;

            integer voxx = cellx * ratio;
            integer voxy = celly * ratio;
            integer voxz = cellz * ratio;

            for(integer z=0; z<ratio; z++)
            {
                for(integer y=0; y<ratio; y++)
                {
                    for(integer x=0; x<ratio; x++)
                    {
                        integer voxid = (voxz+z) * (sheet) + (voxy+y) * (dim.x) + (voxx+x);

                        i_cellToVox.push_back(voxid);
                    }
                }
            }
    }

    i_voxToCell.clear();
    for(integer i=0; i<vox_mc.i_nVertex; i++)
    {
            integer cellz = (i / sheet) / ratio;
            integer celly = (i % sheet / dim.x) / ratio;
            integer cellx = (i % sheet % dim.x) / ratio;

            integer cellId = (cellz) * (geo.i_cell_sheet) + (celly) * (geo.i_cell_dx) + (cellx);

            i_voxToCell.push_back(cellId);
    }
    printf("total voxels: %ld\n", i_voxToCell.size());
    printf("%d, %d\n", geo.i_num_cells, vox_mc.i_nVertex);
    printf("%d\n", i_voxToCell[vox_mc.i_nVertex-1]);
    /*-------------*/

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
    real _mem = 0.0f;

    cudaDeviceReset();

    checkCuda( cudaMalloc(&dev_i_id_tmp,           i_np*sizeof(integer)) ,  "323");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_type_tmp,         i_np*sizeof(integer)) ,  "324");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_cell_list_tmp,    i_np*sizeof(integer)) ,  "325");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_normal_tmp,       i_np*sizeof(integer)) ,  "326");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_d_press_tmp,        i_np*sizeof(real)) ,  "327");    dev_addMem(_mem, sizeof(real), i_np);
    checkCuda( cudaMalloc(&dev_d_n_tmp,            i_np*sizeof(real)) ,  "328");    dev_addMem(_mem, sizeof(real), i_np);
    checkCuda( cudaMalloc(&dev_d3_pos_tmp,         i_np*sizeof(real3)) ,  "329");   dev_addMem(_mem, sizeof(real3), i_np);
    checkCuda( cudaMalloc(&dev_d3_vel_tmp,         i_np*sizeof(real3)) ,  "330");   dev_addMem(_mem, sizeof(real3), i_np);

    checkCuda( cudaMalloc(&dev_i_id,               i_np*sizeof(integer)) ,  "332");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_index,            i_np*sizeof(integer)) ,  "333");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_type,             i_np*sizeof(integer)) ,  "334");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_normal,           i_np*sizeof(integer)) ,  "335");     dev_addMem(_mem, sizeof(integer), i_np);
    checkCuda( cudaMalloc(&dev_i_cell_list,        i_np*sizeof(integer)) ,  "336");     dev_addMem(_mem, sizeof(integer), i_np);

    checkCuda( cudaMalloc(&dev_i_link_cell,     28*geo.i_num_cells*sizeof(integer)) ,  "338");  dev_addMem(_mem, sizeof(integer), 28*geo.i_num_cells);
    checkCuda( cudaMalloc(&dev_i_part_in_cell,     geo.i_num_cells*sizeof(integer)) ,  "339");  dev_addMem(_mem, sizeof(integer), geo.i_num_cells);
    checkCuda( cudaMalloc(&dev_i_cell_start,       geo.i_num_cells*sizeof(integer)) ,  "340");  dev_addMem(_mem, sizeof(integer), geo.i_num_cells);
    checkCuda( cudaMalloc(&dev_i_cell_end,         geo.i_num_cells*sizeof(integer)) ,  "341");  dev_addMem(_mem, sizeof(integer), geo.i_num_cells);

    checkCuda( cudaMalloc(&dev_d_n,                i_np*sizeof(real)) ,  "343");    dev_addMem(_mem, sizeof(real), i_np);
#ifdef RENDER
    /*-----new-----*/
    checkCuda( cudaMalloc(&dev_i_markVox,         vox_mc.i_nVertex*sizeof(integer)), "new");  dev_addMem(_mem, sizeof(integer), vox_mc.i_nVertex);
    //checkCuda( cudaMalloc(&dev_i_cellToVox,        i_cellToVox.size()*sizeof(integer)), "new");  dev_addMem(_mem, sizeof(integer), i_cellToVox.size());
    checkCuda( cudaMalloc(&dev_i_voxToCell,        i_voxToCell.size()*sizeof(integer)), "new");  dev_addMem(_mem, sizeof(integer), i_voxToCell.size());
    checkCuda( cudaMalloc(&dev_i_cellInFluid,      geo.i_num_cells*sizeof(integer)), "new");  dev_addMem(_mem, sizeof(integer), geo.i_num_cells);
    /*-------------*/
    checkCuda( cudaMalloc(&dev_r_vertex_n,         vox_mc.i_nVertex*sizeof(real )) ,  "345");  dev_addMem(_mem, sizeof(real), vox_mc.i_nVertex);
    checkCuda( cudaMalloc(&dev_r3_verList,         vox_mc.i_nVertex*sizeof(real3)) ,  "346");  dev_addMem(_mem, sizeof(real3), vox_mc.i_nVertex);
    //checkCuda( cudaMalloc(&dev_r3_triangle,        vox_mc.getMaxEdge  ()*sizeof(real3)) ,  "347");  dev_addMem(_mem, sizeof(real3), vox_mc.getMaxEdge());
    checkCuda( cudaMalloc(&dev_r3_vertex_norm,     vox_mc.i_nVertex*sizeof(real3)) ,  "348");  dev_addMem(_mem, sizeof(real3), vox_mc.i_nVertex);
    //checkCuda( cudaMalloc(&dev_r3_norm,            vox_mc.getMaxEdge  ()*sizeof(real3)) ,  "348");  dev_addMem(_mem, sizeof(real3), vox_mc.getMaxEdge());
    //checkCuda( cudaMalloc(&dev_r_alpha,            vox_mc.getMaxEdge  ()*sizeof(real)) ,  "349");   dev_addMem(_mem, sizeof(real), vox_mc.getMaxEdge());
    checkCuda( cudaMalloc(&dev_i_voxList,        8*vox_mc.i_nVoxel *sizeof(integer)) ,  "350");    dev_addMem(_mem, sizeof(integer), 8*vox_mc.i_nVoxel);
    checkCuda( cudaMalloc(&dev_u_numVerTable,  256*sizeof(uint)) ,  "351");     dev_addMem(_mem, sizeof(uint), 256);
    checkCuda( cudaMalloc(&dev_u_triTable,  256*16*sizeof(uint)) ,  "352");     dev_addMem(_mem, sizeof(uint), 256*16);
#endif
    checkCuda( cudaMalloc(&dev_d_press,            i_np*sizeof(real)) ,  "354");    dev_addMem(_mem, sizeof(real), i_np);
    checkCuda( cudaMalloc(&dev_d3_pos,             i_np*sizeof(real3)) ,  "355");   dev_addMem(_mem, sizeof(real3), i_np);
    checkCuda( cudaMalloc(&dev_d3_vel,             i_np*sizeof(real3)) ,  "356");   dev_addMem(_mem, sizeof(real3), i_np);

    checkCuda( cudaMemcpy(dev_i_id,             i_id,               i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "358");
    checkCuda( cudaMemcpy(dev_i_index,          i_index,            i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "359");
    checkCuda( cudaMemcpy(dev_i_type,           i_type,             i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "360");
    checkCuda( cudaMemcpy(dev_i_normal,         i_normal,           i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "361");
    checkCuda( cudaMemcpy(dev_i_cell_list,      i_cell_list,        i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "362");

    checkCuda( cudaMemcpy(dev_i_link_cell,      i_link_cell,     28*geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "364");
    checkCuda( cudaMemcpy(dev_i_part_in_cell,   i_part_in_cell,     geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "365");
    checkCuda( cudaMemcpy(dev_i_cell_start,     i_cell_start,       geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "366");
    checkCuda( cudaMemcpy(dev_i_cell_end,       i_cell_end,         geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "367");

#ifdef RENDER
    /*-----new-----*/
    //checkCuda( cudaMemcpy(dev_i_cellToVox,   i_cellToVox.data(),    i_cellToVox.size()*sizeof(integer),  cudaMemcpyHostToDevice) ,  "new");
    checkCuda( cudaMemcpy(dev_i_voxToCell,   i_voxToCell.data(),    i_voxToCell.size()*sizeof(integer),  cudaMemcpyHostToDevice) ,  "new");
    /*-------------*/
    checkCuda( cudaMemcpy(dev_r3_verList,   vox_mc.r3_verList,    vox_mc.i_nVertex*sizeof(real3),  cudaMemcpyHostToDevice) ,  "370");
    checkCuda( cudaMemcpy(dev_i_voxList,    vox_mc.i_voxelList, 8*vox_mc.i_nVoxel *sizeof(integer),cudaMemcpyHostToDevice) ,  "371");
    checkCuda( cudaMemcpy(dev_u_numVerTable,numVertsTable,      256*sizeof(uint),                         cudaMemcpyHostToDevice) ,  "372");
    checkCuda( cudaMemcpy(dev_u_triTable,   triTable,        256*16*sizeof(uint),                         cudaMemcpyHostToDevice) ,  "373");
#endif
    checkCuda( cudaMemcpy(dev_d_n,          d_n,            i_np*sizeof(real),              cudaMemcpyHostToDevice) ,  "375");
    checkCuda( cudaMemcpy(dev_d_press,      d_press,        i_np*sizeof(real),              cudaMemcpyHostToDevice) ,  "376");
    checkCuda( cudaMemcpy(dev_d3_pos,       d3_pos,         i_np*sizeof(real3),             cudaMemcpyHostToDevice) ,  "377");
    checkCuda( cudaMemcpy(dev_d3_vel,       d3_vel,         i_np*sizeof(real3),             cudaMemcpyHostToDevice) ,  "378");

    FILE* _fid_log = fopen(LOG_NAME, "at");
    char  _str[256];
    sprintf(_str, "Allocated device memory:      %.1f M. \n", _mem / 1024);
    throwScn(_str);
    throwLog(_fid_log, _str);
    fclose(_fid_log);
}

///////////////////////////////
///device free
///////////////////////////////
void MPS_GPU::devFree()
{
    checkCuda( cudaFree(dev_i_id_tmp) ,  "393");
    checkCuda( cudaFree(dev_i_type_tmp) ,  "394");
    checkCuda( cudaFree(dev_i_cell_list_tmp) ,  "395");
    checkCuda( cudaFree(dev_i_normal_tmp) ,  "396");
    checkCuda( cudaFree(dev_d_press_tmp) ,  "397");
    checkCuda( cudaFree(dev_d_n_tmp) ,  "398");
#ifdef RENDER
    /*-----new-----*/
    checkCuda( cudaFree(dev_i_markVox) ,  "new");
    checkCuda( cudaFree(dev_i_voxToCell) ,  "new");
    checkCuda( cudaFree(dev_i_cellInFluid) ,  "new");
    /*-------------*/
    checkCuda( cudaFree(dev_r_vertex_n) ,  "400");
    checkCuda( cudaFree(dev_r3_verList) ,  "401");
    //checkCuda( cudaFree(dev_r3_triangle) ,  "402");
    //checkCuda( cudaFree(dev_r_alpha) ,  "403");
    checkCuda( cudaFree(dev_i_voxList) ,  "404");
    checkCuda( cudaFree(dev_u_numVerTable) ,  "405");
    checkCuda( cudaFree(dev_u_triTable) ,  "406");
#endif
    checkCuda( cudaFree(dev_d3_pos_tmp) ,  "408");
    checkCuda( cudaFree(dev_d3_vel_tmp) ,  "409");

    checkCuda( cudaFree(dev_i_id) ,  "411");
    checkCuda( cudaFree(dev_i_index) ,  "412");
    checkCuda( cudaFree(dev_i_type) ,  "413");
    checkCuda( cudaFree(dev_i_cell_list) ,  "414");
    checkCuda( cudaFree(dev_i_link_cell) ,  "415");
    checkCuda( cudaFree(dev_i_part_in_cell) ,  "416");
    checkCuda( cudaFree(dev_i_cell_start) ,  "417");
    checkCuda( cudaFree(dev_i_cell_end) ,  "418");
    checkCuda( cudaFree(dev_i_normal) ,  "419");

    checkCuda( cudaFree(dev_d_press) ,  "421");
    checkCuda( cudaFree(dev_d_n) ,  "422");
    checkCuda( cudaFree(dev_d3_pos) ,  "423");
    checkCuda( cudaFree(dev_d3_vel) ,  "424");

    cudaDeviceReset();
}

//////////////////////////////////////
///communication from host to device
//////////////////////////////////////
void MPS_GPU::commHostToDevice()
{
#ifdef TIMER
    cudaEvent_t _start, _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    cudaEventRecord(_start);
#endif
    checkCuda( cudaMemcpy(dev_i_markVox,        i_markVox.data(),   i_markVox.size()*sizeof(integer),   cudaMemcpyHostToDevice), "new");
    checkCuda( cudaMemcpy(dev_i_cellInFluid,        i_cellInFluid,   geo.i_num_cells*sizeof(integer),   cudaMemcpyHostToDevice), "new");
    checkCuda( cudaMemcpy(dev_i_index,          i_index,            i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "441");
    checkCuda( cudaMemcpy(dev_i_cell_list,      i_cell_list,        i_np*sizeof(integer),               cudaMemcpyHostToDevice) ,  "442");

    checkCuda( cudaMemcpy(dev_i_part_in_cell,   i_part_in_cell,     geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "444");
    checkCuda( cudaMemcpy(dev_i_cell_start,     i_cell_start,       geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "445");
    checkCuda( cudaMemcpy(dev_i_cell_end,       i_cell_end,         geo.i_num_cells*sizeof(integer),        cudaMemcpyHostToDevice) ,  "446");
#ifdef TIMER
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);

    printf("        HtoD: %f \n", milliseconds);
#endif
}

//////////////////////////////////////
///communication from device to host
//////////////////////////////////////
void MPS_GPU::commDeviceToHost()
{
#ifdef TIMER
    cudaEvent_t _start, _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    cudaEventRecord(_start);
#endif

    checkCuda( cudaMemcpy(d3_pos,       dev_d3_pos,         i_np*sizeof(real3),             cudaMemcpyDeviceToHost) ,  "471");
    checkCuda( cudaMemcpy(i_type,       dev_i_type,         i_np*sizeof(integer),           cudaMemcpyDeviceToHost) ,  "472");
    //checkCuda( cudaMemcpy(d3_vel,       dev_d3_vel,         i_np*sizeof(real3),             cudaMemcpyDeviceToHost) ,  "471");
    //checkCuda( cudaMemcpy(d_press,      dev_d_press,        i_np*sizeof(real),              cudaMemcpyDeviceToHost) ,  "471");

#ifdef RENDER
    checkCuda( cudaMemcpy(vox_mc.r_verDensity,       dev_r_vertex_n,         vox_mc.i_nVertex*sizeof(real),           cudaMemcpyDeviceToHost) ,  "475");
    checkCuda( cudaMemcpy(vox_mc.r3_verNorm,       dev_r3_vertex_norm,         vox_mc.i_nVertex*sizeof(real3),           cudaMemcpyDeviceToHost) ,  "475");
#endif

#ifdef TIMER
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);

    printf("        DtoH: %f \n", milliseconds);
#endif
}

/////////////////////////////////////////
///main loop
/////////////////////////////////////////
void MPS_GPU::step()
{
    printf("\n");
    printf("    step: %d\n", i_step);
#ifdef TIMER
    real milliseconds = 0.0f;
    cudaEvent_t _start, _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
#endif

    /*-----boundary motion-----*/
    //motion();
    //divide_cell();
    /*-------------------------*/

    /*-----devide cells-----*/
#ifdef TIMER
    cudaEventRecord(_start);
#endif

    divideCell();//cpu
    makeVox();//cpu

#ifdef TIMER
/*
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&milliseconds, _start, _stop);

    printf("        divi time ->   %.3f \n", milliseconds);
*/
#endif

#ifdef GPU_CUDA
    commHostToDevice();
#endif
    /*----------------------*/

#ifdef TIMER
/*
    cudaEventRecord(_start);
*/
#endif

    sortAll();  //gpu
    calVisc_expl();//gpu
/*
    commDeviceToHost();
    divideCell();
    commHostToDevice();
    sortAll();
*/
    collision   ();//gpu
/*
    commDeviceToHost();
    divideCell();
    commHostToDevice();
    sortAll();
*/
    cal_n       ();//gpu
    calPres_expl();//gpu
    calDash     ();//gpu

#ifdef TIMER
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&milliseconds, _start, _stop);

    printf("        calc time ->   %.3f \n", milliseconds);
    tmp1 += milliseconds;
#endif

#ifdef TIMER
    cudaEventRecord(_start);
#endif

#ifdef RENDER
    calVertex_n ();//gpu
    calNorm     ();//gpu
#endif

#ifdef TIMER
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&milliseconds, _start, _stop);

    printf("        samp time ->   %.3f \n", milliseconds);
    tmp2 += milliseconds;

    printf("cal, samp:  %.3f, %.3f \n", tmp1, tmp2);
#endif

#ifdef GPU_CUDA
    commDeviceToHost();
#endif

    update_dt();
    i_step++;

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
        for(integer i=0;i<geo.i_num_cells;i++)
        {
            integer __offset = 28 * i;

            i_link_cell[__offset + 0] = 9;

            i_link_cell[__offset + 1] = i - geo.i_cell_dx - 1;  //-1,-1
            i_link_cell[__offset + 2] = i - geo.i_cell_dx;      //-1,0
            i_link_cell[__offset + 3] = i - geo.i_cell_dx + 1;  //-1,1

            i_link_cell[__offset + 4] = i - 1;              //0,-1
            i_link_cell[__offset + 5] = i;                  //0,0
            i_link_cell[__offset + 6] = i + 1;              //0,1

            i_link_cell[__offset + 7] = i + geo.i_cell_dx - 1;  //1,-1
            i_link_cell[__offset + 8] = i + geo.i_cell_dx;      //1,0
            i_link_cell[__offset + 9] = i + geo.i_cell_dx + 1;  //1,1
        }
    }
    else if(i_dim == 3)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(integer i=0;i<geo.i_num_cells;i++)
        {
            integer __offset = 28 * i;

            i_link_cell[__offset + 0] = 27;

            i_link_cell[__offset + 1] = i - geo.i_cell_sheet - geo.i_cell_dx - 1;   //-1,-1,-1
            i_link_cell[__offset + 2] = i - geo.i_cell_sheet - geo.i_cell_dx;       //-1,-1,0
            i_link_cell[__offset + 3] = i - geo.i_cell_sheet - geo.i_cell_dx + 1;   //-1,-1,1

            i_link_cell[__offset + 4] = i - geo.i_cell_sheet - 1;               //-1,0,-1
            i_link_cell[__offset + 5] = i - geo.i_cell_sheet;                   //-1,0,0
            i_link_cell[__offset + 6] = i - geo.i_cell_sheet + 1;               //-1,0,1

            i_link_cell[__offset + 7] = i - geo.i_cell_sheet + geo.i_cell_dx - 1;   //-1,1,-1
            i_link_cell[__offset + 8] = i - geo.i_cell_sheet + geo.i_cell_dx;       //-1,1,0
            i_link_cell[__offset + 9] = i - geo.i_cell_sheet + geo.i_cell_dx + 1;   //-1,1,1

            i_link_cell[__offset + 10] = i - geo.i_cell_dx - 1;                 //0,-1,-1
            i_link_cell[__offset + 11] = i - geo.i_cell_dx;                     //0,-1,0
            i_link_cell[__offset + 12] = i - geo.i_cell_dx + 1;                 //0,-1,1

            i_link_cell[__offset + 13] = i - 1;                             //0,0,-1
            i_link_cell[__offset + 14] = i;                                 //0,0,0
            i_link_cell[__offset + 15] = i + 1;                             //0,0,1

            i_link_cell[__offset + 16] = i + geo.i_cell_dx - 1;                 //0,1,-1
            i_link_cell[__offset + 17] = i + geo.i_cell_dx;                     //0,1,0
            i_link_cell[__offset + 18] = i + geo.i_cell_dx + 1;                 //0,1,1

            i_link_cell[__offset + 19] = i + geo.i_cell_sheet - geo.i_cell_dx - 1;  //1,-1,-1
            i_link_cell[__offset + 20] = i + geo.i_cell_sheet - geo.i_cell_dx;      //1,-1,0
            i_link_cell[__offset + 21] = i + geo.i_cell_sheet - geo.i_cell_dx + 1;  //1,-1,1

            i_link_cell[__offset + 22] = i + geo.i_cell_sheet - 1;              //1,0,-1
            i_link_cell[__offset + 23] = i + geo.i_cell_sheet;                  //1,0,0
            i_link_cell[__offset + 24] = i + geo.i_cell_sheet + 1;              //1,0,1

            i_link_cell[__offset + 25] = i + geo.i_cell_sheet + geo.i_cell_dx - 1;  //1,1,-1
            i_link_cell[__offset + 26] = i + geo.i_cell_sheet + geo.i_cell_dx;      //1,1,0
            i_link_cell[__offset + 27] = i + geo.i_cell_sheet + geo.i_cell_dx + 1;  //1,1,1
        }
    }
}

////////////////////////////////
///divide the domain into cells
////////////////////////////////
void MPS_GPU::divideCell()
{
    integer _excl = 0;

    Zero(i_part_in_cell, geo.i_num_cells);
    Zero(i_bd_in_cell, geo.i_num_cells);
    Zero(i_cell_start, geo.i_num_cells);

    //divide particles into cells
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] != 3)
        {
            integer __cellx = integer( (d3_pos[i].x - geo.d_cell_left)   / geo.d_cell_size );
            integer __celly = integer( (d3_pos[i].y - geo.d_cell_back)   / geo.d_cell_size );
            integer __cellz = integer( (d3_pos[i].z - geo.d_cell_bottom) / geo.d_cell_size );
            integer __num   = __cellz * geo.i_cell_sheet + __celly * geo.i_cell_dx + __cellx;

            if(__cellx < 0 || __celly < 0 || __cellz < 0 ||
               __cellx >= geo.i_cell_dx || __celly >= geo.i_cell_dy || __cellz >= geo.i_cell_dz)
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

            if(i_type[i] != 0)
            {
            #ifdef CPU_OMP
                #pragma omp atomic
            #endif
                i_bd_in_cell[__num]++;
            }
        }
    }

    //start in each cell
    //initialize i_cell_end
    //can not be paralleled !
    i_cell_start[0] = i_cell_end[0] = 0;
    for(integer i=1;i<geo.i_num_cells;i++)
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
///make voxels
/////////////////////////////////
void MPS_GPU::makeVox()
{
    /*-----new-----*/
    int ratio = round(1 / VOXEL_SIZE);
    int ratio3 = ratio * ratio * ratio;

    i_markVox.clear();
    i_markCell.clear();
    #ifdef CPU_OMP
        //#pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<geo.i_num_cells;i++)
    {
        const integer _offset = 28 * i;
        const integer _num = i_link_cell[_offset];
/*
        bool _flagi = (i_part_in_cell[i] - i_bd_in_cell[i] > 0);
        bool _flagj = 1;

        for(integer dir=1;dir<=_num;dir++)
        {
            integer __cell = i_link_cell[_offset + dir];

            if(__cell < 0 || __cell >= geo.i_num_cells) continue;
            if(__cell == i) continue;
            _flagj = _flagj && (i_part_in_cell[__cell] - i_bd_in_cell[__cell] > 0);
        }

        if( _flagi ^ _flagj )
        {
            for(int v=0; v<ratio3; v++)
            {
                i_markVox.push_back( i_cellToVox[ratio3*i + v] );
            }
        }
*/
        integer threshold = 20;
        if(i_part_in_cell[i] - i_bd_in_cell[i] > 0)
        {
            i_cellInFluid[i] = 1;
            for(integer dir=1;dir<=_num;dir++)
            {
                integer j = i_link_cell[_offset + dir];

                if(j < 0 || j >= geo.i_num_cells) continue;
                if(j == i) continue;
                if(i_part_in_cell[i] - i_bd_in_cell[i] < threshold || i_part_in_cell[j] - i_bd_in_cell[j] <= 0)
                {
                    #ifdef CPU_OMP
                        //#pragma omp critical(pushback)
                    #endif
                    {
                        i_markCell.push_back(i);
                        for(int v=0; v<ratio3; v++)
                        {
                            i_markVox.push_back( i_cellToVox[ratio3*i + v] );
                        }
                    }
                    break;
                }
            }
        }
        else
        {
            i_cellInFluid[i] = 0;
            for(integer dir=1;dir<=_num;dir++)
            {
                integer j = i_link_cell[_offset + dir];

                if(j < 0 || j >= geo.i_num_cells) continue;
                if(j == i) continue;
                if(i_part_in_cell[j] - i_bd_in_cell[j] > 0)
                {
                    #ifdef CPU_OMP
                        //#pragma omp critical(pushback)
                    #endif
                    {
                        i_markCell.push_back(i);
                        for(int v=0; v<ratio3; v++)
                        {
                            i_markVox.push_back( i_cellToVox[ratio3*i + v] );
                        }
                    }
                    break;
                }
            }
        }

    }
    printf(" marked vox: %ld, total: %d \n", i_markVox.size(), vox_mc.i_nVertex);
    printf(" marked cell: %ld \n", i_markCell.size());
    /*-------------*/
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

#if GPU_CUDA_DEP
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
#if GPU_CUDA_DEP
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
#if GPU_CUDA_DEP
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
#if GPU_CUDA_DEP
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
#if GPU_CUDA_DEP
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

        if(__cell >= 0 && __cell < geo.i_num_cells)
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

        if(__cell >= 0 && __cell < geo.i_num_cells)
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

        if(__cell >= 0 && __cell < geo.i_num_cells)
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

        if(__cell >= 0 && __cell < geo.i_num_cells)
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

        if(__cell >= 0 && __cell < geo.i_num_cells)
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
                          geo.i_num_cells,
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
                        dev_i_type,
                        dev_i_cell_list,
                        dev_i_link_cell,
                        dev_i_cell_start,
                        dev_i_cell_end,
                        d_rzero,
                        geo.i_num_cells,
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
#ifndef GPU_CUDA
    int _ncol = 0;
#endif

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
                         geo.i_num_cells,
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

#ifndef GPU_CUDA
    sprintf(c_log, "        collision count: %4d \n", _ncol);
    throwScn(c_log);
#endif

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
                               geo.i_num_cells,
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

/////////////////////////////////////
///calculate n of each vertex
/////////////////////////////////////
void MPS_GPU::calVertex_n()
{
#ifdef GPU_CUDA
    cudaker::dev_zeroVert(dev_r_vertex_n, dev_r3_vertex_norm, dev_i_cellInFluid, dev_i_voxToCell, vox_mc.i_nVertex);

    cudaker::dev_calVertex_n( dev_r_vertex_n,
                              dev_i_markVox,
                              dev_i_voxToCell,
                              dev_r3_verList,
                              dev_d3_pos,
                              dev_i_type,
                              dev_i_link_cell,
                              dev_i_cell_start,
                              dev_i_cell_end,
                              d_rzero,
                              vox_mc.i_nVertex,
                              i_markVox.size(),
                              geo );

#endif
}

/////////////////////////////////////
///calculate triangles for each cube
/////////////////////////////////////
void MPS_GPU::calNorm()
{
#ifdef GPU_CUDA

    cudaker::dev_calNorm  ( dev_r3_vertex_norm,
                            dev_i_markVox,
                            dev_r_vertex_n,
                            vox_mc.i3_dim,
                            vox_mc.i_nVertex,
                            i_markVox.size());
#endif
}

///////////////////////////////////////
///sortAll
///////////////////////////////////////
void MPS_GPU::sortAll()
{
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
}

#ifdef DEBUG_
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
