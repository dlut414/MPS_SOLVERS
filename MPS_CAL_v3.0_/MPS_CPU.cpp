/*
LICENCE
*/
//MPS_CPU.cpp
//implementation of class MPS_CPU
///receive data from class MPS and functions of main loop
#include "def_incl.h"
#include "MPS_CPU.h"

MPS_CPU::MPS_CPU()
{
    fid_log = NULL;
    fid_out = NULL;
    b = NULL;
    x = NULL;
    A = NULL;
    r = NULL;
    p = NULL;
    Ap = NULL;
    bc = NULL;

    i_tmp = NULL;
    d_tmp = NULL;
    i3_tmp = NULL;
    d3_tmp = NULL;

    d_ini_time = 0;
    d_cal_time = 0;
    t_loop_s = 0;
    t_loop_e = 0;
}

MPS_CPU::~MPS_CPU()
{
    fid_log = NULL;
    fid_out = NULL;
    delete[] b;
    delete[] x;
    delete[] A;
    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] bc;

    delete[] i_tmp;
    delete[] d_tmp;
    delete[] i3_tmp;
    delete[] d3_tmp;
}

///////////////////////////////
///initialization
///////////////////////////////
void MPS_CPU::Initial()
{
    char str[256];

    /*-----time of initialization-----*/
    //t_loop_s = time(NULL);
    t_loop_s = getSystemTime();
    /*--------------------------------*/

    /*-----initialization of parent class-----*/
    LoadCase();//must at first
    CreateGeo();
    CalOnce();
    /*----------------------------------------*/

    /*-----initialization of variables in MPS_CPU-----*/
    strcpy(c_name , "0000.out");
    i_step = 0;
    d_time = 0.0;

    d_ini_time = 0;
    d_cal_time = 0;

    ///-----initialization for b=Ax
    b = new real[i_np];
    x = new real[i_np];
    A = new real*[i_np];

    r = new real[i_np];
    p = new real[i_np];
    Ap = new real[i_np];
    bc = new bool[i_np];

    for(int i=0; i < i_np; i++) A[i] = new real[i_np];

    i_tmp = new int[i_np];
    d_tmp = new real[i_np];
    i3_tmp = new int3[i_np];
    d3_tmp = new real3[i_np];
    /*------------------------------------------------*/

    /*-----make the cell-list-----*/
    divide_cell<DIM>();
    /*----------------------------*/

    /*-----end of initialization-----*/
    //t_loop_e = time(NULL);
    t_loop_e = getSystemTime();
    //i_ini_time = difftime(t_loop_e , t_loop_s);
    d_ini_time = t_loop_e - t_loop_s;
    /*-------------------------------*/

    sprintf(str, "successfully Initialized!\n");
    throwScn(str);
    throwLog(fid_log, str);
}

/////////////////////////////////////////
///out put case
/////////////////////////////////////////
void MPS_CPU::WriteCase()
{
    fid_out = fopen(c_name , "wt");

    for(int i = 0; i < i_np; i++)
    {
        fprintf(fid_out , "%lf %lf %lf %lf %lf %lf %lf %d " ,
                d3_pos[i].x , d3_pos[i].y , d3_pos[i].z ,
                d3_vel[i].x , d3_vel[i].y , d3_vel[i].z ,
                d_press[i], i_id[i]);
    }

    fclose(fid_out);
}

////////////////////////////////////
///set all values in p to zero
////////////////////////////////////
template<typename T> void MPS_CPU::Zero(T* p, int n)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<n;i++) p[i] = 0;
}

////////////////////////////////
///divide the domain into cells
////////////////////////////////
template<int _dim> void MPS_CPU::divide_cell()
{
    Zero(i_part_in_cell, i_num_cells);
    Zero(i_cell_start, i_num_cells);

    //divide particles into cells
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(int i=0;i<i_np;i++)
    {
        register int __cellx = (d3_pos[i].x - d_cell_left) / d_cell_size;
        register int __celly = (d3_pos[i].y - d_cell_back) / d_cell_size;
        register int __cellz = (d3_pos[i].z - d_cell_bottom) / d_cell_size;
        register int __num = __cellz * i_cell_sheet + __celly * i_cell_dx + __cellx;

        if(__num >= i_num_cells || __num < 0)
        {
            printf("particle exceeding cell -> _num: %d, # of cells: %d\n", __num, i_num_cells);
            exit(4);
        }

        #ifdef CPU_OMP
            #pragma omp atomic
        #endif
        i_part_in_cell[__num]++;
        i_cell_list[i] = __num;
    }

    //start in each cell
    //initialize i_cell_end
    //can not be paralleled !
    i_cell_start[0] = i_cell_end[0] = 0;
    for(int i=1;i<i_num_cells;i++)
    {
        i_cell_end[i] = i_cell_start[i] = i_cell_start[i-1] + i_part_in_cell[i-1];
    }

    //put particles into cells
    //can not be paralleled !
    for(int i=0;i<i_np;i++)
    {
        //*1).i_index -> new , i -> old; 2).i_index -> old , i -> new, which is better?
        register unsigned __cell_index = i_cell_list[i];
        i_index[i] = i_cell_end[__cell_index];
        i_cell_end[__cell_index]++;
    }

    //make link_cell list
    if(_dim == 2)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
        #endif
        for(int i=0;i<i_num_cells;i++)
        {
            if(i_part_in_cell[i] == 0) continue;
            //i_link_cell[i][0] = 9;
            i_link_cell[i][0] = 4;

            //i_link_cell[i][1] = i - i_cell_dx - 1; //-1,-1
            //i_link_cell[i][2] = i - i_cell_dx; //-1,0
            //i_link_cell[i][3] = i - i_cell_dx + 1; //-1,1

            //i_link_cell[i][4] = i - 1; //0,-1
            //i_link_cell[i][1] = i; //0,0 //index == 1
            i_link_cell[i][1] = i + 1; //0,1

            i_link_cell[i][2] = i + i_cell_dx - 1; //1,-1
            i_link_cell[i][3] = i + i_cell_dx; //1,0
            i_link_cell[i][4] = i + i_cell_dx + 1; //1,1
        }
    }
    else if(_dim == 3)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
        #endif
        for(int i=0;i<i_num_cells;i++)
        {
            if(i_part_in_cell[i] == 0) continue;
            //i_link_cell[i][0] = 27;
            i_link_cell[i][0] = 13;

            //i_link_cell[i][1] = i - i_cell_sheet - i_cell_dx - 1; //-1,-1,-1
            //i_link_cell[i][2] = i - i_cell_sheet - i_cell_dx; //-1,-1,0
            //i_link_cell[i][3] = i - i_cell_sheet - i_cell_dx + 1; //-1,-1,1

            //i_link_cell[i][4] = i - i_cell_sheet - 1; //-1,0,-1
            //i_link_cell[i][5] = i - i_cell_sheet; //-1,0,0
            //i_link_cell[i][6] = i - i_cell_sheet + 1; //-1,0,1

            i_link_cell[i][1] = i - i_cell_sheet + i_cell_dx - 1; //-1,1,-1
            i_link_cell[i][2] = i - i_cell_sheet + i_cell_dx; //-1,1,0
            i_link_cell[i][3] = i - i_cell_sheet + i_cell_dx + 1; //-1,1,1

            //i_link_cell[i][10] = i - i_cell_dx - 1; //0,-1,-1
            //i_link_cell[i][11] = i - i_cell_dx; //0,-1,0
            //i_link_cell[i][12] = i - i_cell_dx + 1; //0,-1,1

            //i_link_cell[i][13] = i - 1; //0,0,-1
            //i_link_cell[i][1] = i; //0,0,0 //index == 1
            i_link_cell[i][4] = i + 1; //0,0,1

            i_link_cell[i][5] = i + i_cell_dx - 1; //0,1,-1
            i_link_cell[i][6] = i + i_cell_dx; //0,1,0
            i_link_cell[i][7] = i + i_cell_dx + 1; //0,1,1

            //i_link_cell[i][19] = i + i_cell_sheet - i_cell_dx - 1; //1,-1,-1
            //i_link_cell[i][20] = i + i_cell_sheet - i_cell_dx; //1,-1,0
            //i_link_cell[i][21] = i + i_cell_sheet - i_cell_dx + 1; //1,-1,1

            i_link_cell[i][8] = i + i_cell_sheet - 1; //1,0,-1
            i_link_cell[i][9] = i + i_cell_sheet; //1,0,0
            i_link_cell[i][10] = i + i_cell_sheet + 1; //1,0,1

            i_link_cell[i][11] = i + i_cell_sheet + i_cell_dx - 1; //1,1,-1
            i_link_cell[i][12] = i + i_cell_sheet + i_cell_dx; //1,1,0
            i_link_cell[i][13] = i + i_cell_sheet + i_cell_dx + 1; //1,1,1
        }
    }

    //sort variable to new rankings
    sort_i(i_id);
    sort_i(i_type);
    sort_i(i_cell_list);
    sort_d(d_press);
    sort_d(d_n);
    sort_d3(d3_pos);
    sort_d3(d3_vel);
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, int
///////////////////////////////////////////////////////////
void MPS_CPU::sort_i(int* const __p)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        i_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        __p[i] = i_tmp[i];
    }
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, double
///////////////////////////////////////////////////////////
void MPS_CPU::sort_d(real* const __p)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        d_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        __p[i] = d_tmp[i];
    }
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, int3
///////////////////////////////////////////////////////////
void MPS_CPU::sort_i3(int3* const __p)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        i3_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        __p[i] = i3_tmp[i];
    }
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, double3
///////////////////////////////////////////////////////////
void MPS_CPU::sort_d3(real3* const __p)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        d3_tmp[i_index[i]] = __p[i];
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        __p[i] = d3_tmp[i];
    }
}

/////////////////////////////
///gradient of press at i
/////////////////////////////
/*
real3 MPS_CPU::d3_Grad_press(const int& i)
{
    int _num;
    int _self;
    int _cell;
    int _start, _end;

    real3 _ret = {0,0,0};
    real3 _dr = {0,0,0};
    real _r2 = 0.0;
    real _hat_p = d_press[i];

    //searching _hat_p (minimum of p in 27 cells)
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    _self = i_cell_list[i];
    _num = i_link_cell[_self][0];
    for(int dir=1;dir<=_num;dir++)
    {
        _cell = i_link_cell[_self][dir];

        if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

        _start = i_cell_start[_cell];
        _end = i_cell_end[_cell];

        for(int j=_start;j<_end;j++)
        {
            if(i_type[j] == 2) continue; //dealing with boundary part
            _dr = d3_pos[j] - d3_pos[i];
            _r2 = _dr * _dr;
            if( d_press[j] < _hat_p && _r2 <= (d_rzero*d_rzero) ) _hat_p = d_press[j];
        }
    }

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(int dir=1;dir<=_num;dir++)
    {
        _cell = i_link_cell[_self][dir];

        if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

        _start = i_cell_start[_cell];
        _end = i_cell_end[_cell];

        for(int j=_start;j<_end;j++)
        {
            if(i_type[j] == 2 || j == i) continue;
            _dr = d3_pos[j] - d3_pos[i];
            _r2 = _dr * _dr;
            _ret = _ret + (d_press[j] - _hat_p) / _r2 * d_weight(d_rzero,sqrt(_r2)) * _dr;
        }
    }

    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}
*/
//////////////////////////////////////////
///divergence of velocity at i (not used)
//////////////////////////////////////////
/*
real MPS_CPU::d_Div_vel(const int& i)
{
    int _num;
    int _self;
    int _cell;
    int _start, _end;

    real _ret = 0.0;
    real _r2 = 0.0;
    real3 _dr = {0,0,0};
    real3 _du = {0,0,0};

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    _self = i_cell_list[i];
    _num = i_link_cell[_self][0];
    for(int dir=1;dir<=_num;dir++)
    {
        _cell = i_link_cell[_self][dir];

        if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

        _start = i_cell_start[_cell];
        _end = i_cell_end[_cell];

        for(int j=_start;j<_end;j++)
        {
            if(j == i) continue; //dealing with boundary part
            _dr = d3_pos[j] - d3_pos[i];
            _du = d3_vel[j] - d3_vel[i];
            _r2 = _dr * _dr;
            _ret = _ret + d_weight(d_rzero,sqrt(_r2)) / _r2 * (_du * _dr);
        }
    }

    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}
*/
/////////////////////////////////////////
///Laplacian of velocity at i
/////////////////////////////////////////
/*
real3 MPS_CPU::d3_Lap_vel(const int& i)
{
    int _num;
    int _self;
    int _cell;
    int _start, _end;

    real3 _ret = {0.0, 0.0, 0.0};
    real3 _dr;
    real3 _du;

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    _self = i_cell_list[i];
    _num = i_link_cell[i_cell_list[i]][0];
    for(int dir=1;dir<=_num;dir++)
    {
        _cell = i_link_cell[_self][dir];

        if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

        _start = i_cell_start[_cell];
        _end = i_cell_end[_cell];

        for(int j=_start;j<_end;j++)
        {
            if(j == i) continue; //dealing with boundary part
            _dr = d3_pos[j] - d3_pos[i];
            _du = d3_vel[j] - d3_vel[i];

            _ret = _ret + d_weight(d_rlap , sqrt( _dr * _dr )) * _du;
        }
    }

    _ret = (d_2bydim_over_nzerobylambda) * _ret;

    return _ret;
}
*/
//////////////////////////////
///update dt
//////////////////////////////
void MPS_CPU::update_dt()
{
    real _d_dt_tmp;

    d_dt = d_dt_max;
    for(int i = 0; i < i_np; i++)
    {
        _d_dt_tmp = d_CFL * d_dp / sqrt(d3_vel[i] * d3_vel[i]); //note: d3_vel != 0
        if(_d_dt_tmp < d_dt) d_dt = _d_dt_tmp;
    }
    if(d_dt < d_dt_min)
    {
        sprintf(c_log, "error: dt -> %e, too small \n", d_dt);
        throwScn(c_log);
        throwLog(fid_log,c_log);
        exit(1);
    }
}

//////////////////////////////
///calculate implicit part
//////////////////////////////
void MPS_CPU::cal_p_dash_impl()
{
///v2.0
    /*
    real3 _d3_vel_tmp;
    real _dt_over_rho = - d_dt * d_one_over_rho;

    //actually the following is not exactly right, the change of d3_pos[i]
    //will influence d3_Grad_press for other particles
    #ifdef CPU_OMP
        #pragma omp parallel for private(_d3_vel_tmp) schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(int i = 0; i < i_np; i++)
    {
        if(i_type[i]) continue;
        _d3_vel_tmp = _dt_over_rho * d3_Grad_press(i);
        d3_vel[i] = d3_vel[i] + _d3_vel_tmp; //update velocity
        d3_pos[i] = d3_pos[i] + d_dt * _d3_vel_tmp; //update position
    }
    */
    /*-----v3.0-----*/
    //searching for P_hat
    //only search for cells with a larger index
    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        for(int i=0;i<i_np;i++)
        {
            d_omp_tmp[i*nthread+myid] = d_press[i];
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        #ifdef CPU_OMP
            #pragma omp for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(int _icell=0;_icell<i_num_cells;_icell++) //loop over all cells
        {
            //ignore empty cells
            if(i_part_in_cell[_icell] > 0)
            {
                register char __num = i_link_cell[_icell][0];
                register unsigned __istart = i_cell_start[_icell];
                register unsigned __iend = i_cell_end[_icell];

                for(unsigned i=__istart;i<__iend;i++) //loop over i itself
                {
                    if( i_type[i] != 2)
                    {
                        for(unsigned j=i+1;j<__iend;j++)
                        {
                            register unsigned __itid = i*nthread+myid;
                            register unsigned __jtid = j*nthread+myid;
                            register real __rr = (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]);

                            if( __rr <= d_rzero * d_rzero && i_type[j] != 2)
                            {
                                if(d_press[j] < d_omp_tmp[__itid]) d_omp_tmp[__itid] = d_press[j];
                                if(d_press[i] < d_omp_tmp[__jtid]) d_omp_tmp[__jtid] = d_press[i];
                            }
                        }
                    }
                }

                for(int c=1;c<=__num;c++) //loop over surrounding cells
                {
                    register unsigned _jcell = i_link_cell[_icell][c];
                    register unsigned __jstart = i_cell_start[_jcell];
                    register unsigned __jend = i_cell_end[_jcell];

                    for(unsigned i=__istart;i<__iend;i++)
                    {
                        if( i_type[i] != 2)
                        {
                            for(unsigned j=__jstart;j<__jend;j++)
                            {
                                register unsigned __itid = i*nthread+myid;
                                register unsigned __jtid = j*nthread+myid;
                                register real __rr = (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]);

                                if( __rr <= d_rzero * d_rzero && i_type[j] != 2)
                                {
                                    if(d_press[j] < d_omp_tmp[__itid]) d_omp_tmp[__itid] = d_press[j];
                                    if(d_press[i] < d_omp_tmp[__jtid]) d_omp_tmp[__jtid] = d_press[i];
                                }
                            }
                        }
                    }
                }
            }
        }

        #ifdef CPU_OMP
            #pragma omp for
        #endif
        for(int i=0;i<i_np;i++)
        {
            register unsigned __itid = i*nthread;

            for(int j=0;j<nthread;j++)
            {
                if(d_omp_tmp[__itid+j] < d_omp_tmp[__itid]) d_omp_tmp[__itid] = d_omp_tmp[__itid+j];
            }
        }
    }

    //searching for neighbors
    //only search for cells with a larger index
    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif
        for(int i=0;i<i_np;i++)
        {
            d3_omp_tmp[i*nthread+myid].x = 0.0;
            d3_omp_tmp[i*nthread+myid].y = 0.0;
            d3_omp_tmp[i*nthread+myid].z = 0.0;
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        #ifdef CPU_OMP
            #pragma omp for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(int _icell=0;_icell<i_num_cells;_icell++) //loop over all cells
        {
            //ignore empty cells
            if(i_part_in_cell[_icell] > 0)
            {
                register char __num = i_link_cell[_icell][0];
                register unsigned __istart = i_cell_start[_icell];
                register unsigned __iend = i_cell_end[_icell];

                for(unsigned i=__istart;i<__iend;i++) //loop over i itself
                {
                    if(i_type[i] != 2)
                    {
                        for(unsigned j=i+1;j<__iend;j++)
                        {
                            register unsigned __itid = i*nthread+myid;
                            register unsigned __jtid = j*nthread+myid;
                            register real3 __dr = d3_pos[j] - d3_pos[i];
                            register real __drr = __dr * __dr;

                            if( __drr < d_rzero * d_rzero && i_type[j] != 2 )
                            {
                                register real3 __tmp = (d_press[j] - d_omp_tmp[i*nthread]) / __drr * d_weight(d_rlap , sqrt(__drr)) * __dr;

                                d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                                d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;
                            }
                        }
                    }
                }

                for(int c=1;c<=__num;c++) //loop over surrounding cells
                {
                    register unsigned _jcell = i_link_cell[_icell][c];
                    register unsigned __jstart = i_cell_start[_jcell];
                    register unsigned __jend = i_cell_end[_jcell];

                    for(unsigned i=__istart;i<__iend;i++)
                    {
                        if(i_type[i] != 2)
                        {
                            for(unsigned j=__jstart;j<__jend;j++)
                            {
                                register unsigned __itid = i*nthread+myid;
                                register unsigned __jtid = j*nthread+myid;
                                register real3 __dr = d3_pos[j] - d3_pos[i];
                                register real __drr = __dr * __dr;

                                if( __drr < d_rzero * d_rzero && i_type[j] != 2 )
                                {
                                    register real3 __tmp = (d_press[j] - d_omp_tmp[i*nthread]) / __drr * d_weight(d_rlap , sqrt(__drr)) * __dr;

                                    d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                                    d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;
                                }
                            }
                        }
                    }
                }
            }
        }

        #ifdef CPU_OMP
            #pragma omp for
        #endif
        for(int i=0;i<i_np;i++)
        {
            if(i_type[i] == 0)
            {
                register unsigned __itid = i*nthread;
                register real3 __tmp = d3_omp_tmp[__itid];

                for(int j=1;j<nthread;j++)
                {
                    __tmp = __tmp + d3_omp_tmp[__itid+j];
                }

                __tmp = - d_dt * d_one_over_rho * i_dim * d_one_over_nzero * __tmp;

                d3_vel[i] = __tmp + d3_vel[i];
                d3_pos[i] = d3_pos[i] + d_dt * __tmp;
            }
        }
    }
    /*--------------*/
}

//////////////////////////////
///particle number density
//////////////////////////////
void MPS_CPU::cal_n()
{
///v2.0
    /*
    int _num;
    int _self;
    int _cell;
    int _start, _end;

    real _n;
    real3 _dr;
    #ifdef CPU_OMP
        #pragma omp parallel for private(_dr,_n,_num,_self,_cell,_start,_end) schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(int i = 0; i < i_np; i++)//to be optimized
    {
        if(i_type[i] == 2) continue;

        _n = 0.0;

        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        _self = i_cell_list[i];
        _num = i_link_cell[_self][0];
        for(int dir=1;dir<=_num;dir++)
        {
            _cell = i_link_cell[_self][dir];

            if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

            _start = i_cell_start[_cell];
            _end = i_cell_end[_cell];

            for(int j=_start;j<_end;j++)
            {
                if(j == i) continue;
                _dr = d3_pos[j] - d3_pos[i];
                _n += d_weight(d_rzero,sqrt(_dr*_dr));
            }
        }

        d_n[i] = _n;

    }
    */

    /*-----v3.0-----*/
    //searching for neighbor
    //only search for cells with a larger index
    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        for(int i=0;i<i_np;i++)
        {
            d_omp_tmp[i*nthread+myid] = 0.0;
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        #ifdef CPU_OMP
            #pragma omp for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(int _icell=0;_icell<i_num_cells;_icell++) //loop over all cells
        {
            //ignore empty cells
            if(i_part_in_cell[_icell] > 0)
            {
                register char __num = i_link_cell[_icell][0];
                register unsigned __istart = i_cell_start[_icell];
                register unsigned __iend = i_cell_end[_icell];

                for(unsigned i=__istart;i<__iend;i++) //loop over i itself
                {
                    for(unsigned j=i+1;j<__iend;j++)
                    {
                        register unsigned __itid = i*nthread+myid;
                        register unsigned __jtid = j*nthread+myid;
                        register real __ds = sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) );

                        d_omp_tmp[__itid] = d_weight(d_rzero, __ds);
                        d_omp_tmp[__jtid] = d_weight(d_rzero, __ds);
                    }
                }

                for(int c=1;c<=__num;c++) //loop over surrounding cells
                {
                    register unsigned _jcell = i_link_cell[_icell][c];
                    register unsigned __jstart = i_cell_start[_jcell];
                    register unsigned __jend = i_cell_end[_jcell];

                    for(unsigned i=__istart;i<__iend;i++)
                    {
                        for(unsigned j=__jstart;j<__jend;j++) //j is impossible to equal i (not in the same cell)
                        {
                            register unsigned __itid = i*nthread+myid;
                            register unsigned __jtid = j*nthread+myid;
                            register real __ds = sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) );

                            d_omp_tmp[__itid] = d_weight(d_rzero, __ds);
                            d_omp_tmp[__jtid] = d_weight(d_rzero, __ds);
                        }
                    }
                }
            }
        }

        #ifdef CPU_OMP
            #pragma omp for
        #endif
        for(int i=0;i<i_np;i++)
        {
            register unsigned __itid = i*nthread;
            register real __tmp = 0.0;

            if(i_type[i] != 2)
            {
                for(int j=0;j<nthread;j++)
                {
                    __tmp += d_omp_tmp[__itid+j];
                }

                d_n[i] = __tmp;
            }
        }
    }

}

//////////////////////////////////
///make b[] & A[]
//////////////////////////////////
void MPS_CPU::build_poisson()
{
///v2.0
    /*
    int _num;
    int _self;
    int _cell;
    int _start, _end;

    real3 _dr;
    real _weight;
    */

    real _Aii = d_rho * d_lambda * d_alpha * d_nzero / (2 * i_dim * d_dt * d_dt);//compressive flow (minused by Aii)
    real _b_tmp = d_lambda * d_rho * (0.5 * d_one_over_dim * d_one_over_dt * d_one_over_dt);


    /*-----calculate b & initialize A[i][j]=0, x=press, Ap=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static,STATIC_CHUNK)
    #endif
    for(int i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        for(int j=0;j<i_np;j++)
        {
            A[i][j] = 0.0;
        }
        b[i] = _b_tmp * (d_nzero - d_n[i]);
        x[i] = d_press[i];
        Ap[i] = 0.0;
        p[i] = 0.0;
        r[i] = 0.0;
        bc[i] = false;

        if(d_n[i] < d_beta * d_nzero || i_type[i] == 2)
        {
            x[i] = 0.0;    //Dirichlet B.C.
            bc[i] = true; //particle i is at boundary
            A[i][i] = 1.0;
            b[i] = 0.0;
        }

    }
    /*-----------------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    ///v2.0
    /*
    #ifdef CPU_OMP
        #pragma omp parallel for private(_num,_self,_cell,_start,_end,_dr,_weight) schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(int i=0;i<i_np;i++)
    {
        //Dirichlet B.C. modify A[][] & b[]
        if(bc[i] || i_type[i] == 2)
        {
            A[i][i] = 1.0;
            b[i] = 0.0;
            continue;
        }

        _self = i_cell_list[i];
        _num = i_link_cell[_self][0];
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        for(int dir=1;dir<=_num;dir++)
        {
            _cell = i_link_cell[_self][dir];

            if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

            _start = i_cell_start[_cell];
            _end = i_cell_end[_cell];

            for(int j=_start;j<_end;j++)
            {
                if(i_type[j] == 2 || j == i) continue;
                _dr = d3_pos[j] - d3_pos[i];
                _weight = d_weight(d_rlap,sqrt( _dr * _dr ));

                A[i][j] += _weight;
                A[i][i] -= _weight;

                //Dirichlet B.C. modify A[][] & b[]
                if(bc[j])
                {
                    A[i][j] = 0.0;
                }
            }
        }

        A[i][i] -= _Aii;

    }

    ///symmetric copy
    //for(int i=0;i<i_nb1+i_nfluid;i++)
    //{
    //    for(int j=i+1;j<i_nb1+i_nfluid;j++)
    //    {
    //        A[j][i] = A[i][j];
    //    }
    //}
    */
    /*---------------------------------------*/

    /*-----v3.0-----*/
    //searching for neighbor
    //only search for cells with a larger index
    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {

        #ifdef CPU_OMP
            #pragma omp for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(int _icell=0;_icell<i_num_cells;_icell++) //loop over all cells
        {
            //ignore empty cells
            if(i_part_in_cell[_icell] > 0)
            {
                register char __num = i_link_cell[_icell][0];
                register unsigned __istart = i_cell_start[_icell];
                register unsigned __iend = i_cell_end[_icell];

                for(unsigned i=__istart;i<__iend;i++) //loop over i itself
                {
                    if(i != 2)
                    {
                        for(unsigned j=i+1;j<__iend;j++)
                        {
                            if(j != 2)
                            {
                                register real __weight = d_weight( d_rlap, sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) ) );

                                A[i][j] += __weight;
                                A[i][i] -= __weight;

                                A[j][i] += __weight;
                                A[j][j] -= __weight;

                                if(bc[j])
                                {
                                    A[i][j] = 0.0;
                                }
                            }
                        }
                    }
                }

                for(int c=1;c<=__num;c++) //loop over surrounding cells
                {
                    register unsigned _jcell = i_link_cell[_icell][c];
                    register unsigned __jstart = i_cell_start[_jcell];
                    register unsigned __jend = i_cell_end[_jcell];

                    for(unsigned i=__istart;i<__iend;i++)
                    {
                        if(i != 2)
                        {
                            for(unsigned j=__jstart;j<__jend;j++) //j is impossible to equal i (not in the same cell)
                            {
                                if(j != 2)
                                {
                                    register real __weight = d_weight( d_rlap, sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) ) );

                                    A[i][j] += __weight;
                                    A[i][i] -= __weight;

                                    A[j][i] += __weight;
                                    A[j][j] -= __weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        #ifdef CPU_OMP
            #pragma omp for
        #endif
        for(int i=0;i<i_np;i++)
        {
            A[i][i] -= _Aii;
        }
    }

}

//////////////////////////////////
///solve possion equation b=Ax
//////////////////////////////////
void MPS_CPU::solve_poisson()
{
    DoCG(i_np);

    /*-----update pressure-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        d_press[i] = x[i];
    }
    /*-------------------------*/
}

//////////////////////////////////////
///correct minus pressure to zero
//////////////////////////////////////
void MPS_CPU::press_corr()
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<i_np;i++)
    {
        if(d_press[i] < 0.0) d_press[i] = 0.0;
    }
}

//////////////////////////////////
///collision model
//////////////////////////////////
void MPS_CPU::collision()
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

///v2.0
    /*
    int _ncol = 0;
    real _rr;
    real _one_over_rr;
    real _vabs;
    real3 _dr; //rji
    real3 _v_crt;

    int _num;
    int _self;
    int _cell;
    int _start, _end;

    for(int i=0;i<i_np;i++)
    {
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        _self = i_cell_list[i];
        _num = i_link_cell[_self][0];
        for(int dir=1;dir<=_num;dir++)
        {
            _cell = i_link_cell[_self][dir];

            if(_cell < 0 || _cell >= i_num_cells) continue; //exact: if(_cell < 0 || _cell >= i_num_cells)

            _start = i_cell_start[_cell];
            _end = i_cell_end[_cell];
            for(int j=_start;j<_end;j++)
            {
                if(i_type[j] == 2 || j == i) continue;
                _dr = d3_pos[j] - d3_pos[i];
                _rr = sqrt(_dr * _dr);
                _one_over_rr = 1.0 / _rr;

                if( _rr > d_dp * d_col_dis ) continue;
                _vabs = 0.5f * (d3_vel[i] - d3_vel[j]) * _dr * _one_over_rr;

                if(_vabs <= 0.0) continue;
                _v_crt = d_col_rate * _vabs * _one_over_rr * _dr;

                if(i_type[j] == 0)
                {
                    d3_vel[j] = d3_vel[j] + _v_crt;
                    d3_pos[j] = d3_pos[j] + d_dt * _v_crt;
                }

                _ncol++;
                sprintf(c_log, "        collision count: %4d,    distance: %8e\n", _ncol, _rr);
                throwLog(fid_log, c_log);
                throwScn(c_log);
            }
        }

    }
    */

    /*-----v3.0-----*/
    unsigned _ncol = 0;
    //searching for neighbors
    //only search for cells with a larger index
    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif
        for(int i=0;i<i_np;i++)
        {
            d3_omp_tmp[i*nthread+myid].x = 0.0;
            d3_omp_tmp[i*nthread+myid].y = 0.0;
            d3_omp_tmp[i*nthread+myid].z = 0.0;
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel
    #endif
    {
        #ifdef CPU_OMP
        register char myid = omp_get_thread_num();
        register char nthread = omp_get_num_threads();
        #else
        register char myid = 0;
        register char nthread = 1;
        #endif

        #ifdef CPU_OMP
            #pragma omp for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(int _icell=0;_icell<i_num_cells;_icell++) //loop over all cells
        {
            //ignore empty cells
            if(i_part_in_cell[_icell] > 0)
            {
                register char __num = i_link_cell[_icell][0];
                register unsigned __istart = i_cell_start[_icell];
                register unsigned __iend = i_cell_end[_icell];

                //loop over i itself
                for(unsigned i=__istart;i<__iend;i++)
                {
                    if(i_type[i] != 2)
                    {
                        for(unsigned j=i+1;j<__iend;j++)
                        {
                            register unsigned __itid = i*nthread+myid;
                            register unsigned __jtid = j*nthread+myid;
                            register real3 __dr = d3_pos[j] - d3_pos[i];
                            register real3 __du = d3_vel[j] - d3_vel[i];
                            register real __ds = sqrt(__dr * __dr);
                            register real __vabs = 0.5f * __du * __dr / (__ds *__ds);

                            if( i_type[j] != 2 && (__ds < d_dp * d_col_dis) && (__vabs < 0) )
                            {
                                register real3 __tmp = d_col_rate * __vabs * __dr;

                                d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                                d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;

                                #ifdef CPU_OMP
                                    #pragma omp atomic
                                #endif
                                _ncol++;
                            }
                        }
                    }
                }

                //loop over surrounding cells
                for(int c=1;c<=__num;c++)
                {
                    register unsigned _jcell = i_link_cell[_icell][c];
                    register unsigned __jstart = i_cell_start[_jcell];
                    register unsigned __jend = i_cell_end[_jcell];

                    for(unsigned i=__istart;i<__iend;i++)
                    {
                        if(i_type[i] != 2)
                        {
                            for(unsigned j=__jstart;j<__jend;j++) //j is impossible to equals i
                            {
                                register unsigned __itid = i*nthread+myid;
                                register unsigned __jtid = j*nthread+myid;
                                register real3 __dr = d3_pos[j] - d3_pos[i];
                                register real3 __du = d3_vel[j] - d3_vel[i];
                                register real __ds = sqrt(__dr * __dr);
                                register real __vabs = 0.5f * __du * __dr / (__ds *__ds);

                                if( i_type[j] != 2 && (__ds < d_dp * d_col_dis) && (__vabs < 0) )
                                {
                                    register real3 __tmp = d_col_rate * __vabs * __dr;

                                    d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                                    d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;

                                    #ifdef CPU_OMP
                                        #pragma omp atomic
                                    #endif
                                    _ncol++;
                                }
                            }
                        }
                    }
                }
            }
        }

        #ifdef CPU_OMP
            #pragma omp for
        #endif
        for(int i=0;i<i_np;i++)
        {
            if(i_type[i] == 0)
            {
                register unsigned __itid = i*nthread;
                register real3 __tmp = d3_omp_tmp[__itid];

                for(int j=1;j<nthread;j++)
                {
                    __tmp = __tmp + d3_omp_tmp[__itid+j];
                }

                d3_vel[i] = __tmp + d3_vel[i];
                d3_pos[i] = d3_pos[i] + d_dt * __tmp;
            }
        }
    }

    sprintf(c_log, "        collision count: %4d\n", _ncol);
    throwLog(fid_log, c_log);
    throwScn(c_log);
    /*--------------*/

}

/////////////////////////////
///add motion of boundary
/////////////////////////////
void MPS_CPU::motion()
{
    M_motion.doMotion(d3_pos, d3_vel, i_np);
}

//////////////////////////////
///solve 'Ax = b' by CG
//////////////////////////////
void MPS_CPU::DoCG(int __n)
{
    unsigned _num = 0;

    real _alpha = 0.0;//for CG
    real _rrnew = 0.0;//for CG
    real _rrold = 0.0;//for CG
    real _rn_over_ro;

    /*-----main part of Conjugate Gradient-----*/
    //p0 = r0 = b - Ax0
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<__n;i++)
    {
        r[i] = b[i] - (real)std::inner_product(x,x+__n,A[i],0.0);
        p[i] = r[i];
    }
    _rrold = (real)std::inner_product(r,r+__n,r,0.0);

    //repeat
    while(_rrold > EPS_BY_EPS)
    {
        //printf("CONVERGENCE -> RESIDUAL: %.2e\n",_rrnew);
        if(_rrnew > RMAX) {printf("DIVERGENCE -> RESIDUAL: %.2e\n",_rrnew);exit(0);}
        #ifdef CPU_OMP
            #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            Ap[i] = (real)std::inner_product(p,p+__n,A[i],0.0);
        }
        _alpha = _rrold / (real)std::inner_product(Ap,Ap+__n,p,0.0);

        #ifdef CPU_OMP
            #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            x[i] = x[i] + _alpha * p[i];
            r[i] = r[i] - _alpha * Ap[i];
        }

        _rrnew = (real)std::inner_product(r,r+__n,r,0.0);


        _rn_over_ro = _rrnew / _rrold;

        #ifdef CPU_OMP
            #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            p[i] = r[i] + _rn_over_ro * p[i];
        }
        _rrold = _rrnew;

        _num++;
    }

    printf("    CG -> times: %d \n", _num);
    /*-----------------------------------------*/
}

