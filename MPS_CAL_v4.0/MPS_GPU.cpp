/*
LICENCE
*/
//MPS_GPU.cu
//implementation of class MPS_GPU
///receive data from class MPS and functions of main loop

#include "def_incl.h"
#include "MPS_GPU.h"
#include "cuda_ker.h"

using namespace mytype;

MPS_GPU::MPS_GPU()
{
    fid_log = NULL;
    fid_out = NULL;
    b = NULL;
    x = NULL;
    A = NULL;
    bc = NULL;

    i_tmp = NULL;
    d_tmp = NULL;
    i3_tmp = NULL;
    d3_tmp = NULL;

    i_ini_time = 0;
    d_cal_time = 0;
    t_loop_s = 0;
    t_loop_e = 0;
}

MPS_GPU::~MPS_GPU()
{
    fid_log = NULL;
    fid_out = NULL;
    delete[] b;
    delete[] x;
    delete[] A;
    delete[] bc;

    delete[] i_tmp;
    delete[] d_tmp;
    delete[] i3_tmp;
    delete[] d3_tmp;
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
    calOnce();
    /*----------------------------------------*/

    /*-----initialization of variables in MPS_CPU-----*/
    strcpy(c_name , "0000.out");
    i_step = 0;
    d_time = 0.0;

    i_ini_time = 0;
    d_cal_time = 0;

    ///-----initialization for b=Ax
    b = new real[i_np];
    x = new real[i_np];
    A = new real[i_np*i_np];

    bc = new bool[i_np];

    ///-----initialization for gpu

    memAdd(sizeof(real), i_np);
    memAdd(sizeof(real), i_np);
    memAdd(sizeof(real), i_np);
    memAdd(sizeof(real), i_np);
    memAdd(sizeof(real), i_np);
    memAdd(sizeof(bool), i_np);
    memAdd(sizeof(real), i_np*i_np);

    i_tmp = new integer[i_np];
    d_tmp = new real[i_np];
    i3_tmp = new int3[i_np];
    d3_tmp = new real3[i_np];

    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(real), i_np);
    memAdd(sizeof(int3), i_np);
    memAdd(sizeof(real3), i_np);
    /*------------------------------------------------*/

    /*-----make the cell-list-----*/
    divideCell<DIM>();
    /*----------------------------*/

    /*-----end of initialization-----*/
    t_loop_e = time(NULL);
    i_ini_time = difftime(t_loop_e , t_loop_s);
    /*-------------------------------*/

    sprintf(str, "successfully Initialized!\n");
    throwScn(str);
    throwLog(fid_log, str);
    sprintf(str, "memory usage: %.1f M byte.\n", d_mem / 1024);
    throwScn(str);
    throwLog(fid_log, str);
}

/////////////////////////////////////////
///out put case
/////////////////////////////////////////
void MPS_GPU::writeCase()
{
    fid_out = fopen(c_name , "wt");

    for(integer i = 0; i < i_np; i++)
    {
        fprintf(fid_out , "%d %d %lf %lf %lf %lf %lf %lf %lf " ,
                i_id[i], i_type[i],
                d3_pos[i].x , d3_pos[i].y , d3_pos[i].z ,
                d3_vel[i].x , d3_vel[i].y , d3_vel[i].z ,
                d_press[i]);
    }

    fclose(fid_out);
}

////////////////////////////////////
///set all values in p to zero
////////////////////////////////////
template<typename T> void MPS_GPU::Zero(T* p, integer n)
{
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<n;i++) p[i] = 0;
}

////////////////////////////////
///divide the domain into cells
////////////////////////////////
template<char _dim> void MPS_GPU::divideCell()
{
    Zero(i_part_in_cell, i_num_cells);
    Zero(i_cell_start, i_num_cells);

    //divide particles into cells
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        integer __cellx = (d3_pos[i].x - d_cell_left) / d_cell_size;
        integer __celly = (d3_pos[i].y - d_cell_back) / d_cell_size;
        integer __cellz = (d3_pos[i].z - d_cell_bottom) / d_cell_size;
        integer __num = __cellz * i_cell_sheet + __celly * i_cell_dx + __cellx;

        if(__num >= i_num_cells || __num < 0)
        {
            char str[256];
            sprintf(str,"particle exceeding cell -> id: %d, _num: %d, # of cells: %d\n",
                        i_id[i], __num, i_num_cells);
            throwScn(str);
            throwLog(fid_log, str);
            i_type[i] = 2;
            /*exit(4);*/
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
    for(integer i=1;i<i_num_cells;i++)
    {
        i_cell_end[i] = i_cell_start[i] = i_cell_start[i-1] + i_part_in_cell[i-1];
    }

    //put particles into cells
    //can not be paralleled !
    //*1).i_index -> new , i -> old; 2).i_index -> old , i -> new, which is better?
    for(integer i=0;i<i_np;i++)
    {
        integer __cell_index = i_cell_list[i];

        i_index[i] = i_cell_end[__cell_index];
        i_cell_end[__cell_index]++;
    }

    //make link_cell list
    if(_dim == 2)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(integer i=0;i<i_num_cells;i++)
        {
            //ignore empty cells
            if(i_part_in_cell[i] != 0)
            {
                i_link_cell[i][0] = 9;

                i_link_cell[i][1] = i - i_cell_dx - 1; //-1,-1
                i_link_cell[i][2] = i - i_cell_dx; //-1,0
                i_link_cell[i][3] = i - i_cell_dx + 1; //-1,1

                i_link_cell[i][4] = i - 1; //0,-1
                i_link_cell[i][5] = i; //0,0
                i_link_cell[i][6] = i + 1; //0,1

                i_link_cell[i][7] = i + i_cell_dx - 1; //1,-1
                i_link_cell[i][8] = i + i_cell_dx; //1,0
                i_link_cell[i][9] = i + i_cell_dx + 1; //1,1
            }
        }
    }
    else if(_dim == 3)
    {
        #ifdef CPU_OMP
            #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
        #endif
        for(integer i=0;i<i_num_cells;i++)
        {
            //ignore empty cells
            if(i_part_in_cell[i] != 0)
            {
                i_link_cell[i][0] = 27;

                i_link_cell[i][1] = i - i_cell_sheet - i_cell_dx - 1; //-1,-1,-1
                i_link_cell[i][2] = i - i_cell_sheet - i_cell_dx; //-1,-1,0
                i_link_cell[i][3] = i - i_cell_sheet - i_cell_dx + 1; //-1,-1,1

                i_link_cell[i][4] = i - i_cell_sheet - 1; //-1,0,-1
                i_link_cell[i][5] = i - i_cell_sheet; //-1,0,0
                i_link_cell[i][6] = i - i_cell_sheet + 1; //-1,0,1

                i_link_cell[i][7] = i - i_cell_sheet + i_cell_dx - 1; //-1,1,-1
                i_link_cell[i][8] = i - i_cell_sheet + i_cell_dx; //-1,1,0
                i_link_cell[i][9] = i - i_cell_sheet + i_cell_dx + 1; //-1,1,1

                i_link_cell[i][10] = i - i_cell_dx - 1; //0,-1,-1
                i_link_cell[i][11] = i - i_cell_dx; //0,-1,0
                i_link_cell[i][12] = i - i_cell_dx + 1; //0,-1,1

                i_link_cell[i][13] = i - 1; //0,0,-1
                i_link_cell[i][14] = i; //0,0,0
                i_link_cell[i][15] = i + 1; //0,0,1

                i_link_cell[i][16] = i + i_cell_dx - 1; //0,1,-1
                i_link_cell[i][17] = i + i_cell_dx; //0,1,0
                i_link_cell[i][18] = i + i_cell_dx + 1; //0,1,1

                i_link_cell[i][19] = i + i_cell_sheet - i_cell_dx - 1; //1,-1,-1
                i_link_cell[i][20] = i + i_cell_sheet - i_cell_dx; //1,-1,0
                i_link_cell[i][21] = i + i_cell_sheet - i_cell_dx + 1; //1,-1,1

                i_link_cell[i][22] = i + i_cell_sheet - 1; //1,0,-1
                i_link_cell[i][23] = i + i_cell_sheet; //1,0,0
                i_link_cell[i][24] = i + i_cell_sheet + 1; //1,0,1

                i_link_cell[i][25] = i + i_cell_sheet + i_cell_dx - 1; //1,1,-1
                i_link_cell[i][26] = i + i_cell_sheet + i_cell_dx; //1,1,0
                i_link_cell[i][27] = i + i_cell_sheet + i_cell_dx + 1; //1,1,1
            }
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
void MPS_GPU::sort_i(integer* const __p)
{
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
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, double
///////////////////////////////////////////////////////////
void MPS_GPU::sort_d(real* const __p)
{
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
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, int3
///////////////////////////////////////////////////////////
void MPS_GPU::sort_i3(int3* const __p)
{
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
}

///////////////////////////////////////////////////////////
///sorting funtion putting partcles into new index, double3
///////////////////////////////////////////////////////////
void MPS_GPU::sort_d3(real3* const __p)
{
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
}

/////////////////////////////
///gradient of press at i
/////////////////////////////
real3 MPS_GPU::d3_GradPress(const integer& i)
{
    real3 _ret = {0,0,0};
    integer _self = i_cell_list[i];
    integer _num = i_link_cell[_self][0];
    real _hat_p = d_press[i];

    //searching _hat_p (minimum of p in 27 cells)
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_self][dir];

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

                    if( d_press[j] < _hat_p && __rr <= (d_rzero*d_rzero) ) _hat_p = d_press[j];
                }
            }
        }
    }

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_self][dir];

        if(__cell >= 0 && __cell < i_num_cells)
        {
            integer __start = i_cell_start[__cell];
            integer __end = i_cell_end[__cell];

            for(integer j=__start;j<__end;j++)
            {
                //ignore type 2 and i itself
                if(i_type[j] != 2 && j != i)
                {
                    real3 __dr = d3_pos[j] - d3_pos[i];
                    real __rr = __dr * __dr;

                    _ret = _ret + (d_press[j] - _hat_p) / __rr * d_weight(d_rzero,sqrt(__rr)) * __dr;
                }
            }
        }
    }

    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}

//////////////////////////////////////////
///divergence of velocity at i (not used)
//////////////////////////////////////////
real MPS_GPU::d_DivVel(const integer& i)
{
    real _ret = 0.0;
    integer _self = i_cell_list[i];
    integer _num = i_link_cell[_self][0];

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_self][dir];

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
    integer _self = i_cell_list[i];
    integer _num = i_link_cell[i_cell_list[i]][0];
    real3 _ret = {0.0, 0.0, 0.0};

    //searching neighbors
    //loop: surrounding cells including itself (totally 27 cells)
    //loop: from bottom to top, from back to front, from left to right
    for(integer dir=1;dir<=_num;dir++)
    {
        integer __cell = i_link_cell[_self][dir];

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
        throwLog(fid_log,c_log);
        exit(1);
    }
}

//////////////////////////////
///calculate implicit part
//////////////////////////////
void MPS_GPU::cal_Pdash_impl()
{
    //actually the following is not exactly right, the change of d3_pos[i]
    //will influence d3_Grad_press for other particles
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        //only apply to fluid particles
        if(i_type[i] == 0)
        {
            real3 __vel_tmp = - d_dt * d_one_over_rho * d3_GradPress(i);

            d3_vel[i] = d3_vel[i] + __vel_tmp; //update velocity
            d3_pos[i] = d3_pos[i] + d_dt * __vel_tmp; //update position
        }
    }
}

//////////////////////////////
///particle number density
//////////////////////////////
void MPS_GPU::cal_n()
{
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
            integer __self = i_cell_list[i];
            integer __num = i_link_cell[__self][0];

            for(integer dir=1;dir<=__num;dir++)
            {
                integer __cell = i_link_cell[__self][dir];

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
}

//////////////////////////////////
///make b[] & A[]
//////////////////////////////////
void MPS_GPU::buildPoisson()
{
    real _Aii = d_rho * d_lambda * d_alpha * d_nzero / (2 * i_dim * d_dt * d_dt);//compressive flow (minused by Aii)
    real _b_tmp = d_lambda * d_rho * (0.5 * d_one_over_dim * d_one_over_dt * d_one_over_dt);

    /*-----calculate b & initialize A[i][j]=0, x=press, Ap=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        for(integer j=0;j<i_np;j++) A(i,j,i_np) = 0.0;

        b[i] = _b_tmp * (d_nzero - d_n[i]);
        x[i] = d_press[i];
        bc[i] = false;

        if(d_n[i] < d_beta * d_nzero)
        {
            x[i] = 0.0;    //Dirichlet B.C.
            bc[i] = true; //particle i is at boundary
        }
    }
    /*-----------------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        if( !bc[i] && i_type[i] != 2)
        {
            //searching neighbors
            //loop: surrounding cells including itself (totally 27 cells)
            //loop: from bottom to top, from back to front, from left to right
            integer __self = i_cell_list[i];
            integer __num = i_link_cell[__self][0];
            for(integer dir=1;dir<=__num;dir++)
            {
                integer __cell = i_link_cell[__self][dir];

                if(__cell >= 0 && __cell < i_num_cells)
                {
                    integer __start = i_cell_start[__cell];
                    integer __end = i_cell_end[__cell];

                    for(integer j=__start;j<__end;j++)
                    {
                        //ignore type 2 particles but not bc particles
                        if(i_type[j] != 2 && j != i)
                        {
                            real3 __dr = d3_pos[j] - d3_pos[i];
                            real __weight = d_weight(d_rlap,sqrt( __dr * __dr ));

                            A(i,j,i_np) += __weight;
                            A(i,i,i_np) -= __weight;

                            //Dirichlet B.C. modify A[][] & b[]
                            if(bc[j])
                            {
                                A(i,j,i_np) = 0.0;
                            }
                        }
                    }
                }
            }

            A(i,i,i_np) -= _Aii;
        }

        //Dirichlet B.C. modify A[][] & b[]
        else
        {
            A(i,i,i_np) = 1.0;
            b[i] = 0.0;
        }
    }

    ///symmetric copy
    //for(int i=0;i<i_nb1+i_nfluid;i++)
    //{
    //    for(int j=i+1;j<i_nb1+i_nfluid;j++)
    //    {
    //        A[j][i] = A[i][j];
    //    }
    //}
    /*---------------------------------------*/
}

//////////////////////////////////
///solve possion equation b=Ax
//////////////////////////////////
void MPS_GPU::solvePoisson()
{
    DoCG(i_np);
    //DoJacobi(i_np);

    /*-----update pressure-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        d_press[i] = x[i];
    }
    /*-------------------------*/
}

//////////////////////////////////////
///correct minus pressure to zero
//////////////////////////////////////
void MPS_GPU::pressCorr()
{
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        if(d_press[i] < 0.0) d_press[i] = 0.0;
    }
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

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        d3_tmp[i].x = d3_tmp[i].y = d3_tmp[i].z = 0.0;
    }

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        integer __self = i_cell_list[i];
        integer __num = i_link_cell[__self][0];

        for(integer dir=1;dir<=__num;dir++)
        {
            integer __cell = i_link_cell[__self][dir];

            if( (__cell >= 0) && (__cell < i_num_cells) )
            {
                integer __start = i_cell_start[__cell];
                integer __end = i_cell_end[__cell];

                for(integer j=__start;j<__end;j++)
                {
                    if(i_type[j] != 2 && j != i)
                    {
                        real3 __dr = d3_pos[j] - d3_pos[i];
                        real3 __du = d3_vel[j] - d3_vel[i];
                        real __ds = sqrt(__dr * __dr);
                        real __one_over_ds = 1.0 / __ds;
                        real __vabs = 0.5f * __du * __dr * __one_over_ds;

                        if( (__ds <= d_dp * d_col_dis) && (__vabs <= 0.0) )
                        {
                            real3 __v_crt = - d_col_rate * __vabs * __one_over_ds * __dr;

                            if(i_type[j] == 0)
                            {
                                d3_tmp[j] = __v_crt;
                                /*
                                d3_vel[j] = d3_vel[j] + __v_crt;
                                d3_pos[j] = d3_pos[j] + d_dt * __v_crt;
                                */
                            }

                            _ncol++;
                            sprintf(c_log, "        collision count: %4d,    distance: %8e\n", _ncol, __ds);
                            throwLog(fid_log, c_log);
                            throwScn(c_log);
                        }
                    }
                }
            }
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        d3_vel[i] = d3_vel[i] + d3_tmp[i];
        d3_pos[i] = d3_pos[i] + d_dt * d3_tmp[i];
    }

}

/////////////////////////////
///add motion of boundary
/////////////////////////////
void MPS_GPU::motion()
{
    M_motion.doMotion(d3_pos, d3_vel, i_np);
}

//////////////////////////////
///solve 'Ax = b' by CG
//////////////////////////////
void MPS_GPU::DoCG(integer n)
{
    cudaker::CG(A, x, b, n);
}

