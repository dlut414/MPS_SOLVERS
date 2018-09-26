/*
LICENCE
*/
//MPS_CPU_IMPL.cpp
//implementation of class MPS_CPU_IMPL
///do implicit calculation (main loop)
#include "MPS_CPU_IMPL.h"
#include "def_incl.h"

MPS_CPU_IMPL::MPS_CPU_IMPL()
{
}

MPS_CPU_IMPL::~MPS_CPU_IMPL()
{
}

////////////////////////////////////////////////////////////
///do the initialization, main simulation and output
////////////////////////////////////////////////////////////
void MPS_CPU_IMPL::mps_cal()
{
    int _count = 0; //counter of output
    real _cal_dt = 0; //cal time for one output step
    real _cal_time = 0; //total cal time at present
    real _threshold = 0.0; //for limit output
    fid_log = fopen("./out/LOG.txt","at");

    /*-----initialization-----*/
    Initial();
    /*------------------------*/

    /*-----main loop-----*/
    while(d_time < d_tt)
    {
        //t_loop_s = time(NULL); //start
        t_loop_s = getSystemTime();
        /*-----write every tout time-----*/
        if( (d_time - _threshold) >= 0.0 )
        {
            sprintf(c_log,"count: %04d time: %.3f  calculation time: %8.3f s  cal_dt: %8.3f s \n    CFL dt: %8e s \n",
                           _count, d_time, _cal_time, _cal_time-_cal_dt, d_dt);
            sprintf(c_name,"./out/%04d.out",_count);
            throwScn(c_log);
            throwLog(fid_log,c_log);

            writeCase(); //output

            _threshold += d_tout;
            _count++;
            _cal_dt = _cal_time;
        }
        /*-------------------------------*/

        step(); //main loop

        i_step++;
        d_time += d_dt;

        //t_loop_e = time(NULL); //end
        t_loop_e = getSystemTime();
        //_cal_time += difftime(t_loop_e , t_loop_s);
        _cal_time += t_loop_e - t_loop_s;
    }
    /*-------------------*/

    /*-----finalization-----*/
    d_cal_time = _cal_time;
    sprintf(c_log,"calculation complete, total time %8.3f s\n", d_cal_time);
    throwScn(c_log);
    throwLog(fid_log,c_log);
    fclose(fid_log);
    /*----------------------*/
}

/////////////////////////////////////////
///main loop
/////////////////////////////////////////
void MPS_CPU_IMPL::step()
{
    /*-----boundary motion-----*/
    //motion();
    //divide_cell();
    /*-------------------------*/

    /*-----update d_dt-----*/
    update_dt();
    /*----------------------*/

    /*-----calculate gravity and viscosity implicitly u*,
    ---------------------------and first displacement r*-----*/
    //g_expl();
    //divide_cell();
    ///G is included into implicit step
    visCrankNicolson();
    divideCell<DIM>();
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    collision();
    divideCell<DIM>();
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    cal_n();
    buildPoisson();
    solvePoisson();
    pressCorr(); //correct minus pressure to zero
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    cal_Pdash_impl();
    divideCell<DIM>();
    /*-------------------------------------------------------------------------*/
    update_dt();
}

/////////////////////////////////////////
///calculate expicit part by g (in CN)
/////////////////////////////////////////
void MPS_CPU_IMPL::g_expl()
{
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        if(i_type[i] == 0)
        {
            real3 __vel_tmp = d3_vel[i] + d_dt * G;
            d3_vel[i] = __vel_tmp;
            d3_pos[i] = d3_pos[i] + d_dt * __vel_tmp;
        }
    }
}

//////////////////////////////
///update pos and vel
//////////////////////////////
template<unsigned int _dir> void MPS_CPU_IMPL::updatePosVel()
{
    switch(_dir)
    {
        case 1:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static, STATIC_CHUNK)
            #endif
            for(integer i = 0; i < i_np; i++)
            {
                if(i_type[i] == 0)
                {
                    d3_vel[i].x = x[i];
                    d3_pos[i].x += d_dt * x[i];
                }
            }
            break;
        }
        case 2:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static, STATIC_CHUNK)
            #endif
            for(integer i = 0; i < i_np; i++)
            {
                if(i_type[i] == 0)
                {
                    d3_vel[i].y = x[i];
                    d3_pos[i].y += d_dt * x[i];
                }
            }
            break;
        }
        case 3:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static, STATIC_CHUNK)
            #endif
            for(integer i = 0; i < i_np; i++)
            {
                if(i_type[i] == 0)
                {
                    d3_vel[i].z = x[i];
                    d3_pos[i].z += d_dt * x[i];
                }
            }
            break;
        }
    }
}

//////////////////////////////
///build A, x, b for DoCG
//////////////////////////////
template<unsigned int _dir> void MPS_CPU_IMPL::buildCG_CN()
{
    real _ONE_OVER_R = (d_lambda * d_nzero) / (i_dim * d_niu * d_dt); // 1 / R, for b
    real3 _DT_BY_G_OVER_R = (d_lambda * d_nzero) / (i_dim * d_niu) * G; // dt * g / R, for b

    /*-----initialize A[i][j]=0, x=d3_vel.x, Ap=0, b=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static, STATIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        for(integer j=0;j<i_np;j++) A[i][j] = 0.0;

        b[i] = 0.0;
        switch(_dir)
        {
            case 1: x[i] = d3_vel[i].x; break;
            case 2: x[i] = d3_vel[i].y; break;
            case 3: x[i] = d3_vel[i].z; break;
        }
        Ap[i] = 0.0;
        p[i] = 0.0;
        r[i] = 0.0;
        bc[i] = false;

        if(i_type[i])
        {
            bc[i] = true; //particle i is at boundary
        }
    }
    /*-----------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    //to be optimized
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        //only fluid particles
        if(bc[i] == 0)
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
                        if(j != i)
                        {
                            real3 __dr = d3_pos[j] - d3_pos[i];
                            real __weight = d_weight(d_rlap , sqrt( __dr * __dr ));

                            A[i][j] += __weight;
                            A[i][i] -= __weight;

                            switch(_dir)
                            {
                                case 1: b[i] -= (d3_vel[j].x - d3_vel[i].x) * __weight; break;
                                case 2: b[i] -= (d3_vel[j].y - d3_vel[i].y) * __weight; break;
                                case 3: b[i] -= (d3_vel[j].z - d3_vel[i].z) * __weight; break;
                            }

                            //j is boundary particle
                            if(bc[j])
                            {
                                A[i][j] = 0.0;
                                switch(_dir)
                                {
                                    case 1: b[i] -= __weight * d3_vel[j].x; break;
                                    case 2: b[i] -= __weight * d3_vel[j].y; break;
                                    case 3: b[i] -= __weight * d3_vel[j].z; break;
                                }
                            }
                        }
                    }
                }
            }

            A[i][i] -= _ONE_OVER_R;

            switch(_dir)
            {
                case 1: b[i] -= _ONE_OVER_R * d3_vel[i].x + _DT_BY_G_OVER_R.x; break;
                case 2: b[i] -= _ONE_OVER_R * d3_vel[i].y + _DT_BY_G_OVER_R.y; break;
                case 3: b[i] -= _ONE_OVER_R * d3_vel[i].z + _DT_BY_G_OVER_R.z; break;
            }
        }

        //Dirichlet B.C. modify A[][] & b[]
        //i is boundary particle
        else
        {
            A[i][i] = 1.0;
            switch(_dir)
            {
                case 1: b[i] = d3_vel[i].x; break;
                case 2: b[i] = d3_vel[i].y; break;
                case 3: b[i] = d3_vel[i].z; break;
            }
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

//////////////////////////////
///viscous by crank-nicoloson
//////////////////////////////
void MPS_CPU_IMPL::visCrankNicolson()
{
    //x direction
    buildCG_CN<1>();
    DoCG(i_np);
    updatePosVel<1>();

    //y direction
    buildCG_CN<2>();
    DoCG(i_np);
    updatePosVel<2>();

    //z direction
    buildCG_CN<3>();
    DoCG(i_np);
    updatePosVel<3>();
}
