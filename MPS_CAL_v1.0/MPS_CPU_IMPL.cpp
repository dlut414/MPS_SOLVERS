/*
LICENCE
*/
//MPS_CPU_IMPL.h
//implementation of class MPS_CPU_EXPL
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
    int _cal_dt = 0; //cal time for one output step
    int _cal_time = 0; //total cal time at present
    real _threshold = 0.0; //for limit output
    fid_log = fopen("./out/LOG.txt","at");

    /*-----initialization-----*/
    Initial();
    /*------------------------*/

    /*-----main loop-----*/
    while(d_time < d_tt)
    {
        t_loop_s = time(NULL); //start
        /*-----write every tout time-----*/
        if( (d_time - _threshold) >= 0.0 )
        {
            sprintf(c_log,"count: %04d time: %.3f  calculation time: %8d s  cal_dt: %8d s \n    CFL dt: %8e s \n",
                           _count, d_time, _cal_time, _cal_time-_cal_dt, d_dt);
            sprintf(c_name,"./out/%04d.out",_count);
            throwScn(c_log);
            throwLog(fid_log,c_log);

            WriteCase(); //output

            _threshold += d_tout;
            _count++;
            _cal_dt = _cal_time;
        }
        /*-------------------------------*/

        step(); //main loop

        i_step++;
        d_time += d_dt;

        t_loop_e = time(NULL); //end
        _cal_time += difftime(t_loop_e , t_loop_s);
    }
    /*-------------------*/

    /*-----finalization-----*/
    i_cal_time = _cal_time;
    sprintf(c_log,"calculation complete, total time %8d s\n", i_cal_time);
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
    g_expl();
    //divide_cell();
    vis_CrankNicolson();
    //divide_cell();
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    collision();
    //divide_cell();
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    cal_n();
    build_poisson();
    solve_poisson();
    press_corr(); //correct minus pressure to zero
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    impl_part();
    //divide_cell();
    /*-------------------------------------------------------------------------*/
    update_dt();
}

/////////////////////////////////////////
///calculate expicit part by g (in CN)
/////////////////////////////////////////
void MPS_CPU_IMPL::g_expl()
{
    real3 _vel_tmp;
    #ifdef CPU_OMP
        #pragma omp parallel for private(_vel_tmp)
    #endif
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _vel_tmp = d3_vel[i] + d_dt * G;
        d3_vel[i] = _vel_tmp;
        d3_pos[i] = d3_pos[i] + d_dt * _vel_tmp;
    }
}

//////////////////////////////
///update pos and vel
//////////////////////////////
template<unsigned int _dir> void MPS_CPU_IMPL::update_pos_vel()
{
    switch(_dir)
    {
        case 1:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].x = x[i];
                d3_pos[i].x += d_dt * x[i];
            }
            break;
        }
        case 2:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].y = x[i];
                d3_pos[i].y += d_dt * x[i];
            }
            break;
        }
        case 3:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].z = x[i];
                d3_pos[i].z += d_dt * x[i];
            }
            break;
        }
    }
}

//////////////////////////////
///build A, x, b for DoCG
//////////////////////////////
template<unsigned int _dir> void MPS_CPU_IMPL::CN_buildCG()
{
    real3 _dr;
    real _weight;
    real _ONE_OVER_R = (d_lambda * d_nzero) / (i_dim * d_niu * d_dt);

    /*-----initialize A[i][j]=0, x=d3_vel.x, Ap=0, b=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static)
    #endif
    for(int i=0;i<i_np;i++) //size of A,b,x is i_np
    {
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

        if(i < i_nb2 + i_nb1)
        {
            bc[i] = true; //particle i+i_nb2 is at boundary
        }

        for(int j=0;j<i_np;j++) A[i][j] = 0.0;
    }
    /*-----------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for private(_dr,_weight)
    #endif
    for(int i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        //Dirichlet B.C. modify A[][] & b[]
        if(bc[i])
        {
            A[i][i] = 1.0;
            switch(_dir)
            {
                case 1: b[i] = d3_vel[i].x; break;
                case 2: b[i] = d3_vel[i].y; break;
                case 3: b[i] = d3_vel[i].z; break;
            }
            continue;
        }

        for(int j=0;j<i_np;j++)
        {
            if(j == i) continue;
            _dr = d3_pos[j] - d3_pos[i];
            _weight = d_weight(d_rlap , sqrt( _dr * _dr ));

            A[i][j] += _weight;
            A[i][i] -= _weight;

            switch(_dir)
            {
                case 1: b[i] -= (d3_vel[j].x - d3_vel[i].x) * _weight; break;
                case 2: b[i] -= (d3_vel[j].y - d3_vel[i].y) * _weight; break;
                case 3: b[i] -= (d3_vel[j].z - d3_vel[i].z) * _weight; break;
            }

            if(bc[j])
            {
                A[i][j] = 0.0;
                switch(_dir)
                {
                    case 1: b[i] -= _weight * d3_vel[j].x; break;
                    case 2: b[i] -= _weight * d3_vel[j].y; break;
                    case 3: b[i] -= _weight * d3_vel[j].z; break;
                }
            }
        }

        A[i][i] -= _ONE_OVER_R;

        switch(_dir)
        {
            case 1: b[i] -= _ONE_OVER_R * d3_vel[i].x; break;
            case 2: b[i] -= _ONE_OVER_R * d3_vel[i].y; break;
            case 3: b[i] -= _ONE_OVER_R * d3_vel[i].z; break;
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
void MPS_CPU_IMPL::vis_CrankNicolson()
{
    //x direction
    CN_buildCG<1>();
    DoCG(i_np);
    update_pos_vel<1>();

    //y direction
    CN_buildCG<2>();
    DoCG(i_np);
    update_pos_vel<2>();

    //z direction
    CN_buildCG<3>();
    DoCG(i_np);
    update_pos_vel<3>();
}
