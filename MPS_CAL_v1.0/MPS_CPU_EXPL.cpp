/*
LICENCE
*/
//MPS_CPU.h
//implementation of class MPS_CPU_EXPL
///do explicit calculation (main loop)
#include "def_incl.h"
#include "MPS_CPU_EXPL.h"

MPS_CPU_EXPL::MPS_CPU_EXPL()
{
}

MPS_CPU_EXPL::~MPS_CPU_EXPL()
{
}

////////////////////////////////////////////////////////////
///do the initialization, main simulation and output
////////////////////////////////////////////////////////////
void MPS_CPU_EXPL::mps_cal()
{
    int _count = 0; //counter of output
    int _cal_dt = 0; //cal time for one output step
    int _cal_time = 0; //total cal time at present
    real _threshold = 0.0; //for limit output
    fid_log = fopen(LOG_NAME,"at");

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
void MPS_CPU_EXPL::step()
{
    /*-----boundary motion-----*/
    //motion();
    /*-------------------------*/

    /*-----update d_dt-----*/
    update_dt();
    /*----------------------*/

    /*-----calculate gravity and viscosity explicitly u*,
    ---------------------------and first displacement r*-----*/
    expl_part();
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    collision();
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    cal_n();
    build_poisson();
    solve_poisson();
    press_corr(); //correct minus pressure to zero
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    impl_part();
    /*-------------------------------------------------------------------------*/
    update_dt();
}

//////////////////////////////
///calculate explicit part
//////////////////////////////
void MPS_CPU_EXPL::expl_part()
{
    real3 _vel_tmp;
    #ifdef CPU_OMP
        #pragma omp parallel for private(_vel_tmp)
    #endif
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _vel_tmp = d3_vel[i] + d_dt * (d_niu * d3_Lap_vel(i) + G);
        d3_vel[i] = _vel_tmp;
        d3_pos[i] = d3_pos[i] + d_dt * _vel_tmp;
    }
}
