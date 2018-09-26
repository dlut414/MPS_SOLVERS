/*
LICENCE
*/
//MPS_GPU_EXPL.cu
//implementation of class MPS_GPU_EXPL
///do explicit calculation (main loop)
#include "def_incl.h"
#include "MPS_GPU_EXPL.h"

using namespace mytype;

MPS_GPU_EXPL::MPS_GPU_EXPL()
{
}

MPS_GPU_EXPL::~MPS_GPU_EXPL()
{
}

////////////////////////////////////////////////////////////
///do the initialization, main simulation and output
////////////////////////////////////////////////////////////
void MPS_GPU_EXPL::mps_cal()
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
        _cal_time += real(t_loop_e - t_loop_s) / 1000;
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
void MPS_GPU_EXPL::step()
{
    long _start, _end;
    /*-----boundary motion-----*/
    //motion();
    //divide_cell();
    /*-------------------------*/

    /*-----update d_dt-----*/
    update_dt();
    /*----------------------*/

    /*-----calculate gravity and viscosity explicitly u*,
    ---------------------------and first displacement r*-----*/
    _start = getSystemTime();

    cal_vis_expl();
    divideCell<DIM>();

    _end = getSystemTime();
    d_part1 = real(_end-_start)/1000;
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    _start = getSystemTime();

    collision();
    divideCell<DIM>();

    _end = getSystemTime();
    d_part2 = real(_end-_start)/1000;
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    _start = getSystemTime();

    cal_n();
    buildPoisson();

    _end = getSystemTime();
    d_part3 = real(_end-_start)/1000;

    _start = getSystemTime();

    solvePoisson();
    pressCorr(); //correct minus pressure to zero

    _end = getSystemTime();
    d_part4 = real(_end-_start)/1000;
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    _start = getSystemTime();

    cal_Pdash_impl();
    divideCell<DIM>();

    _end = getSystemTime();
    d_part5 = real(_end-_start)/1000;
    /*-------------------------------------------------------------------------*/
    update_dt();

    printf("            part1 time ->   %.3f \n", d_part1);
    printf("            part2 time ->   %.3f \n", d_part2);
    printf("            part3 time ->   %.3f \n", d_part3);
    printf("            part4 time ->   %.3f \n", d_part4);
    printf("            part5 time ->   %.3f \n", d_part5);
}

/////////////////////////////////////
///calculate explicit part vis and g
/////////////////////////////////////
void MPS_GPU_EXPL::cal_vis_expl()
{
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        //only apply to fluid particles
        if(i_type[i] == 0)
        {
            real3 __vel_tmp = d3_vel[i] + d_dt * (d_niu * d3_LapVel(i) + G);
            d3_vel[i] = __vel_tmp;
            d3_pos[i] = d3_pos[i] + d_dt * __vel_tmp;
        }
    }
}
