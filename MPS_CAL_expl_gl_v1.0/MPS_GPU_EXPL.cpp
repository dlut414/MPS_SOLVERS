/*
LICENCE
*/
//MPS_GPU_EXPL.cu
//implementation of class MPS_GPU_EXPL
///do explicit calculation (main loop)
#include "def_incl.h"
#include "MPS_GPU_EXPL.h"
#include "mps_gpu_cudaker.h"

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
    //devInit();
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

    //devFree();
    /*----------------------*/
}

/////////////////////////////////////////
///main loop
/////////////////////////////////////////
void MPS_GPU_EXPL::step()
{
    long _start, _end;
    real d_part1;
    real d_part2;
    real d_part3;
    real d_part4;
    real d_part5;
    /*-----boundary motion-----*/
    //motion();
    //divide_cell();
    /*-------------------------*/

    /*-----update d_dt-----*/
    update_dt();
    /*----------------------*/

    /*-----calculate gravity and viscosity explicitly u*,
    ---------------------------and first displacement r*-----*/
    ///part 1
    _start = getSystemTime();

    calVisc_expl();//gpu
    divideCell();

    _end = getSystemTime();
    d_part1 = real(_end-_start)/1000;
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    ///part 2
    _start = getSystemTime();

    collision();//gpu
    divideCell();

    _end = getSystemTime();
    d_part2 = real(_end-_start)/1000;
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    ///part 3
    _start = getSystemTime();

    cal_n();//gpu
    calPres_expl();//gpu

    _end = getSystemTime();
    d_part3 = real(_end-_start)/1000;
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    ///part 4
    _start = getSystemTime();

    calDash();//gpu
    divideCell();

    _end = getSystemTime();
    d_part4 = real(_end-_start)/1000;
    /*-------------------------------------------------------------------------*/

    printf("            part1 time ->   %.3f \n", d_part1);
    printf("            part2 time ->   %.3f \n", d_part2);
    printf("            part3 time ->   %.3f \n", d_part3);
    printf("            part4 time ->   %.3f \n", d_part4);

}

/////////////////////////////////////
///calculate vis and g explicitly
/////////////////////////////////////
void MPS_GPU_EXPL::calVisc_expl()
{
#ifdef GPU_CUDA

    cudaker::dev_calVisc_expl( d3_vel, d3_pos, d3_tmp,
                               d_press,
                               i_type,
                               i_cell_list,
                               i_link_cell,
                               i_cell_start,
                               i_cell_end,
                               d_dt, d_2bydim_over_nzerobylambda, d_rlap, d_niu,
                               i_num_cells, i_np );

#else

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(integer i = 0; i < i_np; i++)
    {
        //ignore boundary
        if(i_type[i] == 0)
        {
            d3_tmp[i] = d3_vel[i] + d_dt * (d_niu * d3_LapVel(i) + G);
        }
    }

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] == 0)
        {
            d3_vel[i] = d3_tmp[i];
            d3_pos[i] = d3_pos[i] + d_dt * d3_tmp[i];
        }
    }

#endif

}

//////////////////////////////////////
///calculate pressure explicitly
//////////////////////////////////////
void MPS_GPU_EXPL::calPres_expl()
{
#ifdef GPU_CUDA

    cudaker::dev_calPres_expl(d_press, d_n,
                              i_type, i_normal,
                              d_one_over_alpha,
                              d_nzero,
                              d_one_over_nzero,
                              i_np);

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
