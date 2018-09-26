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
            sprintf(c_log,"count: %04d time: %.3f  calculation time: %8f s  cal_dt: %8f s \n    CFL dt: %8e s \n",
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

        //t_loop_e = time(NULL); //end
        t_loop_e = getSystemTime();
        //_cal_time += difftime(t_loop_e , t_loop_s);
        _cal_time += t_loop_e - t_loop_s;
    }
    /*-------------------*/

    /*-----finalization-----*/
    d_cal_time = _cal_time;
    sprintf(c_log,"calculation complete, total time %8f s\n", d_cal_time);
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
    //divide_cell<DIM>();
    /*-------------------------*/

    /*-----update d_dt-----*/
    update_dt();
    /*----------------------*/

    /*-----calculate gravity and viscosity explicitly u*,
    ---------------------------and first displacement r*-----*/
    cal_vis_expl();
    divide_cell<DIM>();
    /*-------------------------------------------------------*/

    /*-----correction by collision model-----*/
    collision();
    divide_cell<DIM>();
    /*---------------------------------------*/

    /*-----calculate n* and build & solve poisson equation P(k+1)-----*/
    cal_n();
    build_poisson();
    solve_poisson();
    press_corr(); //correct minus pressure to zero
    /*----------------------------------------------------------------*/

    /*-----calculate pressure term u' & u(k+1), second displacement r(k+1)-----*/
    cal_p_dash_impl();
    divide_cell<DIM>();
    /*-------------------------------------------------------------------------*/
    update_dt();
}

/////////////////////////////////////
///calculate explicit part vis and g
/////////////////////////////////////
void MPS_CPU_EXPL::cal_vis_expl()
{

///v2.0
    /*
    real3 _vel_tmp;
    #ifdef CPU_OMP
        #pragma omp parallel for private(_vel_tmp) schedule(dynamic, DYNAMIC_CHUNK)
    #endif
    for(int i = 0; i < i_np; i++)
    {
        if(i_type[i]) continue;
        _vel_tmp = d3_vel[i] + d_dt * (d_niu * d3_Lap_vel(i) + G);
        d3_vel[i] = _vel_tmp;
        d3_pos[i] = d3_pos[i] + d_dt * _vel_tmp;
    }
    */
    /*-----v3.0-----*/
    //calculate explicit term
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
                    for(unsigned j=i+1;j<__iend;j++)
                    {
                        register unsigned __itid = i*nthread+myid;
                        register unsigned __jtid = j*nthread+myid;
                        register real __dr = sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) );
                        register real3 __du = d3_vel[j] - d3_vel[i];

                        if( __dr < d_rzero )
                        {
                            register real3 __tmp = d_weight(d_rlap , __dr) * __du;

                            d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                            d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;
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
                        for(unsigned j=__jstart;j<__jend;j++)
                        {
                            register unsigned __itid = i*nthread+myid;
                            register unsigned __jtid = j*nthread+myid;
                            register real __dr = sqrt( (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]) );
                            register real3 __du = d3_vel[j] - d3_vel[i];

                            if( __dr < d_rzero )
                            {
                                register real3 __tmp = d_weight(d_rlap , __dr) * __du;

                                d3_omp_tmp[__itid] = d3_omp_tmp[__itid] + __tmp;
                                d3_omp_tmp[__jtid] = d3_omp_tmp[__jtid] - __tmp;
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

                __tmp = d_dt * ( d_niu * d_2bydim_over_nzerobylambda * __tmp + G );

                d3_vel[i] = d3_vel[i] + __tmp;
                d3_pos[i] = d3_pos[i] + d_dt * __tmp;
            }
        }
    }
    /*--------------*/
}
