/*
LICENCE
*/
//MPS_CPU.h
//defination of class MPS_CPU
///receive data from class MPS and functions of main loop
#ifndef MPS_CPU_H
#define MPS_CPU_H

#include "def_incl.h"
#include "MPS.h"

class MPS_CPU : public MPS
{
public:
    MPS_CPU();
    ~MPS_CPU();

protected:
    void initial();//initialize the variables for calculation
    void WriteCase();//output case

    void update_dt();//calculate new dt
    void expl_part();//calculate explicit part
    void impl_part();//calculate implicit part
    void cal_n();//calculate particle # density
    void build_poisson();//build the poisson equation
    void solve_poisson();//solve the poisson equation
    void press_corr();//correct minus pressure to zero
    void collision();//add a collision model
    void motion();//add motion to boundary
    void g_expl();//calculate expicit part by g (in CN)
    void vis_CrankNicolson();//calculate viscous using crank-nicolson

private:
    void DoCG(int __n);//CG
    template<unsigned int _dir> void CN_buildCG();//build CG for CN method
    template<unsigned int _dir> void update_pos_vel();//update pos and vel
    double d_Div_vel(int i);//divergent of velocity at particle i
    double3 d3_Grad_press(int i);//gradient of pressure at particle i
    double3 d3_Lap_vel(int i);//laplacian of velocity of particle i

protected:
    FILE* fid_log;//file to save log
    FILE* fid_out;//file to store results
    char c_log[256];//buffer for writing log
    char c_name[256];//the name of the output file
    int i_step;//count of the step
    double d_time;//time now in simulation

    int i_ini_time;//total time of initialization (in second)
    int i_cal_time;//total time of simulation (in second)
    time_t t_loop_s , t_loop_e;//start & end time of the loop

private:
    MOTION M_motion;//movement of boundary

    double* b;//b: poisson equation Ax = b
    double* x;//x: poisson equation Ax = b
    double** A;//A: poisson equation Ax = b

    double* r;//used in CG
    double* p;//used in CG
    double* Ap;//used in CG
    bool* bc;//if particle i+i_nb2 is at boundary (Dirichlet B.C.)

private:
    /*-----cell list-----*/
    int i_num_cells;//# of cells
    int* i_cell_list;//# of particles in each cell
    int* i_bd;//record boundary
    /*-------------------*/
};

#endif // MPS_CPU_H
