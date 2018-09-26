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
    void Initial();//initialize the variables for calculation
    void WriteCase();//output case

    void update_dt();//calculate new dt

    void impl_part();//calculate implicit part
    void cal_n();//calculate particle # density
    void build_poisson();//build the poisson equation
    void solve_poisson();//solve the poisson equation
    void press_corr();//correct minus pressure to zero
    void collision();//add a collision model
    void motion();//add motion to boundary
    void DoCG(int __n);//CG

protected:
    real d_Div_vel(int i);//divergent of velocity at particle i
    real3 d3_Grad_press(int i);//gradient of pressure at particle i
    real3 d3_Lap_vel(int i);//laplacian of velocity of particle i

protected:
    FILE* fid_log;//file to save log
    FILE* fid_out;//file to store results
    char c_log[256];//buffer for writing log
    char c_name[256];//the name of the output file
    int i_step;//count of the step
    real d_time;//time now in simulation

    int i_ini_time;//total time of initialization (in second)
    int i_cal_time;//total time of simulation (in second)
    time_t t_loop_s , t_loop_e;//start & end time of the loop

protected:
    real* b;//b: poisson equation Ax = b
    real* x;//x: poisson equation Ax = b
    real** A;//A: poisson equation Ax = b

    real* r;//used in CG
    real* p;//used in CG
    real* Ap;//used in CG
    bool* bc;//if particle i+i_nb2 is at boundary (Dirichlet B.C.)

private:
    MOTION M_motion;//movement of boundary

};

#endif // MPS_CPU_H
