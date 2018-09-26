/*
LICENCE
*/
//MPS_GPU.h
//defination of class MPS_GPU
///receive data from class MPS and functions of main loop
#ifndef MPS_GPU_H
#define MPS_GPU_H

#include "def_incl.h"
#include "MPS.h"

namespace mytype
{

class MPS_GPU : public MPS
{
public:
    MPS_GPU();
    ~MPS_GPU();

protected:
    void Initial();//initialize the variables for calculation
    void writeCase();//output case

    template<char _dim> void divideCell();//divide the domain into cells
    void update_dt();//calculate new dt

    void cal_Pdash_impl();//calculate p' by implicit method
    void cal_n();//calculate particle # density
    void buildPoisson();//build the poisson equation
    void solvePoisson();//solve the poisson equation
    void pressCorr();//correct minus pressure to zero
    void collision();//add a collision model
    void motion();//add motion to boundary
    void DoCG(integer __n);//CG
    void DoJacobi(integer __n);//Jacobi method

protected:
    template<typename T> void Zero(T* p, integer n);//set all values in p to zero

    real d_DivVel(const integer& i);//divergent of velocity at particle i
    real3 d3_GradPress(const integer& i);//gradient of pressure at particle i
    real3 d3_LapVel(const integer& i);//laplacian of velocity of particle i

protected:
    FILE* fid_log;//file to save log
    FILE* fid_out;//file to store results
    char c_log[256];//buffer for writing log
    char c_name[256];//the name of the output file
    int i_step;//count of the step
    real d_time;//time now in simulation

    int i_ini_time;//total time of initialization (in second)
    real d_cal_time;//total time of simulation (in second)
    long t_loop_s , t_loop_e;//start & end time of the loop

protected:
    real* b;//b: poisson equation Ax = b
    real* x;//x: poisson equation Ax = b
    real* A;//A: poisson equation Ax = b

    bool* bc;//if particle i+i_nb2 is at boundary (Dirichlet B.C.)

private:
    MOTION M_motion;//movement of boundary

protected:
    /*-----sorting funtion putting partcles into new index-----*/
    void sort_i(integer* const __p);
    void sort_d(real* const __p);
    void sort_i3(int3* const __p);
    void sort_d3(real3* const __p);
    /*---------------------------------------------------------*/
private:
    /*-----tmp for sorting-----*/
    integer* i_tmp;
    int3* i3_tmp;
    real* d_tmp;
    real3* d3_tmp;
    /*-------------------------*/
};

}
#endif // MPS_GPU_H
