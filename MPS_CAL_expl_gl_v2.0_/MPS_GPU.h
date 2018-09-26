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

public:
    void Initial();             //initialize the variables for calculation
    void step   ();
    void calVisc_expl();
    void calPres_expl();

protected:
    void devInit();             //initialize gpu mem
    void devFree();             //free gpu mem
    void commHostToDevice();    //communication from host to device
    void commDeviceToHost();    //communication from device to host

    void divideCell();          //divide the domain into cells
    void sort_all();            //sort part by cell
    void makeLink();            //make link list
    void calOnce();             //calculate n, n_zero, lambda, alpha, and dt
    void update_dt();           //calculate new dt

    void calDash();             //calculate p' by implicit method
    void cal_n();               //calculate particle # density
    void buildPoisson();        //build the poisson equation
    void solvePoisson();        //solve the poisson equation
    void pressCorr();           //correct minus pressure to zero
    void collision();           //add a collision model
    void motion();              //add motion to boundary
    void DoCG(integer __n);     //CG
    void DoJacobi(integer __n); //Jacobi method
    void check();

protected:
    template<typename T>
    void Zero(T* p, integer n);             //set all values in p to zero

    real d_weight       (const real _r0, const real _r) const;
    real  d_DivVel      (const integer& i); //divergent of velocity at particle i
    real  d_LapPres     (const integer& i); //laplacian of pressure of particle i
    real3 d3_GradPres   (const integer& i); //gradient of pressure at particle i
    real3 d3_LapVel     (const integer& i); //laplacian of velocity of particle i

protected:
    FILE* fid_log;      //file to save log
    FILE* fid_out;      //file to store results
    char  c_log[256];   //buffer for writing log
    char  c_name[256];  //the name of the output file
    int   i_step;       //count of the step
    real  d_time;       //time now in simulation

    int  i_ini_time;    //total time of initialization (in second)
    real d_cal_time;    //total time of simulation (in second)
    long t_loop_s;
    long t_loop_e;      //start & end time of the loop

private:
    MOTION M_motion;//movement of boundary

protected:
    /*-----sorting funtion putting partcles into new index-----*/
    void sort_i     (integer* const __p);
    void sort_d     (real*    const __p);
    void sort_i3    (int3*    const __p);
    void sort_d3    (real3*   const __p);
    void sort_normal();
    /*---------------------------------------------------------*/

protected:
    /*-----tmp for sorting and update vel and pos-----*/
    integer* i_tmp;
    int3*    i3_tmp;
    real*    d_tmp;
    real3*   d3_tmp;
    /*------------------------------------------------*/

protected:
    /*----------------memory on device----------------*/
    integer* dev_i_id_tmp;
    integer* dev_i_type_tmp;
    integer* dev_i_cell_list_tmp;
    integer* dev_i_normal_tmp;
    real*    dev_d_press_tmp;
    real*    dev_d_n_tmp;
    real3*   dev_d3_pos_tmp;
    real3*   dev_d3_vel_tmp;

    integer* dev_i_index;
    integer* dev_i_id;
    integer* dev_i_type;
    integer* dev_i_cell_list;   //cell # of each particle
    integer* dev_i_link_cell;   //neighbor cells of cell i (27 in 3d)
    integer* dev_i_part_in_cell;//# of particles in each cell
    integer* dev_i_cell_start;  //starting index of each cell
    integer* dev_i_cell_end;    //end index of each cell
    integer* dev_i_normal;      //corresponding 1 type particle of 2 type
    real*    dev_d_press;       //pre: pressure list of particles
    real*    dev_d_n;           //n: particle number density of particles
    real3*   dev_d3_pos;        //pos: position list of particles
    real3*   dev_d3_vel;        //vel: velocity list of particles
    /*-------------------------------------------------*/

public:
    /*-----interface for opengl-----*/
    const integer&  getNp  ()const   {return i_np;   }
    const integer&  getNb2 ()const   {return i_nb2;  }
    const integer&  getNb1 ()const   {return i_nb1;  }

    integer* getId  ()const   {return i_id;   }
    integer* getType()const   {return i_type; }
    real3*   getPos ()const   {return d3_pos; }
    real3*   getVel ()const   {return d3_vel; }
    real*    getPres()const   {return d_press;}
    /*------------------------------*/

};

}
#endif // MPS_GPU_H
