/*
LICENCE
*/
//MPS.h
//defination of class MPS
///preparation of data & memory
#ifndef MPS_H
#define MPS_H

#include "def_incl.h"
#include "OBJECT.h"

namespace mytype
{

class MPS : public OBJECT
{
public:
    MPS();
    ~MPS();

public:
    void loadCase();                                     //load case by class input
    void createGeo();                                    //create geometry of the case
    void allocateCellList();                             //allocate memory for cell_list

public:
    /*-----geometry of cells-----*/
    GEOMETRY geo;
    integer* i_cell_list;       //cell # of each particle
    integer* i_link_cell;       //neighbor cells of cell i (27 in 3d)
    integer* i_part_in_cell;    //# of particles in each cell
    /*-----new-----*/
    integer* i_bd_in_cell;  //boundary particles in each cell
    integer* i_cellInFluid; // a texture tells whether a cell is in the fluid
    /*-------------*/
    integer* i_cell_start;      //starting index of each cell
    integer* i_cell_end;        //end index of each cell
    /*---------------------------*/

protected:
    /*-----allocate memory before the first step-----*/
    integer* i_index;   //new index of each particle
    integer* i_id;      //id of each particle (never change)
    integer* i_type;    //type of each particle: from outside to inside 2, 1, 0; 3 is excluded particles
    real*    d_press;   //pre: pressure list of particles
    real*    d_n;       //n: particle number density of particles
    real3*   d3_pos;    //pos: position list of particles
    real3*   d3_vel;    //vel: velocity list of particles
    //real3* d3_vel_dash;//vel_dash: u', implicit vel ter

    integer* i_normal;//corresponding 1 type particle of 2 type
    /*-----------------------------------------------*/

    /*-----calculated before the first step-----*/
    real d_nzero;   //nzero: n_zero of particles
    real d_lambda;  //lambda: obtained once initially
    real d_alpha;   //alpha: calculated by cs (for compressible flow)
    real d_dt;      //time step

    real d_one_over_alpha;              //1.0 / d_alpha
    real d_one_over_rho;                //1.0 / d_rho
    real d_one_over_nzero;              //1.0 / d_nzero
    real d_one_over_dim;                //1.0 / double(i_dim)
    real d_one_over_dt;                 //1.0 / d_dt
    real d_2bydim_over_nzerobylambda;   //(2 * dim) / (d_nzero*d_lambda)
    /*------------------------------------------*/

    /*-----input data from GEO.dat-----*/
    ///data arrangement is i_nb2 -> i_nb1 -> i_nfluid
    integer i_np;       //np: # of particles
    integer i_nb2;      //nb2: # of second kind boundary particles (out-layer)
    integer i_nb1;      //nb1: # of first kind boundary particles (near fluid)
    integer i_nfluid;   //nfluid: # of fluid particles
    integer i_nexcl;    //nexcl: # of excluded particles
    /*---------------------------------*/

    /*-----input data from input.dat-----*/
    int i_dim;//dim: dimention

    real d_rho;         //Rho0: density
    real d_niu;         //miu: viscosity
    real d_dp;          //dp: initial particle distance
    real d_rzero;       //R0: influence domain
    real d_rlap;        //R0 for Laplacian
    real d_beta;        //beta: n* < beta * nzero at boundary
    real d_cs;          //cs: numerical speed of sound
    real d_CFL;         //CFL: dt = dp * CFL / vel_max
    real d_col_dis;     //collision distance
    real d_col_rate;    //collision rate

    real d_dt_max;      //maximum of dt
    real d_dt_min;      //minimum of dt = d_dt_max * d_dt_min
    real d_tt;          //total time
    real d_tout;        //output time step

    real d_max_dis;     //maximum displacement of the geometry, for cell creation
    /*-----------------------------------*/

protected:
    real d_mem;             //memory usage
    void memAdd(const int& __size, const integer& __n);

};

}
#endif // MPS_H
