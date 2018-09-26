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

class MPS : public OBJECT
{
public:
    MPS();
    ~MPS();

public:
    void LoadCase();//load case by class input
    void CreateGeo();//create geometry of the case

    real d_weight(real _r0 , real _r);//weight function
    void CalOnce();//calculate n, n_zero, lambda, alpha, and dt
    void allocate_cell_list();//allocate memory for cell_list

protected:
    /*-----allocate memory before the first step-----*/
    int* i_index;//new index of each particle
    int* i_id;//id of each particle (never change)
    int* i_type;//type of each particle: from outside to inside 2, 1, 0
    real* d_press;//pre: pressure list of particles
    real* d_n;//n: particle number density of particles
    real3* d3_pos;//pos: position list of particles
    real3* d3_vel;//vel: velocity list of particles
    //real3* d3_vel_dash;//vel_dash: u', implicit vel term

    int* i_cell_list;//cell # of each particle
    int** i_link_cell;//neighbor cells of cell i (27 in 3d)
    int* i_part_in_cell;//# of particles in each cell
    int* i_cell_start;//starting index of each cell
    int* i_cell_end;//end index of each cell
    /*-----------------------------------------------*/

    /*-----calculated before the first step-----*/
    real d_nzero;//nzero: n_zero of particles
    real d_lambda;//lambda: obtained once initially
    real d_alpha;//alpha: calculated by cs (for compressible flow)
    real d_dt;//time step

    real d_one_over_rho;//1.0 / d_rho
    real d_one_over_nzero;//1.0 / d_nzero
    real d_one_over_dim;//1.0 / double(i_dim)
    real d_one_over_dt;//1.0 / d_dt
    real d_2bydim_over_nzerobylambda;//1.0 / (d_nzero*d_lambda)
    /*------------------------------------------*/

    /*-----input data from GEO.dat-----*/
    ///data arrangement is i_nb2 -> i_nb1 -> i_nfluid
    int i_np;//np: # of particles
    int i_nb2;//nb2: # of second kind boundary particles (out-layer)
    int i_nb1;//nb1: # of first kind boundary particles (near fluid)
    int i_nfluid;//nfluid: # of fluid particles
    /*---------------------------------*/

    /*-----input data from input.dat-----*/
    int i_dim;//dim: dimention

    real d_rho;//Rho0: density
    real d_niu;//miu: viscosity
    real d_dp;//dp: initial particle distance
    real d_rzero;//R0: influence domain
    real d_rlap;//R0 for Laplacian
    real d_beta;//beta: n* < beta * nzero at boundary
    real d_cs;//cs: numerical speed of sound
    real d_CFL;//CFL: dt = dp * CFL / vel_max
    real d_col_dis;//collision distance
    real d_col_rate;//collision rate

    real d_dt_max;//maximum of dt
    real d_dt_min;//minimum of dt = d_dt_max * d_dt_min
    real d_tt;//total time
    real d_tout;//output time step

    real d_max_dis;//maximum displacement of the geometry, for cell creation
    /*-----------------------------------*/

    /*-----geometry of cells-----*/
    int i_num_cells;//number of cells, cal before the first step
    int i_cell_dx, i_cell_dy, i_cell_dz;//# of cells in each dim, cal before the first step
    int i_cell_sheet;//i_cell_sheet = i_cell_dx * i_cell_dy, cal before the first step
    real d_cell_size;//size of the cell, cal before the first step
    real d_cell_left, d_cell_right, d_cell_back, d_cell_front, d_cell_bottom, d_cell_top;//boundary in each dim (6 facets), cal before the first step
    real d_cell_dx, d_cell_dy, d_cell_dz;//size in each dim, cal before the first step
    /*---------------------------*/
};

#endif // MPS_H
