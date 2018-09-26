/*
LICENCE
*/
//MPS.cpp
//implementation of class MPS
///preparation of data & memory
#include "def_incl.h"
#include "MPS.h"

MPS::MPS()
{
    i_id = NULL;
    i_type = NULL;
    d3_pos = NULL;
    d3_vel = NULL;
    d3_vel = NULL;
    d_press = NULL;
    d_n = NULL;
    i_part_in_cell = NULL;
    i_cell_list = NULL;
    i_cell_start = NULL;
    i_cell_end = NULL;
}

MPS::~MPS()
{
    delete[] d3_pos; d3_pos = NULL;
    delete[] d3_vel; d3_vel = NULL;
    //delete[] d3_vel_dash; d3_vel_dash = NULL;
    delete[] d_press; d_press = NULL;
    delete[] d_n; d_n = NULL;

    delete[] i_id;  i_id = NULL;
    delete[] i_type; i_type = NULL;
    delete[] i_part_in_cell;    i_part_in_cell = NULL;
    delete[] i_cell_list;   i_cell_list = NULL;
    delete[] i_cell_start;  i_cell_start = NULL;
    delete[] i_cell_end;    i_cell_end = NULL;
}

//////////////////////////////////////
///load parameters from input.dat
//////////////////////////////////////
void MPS::LoadCase()
{
    FILE* _fid = fopen("input.dat", "r");
    FILE* _fid_log = fopen("./out/LOG.txt", "wt");

    char buffer[256];
    char str[256];

    /*-----read dimention-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%d", &i_dim);
    sprintf(str, "successfully load \'dimention\': %d\n", i_dim);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------------*/

    /*-----read rho-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_rho);
    sprintf(str, "successfully load \'rho\': %.2e\n", d_rho);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------*/

    /*-----read viscosity-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_niu);
    sprintf(str, "successfully load \'viscosity\': %.2e\n", d_niu);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------------*/

    /*-----read dp-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_dp);
    sprintf(str, "successfully load \'dp\': %.2e\n", d_dp);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read R0-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_rzero);
    sprintf(str, "successfully load \'rzero\': %.3f\n", d_rzero);
    throwScn(str);
    throwLog(_fid_log, str);
    d_rzero *= d_dp;
    /*-----------------*/

    /*-----read RLAP-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_rlap);
    sprintf(str, "successfully load \'rlap\': %.3f\n", d_rlap);
    throwScn(str);
    throwLog(_fid_log, str);
    d_rlap *= d_dp;
    /*-------------------*/

    /*-----read beta-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_beta);
    sprintf(str, "successfully load \'beta\': %.3f\n", d_beta);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-------------------*/

    /*-----read cs-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_cs);
    sprintf(str, "successfully load \'cs\': %.3f\n", d_cs);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read CFL-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_CFL);
    sprintf(str, "successfully load \'CFL\': %.3f\n", d_CFL);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------*/

    /*-----read collision distance-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_col_dis);
    sprintf(str, "successfully load \'collision distance\': %.3f\n", d_col_dis);
    throwScn(str);
    throwLog(_fid_log, str);
    /*---------------------------------*/

    /*-----read collision rate-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_col_rate);
    sprintf(str, "successfully load \'collision rate\': %.3f\n", d_col_rate);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------------------*/

    /*-----read dt_max-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_dt_max);
    sprintf(str, "successfully load \'dt_max\': %.2e\n", d_dt_max);
    throwScn(str);
    throwLog(_fid_log, str);
    /*---------------------*/

    /*-----read dt_min-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_dt_min);
    sprintf(str, "successfully load \'dt_min\': %.2e\n", d_dt_min);
    throwScn(str);
    throwLog(_fid_log, str);
    d_dt_min *= d_dt_max;
    /*---------------------*/

    /*-----read tt-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_tt);
    sprintf(str, "successfully load \'total time\': %.1f\n", d_tt);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read tout-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_tout);
    sprintf(str, "successfully load \'tout\': %.3e\n", d_tout);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-------------------*/

    /*-----read max_dis-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%lf", &d_max_dis);
    sprintf(str, "successfully load \'max_dis\': %.3e\n", d_max_dis);
    throwScn(str);
    throwLog(_fid_log, str);
    /*----------------------*/

    fclose(_fid);
    fclose(_fid_log);
}

////////////////////////////////////////////////////////
///load particle data from GEO.dat and allocate memory
////////////////////////////////////////////////////////
void MPS::CreateGeo()
{
    char str[256];
    FILE* fid_log;
    FILE* fid = fopen("GEO.dat","r");
    if(fid == NULL)
    {
        printf("no file 'GEO.dat' exist!\n");
        exit(3);
    }

    //the order of particles is b2 b1 fluid
    fscanf(fid, "%d %d %d", &i_np, &i_nb2, &i_nb1);

    i_nfluid = i_np - i_nb1 - i_nb2;

    i_id = new int[i_np];
    i_type = new int[i_np];
    d3_pos = new real3[i_np];
    d3_vel = new real3[i_np];
    //d3_vel_dash = new real3[i_np];
    d_press = new real[i_np];
    d_n = new real[i_np];

    for(int i=0;i<i_np;i++)
    {
        i_id[i] = i;
    }
    /*-----type of each particle-----*/
    for(int i=0;i<i_nb2;i++) i_type[i] = 2;
    for(int i=i_nb2;i<i_nb2+i_nb1;i++) i_type[i] = 1;
    for(int i=i_nb2+i_nb1;i<i_np;i++) i_type[i] = 0;
    /*-------------------------------*/
    for(int i=0;i<i_np;i++)
    {
        fscanf(fid, "%lf%lf%lf",&d3_pos[i].x,&d3_pos[i].y,&d3_pos[i].z);
    }
    for(int i=0;i<i_np;i++)
    {
        fscanf(fid, "%lf%lf%lf",&d3_vel[i].x,&d3_vel[i].y,&d3_vel[i].z);
    }
    for(int i=0;i<i_np;i++)
    {
        fscanf(fid, "%lf",&d_press[i]);
    }

    fclose(fid);

    fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Created geometry!\n");
    throwScn(str);
    throwLog(fid_log, str);
    fclose(fid_log);
}

/////////////////////////////////
///calculate weight function
/////////////////////////////////
inline real MPS::d_weight(real _r0, real _r)
{
    //danger when _r == 0
    if(_r >= _r0) return 0.0;
    else return (_r0 / _r - 1.0);
}

/////////////////////////////////////
///calculate constant parameters
/////////////////////////////////////
void MPS::CalOnce()
{
    char str[256];
    FILE* fid_log;
    int _tmp;
    real _r = 0.0;

    d_nzero = 0.0;
    d_lambda = 0.0;

    //nzero
    for(int i=0;i<i_np;i++)
    {
        d_n[i] = 0.0;
        for(int j=0;j<i_np;j++)
        {
            if(j == i) continue;
            _r = (d3_pos[i].x - d3_pos[j].x) * (d3_pos[i].x - d3_pos[j].x)

               + (d3_pos[i].y - d3_pos[j].y) * (d3_pos[i].y - d3_pos[j].y)

               + (d3_pos[i].z - d3_pos[j].z) * (d3_pos[i].z - d3_pos[j].z);

            _r = sqrt(_r);

            d_n[i] += d_weight(d_rzero,_r);
        }
        if(d_n[i] > d_nzero) {d_nzero = d_n[i];_tmp = i;}
    }

    //lambda
    for(int i=0;i<i_np;i++)
    {
        if(i == _tmp) continue;
        _r = (d3_pos[i].x - d3_pos[_tmp].x) * (d3_pos[i].x - d3_pos[_tmp].x)

           + (d3_pos[i].y - d3_pos[_tmp].y) * (d3_pos[i].y - d3_pos[_tmp].y)

           + (d3_pos[i].z - d3_pos[_tmp].z) * (d3_pos[i].z - d3_pos[_tmp].z);

        _r = sqrt(_r);
        d_lambda += _r * _r * d_weight(d_rzero,_r);
    }
    d_lambda = d_lambda / d_nzero;

    //alpha
    d_alpha = 1.0 / (d_rho * d_cs * d_cs);

    //maximum of dt
    d_dt = d_dt_max;

    //1.0 / d_*
    d_one_over_rho = 1.0 / d_rho;
    d_one_over_nzero = 1.0 / d_nzero;
    d_one_over_dim = 1.0 / real(i_dim);
    d_one_over_dt = 1.0 / d_dt;
    d_one_over_nzerobylambda = 1.0 / (d_nzero * d_lambda);

    fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Calculated const parameters!\n");
    throwScn(str);
    throwLog(fid_log, str);
    fclose(fid_log);
}
