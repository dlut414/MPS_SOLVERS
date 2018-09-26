/*
LICENCE
*/
//MPS.cpp
//implementation of class MPS
///preparation of data & memory
#include "def_incl.h"
#include "MPS.h"

using namespace mytype;

MPS::MPS()
{
    d_mem = 0.0;

    i_index = NULL;
    i_id = NULL;
    i_type = NULL;
    d3_pos = NULL;
    d3_vel = NULL;
    d3_vel = NULL;
    d_press = NULL;
    d_n = NULL;
    i_part_in_cell = NULL;
    i_cell_list = NULL;
    i_link_cell = NULL;
    i_cell_start = NULL;
    i_cell_end = NULL;
    i_normal = NULL;
}

MPS::~MPS()
{
    delete[] d3_pos; d3_pos = NULL;
    delete[] d3_vel; d3_vel = NULL;
    //delete[] d3_vel_dash; d3_vel_dash = NULL;
    delete[] d_press; d_press = NULL;
    delete[] d_n; d_n = NULL;

    delete[] i_index;   i_index = NULL;
    delete[] i_id;  i_id = NULL;
    delete[] i_type; i_type = NULL;
    delete[] i_part_in_cell;    i_part_in_cell = NULL;
    delete[] i_cell_list;   i_cell_list = NULL;
    delete[] i_link_cell;   i_link_cell = NULL;
    delete[] i_cell_start;  i_cell_start = NULL;
    delete[] i_cell_end;    i_cell_end = NULL;
    delete[] i_normal;      i_normal = NULL;
}

////////////////////////////////////////
///add to d_mem
////////////////////////////////////////
void MPS::memAdd(const int& __size, const integer& __n)
{
    d_mem += real(__size * __n) / 1024;
}

//////////////////////////////////////
///load parameters from input.dat
//////////////////////////////////////
void MPS::loadCase()
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
    sscanf(buffer, "%f", &d_rho);
    sprintf(str, "successfully load \'rho\': %.2e\n", d_rho);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------*/

    /*-----read viscosity-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_niu);
    sprintf(str, "successfully load \'viscosity\': %.2e\n", d_niu);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------------*/

    /*-----read dp-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_dp);
    sprintf(str, "successfully load \'dp\': %.2e\n", d_dp);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read R0-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_rzero);
    sprintf(str, "successfully load \'rzero\': %.3f\n", d_rzero);
    throwScn(str);
    throwLog(_fid_log, str);
    d_rzero *= d_dp;
    /*-----------------*/

    /*-----read RLAP-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_rlap);
    sprintf(str, "successfully load \'rlap\': %.3f\n", d_rlap);
    throwScn(str);
    throwLog(_fid_log, str);
    d_rlap *= d_dp;
    /*-------------------*/

    /*-----read beta-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_beta);
    sprintf(str, "successfully load \'beta\': %.3f\n", d_beta);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-------------------*/

    /*-----read cs-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_cs);
    sprintf(str, "successfully load \'cs\': %.3f\n", d_cs);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read CFL-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_CFL);
    sprintf(str, "successfully load \'CFL\': %.3f\n", d_CFL);
    throwScn(str);
    throwLog(_fid_log, str);
    /*------------------*/

    /*-----read collision distance-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_col_dis);
    sprintf(str, "successfully load \'collision distance\': %.3f\n", d_col_dis);
    throwScn(str);
    throwLog(_fid_log, str);
    d_col_dis *= d_dp;
    /*---------------------------------*/

    /*-----read collision rate-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_col_rate);
    sprintf(str, "successfully load \'collision rate\': %.3f\n", d_col_rate);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------------------*/

    /*-----read dt_max-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_dt_max);
    sprintf(str, "successfully load \'dt_max\': %.2e\n", d_dt_max);
    throwScn(str);
    throwLog(_fid_log, str);
    /*---------------------*/

    /*-----read dt_min-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_dt_min);
    sprintf(str, "successfully load \'dt_min\': %.2e\n", d_dt_min);
    throwScn(str);
    throwLog(_fid_log, str);
    d_dt_min *= d_dt_max;
    /*---------------------*/

    /*-----read tt-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_tt);
    sprintf(str, "successfully load \'total time\': %.1f\n", d_tt);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-----------------*/

    /*-----read tout-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_tout);
    sprintf(str, "successfully load \'tout\': %.3e\n", d_tout);
    throwScn(str);
    throwLog(_fid_log, str);
    /*-------------------*/

    /*-----read max_dis-----*/
    fgets(buffer, sizeof(buffer), _fid);
    fgets(buffer, sizeof(buffer), _fid);
    sscanf(buffer, "%f", &d_max_dis);
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
void MPS::createGeo()
{
    char str[256];
    FILE* _fid_log;
    FILE* _fid = fopen("GEO.dat","r");
    if(_fid == NULL)
    {
        printf("no file 'GEO.dat' exist!\n");
        exit(3);
    }

    //the order of particles is b2 b1 fluid
    fscanf(_fid, "%d %d %d", &i_np, &i_nb2, &i_nb1);

    i_nfluid = i_np - i_nb1 - i_nb2;
    i_nexcl = 0;

    i_id            = new integer[i_np];
    i_type          = new integer[i_np];
    i_index         = new integer[i_np];
    i_normal        = new integer[i_np];
    d3_pos          = new real3[i_np];
    d3_vel          = new real3[i_np];
    //d3_vel_dash   = new real3[i_np];
    d_press         = new real[i_np];
    d_n             = new real[i_np];

    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(integer), i_np);
    memAdd(sizeof(real3),   i_np);
    memAdd(sizeof(real3),   i_np);
    memAdd(sizeof(real),    i_np);
    memAdd(sizeof(real),    i_np);

    for(integer i=0;i<i_np;i++)
    {
        i_id[i] = i;
    }
    /*-----type of each particle-----*/
    for(integer i=0;i<i_nb2;i++)            i_type[i] = 2;
    for(integer i=i_nb2;i<i_nb2+i_nb1;i++)  i_type[i] = 1;
    for(integer i=i_nb2+i_nb1;i<i_np;i++)   i_type[i] = 0;
    /*-------------------------------*/
    for(integer i=0;i<i_np;i++)
    {
        fscanf(_fid, "%f%f%f",&d3_pos[i].x,&d3_pos[i].y,&d3_pos[i].z);
    }
    for(integer i=0;i<i_np;i++)
    {
        fscanf(_fid, "%f%f%f",&d3_vel[i].x,&d3_vel[i].y,&d3_vel[i].z);
    }
    for(integer i=0;i<i_np;i++)
    {
        fscanf(_fid, "%f",&d_press[i]);
    }

    fclose(_fid);

    _fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Created geometry!\n");
    throwScn(str);
    throwLog(_fid_log, str);

    sprintf(str, "particle num: %d, bd2 num: %d, bd1 num: %d \n",
            i_np, i_nb2, i_nb1);
    throwScn(str);
    throwLog(_fid_log, str);

    fclose(_fid_log);
}

/////////////////////////////////
///calculate weight function
/////////////////////////////////
inline real MPS::d_weight(real _r0, real _r)
{
    //danger when _r == 0
    if(_r >= _r0)   return 0.0f;
    else            return (_r0 / _r - 1.0f);
}

/////////////////////////////////////
///calculate constant parameters
/////////////////////////////////////
void MPS::calOnce()
{
    char str[256];
    FILE*   _fid_log;
    integer _i_tmp;
    real    _r  = 0.0f;

    d_nzero     = 0.0f;
    d_lambda    = 0.0f;

    //nzero
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(integer i=0;i<i_np;i++)
    {
        real __tmp = 0.0f;
        /*
        for(integer j=0;j<i_np;j++)
        {
            if(j != i)
            {
                _r = (d3_pos[i].x - d3_pos[j].x) * (d3_pos[i].x - d3_pos[j].x)

                   + (d3_pos[i].y - d3_pos[j].y) * (d3_pos[i].y - d3_pos[j].y)

                   + (d3_pos[i].z - d3_pos[j].z) * (d3_pos[i].z - d3_pos[j].z);

                __tmp += d_weight(d_rzero, sqrt(_r));
            }
        }

        d_n[i] = __tmp;
        */
        ///-----note: using result of son class !!
        //searching neighbors
        //loop: surrounding cells including itself (totally 27 cells)
        //loop: from bottom to top, from back to front, from left to right
        integer __offset    = 28 * i_cell_list[i];
        integer __num       =      i_link_cell[__offset];

        for(integer dir=1;dir<=__num;dir++)
        {
            integer __cell = i_link_cell[__offset+dir];

            if(__cell >= 0 && __cell < i_num_cells)
            {
                integer __start = i_cell_start[__cell];
                integer __end   = i_cell_end[__cell];

                for(integer j=__start;j<__end;j++)
                {
                    if(j != i)
                    {
                        real3 __dr = d3_pos[j] - d3_pos[i];

                        __tmp += d_weight(d_rzero, sqrt( __dr * __dr ));
                    }
                }
            }
        }

        d_n[i] = __tmp;
        ///-------------------------------------------

        if(__tmp > d_nzero)
        {
            #ifdef CPU_OMP
                #pragma omp critical
            #endif
            {
                d_nzero = __tmp;
                _i_tmp = i;
            }
        }
    }

    //lambda
    for(integer i=0;i<i_np;i++)
    {
        if(i != _i_tmp)
        {
            _r = (d3_pos[i].x - d3_pos[_i_tmp].x) * (d3_pos[i].x - d3_pos[_i_tmp].x)

               + (d3_pos[i].y - d3_pos[_i_tmp].y) * (d3_pos[i].y - d3_pos[_i_tmp].y)

               + (d3_pos[i].z - d3_pos[_i_tmp].z) * (d3_pos[i].z - d3_pos[_i_tmp].z);

            _r = sqrt(_r);

            d_lambda += _r * _r * d_weight(d_rzero,_r);
        }
    }
    d_lambda = d_lambda / d_nzero;

    //alpha
    d_alpha = 1.0 / (d_rho * d_cs * d_cs);

    //maximum of dt
    d_dt = d_dt_max;

    //1.0 / d_*
    d_one_over_alpha            = d_rho * d_cs * d_cs;
    d_one_over_rho              = 1.0 / d_rho;
    d_one_over_nzero            = 1.0 / d_nzero;
    d_one_over_dim              = 1.0 / real(i_dim);
    d_one_over_dt               = 1.0 / d_dt;
    d_2bydim_over_nzerobylambda = (2.0 * i_dim) / (d_nzero * d_lambda);

    #ifdef CPU_OMP
        #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK_S)
    #endif
    //calculate normal
    for(integer i=0;i<i_np;i++)
    {
        if(i_type[i] == 2)
        {
            real __dis = 2 * d_rzero * d_rzero;

            for(integer j=0;j<i_np;j++)
            {
                if(i_type[j] == 1)
                {
                    real __rr = (d3_pos[j] - d3_pos[i]) * (d3_pos[j] - d3_pos[i]);

                    if(__rr < __dis)
                    {
                        __dis = __rr;
                        i_normal[i] = j;
                    }
                }
            }
        }
    }

    _fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Calculated const parameters!\n");
    throwScn(str);
    throwLog(_fid_log, str);
    fclose(_fid_log);
}

///////////////////////////////////////////////////////
///allocate memory for cell lists (called in CalOnce)
///////////////////////////////////////////////////////
void MPS::allocateCellList()
{
    char str[256];
    FILE* _fid_log;

    d_cell_size     = (d_rzero > d_rlap ? d_rzero : d_rlap);

    d_cell_left     = d_cell_right = d3_pos[0].x;
    d_cell_back     = d_cell_front = d3_pos[0].y;
    d_cell_bottom   = d_cell_top   = d3_pos[0].z;

    for(integer i=0;i<i_np;i++)
    {
        if(d3_pos[i].x < d_cell_left)   d_cell_left     = d3_pos[i].x;
        if(d3_pos[i].x > d_cell_right)  d_cell_right    = d3_pos[i].x;

        if(d3_pos[i].y < d_cell_back)   d_cell_back     = d3_pos[i].y;
        if(d3_pos[i].y > d_cell_front)  d_cell_front    = d3_pos[i].y;

        if(d3_pos[i].z < d_cell_bottom) d_cell_bottom   = d3_pos[i].z;
        if(d3_pos[i].z > d_cell_top)    d_cell_top      = d3_pos[i].z;
    }

    d_cell_left     -= d_max_dis;
    d_cell_right    += d_max_dis;
    d_cell_back     -= d_max_dis;
    d_cell_front    += d_max_dis;
    d_cell_bottom   -= d_max_dis;
    d_cell_top      += d_max_dis;

    if(i_dim == 2)
    {
        d_cell_back = d_cell_front = 0.0;
    }

    d_cell_dx = d_cell_right - d_cell_left;
    d_cell_dy = d_cell_front - d_cell_back;
    d_cell_dz = d_cell_top   - d_cell_bottom;

    i_cell_dx = integer(d_cell_dx / d_cell_size) + 1;
    i_cell_dy = integer(d_cell_dy / d_cell_size) + 1;
    i_cell_dz = integer(d_cell_dz / d_cell_size) + 1;

    if(i_cell_dx <=2 || i_cell_dz <=2) //if too thin, error in cell list
    {
        printf("cells are too thin ! -> dx: %d, dz: %d\n", i_cell_dx, i_cell_dz);
        exit(5);
    }
    if(i_cell_dy <=2 && i_dim == 3) //if too thin, error in cell list
    {
        printf("cells are too thin ! -> dy: %d\n", i_cell_dy);
        exit(5);
    }

    i_cell_sheet    = i_cell_dx     * i_cell_dy;
    i_num_cells     = i_cell_sheet  * i_cell_dz;

    i_link_cell     = new integer[28 * i_num_cells];
    i_part_in_cell  = new integer[i_num_cells];
    i_cell_start    = new integer[i_num_cells];
    i_cell_end      = new integer[i_num_cells];
    i_cell_list     = new integer[i_np];

    memAdd(sizeof(integer), 28 * i_num_cells);
    memAdd(sizeof(integer),      i_num_cells);
    memAdd(sizeof(integer),      i_num_cells);
    memAdd(sizeof(integer),      i_num_cells);
    memAdd(sizeof(integer),      i_np);

    _fid_log = fopen(LOG_NAME, "at");
    sprintf(str, "successfully Allocated cell list!\n");
    throwScn(str);
    throwLog(_fid_log, str);
    fclose(_fid_log);
}
