/*
LICENCE
*/
//MPS_CPU.cpp
//implementation of class MPS_CPU
///receive data from class MPS and functions of main loop
#include "def_incl.h"
#include "MPS_CPU.h"

MPS_CPU::MPS_CPU()
{
    fid_log = NULL;
    fid_out = NULL;
    b = NULL;
    x = NULL;
    A = NULL;
    r = NULL;
    p = NULL;
    Ap = NULL;
    bc = NULL;

    i_ini_time = 0;
    i_cal_time = 0;
    t_loop_s = 0;
    t_loop_e = 0;
}

MPS_CPU::~MPS_CPU()
{
    fid_log = NULL;
    fid_out = NULL;
    delete[] b;
    delete[] x;
    delete[] A;
    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] bc;
}

///////////////////////////////
///initialization
///////////////////////////////
void MPS_CPU::initial()
{
    /*-----time of initialization-----*/
    t_loop_s = time(NULL);
    /*--------------------------------*/

    /*-----initialization of parent class-----*/
    LoadCase();//must at first
    CreateGeo();
    CalOnce();
    /*----------------------------------------*/

    /*-----initialization of variables in MPS_CPU-----*/
    strcpy(c_name , "0000.out");
    i_step = 0;
    d_time = 0.0;

    i_ini_time = 0;
    i_cal_time = 0;

    ///-----initialization for b=Ax
    b = new double[i_np];
    x = new double[i_np];
    A = new double*[i_np];

    r = new double[i_np];
    p = new double[i_np];
    Ap = new double[i_np];
    bc = new bool[i_np];

    for(int i=0; i < i_np; i++)
        A[i] = new double[i_np];
    /*------------------------------------------------*/

    /*-----end of initialization-----*/
    t_loop_e = time(NULL);
    i_ini_time = difftime(t_loop_e , t_loop_s);
    /*-------------------------------*/
}

/////////////////////////////////////////
///out put case
/////////////////////////////////////////
void MPS_CPU::WriteCase()
{
    fid_out = fopen(c_name , "wt");

    for(int i = 0; i < i_np; i++)
    {
        fprintf(fid_out , "%lf %lf %lf %lf %lf %lf %lf %d " ,
                d3_pos[i].x , d3_pos[i].y , d3_pos[i].z ,
                d3_vel[i].x , d3_vel[i].y , d3_vel[i].z ,
                d_press[i], i_id[i]);
    }

    fclose(fid_out);
}

/////////////////////////////
///gradient of press at i
/////////////////////////////
double3 MPS_CPU::d3_Grad_press(int i)
{
    double3 _ret = {0,0,0};
    double3 _dr = {0,0,0};
    double _r2 = 0.0;
    double _hat_p = d_press[i];

    if(i < i_nb2 + i_nb1) return _ret;

    for(int j = i_nb2; j < i_np; j++)
    {
        _dr = d3_pos[j] - d3_pos[i];
        _r2 = _dr * _dr;
        if( d_press[j] < _hat_p && _r2 <= (d_rzero*d_rzero) ) _hat_p = d_press[j];
    }

    for(int j = i_nb2; j < i_np; j++)
    {
        if(j == i) continue;

        _dr = d3_pos[j] - d3_pos[i];
        _r2 = _dr * _dr;
        _ret = _ret + (d_press[j] - _hat_p) / _r2 * d_weight(d_rzero,sqrt(_r2)) * _dr;
    }
    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}

///////////////////////////////////
///divergence of velocity at i
///////////////////////////////////
double MPS_CPU::d_Div_vel(int i)
{
    double _ret = 0.0;
    double3 _dr = {0,0,0};
    double3 _du = {0,0,0};
    double _r2 = 0.0;

    for(int j = i_nb2; j<i_np; j++)
    {
        if(j == i) continue;
        _dr = d3_pos[j] - d3_pos[i];
        _du = d3_vel[j] - d3_vel[i];
        _r2 = _dr * _dr;

        _ret = _ret + d_weight(d_rzero,sqrt(_r2)) / _r2 * (_du * _dr);
    }
    _ret = i_dim * d_one_over_nzero * _ret;

    return _ret;
}

/////////////////////////////////////////
///Laplacian of velocity at i
/////////////////////////////////////////
double3 MPS_CPU::d3_Lap_vel(int i)
{
    //without cell list

    double3 _ret = {0.0,0.0,0.0};
    double3 _dr;
    double3 _du;
    for(int j=0;j<i_np;j++)
    {
        if(j == i) continue;
        _dr = d3_pos[j] - d3_pos[i];
        _du = d3_vel[j] - d3_vel[i];

        _ret = _ret + d_weight(d_rlap , sqrt( _dr * _dr )) * _du;
    }
    _ret = (2 * i_dim * d_one_over_nzerobylambda) * _ret;

    return _ret;
}

//////////////////////////////
///update dt
//////////////////////////////
void MPS_CPU::update_dt()
{
    double _d_dt_tmp;

    d_dt = d_dt_max;
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _d_dt_tmp = d_CFL * d_dp / sqrt(d3_vel[i] * d3_vel[i]); //note: d3_vel = 0
        if(_d_dt_tmp < d_dt) d_dt = _d_dt_tmp;
    }
    if(d_dt < d_dt_min)
    {
        sprintf(c_log, "error: dt -> %e, too small \n", d_dt);
        throwScn(c_log);
        throwLog(fid_log,c_log);
        exit(1);
    }
}

//////////////////////////////
///calculate explicit part
//////////////////////////////
void MPS_CPU::expl_part()
{
    //without cell_list

    double3 _vel_tmp;
#ifdef CPU_OMP
    #pragma omp parallel for private(_vel_tmp)
#endif
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _vel_tmp = d3_vel[i] + d_dt * (d_niu * d3_Lap_vel(i) + G);
        d3_vel[i] = _vel_tmp;
        d3_pos[i] = d3_pos[i] + d_dt * _vel_tmp;
    }
}

//////////////////////////////
///calculate implicit part
//////////////////////////////
void MPS_CPU::impl_part()
{
    double3 _d3_vel_tmp;
#ifdef CPU_OMP
    #pragma omp parallel for private(_d3_vel_tmp)
#endif
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _d3_vel_tmp = - d_dt * d_one_over_rho * d3_Grad_press(i);
        d3_vel[i] = d3_vel[i] + _d3_vel_tmp; //update velocity
        d3_pos[i] = d3_pos[i] + d_dt * _d3_vel_tmp; //update position
    }
}

//////////////////////////////
///particle number density
//////////////////////////////
void MPS_CPU::cal_n()
{
    double _n;
    double3 _dr;
#ifdef CPU_OMP
    #pragma omp parallel for private(_dr,_n) schedule(static)
#endif
    for(int i = i_nb2; i < i_np; i++)
    {
        _n = 0.0;
        for(int j = 0; j < i_np; j++)
        {
            if(j == i) continue;
            _dr = d3_pos[j] - d3_pos[i];
            _n += d_weight(d_rzero,sqrt(_dr*_dr));
        }
        d_n[i] = _n;
    }
}

//////////////////////////////////
///make b[] & A[]
//////////////////////////////////
void MPS_CPU::build_poisson()
{
    double3 _dr;
    double _rr;
    double _weight;
    double _Aii = d_rho * d_lambda * d_alpha * d_nzero / (2 * i_dim * d_dt * d_dt);//compressive flow (minused by Aii)

    /*-----calculate b & initialize A[i][j]=0, x=press, Ap=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static)
    #endif
    for(int i=0;i<i_nb1+i_nfluid;i++) //size of A,b,x is i_nb1+i_nfluid
    {
        b[i] = d_lambda * d_rho *
               (0.5 * d_one_over_dim * d_one_over_dt * d_one_over_dt) *
               (d_nzero - d_n[i+i_nb2]);
        x[i] = d_press[i+i_nb2]; //size of A,b,x is i_nb1+i_nfluid
        Ap[i] = 0.0;
        p[i] = 0.0;
        r[i] = 0.0;
        bc[i] = false;

        if(d_n[i+i_nb2] < d_beta * d_nzero)
        {
            x[i] = 0.0;    //Dirichlet B.C.
            bc[i] = true; //particle i+i_nb2 is at boundary
        }

        for(int j=0;j<i_nb1+i_nfluid;j++) A[i][j] = 0.0;
    }
    /*-----------------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    #ifdef OMP_CPU
        #pragma omp parallel for private(_dr,_rr,_weight)
    #endif
    for(int i=0;i<i_nb1+i_nfluid;i++)
    {
        //Dirichlet B.C. modify A[][] & b[]
        if(bc[i])
        {
            A[i][i] = 1.0;
            b[i] = 0.0;
            continue;
        }

        for(int j=0;j<i_nb1+i_nfluid;j++)
        {
            if(j == i) continue;
            _dr = d3_pos[j+i_nb2] - d3_pos[i+i_nb2];
            _rr = sqrt( _dr * _dr );
            _weight = d_weight(d_rlap,_rr);

            A[i][j] += _weight;
            A[i][i] -= _weight;

            //Dirichlet B.C. modify A[][] & b[]
            if(bc[j])
            {
                A[i][j] = 0.0;
            }
        }
        A[i][i] -= _Aii;
    }

    ///symmetric copy
    //for(int i=0;i<i_nb1+i_nfluid;i++)
    //{
    //    for(int j=i+1;j<i_nb1+i_nfluid;j++)
    //    {
    //        A[j][i] = A[i][j];
    //    }
    //}
    /*---------------------------------------*/
}

//////////////////////////////////
///solve possion equation b=Ax
//////////////////////////////////
void MPS_CPU::solve_poisson()
{
    DoCG(i_nb1+i_nfluid);

    /*-----update pressure-----*/
#ifdef CPU_OMP
        #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<i_nb1+i_nfluid;i++)
    {
        d_press[i+i_nb2] = x[i];
    }
    /*-------------------------*/
}

//////////////////////////////////////
///correct minus pressure to zero
//////////////////////////////////////
void MPS_CPU::press_corr()
{
#ifdef CPU_OMP
    #pragma omp parallel for
#endif
    for(int i=i_nb2;i<i_np;i++)
    {
        if(d_press[i] < 0.0) d_press[i] = 0.0;
    }
}

//////////////////////////////////
///collision model
//////////////////////////////////
void MPS_CPU::collision()
{
    /*-----description of collision model-----
    vg = (rhoi*vi + rhoj*vj) / (2 * rhoij_average);
    mvr = rhoi * (vi - vg);
    vabs = (mvr * rji) / abs(rij);
    if vabs < 0, then do nothing
    else then,
        mvm = vrat * vabs * rji / abs(rij);
        vi -= mvm / rhoi;
        xi -= dt * mvm / rhoi;

        vj += mvm / rhoj;
        xj += dt * mvm / rhoj;
    ----------------------------------------*/
    int _ncol = 0;
    double3 _dr; //rji
    double3 _v_crt;
    double _rr;
    double _one_by_rr;
    double _vabs;

    for(int i=i_nb2;i<i_np;i++)
    {
        for(int j=i+1;j<i_np;j++)
        {
            //if(j == i) continue;
            _dr = d3_pos[j] - d3_pos[i];
            _rr = sqrt(_dr * _dr);
            _one_by_rr = 1.0 / _rr;

            if( _rr > d_dp * d_col_dis ) continue;
            _vabs = 0.5f * (d3_vel[i] - d3_vel[j]) * _dr * _one_by_rr;

            if(_vabs <= 0.0) continue;
            _v_crt = d_col_rate * _vabs * _one_by_rr * _dr;

            if(j >= i_nb2+i_nb1)
            {
                d3_vel[j] = d3_vel[j] + _v_crt;
                d3_pos[j] = d3_pos[j] + d_dt * _v_crt;
            }
            if(i >= i_nb2+i_nb1)
            {
                d3_vel[i] = d3_vel[i] - _v_crt;
                d3_pos[i] = d3_pos[i] - d_dt * _v_crt;
            }

            _ncol++;
            sprintf(c_log, "collision count: %4d,    distance: %8e\n", _ncol, _rr);
            throwLog(fid_log, c_log);
            throwScn(c_log);
        }
    }
}

/////////////////////////////
///add motion of boundary
/////////////////////////////
void MPS_CPU::motion()
{
    M_motion.domotion(d3_pos, d3_vel, i_nb2+i_nb1);
}

//////////////////////////////
///viscous by crank-nicoloson
//////////////////////////////
void MPS_CPU::vis_CrankNicolson()
{
    //x direction
    CN_buildCG<1>();
    DoCG(i_np);
    update_pos_vel<1>();

    //y direction
    CN_buildCG<2>();
    DoCG(i_np);
    update_pos_vel<2>();

    //z direction
    CN_buildCG<3>();
    DoCG(i_np);
    update_pos_vel<3>();
}

//////////////////////////////
///build A, x, b for DoCG
//////////////////////////////
template<unsigned int _dir> void MPS_CPU::CN_buildCG()
{
    double3 _dr;
    double _rr;
    double _weight;
    const double _ONE_OVER_R = (d_lambda * d_nzero) / (i_dim * d_niu * d_dt);

    /*-----initialize A[i][j]=0, x=d3_vel.x, Ap=0, b=0-----*/
    #ifdef CPU_OMP
        #pragma omp parallel for schedule(static)
    #endif
    for(int i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        b[i] = 0.0;
        switch(_dir)
        {
            case 1: x[i] = d3_vel[i].x; break;
            case 2: x[i] = d3_vel[i].y; break;
            case 3: x[i] = d3_vel[i].z; break;
        }
        Ap[i] = 0.0;
        p[i] = 0.0;
        r[i] = 0.0;
        bc[i] = false;

        if(i < i_nb2 + i_nb1)
        {
            bc[i] = true; //particle i+i_nb2 is at boundary
        }

        for(int j=0;j<i_np;j++) A[i][j] = 0.0;
    }
    /*-----------------------------------------------------*/

    /*-----calculate A[i][j] (symmetric)-----*/
    #ifdef OMP_CPU
        #pragma omp parallel for private(_dr,_rr,_weight)
    #endif
    for(int i=0;i<i_np;i++) //size of A,b,x is i_np
    {
        //Dirichlet B.C. modify A[][] & b[]
        if(bc[i])
        {
            A[i][i] = 1.0;
            switch(_dir)
            {
                case 1: b[i] = d3_vel[i].x; break;
                case 2: b[i] = d3_vel[i].y; break;
                case 3: b[i] = d3_vel[i].z; break;
            }
            continue;
        }

        for(int j=0;j<i_np;j++)
        {
            if(j == i) continue;
            _dr = d3_pos[j] - d3_pos[i];
            _rr = sqrt( _dr * _dr );
            _weight = d_weight(d_rlap , _rr);

            A[i][j] += _weight;
            A[i][i] -= _weight;

            switch(_dir)
            {
                case 1: b[i] -= (d3_vel[j].x - d3_vel[i].x) * _weight; break;
                case 2: b[i] -= (d3_vel[j].y - d3_vel[i].y) * _weight; break;
                case 3: b[i] -= (d3_vel[j].z - d3_vel[i].z) * _weight; break;
            }

            if(bc[j])
            {
                A[i][j] = 0.0;
                switch(_dir)
                {
                    case 1: b[i] -= _weight * d3_vel[j].x; break;
                    case 2: b[i] -= _weight * d3_vel[j].y; break;
                    case 3: b[i] -= _weight * d3_vel[j].z; break;
                }
            }
        }

        A[i][i] -= _ONE_OVER_R;

        switch(_dir)
        {
            case 1: b[i] -= _ONE_OVER_R * d3_vel[i].x; break;
            case 2: b[i] -= _ONE_OVER_R * d3_vel[i].y; break;
            case 3: b[i] -= _ONE_OVER_R * d3_vel[i].z; break;
        }
    }

    ///symmetric copy
    //for(int i=0;i<i_nb1+i_nfluid;i++)
    //{
    //    for(int j=i+1;j<i_nb1+i_nfluid;j++)
    //    {
    //        A[j][i] = A[i][j];
    //    }
    //}
    /*---------------------------------------*/
}

//////////////////////////////
///solve 'Ax = b' by CG
//////////////////////////////
void MPS_CPU::DoCG(int __n)
{

    double _alpha = 0.0;//for CG
    double _rrnew = 0.0;//for CG
    double _rrold = 0.0;//for CG
    double _rn_over_ro;

    /*-----main part of Conjugate Gradient-----*/
    //p0 = r0 = b - Ax0
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<__n;i++)
    {
        r[i] = b[i] - (double)std::inner_product(x,x+__n,A[i],0.0);
        p[i] = r[i];
    }
    _rrold = (double)std::inner_product(r,r+__n,r,0.0);

    //repeat
    while(_rrold > EPS_BY_EPS)
    {
        //printf("CONVERGENCE -> RESIDUAL: %.2e\n",_rrnew);
        if(_rrnew > RMAX) {printf("DIVERGENCE -> RESIDUAL: %.2e\n",_rrnew);exit(0);}
        #ifdef CPU_OMP
                #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            Ap[i] = (double)std::inner_product(p,p+__n,A[i],0.0);
        }
        _alpha = _rrold / (double)std::inner_product(Ap,Ap+__n,p,0.0);

        #ifdef CPU_OMP
                #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            x[i] = x[i] + _alpha * p[i];
            r[i] = r[i] - _alpha * Ap[i];
        }

        _rrnew = (double)std::inner_product(r,r+__n,r,0.0);


        _rn_over_ro = _rrnew / _rrold;

        #ifdef CPU_OMP
                #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            p[i] = r[i] + _rn_over_ro * p[i];
        }
        _rrold = _rrnew;
    }
    /*-----------------------------------------*/
}

//////////////////////////////
///update pos and vel
//////////////////////////////
template<unsigned int _dir> void MPS_CPU::update_pos_vel()
{
    switch(_dir)
    {
        case 1:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].x = x[i];
                d3_pos[i].x += d_dt * x[i];
            }
            break;
        }
        case 2:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].y = x[i];
                d3_pos[i].y += d_dt * x[i];
            }
            break;
        }
        case 3:
        {
            #ifdef CPU_OMP
                #pragma omp parallel for schedule(static)
            #endif
            for(int i = i_nb2 + i_nb1; i < i_np; i++)
            {
                d3_vel[i].z = x[i];
                d3_pos[i].z += d_dt * x[i];
            }
            break;
        }
    }
}

/////////////////////////////////////////
///calculate expicit part by g (in CN)
/////////////////////////////////////////
void MPS_CPU::g_expl()
{
    double3 _vel_tmp;
    #ifdef CPU_OMP
        #pragma omp parallel for private(_vel_tmp)
    #endif
    for(int i = i_nb2 + i_nb1; i < i_np; i++)
    {
        _vel_tmp = d3_vel[i] + d_dt * G;
        d3_vel[i] = _vel_tmp;
        d3_pos[i] = d3_pos[i] + d_dt * _vel_tmp;
    }
}
