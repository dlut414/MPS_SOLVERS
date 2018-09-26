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
void MPS_CPU::Initial()
{
    char str[256];

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
    b = new real[i_np];
    x = new real[i_np];
    A = new real*[i_np];

    r = new real[i_np];
    p = new real[i_np];
    Ap = new real[i_np];
    bc = new bool[i_np];

    for(int i=0; i < i_np; i++) A[i] = new real[i_np];

    /*------------------------------------------------*/

    /*-----end of initialization-----*/
    t_loop_e = time(NULL);
    i_ini_time = difftime(t_loop_e , t_loop_s);
    /*-------------------------------*/

    sprintf(str, "successfully Initialized!\n");
    throwScn(str);
    throwLog(fid_log, str);
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
real3 MPS_CPU::d3_Grad_press(int i)
{
    real3 _ret = {0,0,0};
    real3 _dr = {0,0,0};
    real _r2 = 0.0;
    real _hat_p = d_press[i];

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
real MPS_CPU::d_Div_vel(int i)
{
    real _ret = 0.0;
    real _r2 = 0.0;
    real3 _dr = {0,0,0};
    real3 _du = {0,0,0};

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
real3 MPS_CPU::d3_Lap_vel(int i)
{
    //without cell list

    real3 _ret = {0.0,0.0,0.0};
    real3 _dr;
    real3 _du;
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
    real _d_dt_tmp;

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
///calculate implicit part
//////////////////////////////
void MPS_CPU::impl_part()
{
    real3 _d3_vel_tmp;
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
    real _n;
    real3 _dr;
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
    real3 _dr;
    real _rr;
    real _weight;
    real _Aii = d_rho * d_lambda * d_alpha * d_nzero / (2 * i_dim * d_dt * d_dt);//compressive flow (minused by Aii)

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
    real _rr;
    real _one_by_rr;
    real _vabs;
    real3 _dr; //rji
    real3 _v_crt;

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
    M_motion.doMotion(d3_pos, d3_vel, i_nb2+i_nb1);
}

//////////////////////////////
///solve 'Ax = b' by CG
//////////////////////////////
void MPS_CPU::DoCG(int __n)
{

    real _alpha = 0.0;//for CG
    real _rrnew = 0.0;//for CG
    real _rrold = 0.0;//for CG
    real _rn_over_ro;

    /*-----main part of Conjugate Gradient-----*/
    //p0 = r0 = b - Ax0
    #ifdef CPU_OMP
        #pragma omp parallel for
    #endif
    for(int i=0;i<__n;i++)
    {
        r[i] = b[i] - (real)std::inner_product(x,x+__n,A[i],0.0);
        p[i] = r[i];
    }
    _rrold = (real)std::inner_product(r,r+__n,r,0.0);

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
            Ap[i] = (real)std::inner_product(p,p+__n,A[i],0.0);
        }
        _alpha = _rrold / (real)std::inner_product(Ap,Ap+__n,p,0.0);

        #ifdef CPU_OMP
            #pragma omp parallel for
        #endif
        for(int i=0;i<__n;i++)
        {
            x[i] = x[i] + _alpha * p[i];
            r[i] = r[i] - _alpha * Ap[i];
        }

        _rrnew = (real)std::inner_product(r,r+__n,r,0.0);


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

