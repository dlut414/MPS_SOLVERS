/*
LICENCE
*/
//MPS_CPU_IMPL.h
//defination of class MPS_CPU_EXPL
///do implicit calculation (main loop)
#ifndef MPS_CPU_IMPL_H
#define MPS_CPU_IMPL_H

#include "MPS_CPU.h"
#include "def_incl.h"

class MPS_CPU_IMPL : public MPS_CPU
{
public:
    MPS_CPU_IMPL();
    ~MPS_CPU_IMPL();

public:
    void mps_cal();//do the main simulation and output
    void step();//main loop

protected:
    void g_expl();//calculate g explicitly
    void vis_CrankNicolson();//calculate vis using CN
    template<unsigned int _dir> void update_pos_vel();//update pos and vel
    template<unsigned int _dir> void CN_buildCG();//build A b x for DoCG

private:
    real3 _dr;
    real _rr;
    real _weight;
    real _ONE_OVER_R;
};

#endif // MPS_CPU_IMPL_H
