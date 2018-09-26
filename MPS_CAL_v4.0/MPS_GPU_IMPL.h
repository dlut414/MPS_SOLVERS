/*
LICENCE
*/
//MPS_GPU_IMPL.h
//defination of class MPS_GPU_IMPL
///do implicit calculation (main loop)
#ifndef MPS_GPU_IMPL_H_INCLUDED
#define MPS_GPU_IMPL_H_INCLUDED

#include "def_incl.h"
#include "MPS_GPU.h"

namespace mytype
{

class MPS_GPU_IMPL : public MPS_GPU
{
public:
    MPS_GPU_IMPL();
    ~MPS_GPU_IMPL();

public:
    void mps_cal();//do the main simulation and output
    void step();//main loop

protected:
    void g_expl();//calculate g explicitly
    void visCrankNicolson();//calculate vis using CN
    template<unsigned int _dir> void updatePosVel();//update pos and vel
    template<unsigned int _dir> void buildCG_CN();//build A b x for DoCG

private:
    real d_part1;
    real d_part2;
    real d_part3;
    real d_part4;
    real d_part5;

};

}
#endif // MPS_GPU_IMPL_H_INCLUDED
