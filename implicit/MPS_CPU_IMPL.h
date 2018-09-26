/*
LICENCE
*/
//MPS_CPU_IMPL.h
//defination of class MPS_CPU_EXPL
///do implicit calculation (main loop)
#ifndef MPS_CPU_IMPL_H
#define MPS_CPU_IMPL_H

#include "MPS_CPU.h"


class MPS_CPU_IMPL : public MPS_CPU
{
public:
    MPS_CPU_IMPL();
    ~MPS_CPU_IMPL();

public:
    void mps_cal();//do the main simulation and output
    void step();//main loop
};

#endif // MPS_CPU_IMPL_H
