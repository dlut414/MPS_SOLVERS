/*
LICENCE
*/
//MPS_CPU.h
//defination of class MPS_CPU_EXPL
///do explicit calculation (main loop)
#ifndef MPS_CPU_EXPL_H
#define MPS_CPU_EXPL_H

#include "MPS_CPU.h"


class MPS_CPU_EXPL : public MPS_CPU
{
public:
    MPS_CPU_EXPL();
    ~MPS_CPU_EXPL();

public:
    void mps_cal();//do the main simulation and output
    void step();//main loop
};

#endif // MPS_CPU_EXPL_H
