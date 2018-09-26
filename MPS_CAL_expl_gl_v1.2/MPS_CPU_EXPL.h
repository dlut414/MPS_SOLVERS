/*
LICENCE
*/
//MPS_CPU.h
//defination of class MPS_CPU_EXPL
///do explicit calculation (main loop)
#ifndef MPS_CPU_EXPL_H
#define MPS_CPU_EXPL_H

#include "def_incl.h"
#include "MPS_CPU.h"

class MPS_CPU_EXPL : public MPS_CPU
{
public:
    MPS_CPU_EXPL();
    ~MPS_CPU_EXPL();

public:
    void mps_cal();//do the main simulation and output
    void step();//main loop

protected:
    void cal_vis_expl();//calculate the explicit part of vis and g

private:
    real d_part1;
    real d_part2;
    real d_part3;
    real d_part4;
    real d_part5;

};

#endif // MPS_CPU_EXPL_H
