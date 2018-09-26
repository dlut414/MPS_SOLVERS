#ifndef MPS_POST_H
#define MPS_POST_H
/*-------------------------------------------
PROGRAM CHANGING OUTPUT FILES INTO VTK FILES
--------------------------------------------*/
#include <cstdio>
class MPS_POST
{
public:
    MPS_POST();
    ~MPS_POST();

public:
    void LoadCase();
    void LoadData();
    void WriteData();

private:
    void WriteHead();
    void WritePoint();
    void WriteCell1();
    void WriteCell5();
    void WriteAttribute();

private:
    int** cellList;
    int** neighbor;
    int* cellType;
    double dp;
    double r0;

protected:
    char data_file[256];
    char out_file[256];
    int np, nb2, nb1;
    int c;
    FILE* fid_read;
    FILE* fid_write;

    int* id;

    double* posx;
    double* posy;
    double* posz;

    double* velx;
    double* vely;
    double* velz;

    double* pres;
};

#endif // MPS_POST_H
