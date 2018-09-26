/*
LICENCE
*/
//typedef.h
//file to define types
#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <sys/timeb.h>

namespace mytype
{

    typedef enum
    {
        x,y,z
    }DIR;

    typedef unsigned integer;
    typedef float    real;

    inline long getSystemTime() {
        timeb t;
        ftime(&t);
        return (1000 * t.time + t.millitm);
    }

    typedef struct
    {
        integer x;
        integer y;
        integer z;
    }int3;

    typedef struct
    {
        real x;
        real y;
        real z;
    }real3;

    inline real3 operator+ (const real3& a , const real3& b)
    {
        real3 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        return c;
    }

    inline real operator* (const real3& a , const real3& b)
    {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    inline real3 operator* (const real& a , const real3& b)
    {
        real3 c;
        c.x = a * b.x;
        c.y = a * b.y;
        c.z = a * b.z;
        return c;
    }

    inline real3 operator- (const real3& a , const real3& b)
    {
        real3 c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        c.z = a.z - b.z;
        return c;
    }

    inline void operator+= (real3& a , const real3& b)
    {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    inline integer ROUND(real x)
    {
        return integer(x+0.5f);
    }

    struct GEOMETRY
    {
        integer i_num_cells;    //number of cells, cal before the first step
        integer i_cell_dx;
        integer i_cell_dy;
        integer i_cell_dz;      //# of cells in each dim, cal before the first step
        integer i_cell_sheet;   //i_cell_sheet = i_cell_dx * i_cell_dy, cal before the first step
        real d_cell_size;       //size of the cell, cal before the first step
        real d_cell_left;
        real d_cell_right;
        real d_cell_back;
        real d_cell_front;
        real d_cell_bottom;
        real d_cell_top;        //boundary in each dim (6 facets), cal before the first step
        real d_cell_dx;
        real d_cell_dy;
        real d_cell_dz;         //size in each dim, cal before the first step
    };

}

#endif //TYPEDEF_H
