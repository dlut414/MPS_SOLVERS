#ifndef VOXEL_H
#define VOXEL_H
/*
LICENCE
*/
//Voxel.h
//defination of class Voxel
#include <vector>
#include "common.h"

template <typename I, typename I3, typename R, typename R3>
class Voxel
{
public:
    Voxel()
    {
        Initialize();
    }
    ~Voxel()
    {
        Finalize();
    }

public:
    void    setGeo       (I3 dim, R dx, R left, R right, R back, R front, R bottom, R top)
    {
        i3_dim      = dim;
        r_dx        = dx;
        r_left      = left;
        r_right     = right;
        r_back      = back;
        r_front     = front;
        r_bottom    = bottom;
        r_top       = top;

        reset();
    }

public:
    I3   i3_dim;

    I   i_nVertex;
    I   i_nVoxel;
    //I   i_nMaxEdge;

    R3* r3_verList;
    R*  r_verDensity;
    R3* r3_verNorm;
    I*  i_voxelList;

    //R3* r3_triangle;
    //R3* r3_norm;
    //R*  r_alpha;

    R   r_dx;
    R   r_left;
    R   r_right;
    R   r_back;
    R   r_front;
    R   r_bottom;
    R   r_top;

private:
    void    Initialize  ()
    {
        r3_verList      = NULL;
        r_verDensity    = NULL;
        r3_verNorm      = NULL;
        i_voxelList     = NULL;
        //r3_triangle     = NULL;
        //r3_norm         = NULL;
        //r_alpha         = NULL;
    }
    void    Finalize    ()
    {
        delete[] r3_verList;     r3_verList      = NULL;
        delete[] r_verDensity;   r_verDensity    = NULL;
        delete[] r3_verNorm;     r3_verNorm      = NULL;
        delete[] i_voxelList;    i_voxelList     = NULL;
        //delete[] r3_triangle;    r3_triangle     = NULL;
        //delete[] r3_norm;        r3_norm         = NULL;
        //delete[] r_alpha;        r_alpha         = NULL;
    }
    void    reset       ();

}; //class


template <typename I, typename I3, typename R, typename R3>
void Voxel<I, I3, R, R3>::reset()
{
    Finalize();

    I _dx = i3_dim.x;//ceil( (r_right - r_left) / r_dx ); //-1 out-of-bound proof
    I _dy = i3_dim.y;//ceil( (r_front - r_back) / r_dx ); //-1 out-of-bound proof
    I _dz = i3_dim.z;//ceil( (r_top - r_bottom) / r_dx ); //-1 out-of-bound proof
    I _dSheet = _dx * _dy;

    i_nVoxel  = _dx * _dy * _dz;
    i_nVertex = (_dx) * (_dy) * (_dz);
    //i_nMaxEdge = 12 * i_nVoxel;

    r3_verList      = new R3[i_nVertex];
    r_verDensity    = new R [i_nVertex];
    r3_verNorm      = new R3[i_nVertex];
    i_voxelList     = new I [i_nVoxel*8];

    //r3_triangle     = new R3[i_nMaxEdge];
    //r3_norm         = new R3[i_nMaxEdge];
    //r_alpha         = new R [i_nMaxEdge];

    for(I k=0; k<_dz; k++)
    {
        for(I j=0; j<_dy; j++)
        {
            for(I i=0; i<_dx; i++)
            {
                I __num = k*_dSheet + j*(_dx) + i;

                r3_verList[__num].x = r_left + i * r_dx;
                r3_verList[__num].y = r_back + j * r_dx;
                r3_verList[__num].z = r_bottom + k * r_dx;

                r_verDensity[__num] = 0.f;
            }
        }
    }

    for(I k=0; k<_dz; k++)
    {
        for(I j=0; j<_dy; j++)
        {
            for(I i=0; i<_dx; i++)
            {
                I __num = k*_dSheet + j*_dx + i;

                R3 __v[8];

                __v[0].x = i * r_dx;
                __v[0].y = j * r_dx;
                __v[0].z = k * r_dx;

                __v[1].x = (i+1) * r_dx;
                __v[1].y = j * r_dx;
                __v[1].z = k * r_dx;

                __v[2].x = (i+1) * r_dx;
                __v[2].y = (j+1) * r_dx;
                __v[2].z = k * r_dx;

                __v[3].x = i * r_dx;
                __v[3].y = (j+1) * r_dx;
                __v[3].z = k * r_dx;

                __v[4].x = i * r_dx;
                __v[4].y = j * r_dx;
                __v[4].z = (k+1) * r_dx;

                __v[5].x = (i+1) * r_dx;
                __v[5].y = j * r_dx;
                __v[5].z = (k+1) * r_dx;

                __v[6].x = (i+1) * r_dx;
                __v[6].y = (j+1) * r_dx;
                __v[6].z = (k+1) * r_dx;

                __v[7].x = i * r_dx;
                __v[7].y = (j+1) * r_dx;
                __v[7].z = (k+1) * r_dx;

                for(I p=0; p<8; p++)
                {
                    I __vi = mytype::ROUND(__v[p].z / r_dx) * ((_dx+1) * (_dy+1))
                           + mytype::ROUND(__v[p].y / r_dx) * (_dx+1)
                           + mytype::ROUND(__v[p].x / r_dx);

                    i_voxelList[8*__num + p] = __vi;
                }
            }
        }
    }
}

#endif // VOXEL_H
