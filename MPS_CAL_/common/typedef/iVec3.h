/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/

#ifndef IVEC3_H_INCLUDED
#define IVEC3_H_INCLUDED

#include <cmath>

class iVec3
{
public:
    iVec3(const int& _x = 0, const int& _y = 0, const int& _z = 0) : x(_x), y(_y), z(_z)
    {}
    ~iVec3()
    {}

    inline const iVec3 operator+ (const iVec3& v) const
    {
        return iVec3(x+v.x, y+v.y, z+v.z);
    }

    inline const iVec3 operator- (const iVec3& v) const
    {
        return iVec3(x-v.x, y-v.y, z-v.z);
    }

    inline void operator+= (const iVec3& v)
    {
        x += v.x;   y += v.y;   z += v.z;
    }
    inline void operator-= (const iVec3& v)
    {
        x -= v.x;   y -= v.y;   z -= v.z;
    }

    inline const iVec3 operator* (const int& s) const
    {
        return iVec3(s*x, s*y, s*z);
    }

    inline const int operator* (const iVec3& v) const
    {
        return v.x*x + v.y*y + v.z*z;
    }

    inline int mag2() const
    {
        return (x*x + y*y + z*z);
    }

    inline int mag() const
    {
        return sqrt(mag2());
    }

public:
    int x, y, z;
};

inline const iVec3 operator* (const int& s, iVec3& p)
{
    return iVec3(s*p.x, s*p.y, s*p.z);
}

#endif // IVEC3_H_INCLUDED
