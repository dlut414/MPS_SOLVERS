/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#ifndef VEC3_H
#define VEC3_H

#include <cmath>

template <typename real>
class Vec3
{
public:
    Vec3(const real& _x = 0, const real& _y = 0, const real& _z = 0) : x(_x), y(_y), z(_z)
    {}
    ~Vec3()
    {}

    inline const Vec3 operator+ (const Vec3& v) const
    {
        return Vec3(x+v.x, y+v.y, z+v.z);
    }

    inline const Vec3 operator- (const Vec3& v) const
    {
        return Vec3(x-v.x, y-v.y, z-v.z);
    }

    inline void operator+= (const Vec3& v)
    {
        x += v.x;   y += v.y;   z += v.z;
    }
    inline void operator-= (const Vec3& v)
    {
        x -= v.x;   y -= v.y;   z -= v.z;
    }

    inline const Vec3 operator* (const real& s) const
    {
        return Vec3(s*x, s*y, s*z);
    }

    inline const real operator* (const Vec3& v) const
    {
        return v.x*x + v.y*y + v.z*z;
    }

    inline real mag2() const
    {
        return (x*x + y*y + z*z);
    }

    inline real mag() const
    {
        return sqrt(mag2());
    }

public:
    real x, y, z;
};

template <typename real>
inline const Vec3<real> operator* (const real& s, const Vec3<real>& p)
{
    return Vec3<real>(s*p.x, s*p.y, s*p.z);
}

#endif // VEC3_H
