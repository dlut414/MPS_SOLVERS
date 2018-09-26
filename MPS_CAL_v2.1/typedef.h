/*
LICENCE
*/
//typedef.h
//file to define types
#ifndef TYPEDEF_H
#define TYPEDEF_H
typedef enum
{
    x,y,z
}DIR;

typedef double real;

typedef struct
{
    int x;
    int y;
    int z;
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
#endif //TYPEDEF_H
