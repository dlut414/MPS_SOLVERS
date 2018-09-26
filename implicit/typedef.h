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

typedef struct
{
    int x;
    int y;
    int z;
}int3;

typedef struct
{
    double x;
    double y;
    double z;
}double3;

inline double3 operator+ (const double3& a , const double3& b)
{
    double3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

inline double operator* (const double3& a , const double3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline double3 operator* (const double& a , const double3& b)
{
    double3 c;
    c.x = a * b.x;
    c.y = a * b.y;
    c.z = a * b.z;
    return c;
}

inline double3 operator- (const double3& a , const double3& b)
{
    double3 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}
#endif //TYPEDEF_H
