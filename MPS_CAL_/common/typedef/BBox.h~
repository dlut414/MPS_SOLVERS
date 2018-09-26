/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Published under CC BY-NC
*/
#ifndef BBOX_H
#define BBOX_H

#include <limits>
#include "Point.h"
#include "Vector.h"

class BBox
{
public:
    BBox()
    {
        float _infinity = std::numeric_limits<float>::infinity();

        pMin = Point( _infinity,  _infinity,  _infinity);
        pMax = Point(-_infinity, -_infinity, -_infinity);
    }

    BBox(const Point &p) : pMin(p), pMax(p)
    {}

    BBox(const Point &p1, const Point &p2)
    {
        pMin = Point(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
        pMax = Point(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
    }

    void operator+=(const Point &p)
    {
        pMin.x = std::min(pMin.x, p.x);
        pMin.y = std::min(pMin.y, p.y);
        pMin.z = std::min(pMin.z, p.z);
        pMax.x = std::max(pMax.x, p.x);
        pMax.y = std::max(pMax.y, p.y);
        pMax.z = std::max(pMax.z, p.z);
    }

    BBox Union(const BBox &b, const Point &p) const
    {
        BBox ret = b;
        ret.pMin.x = std::min(b.pMin.x, p.x);
        ret.pMin.y = std::min(b.pMin.y, p.y);
        ret.pMin.z = std::min(b.pMin.z, p.z);
        ret.pMax.x = std::max(b.pMax.x, p.x);
        ret.pMax.y = std::max(b.pMax.y, p.y);
        ret.pMax.z = std::max(b.pMax.z, p.z);
        return ret;
    }

    BBox Union(const BBox &b1, const BBox &b2) const
    {
        BBox ret = b1;
        ret.pMin.x = std::min(b1.pMin.x, b2.pMin.x);
        ret.pMin.y = std::min(b1.pMin.y, b2.pMin.y);
        ret.pMin.z = std::min(b1.pMin.z, b2.pMin.z);
        ret.pMax.x = std::max(b1.pMax.x, b2.pMax.x);
        ret.pMax.y = std::max(b1.pMax.y, b2.pMax.y);
        ret.pMax.z = std::max(b1.pMax.z, b2.pMax.z);
        return ret;
    }

    bool bOverlaps(const BBox &b) const
    {
        bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
        bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
        bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
        return (x && y && z);
    }

    bool bInside(const Point &p) const
    {
        return (p.x >= pMin.x && p.x <=pMax.x &&
                p.y >= pMin.y && p.y <=pMax.y &&
                p.z >= pMin.z && p.z <=pMax.z);
    }

    Point pCenter() const
    {
        return 0.5f * Point( (pMax.x+pMin.x), (pMax.y+pMin.y), (pMax.z+pMin.z) );
    }

    void Expand(const float& s)
    {
        Vector v = s * (pMax - pMin);
        pMax += v;
        pMin -= v;
    }

public:
    Point pMin, pMax;

};

#endif // BBOX_H
