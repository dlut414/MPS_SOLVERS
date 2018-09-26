#ifndef TIMEX_H_INCLUDED
#define TIMEX_H_INCLUDED

/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/

#include <sys/timeb.h>
inline long getSystemTime()
{
    timeb t;
    ftime(&t);
    return (1000 * t.time + t.millitm);
}

class Ips
{
public:
    Ips(unsigned _iterMax = 50) : iter(0), iterMax(_iterMax), tStart(0), tStop(0)
    {}
    ~Ips(){}

public:
    unsigned iter;
    unsigned iterMax;
    long tStart;
    long tStop;
};

#endif // TIMEX_H_INCLUDED
