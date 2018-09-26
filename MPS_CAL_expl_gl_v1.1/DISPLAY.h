#ifndef DISPLAY_H
#define DISPLAY_H
/*
LICENCE
*/
//DISPLAY.h
//defination of class DISPLAY

#include <GL/glut.h>
#include "MPS_GPU.h"

namespace mytype
{

class DISPLAY
{
public:
    virtual ~DISPLAY() = 0;

public:
    static void myInit(int argc, char** argv);
    static void myMainLoop();

protected:
    static void setLightPosition();

private:
    static void myDraw();
    static void myAct();
    static void myKeyboard(unsigned char key,int,int);
    static void myReshape(int,int);


private:
    static bool stop;
    static MPS_GPU mps;

};

}

#endif // DISPLAY_H
