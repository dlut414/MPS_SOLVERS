#ifndef DISPLAY_H
#define DISPLAY_H
/*
LICENCE
*/
//DISPLAY.h
//defination of class DISPLAY

#include "header_PCH.h"
#include "Camera.h"
#include "MPS_GPU.h"
//#include "Quaternion.h"

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
    static void myDraw      ();
    static void myAct       ();
    static void myKeyboard  (unsigned char key,int,int);
    static void myReshape   (int,int);
    static void myMouse     (int button, int state, int x, int y);
    static void myMotion    (int x, int y);

private:
    static bool         stop;
    static MPS_GPU      mps;
    static float        f_width;
    static float        f_height;
    static Camera       g_Camera;
    static glm::ivec2   g_MousePos;
    static glm::quat    g_Rotation;
    static glm::vec3    g_InitialCameraPosition;
    static glm::quat    g_InitialCameraRotation;

};

}

#endif // DISPLAY_H
