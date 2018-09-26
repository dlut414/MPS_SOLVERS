/*
LICENCE
*/
//DISPLAY.cpp
//implementation of class DISPLAY
#include "DISPLAY.h"

using namespace mytype;

MPS_GPU DISPLAY::mps;
bool DISPLAY::stop = true;


void DISPLAY::myInit(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(1000, 1000);
    glutCreateWindow("mps");
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glutReshapeFunc(myReshape);
    glutKeyboardFunc(myKeyboard);
    glutDisplayFunc(myAct);

    mps.Initial();
}

void DISPLAY::myReshape(int width, int height)
{
    glViewport(0.0f, 0.0f, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluPerspective(-90.0f, width/height, 10.0f, 100.0f);
}

void DISPLAY::myKeyboard(unsigned char key,int,int)
{
    glutPostRedisplay();

    switch(key)
    {
        case 27: stop = true; break;
        case 13: stop = false; break;
    }
}

void DISPLAY::setLightPosition()
{
    int pos[] = {0.0f, 0.0f, 1.0f, 0.0f};

    glLightiv(GL_LIGHT0, GL_POSITION, pos);
}

void DISPLAY::myDraw()
{
    //const float materialFloor[][4] = {{0.0f, 1.0f, 0.8f, 1.0f} , {0.0f, 0.8f, 1.0f, 1.0f} , };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //setLightPosition();

    glRotatef(-90.0,1.0,0.0,0.0);
    glScalef(1.8f, 1.8f, 1.8f);
    glTranslated(-0.5f, 0.0f, 0.0f);

    real3* _pos = mps.getPos();

    for(integer i = 0; i < mps.getNp(); i++)
    {
        glColor3f(1.0f, 1.0f, 1.0f);

        glBegin(GL_POINTS);
            glVertex3f(_pos[i].x, _pos[i].y, _pos[i].z);
        glEnd();
    }

    glutSwapBuffers();
}

void DISPLAY::myAct()
{
    if(!stop)
    {
        glutPostRedisplay();
        mps.step();
    }

    myDraw();

}

void DISPLAY::myMainLoop()
{
    glutMainLoop();
}
