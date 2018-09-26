/*
LICENCE
*/
//main.cpp
//main function
///main function
#include <GL/glut.h>
#include "def_incl.h"
#include "MPS_GPU.h"
#include "DISPLAY.h"

using namespace std;
using namespace mytype;

/*
MPS_GPU glTest;

void myInit()
{
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(1000, 500);
    glutCreateWindow("mps");
    glClearColor(0.2f, 0.2f, 0.8f, 1.0f);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glLoadIdentity();

    glTest.Initial();
}

void setLightPosition()
{
    int pos[] = {0.0f, 0.0f, 1.0f, 0.0f};
    glLightiv(GL_LIGHT0, GL_POSITION, pos);
}

void myDraw()
{
    //const float materialFloor[][4] = {{0.0f, 1.0f, 0.8f, 1.0f} , {0.0f, 0.8f, 1.0f, 1.0f} , };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    setLightPosition();

    for(integer i = 0; i < glTest.getNp(); i++)
    {
        real3* pos = glTest.getPos();

        glColor3f(0.5f, 0.5f, 1.0f);

        glBegin(GL_POINTS);
            glVertex3f(pos[i].x, pos[i].y, pos[i].z);
        glEnd();
    }

    glutSwapBuffers();
}

void myAct()
{
    glTest.step();
}

void myTimer(int aArg)
{
    myAct();
    glutPostRedisplay();
    glutTimerFunc(33, myTimer, 0);
}
*/

int main(int argc, char** argv)
{
    system("mkdir out");
#ifdef CPU_OMP
    omp_set_num_threads(OMP_THREADS);
#endif
/*
    glutInit(&argc, argv);
    myInit();
    glutTimerFunc(33, myTimer, 0);
    glutDisplayFunc(myDraw);
    glutMainLoop();
*/

    DISPLAY::myInit(argc, argv);
    DISPLAY::myMainLoop();

    return 0;
}









