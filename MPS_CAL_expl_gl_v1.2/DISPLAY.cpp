/*
LICENCE
*/
//DISPLAY.cpp
//implementation of class DISPLAY
#include "DISPLAY.h"

using namespace mytype;

MPS_GPU      DISPLAY::mps;
bool         DISPLAY::stop = true;
float        DISPLAY::f_width = 1024;
float        DISPLAY::f_height = 512;
Camera       DISPLAY::g_Camera;
glm::ivec2   DISPLAY::g_MousePos;
glm::quat    DISPLAY::g_Rotation = glm::angleAxis<float>( -PI * 0.5f, glm::vec3(1, 0, 0) );
glm::vec3    DISPLAY::g_InitialCameraPosition;
glm::quat    DISPLAY::g_InitialCameraRotation;

void DISPLAY::myInit(int argc, char** argv)
{
    glutInit                (&argc, argv);
    glutInitDisplayMode     (GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition  (0, 0);
    glutInitWindowSize      (f_width, f_height);
    glutCreateWindow        ("mps");
    glClearColor            (0.1f, 0.1f, 0.1f, 1.0f);
    glEnable                (GL_DEPTH_TEST);
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glutMouseFunc           (myMouse);
    glutMotionFunc          (myMotion);
    glutReshapeFunc         (myReshape);
    glutKeyboardFunc        (myKeyboard);
    glutDisplayFunc         (myAct);

    mps.Initial();
}

void DISPLAY::myReshape(int width, int height)
{
    glViewport      (0, 0, width, width);
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();
    gluPerspective  (-90.0f, f_width/f_height, 10.0f, 100.0f);
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

void DISPLAY::myMouse(int button, int state, int x, int y)
{
    g_MousePos = glm::ivec2(x, y);
}

void DISPLAY::myMotion(int x, int y)
{
    glm::ivec2 mousePos = glm::ivec2( x, y );
    glm::vec2 delta = glm::vec2( mousePos - g_MousePos );
    g_MousePos = mousePos;

    glm::quat rotX = glm::angleAxis<float>( glm::radians(delta.y) * 0.5f, glm::vec3(1, 0, 0) );
    glm::quat rotY = glm::angleAxis<float>( glm::radians(delta.x) * 0.5f, glm::vec3(0, 1, 0) );
    g_Rotation = ( rotX * rotY ) * g_Rotation;
}

void DISPLAY::setLightPosition()
{
    int pos[] = {0.0f, 0.0f, 1.0f, 0.0f};

    glLightiv(GL_LIGHT0, GL_POSITION, pos);
}

void DISPLAY::myDraw()
{
    //const float materialFloor[][4] = {{0.0f, 1.0f, 0.8f, 1.0f} , {0.0f, 0.8f, 1.0f, 1.0f} , };

    glClear         (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();

    //setLightPosition();
    glm::mat4 mvp = g_Camera.GetProjectionMatrix() * g_Camera.GetViewMatrix() * glm::toMat4(g_Rotation);

    //glTranslatef    (-f_width*0.3, 0.0f, -f_height*0.8);
    glMultMatrixf   ( glm::value_ptr(mvp) );
    glScalef        (1.8f, 1.8f, 1.8f);

    gluLookAt(0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

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
