/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#ifndef RENDERER_H_INCLUDED
#define RENDERER_H_INCLUDED

#include <cstdio>
#include "header.h"
#include "Camera.h"
#include "./common/para.h"
#include "./common/typedef/Point.h"
#include "./common/typedef/BBox.h"
#include "./common/typedef/TimeX.h"
#include "./common/typedef/Bitmap.h"
#include "Controller.h"
#include "DrawVox.h"
#include "MPS_GPU.h"
#include "typedef.h"

namespace Renderer
{

std::vector<Point> BBVertPos;
std::vector<Point> tri;

Controller*     stateObj;
Bitmap*         bmpObj;
mytype::MPS_GPU*        mps;
DrawVox*        renderObj;
Ips counter(50);

void InitGL(int argc, char** argv);
void MainLoop();
void Final();

void InitOBJ();

void Fps();
void myMouse(int, int, int, int);
void myMotion(int, int);
void myMouseWheel(int, int, int, int);
void myReshape(int, int);
void myKeyboard(unsigned char, int, int);
void myDisplay();


void InitOBJ()
{
    stateObj  = new Controller(0);
    mps       = new mytype::MPS_GPU();

    mps->Initial();
}

void InitRender()
{
    renderObj = new DrawVox(stateObj);

    renderObj->volData = (void*)(mps->vox_mc.r_verDensity);
    renderObj->volNorm = (void*)(mps->vox_mc.r3_verNorm);
    //renderObj->cellInFluid = (void*)(mps->i_cellInFluid);

    renderObj->box = BBox(Point(mps->vox_mc.r_left, mps->vox_mc.r_back, mps->vox_mc.r_bottom),
                          Point(mps->vox_mc.r_right, mps->vox_mc.r_front, mps->vox_mc.r_top));
    renderObj->ix = mps->vox_mc.i3_dim.x;
    renderObj->iy = mps->vox_mc.i3_dim.y;
    renderObj->iz = mps->vox_mc.i3_dim.z;

    renderObj->icx = mps->geo.i_cell_dx;
    renderObj->icy = mps->geo.i_cell_dy;
    renderObj->icz = mps->geo.i_cell_dz;

    renderObj->init();
}

void InitGL(int argc, char** argv)
{
    InitOBJ();

    glutInit                (&argc, argv);
    glutInitDisplayMode     (GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition  (0, 0);
    glutInitWindowSize      (stateObj->u_width, stateObj->u_height);
    glutCreateWindow        ("RTRenderer");
    //glLightModeli           (GL_LIGHT_MODEL_TWO_SIDE, 1);
    glutMouseFunc           (myMouse);
    glutMotionFunc          (myMotion);
    //glutMouseWheelFunc      (myMouseWheel);
    glutReshapeFunc         (myReshape);
    glutKeyboardFunc        (myKeyboard);
    glutDisplayFunc         (myDisplay);

    glEnable                (GL_TEXTURE_1D);
    glEnable                (GL_TEXTURE_2D);
    glEnable                (GL_TEXTURE_3D);
    glEnable                (GL_CULL_FACE);
    //glDisable               (GL_CULL_FACE);
    glFrontFace             (GL_CCW);
    glEnable                (GL_POINT_SPRITE_ARB);
    glEnable                (GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable                (GL_DEPTH_TEST);
    glDepthFunc             (GL_LESS);
    glEnable                (GL_ALPHA_TEST);
    glAlphaFunc             (GL_GREATER, 0.f);
    //glEnable                (GL_BLEND);
    //glBlendFunc             (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor            (1.f, 1.f, 1.f, 0.f);

    glewInit              ();

    InitRender();
}

void Fps()
{
    counter.iter++;

    if(counter.iter == counter.iterMax)
    {
        char    _str[256];

        counter.tStop = getSystemTime();

        sprintf( _str, "fps: %4.1f",  counter.iterMax * 1000.0f / (counter.tStop - counter.tStart) );
        glutSetWindowTitle(_str);

        counter.iter = 0;
        counter.tStart = getSystemTime();
    }
}

void MainLoop() { glutMainLoop(); }

void Final() {}

void myMouse(int button, int s, int x, int y)
{
    stateObj->clickMouse(button, s, x, y);
}

void myMotion(int x, int y)
{
    stateObj->moveMouse(x, y);

    glutPostRedisplay();
}

void myMouseWheel(int button, int dir, int x, int y)
{
    stateObj->rollMouse(button, dir, x, y);
}

void myReshape(int width, int height)
{
    glViewport      (0, 0, width, width);
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();
    gluPerspective  (-90.0f, float(stateObj->u_width)/float(stateObj->u_height), 1.0f, 100.0f);

    stateObj->reshapeWindow();
}

void myKeyboard(unsigned char key, int a, int b)
{
    glutPostRedisplay();

    stateObj->pressKey(key, a, b);

    if(stateObj->b_init) InitOBJ();

    stateObj->b_init = false;
}

void myDisplay()
{
    /*
    if(!b_stop)
    {
        b_dirty = true;
    }
    */
    glm::mat4 modelMatrix = glm::translate  ( glm::mat4(1.0f), stateObj->m_pan )
                          * glm::toMat4     ( stateObj->m_rotation )
                          * glm::scale      ( glm::mat4(1.0f), stateObj->m_scale );

    stateObj->m_viewModelMat      = stateObj->m_camera.GetViewMatrix() * modelMatrix;
    stateObj->m_projectionMat     = stateObj->m_camera.GetProjectionMatrix();
    stateObj->m_projectionMatInv  = glm::inverse( stateObj->m_projectionMat );
    stateObj->m_mvp               = stateObj->m_projectionMat * stateObj->m_viewModelMat;
    stateObj->m_mvpInv            = glm::inverse( stateObj->m_mvp );

    BBVertPos.clear();
    /*
    for(unsigned i=0; i<(mps->i_markVox.size()); i++)
    {
        mytype::real3 vox = mps->vox_mc.r3_verList[mps->i_markVox[i]];
        Point ver = Point(vox.x, vox.y, vox.z);
        BBVertPos.push_back(ver);
    }
    */

    for(unsigned i=0; i<(mps->i_markCell.size()); i++)
    {
        //int iz = mps->i_markCell[i] / mps->geo.i_cell_sheet;
        //int iy = mps->i_markCell[i] % mps->geo.i_cell_sheet / mps->geo.i_cell_dx;
        //int ix = mps->i_markCell[i] % mps->geo.i_cell_sheet % mps->geo.i_cell_dx;
        //float size = mps->geo.d_cell_size;

        //Point vox0 = Point(ix * size,        iy * size,        iz * size);
        //Point vox1 = Point(ix * size + size, iy * size,        iz * size);
        //Point vox2 = Point(ix * size + size, iy * size + size, iz * size);
        //Point vox3 = Point(ix * size,        iy * size + size, iz * size);
        //Point vox4 = Point(ix * size,        iy * size,        iz * size + size);
        //Point vox5 = Point(ix * size + size, iy * size,        iz * size + size);
        //Point vox6 = Point(ix * size + size, iy * size + size, iz * size + size);
        //Point vox7 = Point(ix * size,        iy * size + size, iz * size + size);

        int id = mps->i_markCell[i];

        Point vox0 = mps->cellBox[8*id];
        Point vox1 = mps->cellBox[8*id+1];
        Point vox2 = mps->cellBox[8*id+2];
        Point vox3 = mps->cellBox[8*id+3];
        Point vox4 = mps->cellBox[8*id+4];
        Point vox5 = mps->cellBox[8*id+5];
        Point vox6 = mps->cellBox[8*id+6];
        Point vox7 = mps->cellBox[8*id+7];

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox2); BBVertPos.push_back(vox1);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox3); BBVertPos.push_back(vox2);

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox1); BBVertPos.push_back(vox5);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox5); BBVertPos.push_back(vox4);

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox4); BBVertPos.push_back(vox7);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox7); BBVertPos.push_back(vox3);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox1); BBVertPos.push_back(vox2);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox5); BBVertPos.push_back(vox1);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox3); BBVertPos.push_back(vox7);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox2); BBVertPos.push_back(vox3);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox4); BBVertPos.push_back(vox5);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox7); BBVertPos.push_back(vox4);
    }
/*
        BBox box = renderObj->box;
        Point vox0 = Point(box.pMin);
        Point vox1 = Point(box.pMax.x, box.pMin.y, box.pMin.z);
        Point vox2 = Point(box.pMax.x, box.pMax.y, box.pMin.z);
        Point vox3 = Point(box.pMin.x, box.pMax.y, box.pMin.z);
        Point vox4 = Point(box.pMin.x, box.pMin.y, box.pMax.z);
        Point vox5 = Point(box.pMax.x, box.pMin.y, box.pMax.z);
        Point vox6 = Point(box.pMax.x, box.pMax.y, box.pMax.z);
        Point vox7 = Point(box.pMin.x, box.pMax.y, box.pMax.z);

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox2); BBVertPos.push_back(vox1);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox3); BBVertPos.push_back(vox2);

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox1); BBVertPos.push_back(vox5);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox5); BBVertPos.push_back(vox4);

        BBVertPos.push_back(vox0); BBVertPos.push_back(vox4); BBVertPos.push_back(vox7);
        BBVertPos.push_back(vox0); BBVertPos.push_back(vox7); BBVertPos.push_back(vox3);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox1); BBVertPos.push_back(vox2);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox5); BBVertPos.push_back(vox1);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox3); BBVertPos.push_back(vox7);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox2); BBVertPos.push_back(vox3);

        BBVertPos.push_back(vox6); BBVertPos.push_back(vox4); BBVertPos.push_back(vox5);
        BBVertPos.push_back(vox6); BBVertPos.push_back(vox7); BBVertPos.push_back(vox4);
*/
    renderObj->draw(BBVertPos.size(), BBVertPos.data());

    glutSwapBuffers     ();
    glutReportErrors    ();

    if(!stateObj->b_stop)
    {
/*
        ///write to bmp
        static int i = 0;
        char name[256];
        sprintf(name, "./out/bm%04d.bmp", i++);
        bmpObj->SaveAsBMP(name);

        char str[256];
        sprintf(str, "../input/out/%04d.out", (stateObj->i_file++));

        if( !(*sceneObj << str) ) {stateObj->m_scale += glm::vec3(0.0025f);}//exit(100);
        else
        {
            printf("%d\n", i);
            sceneObj->update();
        }

        glBindTexture(GL_TEXTURE_3D, renderObj1->volDataTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, renderObj1->i_vx, renderObj1->i_vy, renderObj1->i_vz, 0, GL_RED, GL_FLOAT, (void*)(sceneObj->scalar.data()));
        glBindTexture(GL_TEXTURE_3D, renderObj1->volNormTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, renderObj1->i_vx, renderObj1->i_vy, renderObj1->i_vz, 0, GL_RGB, GL_FLOAT, (void*)(sceneObj->norm.data()));
        glBindTexture(GL_TEXTURE_3D, 0);

        stateObj->m_rotation = glm::angleAxis<float>( Common::PI * 0.001f * i, glm::vec3(0, 0, 1) );
*/
        mps->step();

        glBindTexture(GL_TEXTURE_3D, renderObj->volDataTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, renderObj->ix, renderObj->iy, renderObj->iz, 0, GL_RED, GL_FLOAT, (void*)(mps->vox_mc.r_verDensity));
        glBindTexture(GL_TEXTURE_3D, renderObj->volNormTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, renderObj->ix, renderObj->iy, renderObj->iz, 0, GL_RGB, GL_FLOAT, (void*)(mps->vox_mc.r3_verNorm));
        glBindTexture(GL_TEXTURE_3D, 0);

        stateObj->b_dirty = true;
    }

    Fps();

    if(stateObj->b_dirty)
    {
        glutPostRedisplay();
        stateObj->b_dirty = false;
    }

    if(stateObj->b_leave)
    {
        glutLeaveMainLoop();
    }
}


} //namespace Renderer

#endif // RENDERER_H_INCLUDED
