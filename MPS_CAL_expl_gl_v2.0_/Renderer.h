#ifndef DISPLAY_H
#define DISPLAY_H
/*
LICENCE
*/
//DISPLAY.h
//defination of class DISPLAY
#include <cuda_runtime.h>

#include "header_PCH.h"
#include "Camera.h"
#include "MPS_GPU.h"

namespace mytype
{

enum e_key
{
    E_KEY_ESC = 0x1b,
    E_KEY_ENTER = 0x0d,
    E_KEY_SPACE = 0x20
};

class Renderer
{
public:
    virtual ~Renderer() = 0;

public:
    static void myInit      (int argc, char** argv);
    static void myMainLoop  ();
    static void myFinal     ();

protected:
    static void setLightPosition();

private:
    static void myDraw              ();
    static void myAct               ();
    static void myKeyboard          (unsigned char key,int,int);
    static void myReshape           (int,int);
    static void myMouse             (int button, int state, int x, int y);
    static void myMotion            (int x, int y);
    static void myMouseWheel        (int button, int dir, int x, int y);
    static void myDrawBuffer_frame  ();
    static void myDrawBuffer_screen ();
    static void myInitVAO           ();
    static void myInitVBO           ();
    static void myInitFramebuffer   ();
    static void myInitTexture       ();
    static void myFps               ();

    static GLuint myLoadShader(const char* vertex_file_path, const char* fragment_file_path);

private:
    static GLuint       programID_depth;
    static GLuint       programID_normal;
    static GLuint       matrixID[5];
    static GLuint       scalaID[4];
    static GLuint       vertexArrayID;
    static GLuint       vertexBufferPos;
    static GLuint       vertexBufferType;
    static GLuint       vertexBufferScreen;
    static real3*       r3_vertex_buffer_data;
    static integer*     u_vertex_buffer_data;
    static real         r_screen_vertex_buffer_data[18];

    static GLuint       frameBuffer;
    static GLuint       depthTex;
    static GLuint       colorTex;
    static GLuint       depth;

private:
    static Camera       g_Camera;
    static glm::ivec2   g_MousePos;
    static glm::quat    g_Rotation;
    static glm::vec3    g_Scale;
    static glm::vec3    g_Pan;
    static glm::vec3    g_InitialCameraPosition;
    static glm::quat    g_InitialCameraRotation;
    static glm::mat4    g_mvp;
    static glm::mat4    g_modelMat;
    static glm::mat4    g_viewMat;
    static glm::mat4    g_projectionMat;
    static glm::mat4    g_viewModelMat;
    static glm::mat4    g_projectionMatInv;
    static GLfloat      f_pointRadius;
    static GLfloat      f_pointScale;
    static GLfloat      f_near;
    static GLfloat      f_far;

private:
    static bool         stop;
    static MPS_GPU      mps;
    static float        f_width;
    static float        f_height;
    static float        f_scaleVel;
    static float        f_panVel;

    static int          i_mouseButton;

private:
    static e_key    myKey;
    static float    f_timer;
    static int      i_count;
    static int      i_limit;
    static cudaEvent_t     cuda_start;
    static cudaEvent_t     cuda_stop;

};

}

#endif // DISPLAY_H
