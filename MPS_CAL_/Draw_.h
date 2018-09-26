/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#ifndef DRAWER_H
#define DRAWER_H

#include "header.h"
#include "./common/typedef/Point.h"
#include "./common/typedef/Vector.h"
#include "./common/typedef/BBox.h"
#include "Controller.h"

class Draw_
{
public:
    Draw_(Controller* state) : stateObj(state)
    {
        r_screenData[0] = -1.0f;    r_screenData[1] =  -1.0f;    r_screenData[2] =  0.0f;
        r_screenData[3] =  1.0f;    r_screenData[4] =  -1.0f;    r_screenData[5] =  0.0f;
        r_screenData[6] = -1.0f;    r_screenData[7] =   1.0f;    r_screenData[8] =  0.0f;
        r_screenData[9] = -1.0f;    r_screenData[10] =  1.0f;    r_screenData[11] = 0.0f;
        r_screenData[12] = 1.0f;    r_screenData[13] = -1.0f;    r_screenData[14] = 0.0f;
        r_screenData[15] = 1.0f;    r_screenData[16] =  1.0f;    r_screenData[17] = 0.0f;
    }
    ~Draw_() {}

    void initVAO(GLuint& vao)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }

    void initVBO(GLuint& vbo)
    {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }

    void initFramebuffer(GLuint& fb)
    {
        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
    }

    void initRenderbuffer(GLuint& db, GLuint& width, GLuint& height)
    {
        glGenRenderbuffers(1, &db);
        glBindRenderbuffer(GL_RENDERBUFFER, db);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    void initTexture_3D_1f(GLuint& tex, GLuint& xdim, GLuint& ydim, GLuint& zdim, void* data)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_3D, tex);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, xdim, ydim, zdim, 0, GL_RED, GL_FLOAT, (void*)data);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        glBindTexture(GL_TEXTURE_3D, 0);
    }
    void updateTexture_3D_1f()
    {}

    void initTexture_3D_3f(GLuint& tex, GLuint& xdim, GLuint& ydim, GLuint& zdim, void* data)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_3D, tex);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, xdim, ydim, zdim, 0, GL_RGB, GL_FLOAT, (void*)data);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        glBindTexture(GL_TEXTURE_3D, 0);
    }
    void updateTexture_3D_3f()
    {}

    void initTexture_2D(GLuint& tex, GLuint& width, GLuint& height)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void initTexture_1D_1f(GLuint& tex, GLuint& width, void* data)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_1D, tex);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, width, 0, GL_RED, GL_FLOAT, (void*)data);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);

        glBindTexture(GL_TEXTURE_1D, 0);
    }

    void initTexture_1D_3f(GLuint& tex, GLuint& width, void* data)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_1D, tex);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, width, 0, GL_RGB, GL_FLOAT, (void*)data);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);

        glBindTexture(GL_TEXTURE_1D, 0);
    }

    void attachTex_2D_color0(GLuint& tex)
    {
        glBindTexture(GL_TEXTURE_2D, tex);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    void attachTex_2D_color1(GLuint& tex)
    {
        glBindTexture(GL_TEXTURE_2D, tex);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex, 0);
        GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(1, drawBuffers);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    void attachRenderbuffer_depth(GLuint& db)
    {
        glBindRenderbuffer(GL_RENDERBUFFER, db);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, db);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

public:
    Controller* stateObj;

    GLuint      vao;
    GLuint      fbo[3];

    GLuint      vbo[6];

    GLfloat     r_screenData[18];

private:
    void CreateVBO(GLuint* vbo, unsigned size)
    {
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glutReportErrors();
    }

    void DeleteVBO(GLuint* vbo)
    {
        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);
        *vbo = 0;
    }

};

#endif // DRAWER_H
