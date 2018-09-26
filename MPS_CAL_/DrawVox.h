/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#ifndef DRAWVOX_H
#define DRAWVOX_H

#include "header.h"
#include "Draw_.h"
#include "Shader.h"
#include "typedef.h"

class DrawVox : public Draw_
{
public:
    DrawVox(Controller* state) : Draw_(state) {}
    ~DrawVox() {}

    void init()
    {
        Draw_::initFramebuffer(Draw_::fbo[0]);
        Draw_::initTexture_2D(frontFaceTex, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        Draw_::attachTex_2D_color0(frontFaceTex);
        Draw_::initRenderbuffer(dbo[0], Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        Draw_::attachRenderbuffer_depth(dbo[0]);

        Draw_::initFramebuffer(Draw_::fbo[1]);
        Draw_::initTexture_2D(dirTex, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        Draw_::attachTex_2D_color0(dirTex);
        Draw_::initRenderbuffer(dbo[1], Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        Draw_::attachRenderbuffer_depth(dbo[1]);

        Draw_::initFramebuffer(Draw_::fbo[2]);
        Draw_::initTexture_2D(bgTex, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        Draw_::attachTex_2D_color0(bgTex);

        Draw_::initVAO(Draw_::vao);
        Draw_::initVBO(Draw_::vbo[0]);
        Draw_::initVBO(Draw_::vbo[1]);
        Draw_::initVBO(Draw_::vbo[2]);
        Draw_::initVBO(Draw_::vbo[3]);
        Draw_::initVBO(Draw_::vbo[4]);
        Draw_::initVBO(Draw_::vbo[5]);

        int max_size = 0;
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_size);
        printf("max texture dimention: %d\n", max_size);
        printf("vx: %d, vy: %d, vz: %d\n", ix, iy, iz);

        Draw_::initTexture_3D_3f(volNormTex, ix, iy, iz, (void*)volNorm);
        Draw_::initTexture_3D_1f(volDataTex, ix, iy, iz, (void*)volData);
        //Draw_::initTexture_3D_1f(cellInFluidTex, icx, icy, icz, (void*)cellInFluid);

        r_bg_chessboard[0] = -1.f; r_bg_chessboard[1] = -1.f; r_bg_chessboard[2] =  0.f;
        r_bg_chessboard[3] =  1.f; r_bg_chessboard[4] = -1.f; r_bg_chessboard[5] =  0.f;
        r_bg_chessboard[6] = -1.f; r_bg_chessboard[7] =  1.f; r_bg_chessboard[8] =  0.f;

        r_bg_chessboard[9] =  1.f; r_bg_chessboard[10] = -1.f; r_bg_chessboard[11] =  0.f;
        r_bg_chessboard[12] =  1.f; r_bg_chessboard[13] =  box.pMax.y; r_bg_chessboard[14] =  0.f;
        r_bg_chessboard[15] = -1.f; r_bg_chessboard[16] =  box.pMax.y; r_bg_chessboard[17] =  0.f;

        r_bg_chessboard[18] = -1.f; r_bg_chessboard[19] = box.pMax.y; r_bg_chessboard[20] =  0.f;
        r_bg_chessboard[21] = 1.f; r_bg_chessboard[22] =  box.pMax.y; r_bg_chessboard[23] =  0.f;
        r_bg_chessboard[24] = 1.f; r_bg_chessboard[25] =  box.pMax.y; r_bg_chessboard[26] =  1.f;

        r_bg_chessboard[27] =  -1.f; r_bg_chessboard[28] =  box.pMax.y; r_bg_chessboard[29] =  0.f;
        r_bg_chessboard[30] =  1.f; r_bg_chessboard[31] =  box.pMax.y; r_bg_chessboard[32] =  1.f;
        r_bg_chessboard[33] = -1.f; r_bg_chessboard[34] =  box.pMax.y; r_bg_chessboard[35] =  1.f;

        initShader();
    }

    void draw(unsigned numVert, Point* vert)
    {
        ///clear bg texture
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[2]);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("fbo[0] not ready\n");
        glViewport(0, 0, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        glFrontFace(GL_CCW);

        glClearDepth(1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderObj.programID[3]);

        glUniformMatrix4fv( glGetUniformLocation(shaderObj.programID[3], "vMvp"), 1, GL_FALSE, &(Draw_::stateObj->m_mvp[0][0]) );

        glBindBuffer(GL_ARRAY_BUFFER, Draw_::vbo[3]);
        glBufferData(GL_ARRAY_BUFFER, 36*sizeof(GLfloat), r_bg_chessboard, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, 12);

        glEnableVertexAttribArray(0);

        ///clear framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("fbo_def not ready\n");
        glViewport(0, 0, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        glFrontFace(GL_CCW);

        glClearDepth(1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderObj.programID[3]);

        glUniformMatrix4fv( glGetUniformLocation(shaderObj.programID[3], "vMvp"), 1, GL_FALSE, &(Draw_::stateObj->m_mvp[0][0]) );

        glBindBuffer(GL_ARRAY_BUFFER, Draw_::vbo[3]);
        glBufferData(GL_ARRAY_BUFFER, 36*sizeof(GLfloat), r_bg_chessboard, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, 12);

        glEnableVertexAttribArray(0);
        ///use program0
        glBindFramebuffer(GL_FRAMEBUFFER, Draw_::fbo[0]);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("fbo[0] not ready\n");
        glViewport(0, 0, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        glFrontFace(GL_CCW);
        glDepthFunc(GL_LESS);

        glClearDepth(1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderObj.programID[0]);

        glUniformMatrix4fv( shaderObj.matrixID[0], 1, GL_FALSE, &(Draw_::stateObj->m_mvp[0][0]) );
        glUniformMatrix4fv( shaderObj.matrixID[1], 1, GL_FALSE, &(Draw_::stateObj->m_mvpInv[0][0]) );

        glBindBuffer(GL_ARRAY_BUFFER, Draw_::vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, numVert*sizeof(Point), vert, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, numVert);

        glDisableVertexAttribArray(0);

        ///use program1
        glBindFramebuffer(GL_FRAMEBUFFER, Draw_::fbo[1]);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("fbo[1] not ready\n");
        glViewport(0, 0, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        glFrontFace(GL_CW);
        glDepthFunc(GL_GREATER);

        glClearDepth(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderObj.programID[1]);

        glUniformMatrix4fv( shaderObj.matrixID[2], 1, GL_FALSE, &(Draw_::stateObj->m_mvp[0][0]) );
        glUniformMatrix4fv( shaderObj.matrixID[3], 1, GL_FALSE, &(Draw_::stateObj->m_mvpInv[0][0]) );

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, frontFaceTex);
        glUniform1i(shaderObj.textureID[0], 0);

        glBindBuffer(GL_ARRAY_BUFFER, Draw_::vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, numVert*sizeof(Point), vert, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, numVert);

        glDisableVertexAttribArray(0);

        ///use program2
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("fbo_def not ready\n");
        glViewport(0, 0, Draw_::stateObj->u_width, Draw_::stateObj->u_height);
        glFrontFace(GL_CCW);
        glDepthFunc(GL_LESS);

        //glClearDepth(1);
        //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderObj.programID[2]);

        glUniformMatrix4fv( shaderObj.matrixID[4], 1, GL_FALSE, &(Draw_::stateObj->m_mvp[0][0]) );
        glUniformMatrix4fv( shaderObj.matrixID[5], 1, GL_FALSE, &(Draw_::stateObj->m_viewModelMat[0][0]) );
        glUniform3f(glGetUniformLocation(shaderObj.programID[2], "pMin"), box.pMin.x, box.pMin.y, box.pMin.z);
        glUniform3f(glGetUniformLocation(shaderObj.programID[2], "pMax"), box.pMax.x, box.pMax.y, box.pMax.z);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, frontFaceTex);
        glUniform1i(shaderObj.textureID[1], 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, dirTex);
        glUniform1i(shaderObj.textureID[2], 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, volNormTex);
        glUniform1i(shaderObj.textureID[3], 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_3D, volDataTex);
        glUniform1i(shaderObj.textureID[4], 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, bgTex);
        glUniform1i(shaderObj.textureID[5], 4);

        glBindBuffer(GL_ARRAY_BUFFER, Draw_::vbo[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Draw_::r_screenData), (void*)Draw_::r_screenData, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glDisableVertexAttribArray(0);
    }

public:
    GLfloat     r_bg_chessboard[36];

    GLuint bgTex;
    GLuint frontFaceTex;
    GLuint dirTex;
    GLuint dbo[2];

    GLuint volNormTex;
    GLuint volDataTex;
    GLuint cellInFluidTex;

    Shader shaderObj;

    BBox box;
    GLuint ix, iy, iz;
    GLuint icx, icy, icz;

    void* volNorm;
    void* volData;
    void* cellInFluid;

private:
    void initShader()
    {
        shaderObj.programID[0] = shaderObj.LoadShader("./shader0/vertex.glsl", "./shader0/fragment.glsl");
        shaderObj.matrixID[0] = glGetUniformLocation( shaderObj.programID[0], "vMvp" );
        shaderObj.matrixID[1] = glGetUniformLocation( shaderObj.programID[0], "fMvpInv" );

        shaderObj.programID[1] = shaderObj.LoadShader("./shader1/vertex.glsl", "./shader1/fragment.glsl");
        shaderObj.matrixID[2] = glGetUniformLocation( shaderObj.programID[1], "vMvp" );
        shaderObj.matrixID[3] = glGetUniformLocation( shaderObj.programID[1], "fMvpInv" );
        shaderObj.textureID[0] = glGetUniformLocation( shaderObj.programID[1], "frontFaceTex" );

        shaderObj.programID[2] = shaderObj.LoadShader("./shader2/vertex.glsl", "./shader2/fragment.glsl");
        shaderObj.matrixID[4] = glGetUniformLocation( shaderObj.programID[2], "vMvp" );
        shaderObj.matrixID[5] = glGetUniformLocation( shaderObj.programID[2], "fMv" );
        shaderObj.textureID[1] = glGetUniformLocation( shaderObj.programID[2], "frontFaceTex" );
        shaderObj.textureID[2] = glGetUniformLocation( shaderObj.programID[2], "dirTex" );
        shaderObj.textureID[3] = glGetUniformLocation( shaderObj.programID[2], "volNormTex" );
        shaderObj.textureID[4] = glGetUniformLocation( shaderObj.programID[2], "volDataTex" );
        shaderObj.textureID[5] = glGetUniformLocation( shaderObj.programID[2], "bgTex" );
        shaderObj.textureID[6] = glGetUniformLocation( shaderObj.programID[2], "cellInFluidTex" );

        shaderObj.programID[3] = shaderObj.LoadShader("./shader5/vertex.glsl", "./shader5/fragment.glsl");
    }
};

#endif // DRAWVOX_H
