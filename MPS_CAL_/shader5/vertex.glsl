/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

layout(location = 0) in vec3 vert;

out vec4 vVert;
out mat4 fMvp;

uniform mat4 vMvp;

void main()
{
    gl_Position = vMvp * vec4(vert, 1.f);
    fMvp = vMvp;

    vVert = vec4(vert, 1.f);
}

