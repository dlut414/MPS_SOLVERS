/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

layout(location = 0) in vec3 vPos;

out vec4 fPos;
out float radius;

uniform mat4 vMvp;
uniform mat4 vMv;

float scale = 30.f;

void main()
{
    fPos = vec4(vPos, 1.f);
    gl_Position = vMvp * fPos;

    float dist = length(vec3(vMv * fPos));

    radius =  0.01f / dist;

    gl_PointSize = radius * scale;
}

