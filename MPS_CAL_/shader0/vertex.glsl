/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

layout(location = 0) in vec3 vPos;

out vec4 mvpPos;

uniform mat4 vMvp;

void main()
{
    mvpPos = vMvp * vec4(vPos, 1.0f);

    gl_Position = mvpPos;
}
