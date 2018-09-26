/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

layout(location = 0) in vec3 vBBVertPos;

out vec4 mvpPos;

uniform mat4 vMvp;

void main()
{
    mvpPos = vMvp * vec4(vBBVertPos, 1.f);

    gl_Position = mvpPos;
}
