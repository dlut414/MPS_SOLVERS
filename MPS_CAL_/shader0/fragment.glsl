/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

layout(early_fragment_tests) in;

in vec4 mvpPos;

out vec4 color;

uniform mat4 fMvpInv;

void main()
{
    color = fMvpInv * mvpPos;
}
