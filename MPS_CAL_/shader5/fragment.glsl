/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

in vec4 vVert;
in mat4 fMvp;

out vec4 color;

const vec4 black = vec4( 0.9f, 0.9f, 0.9f, 1.f);
const vec4 white = vec4( 1.f, 1.f, 1.f, 1.f );

const int block = 2;

void main()
{
    if(vVert.z > 0.f)
    {
        int u = int(vVert.x * 100 + 1000);
        int v = int(vVert.z * 100);

        if(u/block%2 == v/block%2) color = white;
        else color = black;
    }
    else
    {
        int u = int(vVert.x * 100 + 1000);
        int v = int(vVert.y * 100 + 1000);

        if(u/block%2 != v/block%2) color = white;
        else color = black;
    }
}
