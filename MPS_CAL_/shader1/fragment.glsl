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
uniform sampler2D frontFaceTex;

void main()
{
    vec2 uv = ( mvpPos.xy/mvpPos.w + vec2(1.f, 1.f) ) / 2.f;
    color = fMvpInv * mvpPos - texture(frontFaceTex, uv);
    color.a = length(color.rgb);
    color.rgb = normalize(color.rgb);
}
