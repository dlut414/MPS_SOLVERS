/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

in vec4 fPos;
in float radius;

out vec4 color;

uniform mat4 vMv;
uniform mat4 fPro;
uniform vec3 pMin;
uniform vec3 pMax;

vec3 materialColor = vec3( 0.03137f, 0.40392f, 0.57255f );
vec3 lightDir = vec3( 1.f, 1.f, 1.f );

void main()
{

    vec3 pixPos;
    pixPos.xy = gl_PointCoord.st * vec2(2.f, -2.f) - vec2(1.f, -1.f);

    float mag = length(pixPos.xy);
    if(mag >= 1.f) discard;

    pixPos.z = sqrt(1.f - mag);

    vec4 mvPix = vMv * fPos + vec4(radius * pixPos, 0.f);
    vec4 mvpPix = fPro * mvPix;

    float ndcDepth = mvpPix.z / mvpPix.w;

    gl_FragDepth = ndcDepth * 0.5f + 0.5f; //dealing with intersecting spheres

    vec3 norm = pixPos;
    vec3 eyeDir = -mvPix.xyz;

    float cosTheta = max( 0.0f, dot(norm, lightDir) );
    float cosBeta = max( 0.0f, dot(reflect(-lightDir, norm), eyeDir) );

    color.rgb = (cosTheta * materialColor);
    color.a = 1.f;
}
