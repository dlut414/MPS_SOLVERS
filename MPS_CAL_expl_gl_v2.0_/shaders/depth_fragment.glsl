#version 330 core

in vec3 posView;
in vec4 fragmentColor;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 depth;

uniform mat4 projectionMat;
uniform float pointRadius;
uniform float near;
uniform float far;

void main()
{
    vec3 _pixelPos;

    _pixelPos.xy = gl_PointCoord.st * vec2(2.0f, 2.0f) - vec2(1.0f, 1.0f);

    float _mag = dot(_pixelPos.xy, _pixelPos.xy);
    if(_mag > 1.0f) discard;

    _pixelPos.z = sqrt(1.0f - _mag);

    vec4 _spherePosView         = vec4(posView + _pixelPos * pointRadius , fragmentColor.w);
    vec4 _projectionPosView     = projectionMat * _spherePosView;

    ///spere rendering
    //const vec3  _lightDir = normalize( vec3(0.5f, 1.5f, 5.0f) );
    //float _diffuse = max(0.0f, dot(_pixelPos, _lightDir));

    color       = fragmentColor;
    //color.rgb  *= _diffuse;

    //depth.z = ( ((far - near) / 2.0f) * (_projectionPosView.z) + ((far + near) / 2.0f) );
    depth.z = (_projectionPosView.z - near) / (far - near);
/*
float far=gl_DepthRange.far; float near=gl_DepthRange.near;

vec4 eye_space_pos = gl_ModelViewMatrix * /*something*/
vec4 clip_space_pos = gl_ProjectionMatrix * eye_space_pos;

float ndc_depth = clip_space_pos.z / clip_space_pos.w;

float depth = (((far-near) * ndc_depth) + near + far) / 2.0;
gl_FragDepth = depth;
*/
}
