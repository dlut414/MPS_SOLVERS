/*
LICENCE
*/
#version 330 core

layout(location = 0) in vec3 v3_vertexPos;
layout(location = 1) in vec3 v3_norm_world;
layout(location = 2) in float f_alpha;

out vec3 v3_norm_view;
out float f_alpha_;

uniform mat4 projectionMat;
uniform mat4 viewModelMat;

void main()
{
    gl_Position = projectionMat * viewModelMat * vec4(v3_vertexPos, 1.0f);

    v3_norm_view = vec3( viewModelMat * vec4(v3_norm_world, 0.0f) );

    f_alpha_ = f_alpha * 1.0f;
}

