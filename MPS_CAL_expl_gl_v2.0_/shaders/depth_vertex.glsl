#version 330 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in uint vertexType;

out vec3 posView;
out vec4 fragmentColor;

uniform mat4 projectionMat;
uniform mat4 viewModelMat;
uniform float pointRadius;
uniform float pointScale;

void main()
{
    posView = vec3( viewModelMat * vec4(vertexPosition, 1.0f) );

    float _dist = length(posView);

    gl_PointSize = pointRadius * (pointScale / _dist);

    gl_Position = projectionMat * vec4( posView, 1.0f );

    fragmentColor = vec4(0.0f, 0.5f, 0.0f, 1.0f);

    if(vertexType > 0.0f) fragmentColor.a = 0.0f;
}

