#version 330 core

layout(location = 0) in vec3 vertexPosition;

out vec2 UV;

void main()
{
    gl_Position = vec4(vertexPosition, 1.0f);

    UV = (vertexPosition.xy + vec2(1.0f, 1.0f)) / 2.0f;

}
