#version 330 core

in vec2 UV;

layout(location = 0) out vec4 color;

uniform mat4 projectionMatInv;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

const float f_diff_x = 0.1f;
const float f_diff_y = 0.1f;

vec3 uvToView(vec2 pos, float z)
{
	vec4 _clipPos = vec4( pos, z, 1.0f );
	vec4 _viewPos = projectionMatInv * _clipPos;

	return _viewPos.xyz;
}

void main()
{
    float _depth = texture(depthTexture, UV).z;
    float _depthMax = 0.999f;
    if(_depth > _depthMax) discard;
//debug
//    color = vec4(texture(colorTexture, UV).xyz, 1.0f);
//    color = vec4(10*_depth * vec3(1,1,1), 1.0f);

    if(UV.x <= 0.5f)
    {
        color = vec4(0,1,0,1);
    }
    else
    {
        color = vec4(1,0,0,1);
    }

    vec3 _posView = uvToView(UV, _depth);

    vec2 _texCoord1 = vec2(UV.x + f_diff_x, UV.y);
    vec2 _texCoord2 = vec2(UV.x - f_diff_x, UV.y);

    vec3 ddx1 = uvToView(_texCoord1, texture(depthTexture, _texCoord1).z ) - _posView;
    vec3 ddx2 = _posView - uvToView(_texCoord2, texture(depthTexture, _texCoord2).z );

    if(abs(ddx1.z) > abs(ddx2.z)) ddx1 = ddx2;

    _texCoord1 = vec2(UV.x, UV.y + f_diff_y);
    _texCoord2 = vec2(UV.x, UV.y - f_diff_y);

    vec3 ddy1 = uvToView(_texCoord1, texture(depthTexture, _texCoord1).z ) - _posView;
    vec3 ddy2 = _posView - uvToView(_texCoord2, texture(depthTexture, _texCoord2).z );

    if(abs(ddy1.z) > abs(ddy2.z)) ddy1 = ddy2;

    vec3 _n = normalize(cross(ddx1, ddy1));

    ///spere rendering
    const vec3  _lightDir = normalize( vec3(-0.5f, 1.5f, 5.0f) );
    float _diffuse = max(0.0f, dot(_n, _lightDir));

    color       = vec4(texture(colorTexture, UV).rgb, 1.0f);
    color.rgb  *= _diffuse;

}
