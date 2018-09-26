#version 330 core

in vec3 v3_norm_view;
in float f_alpha_;

out vec4 color;

void main()
{
    if(f_alpha_ < 0.01f) discard;
    //const vec4 _lightColor = vec4( 0.0f, 0.5f, 0.0f, f_alpha_color );
    const vec3 _lightDir = normalize( vec3(3.0f, 3.0f, 3.0f) );
    const vec3 _eyeDir = normalize( vec3(0.0f, 0.0f, 1.0f) );
    const vec3 _materialColor = vec3( 0.0f, 0.7f, 0.0f );
    const vec3 _materialAmbientColor = vec3( 0.2f, 0.2f, 0.2f ) * _materialColor;
    //const vec3 _materialSpecularColor = vec3( 0.3f, 0.3f, 0.3f );

    vec3  _n = normalize( v3_norm_view );

    //clamp(dot(_n, _lightDir), 0, 1);
    float _cosTheta = max( 0.0f, abs(dot(_n, _lightDir)) );
    float _cosBeta = max( 0.0f, dot(reflect(-_lightDir, _n), _eyeDir) );

    color = vec4( _materialAmbientColor + (_cosTheta * _materialColor) + (pow(_cosBeta, 2.0f) * _materialColor), f_alpha_ );
    //color = vec4( _materialAmbientColor + (_cosTheta * _materialColor), f_alpha_ );
}
