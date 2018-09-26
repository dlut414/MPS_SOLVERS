/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#version 330 core

in vec4 mvpPos;
in mat4 fMvp;

out vec4 color;

uniform sampler2D frontFaceTex;
uniform sampler2D dirTex;
uniform sampler3D volPosTex;
uniform sampler3D volNormTex;
uniform sampler3D volDataTex;
uniform sampler2D bgTex;

uniform mat4 fMv;
uniform vec3 pMin;
uniform vec3 pMax;

vec2 uv     = (mvpPos.xy/mvpPos.w + vec2(1.f, 1.f) ) / 2.f;
vec3 f_v    = pMax - pMin;
/*
vec3 sunsetOrg = vec3( 0.9961f, 0.3569f, 0.2078f);
vec3 skyBlue   = vec3( 0.5294f, 0.8078f, 0.9216f );
*/
const vec3 lightColor              = vec3( 0.9f, 0.9f, 0.9f);
const vec3 lightDir                = normalize( vec3(0.f, 0.f, -1.f) );
const vec3 eyeDir                  = normalize( vec3(0.f, 0.f, 1.f) );
const vec3 materialColor           = vec3( 0.5294f, 0.8078f, 0.9216f );
const vec3 materialAmbientColor    = vec3( 0.3f, 0.3f, 0.3f ) * materialColor;
//const vec3 materialSpecularColor = vec3( 0.3f, 0.3f, 0.3f );
const vec3 BBoxColor               = vec3( 0.95f, 0.95f, 0.95f);
const float waterOpacity           = 0.8f;
const float boundOpacity           = 0.05f;

const float dt         = 0.0002f;
const float iso        = 6.f;
const float velMax     = 6.f;
const int   newton     = 5;
const float idRafAir   = 1.00f;
const float idRafWater = 1.10f;
float eta = idRafAir / idRafWater;
const int   maxRef     = 5;
const float beta       = 0.1f;

vec3      str(const vec3 p);
float     interpS(const vec3 p);
vec3      interpN(const vec3 p);
vec3      interpV(const vec3 p);
vec3      phong1(const vec3 n);
vec3      phong2(const vec3 eyeDir);
vec3      intercept(const vec3 p, const bool flag);

void main()
{
    vec3  origin  = texture(frontFaceTex, uv).rgb;
    vec3  dir     = texture(dirTex, uv).rgb;
    float end     = texture(dirTex, uv).a;
    float t       = 0.f;
    vec3  surface = origin;
    bool  inAir = true;

    color = vec4(0.f, 0.f, 0.f, 0.f);

    if(end < dt)
    {
        discard;
    }

    while(t < end)
    {
        if(color.a >= 1.f) {break;}

        vec3 _pt1 = origin + t * dir;
        vec3 _pt2 = origin + (t + dt) * dir;

        float _s1 = interpS(_pt1) - iso;
        float _s2 = interpS(_pt2) - iso;

        float _s = length(interpV(_pt2)) / velMax;
        if(!inAir)
        {
            if( _s > 0.f && _s <= 0.5f)
            {
                color.rgb += (1 - color.a) * 2 * ( _s*vec3(0.f, 1.f, 0.f) + (0.5f-_s)*vec3(0.f, 0.f, 1.f) );
            }
            else if(_s > 0.5f && _s < 1.f)
            {
                color.rgb += (1 - color.a) * 2 * ( (_s-0.5f)*vec3(1.f, 0.f, 0.f) + (1.f-_s)*vec3(0.f, 1.f, 0.f) );
            }
            color.a   += (1 - color.a) * waterOpacity;
        }

        if( _s1 * _s2 >= 0.f )
        {
            t += dt;
            continue;
        }
        else
        {
            inAir = !inAir;
            t += dt;
        }
    }
}

vec3 intercept(const vec3 n, const bool flag)
{
    if(flag)
    {
        return phong1( vec3( fMv * vec4(n, 0.f) ) );
    }
    else return vec3(0.f, 0.f, 0.f);
}

vec3 phong1(const vec3 n)
{
    vec3  _n = normalize( n );
    float _cosTheta = max( 0.0f, dot(_n, lightDir) );
    float _cosBeta = max( 0.0f, dot(reflect(-lightDir, _n), eyeDir) );

    //return materialAmbientColor + (_cosTheta * materialColor) + (pow(_cosBeta, 5) * materialColor);
    //return materialAmbientColor + (_cosTheta * materialColor);
    //return materialColor;
    return lightColor * ( (_cosTheta * materialColor) + (pow(_cosBeta, 2) * materialColor) );
}

vec3 phong2(const vec3 rayDir)
{
    vec3 _n = normalize( rayDir );
    float _cosTheta = max( 0.0f, dot(_n, lightDir) );

    return _cosTheta * lightColor * materialColor;
}

vec3 str(const vec3 p)
{
    return (p - pMin) / f_v;
}

float interpS(const vec3 p)
{
    return texture(volDataTex, str(p)).r;
}

vec3 interpN(const vec3 p)
{
    return texture(volNormTex, str(p)).xyz;
}

vec3 interpV(const vec3 p)
{
    return texture(volPosTex, str(p)).xyz;
}
