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
in mat4 fMvp;

out vec4 color;

uniform sampler2D frontFaceTex;
uniform sampler2D dirTex;

uniform sampler3D volNormTex;
uniform sampler3D volDataTex;
uniform sampler2D bgTex;
//uniform sampler3D cellInFluidTex;

uniform mat4 fMv;
uniform vec3 pMin;
uniform vec3 pMax;

vec2 uv     = (mvpPos.xy/*/mvpPos.w*/ + vec2(1.f, 1.f) ) / 2.f;
vec3 f_v    = pMax - pMin;
/*
vec3 sunsetOrg = vec3( 0.9961f, 0.3569f, 0.2078f);
vec3 skyBlue   = vec3( 0.5294f, 0.8078f, 0.9216f );
*/
const vec3 lightColor              = vec3( 0.1f, 0.1f, 0.1f );
const vec3 lightDir                = normalize( vec3(0.f, 0.5f, -1.f) );
const vec3 eyeDir                  = normalize( vec3(0.f, 0.f, 1.f) );
const vec3 materialColor           = vec3( 1.f, 1.f, 1.f );
const vec3 materialAmbientColor    = vec3( 0.3f, 0.3f, 0.3f ) * materialColor;
//const vec3 materialSpecularColor = vec3( 0.3f, 0.3f, 0.3f );
const vec3 BBoxColor               = vec3( 0.95f, 0.95f, 0.95f);
const float waterOpacity           = 2.f;
const float boundOpacity           = 0.05f;

const float dt         = 0.002f;
const float iso        = 8.f;
const int   newton     = 5;
const float idRafAir   = 1.00f;
const float idRafWater = 1.10f;
float eta = idRafAir / idRafWater;
const int   maxRef     = 3;
const float beta       = 0.1f;

vec3      str(const vec3 p);
float     interpS(const vec3 p);
vec3      interpN(const vec3 p);
//float     interpC(const vec3 p);
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

    int count = 0;
    float range = distance(pMax, pMin);
    while(t < range && count < maxRef)
    {
        if(color.a >= 1.f) return;
        if(!inAir)
        {
            color.a   += (1 - color.a) * waterOpacity * dt;
        }

        vec3 _pt1 = origin + t * dir;
        vec3 _pt2 = origin + (t + dt) * dir;

        if(_pt2.x < pMin.x || _pt2.x > pMax.x || _pt2.y < pMin.y || _pt2.y > pMax.y || _pt2.z < pMin.z || _pt2.z > pMax.z) break;

        float _s1 = interpS(_pt1) - iso;
        float _s2 = interpS(_pt2) - iso;

        if( _s1 * _s2 < 0.f )
        {
            count++;

            for(int i=0; i<newton; i++)
            {
                vec3 _pt3 = (_pt1 + _pt2) / 2.f;

                float _s3 = interpS(_pt3) - iso;

                if(_s1 * _s3 < 0.f) _pt2 = _pt3;
                else _pt1 = _pt3;
            }

            surface = (_pt1 + _pt2) / 2.f;

            vec3 norm    = normalize(interpN(surface));

            color.rgb += (1 - color.a) * intercept(norm, inAir);
            color.a   += (1 - color.a) * boundOpacity;

            origin = _pt2;
            t = 0.f;

            vec3  dirTmp = dir;
            if(inAir) dirTmp = refract(dir, norm, eta);
            else      dirTmp = refract(dir, -norm, 1.f/eta);

            if(length(dirTmp) == 0.f)
            {
                dirTmp = reflect(dir, -norm);
            }
            else
            {
                color.a   += (1 - color.a) * boundOpacity;
                inAir = !inAir;
            }
            dir = dirTmp;
        }

        //else if(_pt1.y > (pMax.y + pMin.y)/2.f) break;

        t += dt;
    }

    if(count == 0)
    {
        color.rgb += (1 - color.a) * texture(bgTex, uv).rgb;
        color.a = 1.f;
    }
    else
    {
        vec3 mvDir = normalize( vec3(fMv * vec4(dir, 0.f)) );
        vec2 offsetUv = uv - beta * vec2(dot(mvDir, vec3(1.f, 0.f, 0.f)), dot(mvDir, vec3(0.f, 1.f, 0.f)));

        color.rgb += (1 - color.a) * texture(bgTex, offsetUv).rgb * materialColor;
        color.a = 1.f;
    }
/*
    if(count == 0)
    {
        color = vec4(BBoxColor, 0.1f);
    }
    else  //inAir || inWater
    {
        color.rgb += (1 - color.a) * phong2( vec3(fMv*vec4(dir, 0.f)) );
        color.a = 1.f;
    }
*/
    //color = vec4(texture(dirTex, uv).a * 3.f, 0, 0, 1.f);
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
    //return lightColor * ( (_cosTheta * materialColor) + (pow(_cosBeta, 2) * materialColor) );
    return lightColor * pow(_cosBeta, 10);
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
/*
float interpC(const vec3 p)
{
    return texture(cellInFluidTex, str(p)).r;
}
*/
