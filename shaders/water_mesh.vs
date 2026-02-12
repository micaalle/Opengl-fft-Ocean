#version 330 core

layout(location=0) in vec2 aXZ;
layout(location=1) in vec2 aUV;

uniform mat4 uView;
uniform mat4 uProj;

uniform sampler2D uFFT;
uniform float uPatchSize;
uniform vec2  uWorldOffset;
uniform float uHeightScale;
uniform float uChoppy;
uniform float uTime;

uniform float uSwellAmp;
uniform float uSwellSpeed;

out vec3  vWorldPos;
out vec3  vNormal;
out vec2  vWorldXZ;
out float vCrest;   
out float vBreak;   
out float vCurv;    

#define FREQ_SIZE 256
#define PI 3.14159265359

vec2 sampleTileNearest(int tile, vec2 uv01) {
    uv01 = fract(uv01);
    ivec2 t = ivec2(int(uv01.x * float(FREQ_SIZE)),
                    int(uv01.y * float(FREQ_SIZE)));
    t = clamp(t, ivec2(0), ivec2(FREQ_SIZE-1));
    t.x += tile * FREQ_SIZE;
    return texelFetch(uFFT, t, 0).xy;
}

struct SwellOut { float h; vec2 d; };

SwellOut gerstner(vec2 xz, vec2 dir, float amp, float wl, float spd, float chop)
{
    dir = normalize(dir);
    float k = 2.0 * PI / max(wl, 1e-3);
    float w = spd * k;
    float ph = k * dot(dir, xz) + uTime * w;
    float s = sin(ph);
    float c = cos(ph);

    SwellOut o;
    o.h = amp * s;
    o.d = dir * (amp * chop * c);
    return o;
}

SwellOut swellField(vec2 xz)
{
    float A = uSwellAmp;
    float S = uSwellSpeed;

    SwellOut a = gerstner(xz, vec2( 1.0,  0.25), 1.15*A, 260.0, 0.85*S, 0.75);
    SwellOut b = gerstner(xz, vec2( 0.6,  1.00), 0.85*A, 180.0, 1.05*S, 0.70);
    SwellOut c = gerstner(xz, vec2(-0.25, 1.0), 0.55*A, 120.0, 1.25*S, 0.65);
    SwellOut d = gerstner(xz, vec2( 0.9, -0.7), 0.35*A,  70.0, 1.55*S, 0.55);

    SwellOut o;
    o.h = a.h + b.h + c.h + d.h;
    o.d = a.d + b.d + c.d + d.d;

    return o;
}

vec3 displacedPos(vec2 worldXZ)
{
    vec2 uv = worldXZ / uPatchSize;

    float h  = sampleTileNearest(0, uv).x * uHeightScale;
    float dx = sampleTileNearest(1, uv).x * uChoppy;
    float dz = sampleTileNearest(2, uv).x * uChoppy;

    SwellOut sw = swellField(worldXZ);
    h  += sw.h;
    dx += sw.d.x;
    dz += sw.d.y;

    return vec3(worldXZ.x - dx, h, worldXZ.y - dz);
}

float saturate(float x){ return clamp(x, 0.0, 1.0); }

void main()
{
    vec2 worldXZ = aXZ + uWorldOffset;

    vec3 P  = displacedPos(worldXZ);

    float stepW = uPatchSize / float(FREQ_SIZE);
    vec3 Px = displacedPos(worldXZ + vec2(stepW, 0.0));
    vec3 Pz = displacedPos(worldXZ + vec2(0.0, stepW));

    vec3 N = normalize(cross(Pz - P, Px - P));

    // Jacobian breaking proxy works?? i think :/
    float dXdx = (Px.x - P.x) / stepW;
    float dXdz = (Pz.x - P.x) / stepW;
    float dZdx = (Px.z - P.z) / stepW;
    float dZdz = (Pz.z - P.z) / stepW;

    float J = dXdx * dZdz - dXdz * dZdx; // small/neg => folding => breaking

    float br = smoothstep(0.22, 0.82, (1.0 - J));
    br = max(br, smoothstep(0.0, -0.12, J));
    vBreak = saturate(br);

    float slope = saturate(1.0 - N.y);
    vCrest = smoothstep(0.18, 0.62, slope);

    float curv = abs(Px.y + Pz.y - 2.0 * P.y) / max(stepW, 1e-3);
    vCurv = saturate(curv * 0.65); // tune

    vWorldPos = P;
    vNormal   = N;
    vWorldXZ  = worldXZ;

    gl_Position = uProj * uView * vec4(P, 1.0);
}
