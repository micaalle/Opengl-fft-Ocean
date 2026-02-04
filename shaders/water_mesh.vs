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

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vWorldXZ;

#define FREQ_SIZE 256
#define PI 3.14159265359

vec2 sampleTile(int tile, vec2 uv01) {
    uv01 = fract(uv01);
    ivec2 t = ivec2(int(uv01.x * float(FREQ_SIZE)),
                    int(uv01.y * float(FREQ_SIZE)));
    t = clamp(t, ivec2(0), ivec2(FREQ_SIZE-1));
    t.x += tile * FREQ_SIZE;
    return texelFetch(uFFT, t, 0).xy;
}

// help normalize across tiles 
float swell(vec2 w)
{
    float t = uTime * uSwellSpeed;

    float s1 = sin(dot(w, normalize(vec2(1.0, 0.2))) * 0.0040 + t * 1.2);
    float s2 = sin(dot(w, normalize(vec2(-0.3, 1.0))) * 0.0025 + t * 0.9);
    float s3 = sin(dot(w, normalize(vec2(0.7, -0.7))) * 0.0016 + t * 0.6);

    return (s1 * 0.55 + s2 * 0.35 + s3 * 0.25);
}

vec3 displacedPos(vec2 worldXZ){
    vec2 uv = worldXZ / uPatchSize;

    float h  = sampleTile(0, uv).x * uHeightScale;
    float dx = sampleTile(1, uv).x * uChoppy;
    float dz = sampleTile(2, uv).x * uChoppy;

    // NEW: add long swells to height (continuous across world)
    h += swell(worldXZ) * uSwellAmp;

    return vec3(worldXZ.x - dx, h, worldXZ.y - dz);
}

void main() {
    vec2 worldXZ = aXZ + uWorldOffset;

    vec3 P  = displacedPos(worldXZ);

    float stepW = uPatchSize / float(FREQ_SIZE);
    vec3 Px = displacedPos(worldXZ + vec2(stepW, 0.0));
    vec3 Pz = displacedPos(worldXZ + vec2(0.0, stepW));

    vec3 N = normalize(cross(Pz - P, Px - P));

    vWorldPos = P;
    vNormal   = N;
    vWorldXZ  = worldXZ;

    gl_Position = uProj * uView * vec4(P, 1.0);
}
