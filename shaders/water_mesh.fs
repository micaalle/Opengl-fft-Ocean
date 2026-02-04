#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vWorldXZ;
out vec4 FragColor;

uniform vec3  uCameraPos;
uniform float uTime;
uniform float uDayNight;
uniform int   uDebug;

uniform sampler2D uFFT;
uniform float uPatchSize;
uniform float uHeightScale;
uniform float uChoppy;

uniform float uSwellAmp;     
uniform float uSwellSpeed; 

uniform sampler2D uEnvHDR;
uniform float     uEnvExposure;
uniform float     uEnvMaxMip;

const vec3 WATER_COLOR_DAY   = vec3(0.012, 0.085, 0.155);
const vec3 WATER_COLOR_NIGHT = vec3(0.004, 0.020, 0.055);

#define PI 3.14159265359
#define FREQ_SIZE 256

const vec3 SUN_DIR = normalize(vec3(0.60, 0.35, -0.72));
const float FOG_DENSITY = 0.00125;

const float FOAM_INTENSITY = 0.18;
const float FOAM_MAX_DIST  = 260.0;

float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3 x){ return clamp(x, vec3(0.0), vec3(1.0)); }

float hash(vec2 p){
    p = 50.0 * fract(p * 0.3183099 + vec2(0.71, 0.113));
    return -1.0 + 2.0 * fract(p.x * p.y * (p.x + p.y));
}
float valueNoise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(
        mix(hash(i+vec2(0.0,0.0)), hash(i+vec2(1.0,0.0)), u.x),
        mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), u.x),
        u.y
    );
}

vec3 tonemapACES(vec3 x){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x*(a*x+b)) / (x*(c*x+d)+e));
}

vec2 envUV(vec3 dir){
    dir = normalize(dir);
    float phi   = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0*PI) + 0.5, theta / PI + 0.5);
}

vec3 sampleEnvEquirectLod(vec3 dir, float lod)
{
    vec2 uv = envUV(dir);
    vec3 hdr = textureLod(uEnvHDR, uv, lod).rgb;
    return hdr * max(uEnvExposure, 0.0);
}

vec2 fetchTile(int tile, ivec2 t){
    t = ivec2((t.x % FREQ_SIZE + FREQ_SIZE) % FREQ_SIZE,
              (t.y % FREQ_SIZE + FREQ_SIZE) % FREQ_SIZE);
    t.x += tile * FREQ_SIZE;
    return texelFetch(uFFT, t, 0).xy;
}
vec2 sampleTileBilinear(int tile, vec2 uv01){
    uv01 = fract(uv01);
    vec2 p = uv01 * float(FREQ_SIZE) - 0.5;
    ivec2 i0 = ivec2(floor(p));
    vec2 f = fract(p);

    vec2 a = fetchTile(tile, i0);
    vec2 b = fetchTile(tile, i0 + ivec2(1,0));
    vec2 c = fetchTile(tile, i0 + ivec2(0,1));
    vec2 d = fetchTile(tile, i0 + ivec2(1,1));

    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}

// hide the repetition looks pretty good enough until waves are really big
vec2 warpUV(vec2 worldXZ){
    float w1 = valueNoise(worldXZ * 0.005);
    float w2 = valueNoise(worldXZ * 0.005 + 21.3);
    return vec2(w1, w2) * 0.05;
}

float swell(vec2 w)
{
    float t = uTime * uSwellSpeed;

    float s1 = sin(dot(w, normalize(vec2(1.0, 0.2))) * 0.0040 + t * 1.2);
    float s2 = sin(dot(w, normalize(vec2(-0.3, 1.0))) * 0.0025 + t * 0.9);
    float s3 = sin(dot(w, normalize(vec2(0.7, -0.7))) * 0.0016 + t * 0.6);

    return (s1 * 0.55 + s2 * 0.35 + s3 * 0.25);
}

float heightAt(vec2 worldXZ){
    vec2 uv = (worldXZ / uPatchSize) + warpUV(worldXZ);
    float fftH = sampleTileBilinear(0, uv).x * uHeightScale;

    fftH += swell(worldXZ) * uSwellAmp;

    return fftH;
}
vec2 dispXZAt(vec2 worldXZ){
    vec2 uv = (worldXZ / uPatchSize) + warpUV(worldXZ);
    float dx = sampleTileBilinear(1, uv).x * uChoppy;
    float dz = sampleTileBilinear(2, uv).x * uChoppy;
    return vec2(dx, dz);
}
vec3 displacedPos(vec2 worldXZ){
    float h = heightAt(worldXZ);
    vec2 d  = dispXZAt(worldXZ);
    return vec3(worldXZ.x - d.x, h, worldXZ.y - d.y);
}
vec3 normalFromFFT(vec2 worldXZ, float stepW){
    vec3 P  = displacedPos(worldXZ);
    vec3 Px = displacedPos(worldXZ + vec2(stepW, 0.0));
    vec3 Pz = displacedPos(worldXZ + vec2(0.0, stepW));
    return normalize(cross(Pz - P, Px - P));
}

// FIX LATER FOR FOAM!!!!!!!!!!
float curvatureProxy(vec2 worldXZ, float stepW){
    float hC = heightAt(worldXZ);
    float hL = heightAt(worldXZ - vec2(stepW, 0.0));
    float hR = heightAt(worldXZ + vec2(stepW, 0.0));
    float hD = heightAt(worldXZ - vec2(0.0, stepW));
    float hU = heightAt(worldXZ + vec2(0.0, stepW));
    float lap = (hL + hR + hD + hU - 4.0*hC) / max(stepW*stepW, 1e-6);
    return abs(lap);
}

//a bunch of cool shader approx functions that looks somewhat good 
// maybe look into better ones on shaderlab!!!
float D_GGX(float NoH, float a){
    float a2 = a*a;
    float d = (NoH*NoH)*(a2-1.0) + 1.0;
    return a2 / max(PI*d*d, 1e-6);
}
float G_Schlick(float NoV, float k){
    return NoV / (NoV*(1.0-k) + k);
}
float G_Smith(float NoV, float NoL, float k){
    return G_Schlick(NoV,k) * G_Schlick(NoL,k);
}
vec3 F_Schlick(float VoH, vec3 F0){
    return F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
}

void main(){
    float night01 = uDayNight;

    vec3 V = normalize(uCameraPos - vWorldPos);
    float distCam = length(uCameraPos - vWorldPos);

    float baseStep = uPatchSize / float(FREQ_SIZE);

    float stepBig   = max(baseStep * 2.0,  distCam * 0.0015);
    float stepSmall = max(baseStep * 0.75, distCam * 0.0007);

    vec3 N_geo   = normalize(vNormal);
    vec3 N_big   = normalFromFFT(vWorldXZ, stepBig);
    vec3 N_small = normalFromFFT(vWorldXZ, stepSmall);

    float near01  = saturate(1.0 - distCam / 200.0);
    float grazing = pow(1.0 - saturate(dot(N_geo, V)), 2.0);
    float microAmt = saturate(0.30 + 0.50*near01 + 0.35*grazing);

    vec3 N_fft = normalize(mix(N_big, N_small, microAmt));
    vec3 N = normalize(mix(N_geo, N_fft, 0.92));

    vec2 wdir = normalize(vec2(1.0, 1.0));
    vec2 mP = vWorldXZ * 0.45 + wdir * (uTime * 1.6);
    float n1 = valueNoise(mP) * 2.0 - 1.0;
    float n2 = valueNoise(mP * 1.9 + 13.7) * 2.0 - 1.0;
    vec3 microN = normalize(vec3(n1, 1.0, n2));
    float microStrength = 0.10 + 0.10 * grazing;
    N = normalize(mix(N, microN, microStrength));

    if(uDebug == 2){
        FragColor = vec4(N*0.5 + 0.5, 1.0);
        return;
    }
    if(uDebug == 1){
        float h = vWorldPos.y;
        FragColor = vec4(vec3(0.5 + 0.5*sin(h*10.0)), 1.0);
        return;
    }

    vec3 L = SUN_DIR;
    vec3 H = normalize(L + V);

    float NoV = saturate(dot(N, V));
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    vec3 waterBase = mix(WATER_COLOR_DAY, WATER_COLOR_NIGHT, night01);

    float subsurface = pow(1.0 - NoV, 2.2);
    subsurface *= (0.25 + 0.75 * saturate(1.0 - N.y));
    vec3 hazeCol = mix(vec3(0.10, 0.30, 0.45), vec3(0.05, 0.10, 0.20), night01);
    vec3 body = waterBase + hazeCol * subsurface * 0.18;

    vec3 F0 = vec3(0.020);
    vec3 F  = F_Schlick(VoH, F0);

    // roughness
    float slope = saturate(1.0 - N.y);
    float normalVar = saturate(length(fwidth(N)) * 3.0);
    float rough = clamp(0.06 + slope*0.16 + normalVar*0.28, 0.03, 0.92);
    rough += 0.18 * saturate(distCam / 320.0);
    rough = clamp(rough, 0.03, 0.95);

    float envLod = clamp(rough * uEnvMaxMip, 0.0, uEnvMaxMip);

    vec3 R = reflect(-V, N);
    vec3 env = sampleEnvEquirectLod(R, envLod);

    // avoid the super bright sparkles from my hdr
    env = min(env, vec3(8.0));
    float envLum = dot(env, vec3(0.2126, 0.7152, 0.0722));
    env = mix(env, vec3(envLum), 0.22);

    float reflBoost = 0.75;

    float a = rough*rough;
    float D = D_GGX(NoH, a);
    float k = (rough + 1.0); k = (k*k) / 8.0;
    float G = G_Smith(NoV, NoL, k);
    vec3  spec = (D * G) * F / max(4.0*NoV*NoL, 1e-5);

    vec3 sunCol = sampleEnvEquirectLod(SUN_DIR, 0.0);
    sunCol = min(sunCol, vec3(30.0));
    sunCol = mix(sunCol, sunCol * 0.18, night01);

    vec3 col = body * (vec3(1.0) - F) + env * F * reflBoost;
    col += sunCol * spec * NoL * 0.85;

    // FOAM barely works FIX!!!!!!!!!!!
    vec2 disp = dispXZAt(vWorldXZ);
    vec2 adv  = disp * 0.35 + wdir * (uTime * 0.25);

    float curv = curvatureProxy(vWorldXZ, baseStep*2.0);
    float curvMask = smoothstep(0.003, 0.016, curv);

    float crestMask = smoothstep(0.18, 0.58, slope);
    float whitecapBase = saturate(crestMask * 0.55 + curvMask * 0.95);

    vec2 p = (vWorldXZ + adv);
    float foamNoise  = valueNoise(p * 0.06 + vec2(uTime*0.10, -uTime*0.07)) * 0.6 + 0.4;
    float foamDetail = valueNoise(p * 0.18 + vec2(-uTime*0.05, uTime*0.04)) * 0.5 + 0.5;

    float foam = whitecapBase * foamNoise * foamDetail;
    foam *= (1.0 - saturate(distCam / FOAM_MAX_DIST));
    foam = saturate(pow(foam, 1.45));

    vec3 foamCol = mix(vec3(1.0), vec3(0.90, 0.96, 1.0), 0.35);
    col = mix(col, foamCol, foam * FOAM_INTENSITY);

    // fog barely works FIX!!!!!!!!!
    float haze = 1.0 - exp(-distCam * FOG_DENSITY);
    vec3 fogCol = mix(env, waterBase, 0.85);
    col = mix(col, fogCol, haze);

    col = tonemapACES(col);
    col = pow(col, vec3(1.0/2.2));

    FragColor = vec4(col, 1.0);
}
