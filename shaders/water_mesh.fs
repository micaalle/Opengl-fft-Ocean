#version 330 core

in vec3  vWorldPos;
in vec3  vNormal;
in vec2  vWorldXZ;
in float vCrest;
in float vBreak;
in float vCurv;

out vec4 FragColor;

uniform vec3  uCameraPos;
uniform float uTime;
uniform float uDayNight;
uniform int   uDebug;

uniform sampler2D uFFT;
uniform float uPatchSize;
uniform float uHeightScale;
uniform float uChoppy;

uniform sampler2D uEnvHDR;
uniform float     uEnvExposure;
uniform float     uEnvMaxMip;

#define PI 3.14159265359
#define FREQ_SIZE 256

const vec3  SUN_DIR          = normalize(vec3(0.60, 0.35, -0.72));
const float FOG_DENSITY      = 0.0005;

// color
const float FINAL_EXPOSURE   = 1.08; 
const float FINAL_SAT        = 1.0;  
const float FINAL_VIBRANCE   = 0.28; 

// Foam / spray
const float FOAM_INTENSITY   = 1.22;
const float FOAM_MAX_DIST    = 440.0;

const float SPRAY_STRENGTH   = 1.05;  
const float SPRAY_BRIGHT     = 2.70;  
const float SPRAY_MAX_DIST   = 560.0;


const float REFL_BOOST_BASE  = 0.86;
const float GLITTER_STRENGTH = 0.34;


const float ENV_CLAMP        = 11.0;
const float SOFTCLIP_MAX     = 9.0;

// ------------------------------------------------------------------

float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3 x){ return clamp(x, vec3(0.0), vec3(1.0)); }
float luminance(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

mat2 rot2(float a){
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

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
float fbm(vec2 p){
    float s = 0.0;
    float a = 0.5;
    float f = 1.0;
    for(int i=0;i<5;i++){
        s += a * valueNoise(p*f);
        f *= 2.0;
        a *= 0.5;
    }
    return s;
}
float ridgedFbm(vec2 p){
    float s = 0.0;
    float a = 0.55;
    float f = 1.0;
    for(int i=0;i<5;i++){
        float n = valueNoise(p*f);
        n = 1.0 - abs(n*2.0 - 1.0);
        s += a * (n*n);
        f *= 2.0;
        a *= 0.5;
    }
    return s;
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
vec2 warpUV(vec2 worldXZ){
    float w1 = valueNoise(worldXZ * 0.005);
    float w2 = valueNoise(worldXZ * 0.005 + 21.3);
    return vec2(w1, w2) * 0.05;
}

vec3 detailNormalFFT(vec2 worldXZ, float stepW)
{
    vec2 uv = (worldXZ / uPatchSize) + warpUV(worldXZ);

    float hC = sampleTileBilinear(0, uv).x * uHeightScale;
    float hX = sampleTileBilinear(0, uv + vec2(stepW / uPatchSize, 0.0)).x * uHeightScale;
    float hZ = sampleTileBilinear(0, uv + vec2(0.0, stepW / uPatchSize)).x * uHeightScale;

    float dhdx = (hX - hC) / max(stepW, 1e-4);
    float dhdz = (hZ - hC) / max(stepW, 1e-4);

    return normalize(vec3(-dhdx, 1.0, -dhdz));
}

vec2 dispFFT(vec2 worldXZ)
{
    vec2 uv = (worldXZ / uPatchSize) + warpUV(worldXZ);
    float dx = sampleTileBilinear(1, uv).x * uChoppy;
    float dz = sampleTileBilinear(2, uv).x * uChoppy;
    return vec2(dx, dz);
}

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
float D_GGX_Aniso(float NoH, float ToH, float BoH, float ax, float ay)
{
    float a2 = ax*ay;
    vec3  v  = vec3(ToH/ax, BoH/ay, NoH);
    float d  = dot(v, v);
    return 1.0 / max(PI * a2 * d * d, 1e-6);
}

// kill the white streaks using wind but rm later when i add foam that actually looks good
// also maybe change because the streams are kinda jsut gone
vec2 variedWind(vec2 base, vec2 worldXZ, float seed)
{
    float n   = valueNoise(worldXZ * 0.00085 + vec2(7.3 + seed, 2.1 - seed));
    float ang = (n * 2.0 - 1.0) * 0.75; // +- ~43 degrees
    return normalize(rot2(ang) * base);
}

float foamStreak(vec2 worldXZ, vec2 wind, vec2 adv, float t)
{
    vec2 perp = vec2(-wind.y, wind.x);
    vec2 p = worldXZ + adv;
    vec2 q = vec2(dot(p, wind), dot(p, perp));

    float clumps = ridgedFbm(q * vec2(0.012, 0.060) + vec2(t*0.02, -t*0.01));
    float detail = fbm(q * vec2(0.060, 0.240) + vec2(-t*0.08,  t*0.05));

    float s = smoothstep(0.34, 0.90, clumps) * (0.55 + 0.45*detail);

    s *= (0.78 + 0.22 * valueNoise(q * 0.015 + vec2(13.1, 9.7)));
    return s;
}

vec3 softClip(vec3 x, float m){
    return x / (vec3(1.0) + x / max(m, 1e-3));
}

vec3 tonemapACES(vec3 x){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x*(a*x+b)) / (x*(c*x+d)+e));
}

// I wanted a sea of thieves like look so these are all kinda vibrant which is fine untill you look at it without the skybox
// reflection or if the waves are made really big via spamming ]]]]]]]]], tune this once i fix height issues or add limit!!!!!!!
vec3 applyVibrance(vec3 c, float v)
{
    float l = luminance(c);
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float sat = (mx - mn) / max(mx, 1e-4);
    float k = 1.0 + v * (1.0 - sat);
    return mix(vec3(l), c, clamp(k, 0.0, 2.0));
}

void main()
{
    float night01 = uDayNight;

    vec3 V = normalize(uCameraPos - vWorldPos);
    float distCam = length(uCameraPos - vWorldPos);

    vec3 N_geo = normalize(vNormal);

    float texelW = uPatchSize / float(FREQ_SIZE);
    float stepW  = max(texelW, distCam * 0.0020);
    vec3  N_det  = detailNormalFFT(vWorldXZ, stepW);

    float near01  = saturate(1.0 - distCam / 300.0);
    float grazing = pow(1.0 - saturate(dot(N_geo, V)), 2.0);

    float detailAmt = saturate(0.58 + 0.32*near01 + 0.34*grazing);
    vec3 N = normalize(mix(N_geo, N_det, detailAmt));

    {
        vec2 baseWind = normalize(vec2(1.0, 1.0));
        vec2 w = variedWind(baseWind, vWorldXZ, 5.1);
        vec2 perp = vec2(-w.y, w.x);

        vec2 q = vec2(dot(vWorldXZ, w), dot(vWorldXZ, perp));
        float n1 = valueNoise(q * vec2(0.45, 1.15) + vec2(uTime*1.10, -uTime*0.70)) * 2.0 - 1.0;
        float n2 = valueNoise(q * vec2(0.90, 2.10) + vec2(-uTime*0.80,  uTime*0.60)) * 2.0 - 1.0;

        vec3 microN = normalize(vec3(n1, 1.0, n2));
        float microStrength = 0.08 + 0.09*grazing;
        N = normalize(mix(N, microN, microStrength));
    }

    if(uDebug == 2){
        FragColor = vec4(N*0.5 + 0.5, 1.0);
        return;
    }
    if(uDebug == 1){
        FragColor = vec4(vec3(0.5 + 0.5*sin(vWorldPos.y*10.0)), 1.0);
        return;
    }

    vec3 L = SUN_DIR;
    vec3 H = normalize(L + V);

    float NoV = saturate(dot(N, V));
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    vec3 deepDay      = vec3(0.006, 0.050, 0.140);
    vec3 shallowDay   = vec3(0.040, 0.390, 0.610);
    vec3 deepNight    = vec3(0.004, 0.018, 0.060);
    vec3 shallowNight = vec3(0.014, 0.085, 0.160);

    vec3 deepCol    = mix(deepDay,    deepNight,    night01);
    vec3 shallowCol = mix(shallowDay, shallowNight, night01);

    float slope = saturate(1.0 - N_det.y);

    float thickness = (0.19 / max(NoV, 0.10)) * (0.85 + 0.70*slope);
    vec3 absorb = mix(vec3(4.8, 2.3, 1.1), vec3(6.5, 3.3, 1.7), night01);
    vec3 atten  = exp(-absorb * thickness);

    vec3 volumeCol = shallowCol * atten + deepCol * (vec3(1.0) - atten);

    float wrap = 0.42;
    float NoL_wrap = saturate((NoL + wrap) / (1.0 + wrap));
    float forward  = pow(saturate(dot(-V, L)), 4.0) * (1.0 - night01);

    vec3 scatterTint = mix(vec3(0.10, 0.64, 0.72), vec3(0.05, 0.16, 0.24), night01);

    vec3 body = volumeCol * (0.18 + 0.82*NoL_wrap);
    body += scatterTint * forward * 0.30;

    float crestTint = saturate(vCrest * (0.70 + 0.30*slope) + vCurv*0.35);
    body += vec3(0.03, 0.34, 0.36) * crestTint * 0.85;

    vec3 skyDir = normalize(vec3(0.15, 0.85, 0.05));
    vec3 skyAmb = sampleEnvEquirectLod(skyDir, uEnvMaxMip);
    skyAmb = min(skyAmb, vec3(2.2));
    body += skyAmb * (0.06 + 0.12*(1.0-night01));

    vec3 F0 = vec3(0.020);
    vec3 F  = F_Schlick(VoH, F0);

    float normalVar = saturate(length(fwidth(N)) * 3.0);
    float rough = clamp(0.050 + slope*0.26 + normalVar*0.28, 0.03, 0.95);
    rough += 0.14 * saturate(distCam / 470.0);
    rough = clamp(rough, 0.03, 0.97);

    float envLod = clamp(rough * uEnvMaxMip, 0.0, uEnvMaxMip);
    vec3 R = reflect(-V, N);
    vec3 env = sampleEnvEquirectLod(R, envLod);

    env = min(env, vec3(ENV_CLAMP));
    float envLum = dot(env, vec3(0.2126, 0.7152, 0.0722));
    env = mix(env, vec3(envLum), 0.12);

    float reflBoost = REFL_BOOST_BASE;

    vec3 sunCol = sampleEnvEquirectLod(SUN_DIR, 0.0);
    sunCol = min(sunCol, vec3(40.0));
    sunCol = mix(sunCol, sunCol * 0.16, night01);

    vec2 baseWind = normalize(vec2(1.0, 1.0));
    vec2 wSpec = variedWind(baseWind, vWorldXZ, 0.0);

    vec3 T = normalize(vec3(wSpec.x, 0.0, wSpec.y));
    vec3 B = normalize(cross(N, T));
    T = normalize(cross(B, N));

    float ToH = dot(T, H);
    float BoH = dot(B, H);

    float a = rough*rough;
    float D_iso = D_GGX(NoH, a);

    float ax = max(a * 0.92, 0.03);
    float ay = max(a * 1.08, 0.03);
    float D_an = D_GGX_Aniso(NoH, ToH, BoH, ax, ay);

    float anisoAmt = saturate(grazing * (1.0 - rough) * 1.0) * 0.65; 
    float D = mix(D_iso, D_an, anisoAmt);

    float k = (rough + 1.0); k = (k*k) / 8.0;
    float G = G_Smith(NoV, NoL, k);

    vec3 spec = (D * G) * F / max(4.0*NoV*NoL, 1e-5);

    vec3 col = body * (vec3(1.0) - F) + env * F * reflBoost;
    col += sunCol * spec * NoL * 1.02;

    // fix foam later!!!!
    float curvMask  = smoothstep(0.12, 0.70, vCurv);
    float breakMask = smoothstep(0.15, 0.90, vBreak);

    float foamSpawn = saturate(
        (vCrest * 0.85) +
        (curvMask * 0.55) +
        (breakMask * 0.95)
    );

    foamSpawn *= smoothstep(0.10, 0.75, vCrest + vBreak*0.6);

    vec2 disp = dispFFT(vWorldXZ);

    vec2 wA = variedWind(baseWind, vWorldXZ, 1.3);
    vec2 wB = variedWind(rot2(1.05) * baseWind, vWorldXZ + vec2(500.0, 900.0), -2.1);
    vec2 wC = variedWind(rot2(-0.85) * baseWind, vWorldXZ + vec2(-800.0, 200.0), 4.7);

    vec2 advA = disp * 0.28 + wA * (uTime * 0.30);
    vec2 advB = disp * 0.22 + wB * (uTime * 0.26);
    vec2 advC = disp * 0.18 + wC * (uTime * 0.22);

    float sA = foamStreak(vWorldXZ, wA, advA, uTime);
    float sB = foamStreak(vWorldXZ, wB, advB, uTime);
    float sC = foamStreak(vWorldXZ, wC, advC, uTime);

    float streak = (sA*0.45 + sB*0.35 + sC*0.20);

    float cell = ridgedFbm((vWorldXZ + advA) * 0.035 + vec2(uTime*0.05, -uTime*0.04));
    cell = smoothstep(0.36, 0.96, cell);

    float foam = foamSpawn * streak * (0.72 + 0.28*cell);
    foam *= (1.0 - saturate(distCam / FOAM_MAX_DIST));
    foam = saturate(pow(foam, 1.05));

    rough = mix(rough, 0.93, foam * 0.85);
    reflBoost *= (1.0 - foam * 0.70);

    vec3 foamCol = mix(vec3(1.10, 1.24, 1.34), vec3(0.94, 1.05, 1.18), night01);
    float foamSun = foam * (0.22 + 0.78*NoL_wrap);

    col *= (1.0 - foam * 0.10);

    col = mix(col, foamCol, FOAM_INTENSITY * foam);
    col += foamCol * foamSun * 0.14;

    // attempt at spray, doesnt work well fix next
    float sprayBase = saturate(breakMask * (0.35 + 0.65*vCrest));
    sprayBase *= (1.0 - saturate(distCam / SPRAY_MAX_DIST));
    sprayBase *= (0.40 + 0.60*pow(1.0 - NoV, 1.6));

    vec2 wS = normalize(wA*0.50 + wB*0.35 + wC*0.15);
    vec2 perpS = vec2(-wS.y, wS.x);
    vec2 qS = vec2(dot(vWorldXZ, wS), dot(vWorldXZ, perpS));

    float wisp1 = ridgedFbm(qS * vec2(0.022, 0.120) + vec2(uTime*0.35, -uTime*0.18));
    float wisp2 = ridgedFbm(qS * vec2(0.040, 0.220) + vec2(-uTime*0.22, uTime*0.14));
    float wisp  = smoothstep(0.55, 0.95, wisp1) * (0.60 + 0.40*smoothstep(0.35, 0.90, wisp2));

    float spray = sprayBase * wisp * SPRAY_STRENGTH;

    vec3 sprayCol = mix(vec3(1.28, 1.48, 1.70), vec3(0.98, 1.08, 1.22), night01);
    col += sprayCol * (spray * SPRAY_BRIGHT);

    float g = valueNoise(vWorldXZ * 0.85 + vec2(uTime*1.7, -uTime*1.2));
    g = pow(saturate(g), 18.0);

    float glitter = g * pow(1.0 - NoV, 2.1) * pow(NoL, 1.15) * (1.0 - rough);
    col += sunCol * glitter * GLITTER_STRENGTH;

    float crestSpark = pow(saturate(vCrest * (1.0 - rough)), 2.0) * pow(1.0 - NoV, 2.0);
    col += sunCol * crestSpark * 0.10;

    // adjust the fog its kinda ugly
    float haze = 1.0 - exp(-distCam * FOG_DENSITY);
    vec3 fogDir = normalize(vec3(V.x, max(V.y, 0.15), V.z));
    vec3 fogCol = sampleEnvEquirectLod(fogDir, uEnvMaxMip);
    fogCol = min(fogCol, vec3(2.6));
    fogCol = mix(fogCol, deepCol, 0.55);
    col = mix(col, fogCol, haze);

    col = softClip(col, SOFTCLIP_MAX);

    col *= FINAL_EXPOSURE;

    {
        float l = luminance(col);
        col = mix(vec3(l), col, FINAL_SAT);
        col = applyVibrance(col, FINAL_VIBRANCE);
    }

    vec3 ldr = tonemapACES(col);

    ldr = pow(ldr, vec3(1.0/2.2));

    FragColor = vec4(ldr, 1.0);
}
