#version 330 core
out vec4 FragColor;
in vec2 vTex;

uniform float iTime;
uniform vec2 iResolution;

#define PI 3.14159265359
#define G 9.81
#define FREQ_SIZE 256

const float Amp       = 8.0;
const vec2  WindDir   = normalize(vec2(1.0, 1.0));
const float WindSpeed = 16.0;

const float K_SCALE   = 1.6;

vec2 ComplexMult(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 ComplexExp(float theta){ return vec2(cos(theta), sin(theta)); }

// some noise 
vec2 hash22(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031,0.1030,0.0973));
    p3 += dot(p3,p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

// convert the random to gauss rand
vec2 gaussian(vec2 n){
    vec2 u = hash22(n);
    u = clamp(u, vec2(1e-6), vec2(1.0));
    float r = sqrt(-2.0 * log(u.x));
    float a = 2.0 * PI * u.y;
    return r * vec2(cos(a), sin(a));
}

float dirSpread(vec2 k, vec2 wdir, float s)
{
    float klen = length(k);
    if(klen < 1e-6) return 0.0;
    float c = max(dot(normalize(k), wdir), 0.0);
    return pow(c, s);
}

// adjusted philips that mostly works still need to tweak speed!!!!!!
float phillipsBand(vec2 k, vec2 wdir, float wSpeed, float amp, float hfDamp, float spreadPow)
{
    float klen = length(k);
    if(klen < 1e-6) return 0.0;

    float k2 = klen*klen;
    float k4 = k2*k2;

    float L  = (wSpeed*wSpeed) / G;
    float L2 = L*L;

    // base Phillips
    float P = amp * exp(-1.0/(k2*L2)) / max(k4, 1e-12);

    P *= dirSpread(k, wdir, spreadPow);

    // damp against wind!!
    if(dot(k, wdir) < 0.0) P *= 0.15;

    P *= exp(-k2 * hfDamp);

    return P;
}

void main(){
    ivec2 uv = ivec2(vTex * iResolution);
    if(uv.x >= FREQ_SIZE || uv.y >= FREQ_SIZE){
        FragColor = vec4(0.0);
        return;
    }

    float N = float(FREQ_SIZE);
    vec2 n = vec2(uv) - 0.5*N;

    vec2 K = (2.0*PI*n/N) * K_SCALE;

    // phase
    float omega = sqrt(max(0.0, length(K) * G));
    float t = iTime * 2.0;
    vec2 cs = vec2(cos(omega*t), sin(omega*t));

    // 3 band ocean
    vec2 w0 = normalize(WindDir);
    vec2 w1 = normalize(vec2(WindDir.x*0.85 - WindDir.y*0.35, WindDir.x*0.35 + WindDir.y*0.85)); // ~20°
    vec2 w2 = normalize(vec2(WindDir.x*0.70 + WindDir.y*0.70, -WindDir.x*0.70 + WindDir.y*0.70)); // ~45°

    float P0 = phillipsBand(K, w0, WindSpeed*1.05, Amp*1.00, 0.00055, 6.0); // swell
    float P1 = phillipsBand(K, w1, WindSpeed*0.75, Amp*0.55, 0.00085, 3.5); // mid
    float P2 = phillipsBand(K, w2, WindSpeed*0.45, Amp*0.22, 0.00140, 1.6); // short

    float P  = P0 + P1 + P2;

    // h0(k) and h0(-k)
    vec2 xi  = gaussian(n);
    vec2 xi2 = gaussian(-n);

    vec2 h0  = xi  * sqrt(P  * 0.5);

    vec2 h0m = xi2 * sqrt(P * 0.5);

    vec2 h  = ComplexMult(h0, cs);
    vec2 hm = ComplexMult(vec2(h0m.x, -h0m.y), vec2(cs.x, -cs.y));

    FragColor = vec4(h + hm, 0.0, 1.0);
}
