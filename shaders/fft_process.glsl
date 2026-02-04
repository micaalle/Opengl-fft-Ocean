#version 330 core
out vec4 FragColor;
in vec2 vTex;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform vec2 iResolution;

uniform int uPass; 

#define PI 3.14159265359
#define FREQ_SIZE 256
#define DISPLAYMENT_X 1
#define DISPLAYMENT_Z 2

vec2 ComplexMult(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 ComplexExp(float theta){ return vec2(cos(theta), sin(theta)); }

void main(){
    ivec2 uv = ivec2(vTex * iResolution);

    if(uv.x >= 3*FREQ_SIZE || uv.y >= FREQ_SIZE){
        FragColor = vec4(0.0);
        return;
    }

    int index  = uv.x / FREQ_SIZE;
    int localX = uv.x - index * FREQ_SIZE;

    float N = float(FREQ_SIZE);
    float TwoPi = 2.0*PI;

    vec2 sum = vec2(0.0);

    if(uPass == 0)
    {
        // pass 0
        for(int i=0;i<FREQ_SIZE;i++){
            float kx = float(i) - 0.5*N;

            vec2 h_k = texelFetch(iChannel0, ivec2(i, uv.y), 0).xy;

            if(index >= DISPLAYMENT_X){
                float ky = float(uv.y) - 0.5*N;
                float len = length(vec2(kx, ky));
                float inv_len = (len > 1e-6 ? 1.0/len : 0.0);

                float k1 = (index >= DISPLAYMENT_Z ? ky : kx);

                h_k = vec2(h_k.y, -h_k.x) * (k1 * inv_len);

                float shortKill = smoothstep(0.0, 0.12, inv_len);
                h_k *= shortKill;
            }

            float angleH = TwoPi * kx * float(localX) / N;
            sum += ComplexMult(h_k, ComplexExp(angleH));
        }

        FragColor = vec4(0.0, 0.0, sum.x, sum.y);
        return;
    }
    else
    {
        // pass 1
        for(int i=0;i<FREQ_SIZE;i++){
            float ky = float(i) - 0.5*N;

            vec2 h_prev = texelFetch(iChannel1, ivec2(uv.x, i), 0).zw;
            float angleV = TwoPi * ky * float(uv.y) / N;
            sum += ComplexMult(h_prev, ComplexExp(angleV));
        }

        // nomralize
        vec2 finalVal = (sum / (N*N)) * 25.0; // was 35; start lower now that transform is correct

        FragColor = vec4(finalVal.xy, 0.0, 0.0);
        return;
    }
}
