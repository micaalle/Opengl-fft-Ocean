#version 330 core
in vec3 vDir;
out vec4 FragColor;

uniform samplerCube uEnv;
uniform float uExposure;

vec3 tonemapACES(vec3 x){
    float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main(){
    vec3 col = texture(uEnv, normalize(vDir)).rgb * uExposure;
    col = tonemapACES(col);
    col = pow(col, vec3(1.0/2.2));
    FragColor = vec4(col, 1.0);
}
