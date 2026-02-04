#version 330 core
in vec3 vDir;
out vec4 FragColor;

uniform sampler2D uEquirect;

const vec2 invAtan = vec2(0.15915494, 0.318309886); 

vec2 dirToEquirectUV(vec3 d){
    d = normalize(d);
    float u = atan(d.z, d.x);
    float v = asin(clamp(d.y, -1.0, 1.0));
    u = u * invAtan.x + 0.5;
    v = v * invAtan.y + 0.5;
    return vec2(u, v);
}

void main(){
    vec2 uv = dirToEquirectUV(vDir);
    vec3 hdr = texture(uEquirect, uv).rgb;
    FragColor = vec4(hdr, 1.0);
}
