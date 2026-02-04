#version 330 core
layout(location=0) in vec3 aPos;
out vec3 vDir;
uniform mat4 uView;
uniform mat4 uProj;
void main(){
    vDir = aPos;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
