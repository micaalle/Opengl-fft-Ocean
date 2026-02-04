#version 330 core
layout(location=0) in vec3 aPos;
out vec3 vDir;
uniform mat4 uViewNoTrans;
uniform mat4 uProj;
void main(){
    vDir = aPos;
    vec4 p = uProj * uViewNoTrans * vec4(aPos, 1.0);
    gl_Position = p.xyww;
}
