#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <array>
#include <algorithm>

#define STBI_NO_SIMD
#include "stb_image.h"

namespace fs = std::filesystem;

// -----------------------------
// Camera globals
// -----------------------------
glm::vec3 cameraPos = glm::vec3(0.0f, 1.5f, 5.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;
float fov = 45.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// parms for live tweaks 
float gHeightScale = 14.0f;
float gChoppy = 6.0f;
float gSwellAmp = 1.5f;
float gSwellSpeed = 0.25f;

// Day/night toggle : kinda redundent now that ive added the skybox
float dayNight = 0.0f;

bool wireframe = true;

int shaderDebug = 0; // 1 is still super scuffed FIX

// NEW: Infinite ocean toggle (P)
bool infiniteOcean = true; // true = tiled grid, false = single patch
int oceanRadius = 3;       // R=3 => 7x7 patches

std::string readFile(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << "\n";
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

GLuint compileShader(const std::string &source, GLenum type, const char *debugName)
{
    GLuint shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[4096];
        glGetShaderInfoLog(shader, 4096, nullptr, infoLog);
        std::cerr << "Shader compile error (" << debugName << "):\n"
                  << infoLog << "\n";
    }
    return shader;
}

GLuint createProgram(const std::string &vsSrc, const std::string &fsSrc, const char *progName)
{
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(vsSrc, GL_VERTEX_SHADER, (std::string(progName) + " VS").c_str());
    GLuint fs = compileShader(fsSrc, GL_FRAGMENT_SHADER, (std::string(progName) + " FS").c_str());
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[4096];
        glGetProgramInfoLog(program, 4096, nullptr, infoLog);
        std::cerr << "Program link error (" << progName << "):\n"
                  << infoLog << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

static void requireNonEmpty(const std::string &src, const char *name, const std::string &path)
{
    if (src.empty())
    {
        std::cerr << "FATAL: shader source empty for " << name << "\n";
        std::cerr << "Path tried: " << path << "\n";
        std::exit(1);
    }
}


void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    (void)window;
    float sensitivity = 0.1f;
    if (firstMouse)
    {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
    }

    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    lastX = (float)xpos;
    lastY = (float)ypos;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

float cameraSpeedBase = 10.0f; 

static void printWaveParams()
{
    std::cout
        << "Wave params: height=" << gHeightScale
        << "  choppy=" << gChoppy
        << "  swellAmp=" << gSwellAmp
        << "  swellSpeed=" << gSwellSpeed
        << "\n";
}

void processInput(GLFWwindow *window, float dt)
{
    float speed = cameraSpeedBase * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        speed *= 3.0f;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;

    static bool nPressed = false;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
    {
        if (!nPressed)
        {
            dayNight = 1.0f - dayNight;
            nPressed = true;
        }
    }
    else
        nPressed = false;

    static bool mPressed = false;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
    {
        if (!mPressed)
        {
            wireframe = !wireframe;
            mPressed = true;
        }
    }
    else
        mPressed = false;

    static bool pPressed = false;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    {
        if (!pPressed)
        {
            infiniteOcean = !infiniteOcean;
            pPressed = true;
            std::cout << "infiniteOcean now = " << (infiniteOcean ? "ON" : "OFF") << "\n";
        }
    }
    else
        pPressed = false;

    // still kinda broken but i like the green
    static bool d0 = false, d1 = false, d2 = false;
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
    {
        if (!d0)
        {
            shaderDebug = 0;
            d0 = true;
        }
    }
    else
        d0 = false;
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        if (!d1)
        {
            shaderDebug = 1;
            d1 = true;
        }
    }
    else
        d1 = false;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        if (!d2)
        {
            shaderDebug = 2;
            d2 = true;
        }
    }
    else
        d2 = false;


    static float keyRepeat = 0.0f;
    keyRepeat -= dt;

    // maybe add limits to height idk 
    if (keyRepeat <= 0.0f)
    {
        bool changed = false;

        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
        {
            gHeightScale = std::max(0.0f, gHeightScale - 0.5f);
            changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
        {
            gHeightScale = std::min(200.0f, gHeightScale + 0.5f);
            changed = true;
        }

        if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
        {
            gChoppy = std::max(0.0f, gChoppy - 0.25f);
            changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
        {
            gChoppy = std::min(50.0f, gChoppy + 0.25f);
            changed = true;
        }

        if (glfwGetKey(window, GLFW_KEY_SEMICOLON) == GLFW_PRESS)
        {
            gSwellAmp = std::max(0.0f, gSwellAmp - 0.1f);
            changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_APOSTROPHE) == GLFW_PRESS)
        {
            gSwellAmp = std::min(20.0f, gSwellAmp + 0.1f);
            changed = true;
        }

        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)
        {
            gSwellSpeed = std::max(0.0f, gSwellSpeed - 0.02f);
            changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)
        {
            gSwellSpeed = std::min(5.0f, gSwellSpeed + 0.02f);
            changed = true;
        }

        if (changed)
        {
            printWaveParams();
            keyRepeat = 0.05f; 
        }
    }
}

struct MeshVert
{
    float xz[2];
    float uv[2];
};


glm::vec2 worldOrigin(0.0f, 0.0f);

GLuint loadHDRTexture2D(const std::string &path, int &outW, int &outH)
{
    stbi_set_flip_vertically_on_load(true);
    int n = 0;
    float *data = stbi_loadf(path.c_str(), &outW, &outH, &n, 3);
    if (!data)
    {
        std::cerr << "Failed to load HDR: " << path << "\n";
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, outW, outH, 0, GL_RGB, GL_FLOAT, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    std::cout << "Loaded HDR: " << path << " (" << outW << "x" << outH << ")\n";
    return tex;
}

static GLuint createCubemapFromEquirect(GLuint hdrEquirectTex, int cubeSize, GLuint progEquirectToCube,
                                        GLuint cubeVAO, GLuint captureFBO, GLuint captureRBO)
{
    GLuint envCubemap = 0;
    glGenTextures(1, &envCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    for (int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, cubeSize, cubeSize, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glm::mat4 captureProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] =
        {
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0)),
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0)),
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)),
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, -1)),
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0)),
            glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0)),
        };

    glUseProgram(progEquirectToCube);
    glUniform1i(glGetUniformLocation(progEquirectToCube, "uEquirect"), 0);
    glUniformMatrix4fv(glGetUniformLocation(progEquirectToCube, "uProj"), 1, GL_FALSE, glm::value_ptr(captureProj));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrEquirectTex);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, cubeSize, cubeSize);

    glViewport(0, 0, cubeSize, cubeSize);

    glBindVertexArray(cubeVAO);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glUniformMatrix4fv(glGetUniformLocation(progEquirectToCube, "uView"), 1, GL_FALSE, glm::value_ptr(captureViews[i]));
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    return envCubemap;
}

int main(int argc, char **argv)
{
    fs::path exeDir = fs::absolute(argv[0]).parent_path();
    fs::path shadersDir = exeDir / "shaders";
    std::cout << "ExeDir:     " << exeDir.string() << "\n";
    std::cout << "ShadersDir: " << shadersDir.string() << "\n";

    if (!glfwInit())
    {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1920, 1080, "Ocean Render", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Window creation failed\n";
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    float quadVertices[] = {
        -1.f, -1.f, 0.f, 0.f,
        1.f, -1.f, 1.f, 0.f,
        -1.f, 1.f, 0.f, 1.f,
        -1.f, 1.f, 0.f, 1.f,
        1.f, -1.f, 1.f, 0.f,
        1.f, 1.f, 1.f, 1.f};
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glBindVertexArray(0);


    const int GRID_N = 256;
    const float PATCH_SIZE = 512.0f;

    std::vector<MeshVert> verts;
    verts.reserve((GRID_N + 1) * (GRID_N + 1));

    for (int z = 0; z <= GRID_N; z++)
        for (int x = 0; x <= GRID_N; x++)
        {
            float u = (float)x / (float)GRID_N;
            float v = (float)z / (float)GRID_N;

            float px = (u - 0.5f) * PATCH_SIZE;
            float pz = (v - 0.5f) * PATCH_SIZE;

            MeshVert mv{};
            mv.xz[0] = px;
            mv.xz[1] = pz;
            mv.uv[0] = u;
            mv.uv[1] = v;
            verts.push_back(mv);
        }

    std::vector<uint32_t> indices;
    indices.reserve(GRID_N * GRID_N * 6);

    auto idx = [GRID_N](int x, int z) -> uint32_t
    {
        return (uint32_t)(z * (GRID_N + 1) + x);
    };

    for (int z = 0; z < GRID_N; z++)
        for (int x = 0; x < GRID_N; x++)
        {
            uint32_t i0 = idx(x, z);
            uint32_t i1 = idx(x + 1, z);
            uint32_t i2 = idx(x, z + 1);
            uint32_t i3 = idx(x + 1, z + 1);

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);
            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }

    GLuint waterVAO, waterVBO, waterEBO;
    glGenVertexArrays(1, &waterVAO);
    glGenBuffers(1, &waterVBO);
    glGenBuffers(1, &waterEBO);

    glBindVertexArray(waterVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(MeshVert)), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(indices.size() * sizeof(uint32_t)), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVert), (void *)offsetof(MeshVert, xz));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVert), (void *)offsetof(MeshVert, uv));
    glBindVertexArray(0);

    float cubeVerts[] = {
        -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1,
        -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1,
        -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1,
        1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1,
        -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1,
        -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1};

    GLuint cubeVAO = 0, cubeVBO = 0;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glBindVertexArray(0);

    // probably should make cmake put the shaders in the Release but dragging in for now works
    auto sp = [&](const char *name)
    { return (shadersDir / name).string(); };

    std::string fullscreen_vs = readFile(sp("fullscreen_quad.vs"));
    std::string bufferA_fs = readFile(sp("fft_init.glsl"));
    std::string bufferB_fs = readFile(sp("fft_process.glsl"));
    std::string water_vs = readFile(sp("water_mesh.vs"));
    std::string water_fs = readFile(sp("water_mesh.fs"));

    std::string skybox_vs = readFile(sp("skybox.vs"));
    std::string skybox_fs = readFile(sp("skybox.fs"));
    std::string capture_vs = readFile(sp("cube_capture.vs"));
    std::string eq2cube_fs = readFile(sp("equirect_to_cubemap.fs"));

    requireNonEmpty(fullscreen_vs, "fullscreen_quad.vs", sp("fullscreen_quad.vs"));
    requireNonEmpty(bufferA_fs, "fft_init.glsl", sp("fft_init.glsl"));
    requireNonEmpty(bufferB_fs, "fft_process.glsl", sp("fft_process.glsl"));
    requireNonEmpty(water_vs, "water_mesh.vs", sp("water_mesh.vs"));
    requireNonEmpty(water_fs, "water_mesh.fs", sp("water_mesh.fs"));

    requireNonEmpty(skybox_vs, "skybox.vs", sp("skybox.vs"));
    requireNonEmpty(skybox_fs, "skybox.fs", sp("skybox.fs"));
    requireNonEmpty(capture_vs, "cube_capture.vs", sp("cube_capture.vs"));
    requireNonEmpty(eq2cube_fs, "equirect_to_cubemap.fs", sp("equirect_to_cubemap.fs"));

    GLuint progA = createProgram(fullscreen_vs, bufferA_fs, "FFT Init (A)");
    GLuint progB = createProgram(fullscreen_vs, bufferB_fs, "FFT Process (B)");
    GLuint progWater = createProgram(water_vs, water_fs, "Water Mesh");
    GLuint progSkybox = createProgram(skybox_vs, skybox_fs, "Skybox");
    GLuint progEq2Cube = createProgram(capture_vs, eq2cube_fs, "Equirect->Cubemap");

    const int FREQ_SIZE = 256;

    GLuint texA, texB[2];
    glGenTextures(1, &texA);
    glBindTexture(GL_TEXTURE_2D, texA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, FREQ_SIZE, FREQ_SIZE, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenTextures(2, texB);
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, texB[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 3 * FREQ_SIZE, FREQ_SIZE, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }

    GLuint fboA, fboB[2];
    glGenFramebuffers(1, &fboA);
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texA, 0);

    glGenFramebuffers(2, fboB);
    for (int i = 0; i < 2; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texB[i], 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (int i = 0; i < 2; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[i]);
        glViewport(0, 0, 3 * FREQ_SIZE, FREQ_SIZE);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const int FINAL_B = 0;
    const int TMP_B = 1;

    int hdrW = 0, hdrH = 0;
    GLuint hdrTex = loadHDRTexture2D(sp("sky.hdr"), hdrW, hdrH);
    float envMaxMip = 0.0f;

    if (!hdrTex)
    {
        std::cerr << "HDR missing from build/Release \n";
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, hdrTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        int maxDim = std::max(hdrW, hdrH);
        int mipCount = 1;
        while ((maxDim >>= 1) > 0)
            mipCount++;
        envMaxMip = float(mipCount - 1);

        glBindTexture(GL_TEXTURE_2D, 0);
        std::cout << "HDR mipmaps generated. uEnvMaxMip = " << envMaxMip << "\n";
    }


    GLuint captureFBO = 0, captureRBO = 0;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    GLuint envCube = 0;
    if (hdrTex)
    {
        envCube = createCubemapFromEquirect(hdrTex, 512, progEq2Cube, cubeVAO, captureFBO, captureRBO);
    }

    float time = 0.0f;
    float dbgTimer = 0.0f;

    const float SWELL_AMP = 1.5f;
    const float SWELL_SPEED = 0.25f;

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, deltaTime);
        glfwPollEvents();

        time += deltaTime * 0.4f;

        //buffer A
        glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        glViewport(0, 0, FREQ_SIZE, FREQ_SIZE);
        glUseProgram(progA);
        glUniform1f(glGetUniformLocation(progA, "iTime"), time);
        glUniform2f(glGetUniformLocation(progA, "iResolution"), (float)FREQ_SIZE, (float)FREQ_SIZE);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // buffer b
        glUseProgram(progB);
        glUniform2f(glGetUniformLocation(progB, "iResolution"), (float)(3 * FREQ_SIZE), (float)FREQ_SIZE);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texA);
        glUniform1i(glGetUniformLocation(progB, "iChannel0"), 0);

        // pass 0
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[TMP_B]);
        glViewport(0, 0, 3 * FREQ_SIZE, FREQ_SIZE);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texB[TMP_B]);
        glUniform1i(glGetUniformLocation(progB, "iChannel1"), 1);

        glUniform1i(glGetUniformLocation(progB, "uPass"), 0);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // pass 1
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[FINAL_B]);
        glViewport(0, 0, 3 * FREQ_SIZE, FREQ_SIZE);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texB[TMP_B]);
        glUniform1i(glGetUniformLocation(progB, "iChannel1"), 1);

        glUniform1i(glGetUniformLocation(progB, "uPass"), 1);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // i originally had this because my plane was completely flat but i like it so im going to keep it
        // if for some reason you are testing this code just comment this out and you wont be spammed 
        dbgTimer += deltaTime;
        if (dbgTimer > 1.0f)
        {
            dbgTimer = 0.0f;

            glBindFramebuffer(GL_FRAMEBUFFER, fboB[FINAL_B]);
            auto read = [&](int x, int y)
            {
                float p[4] = {0, 0, 0, 0};
                glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, p);
                return std::array<float, 4>{p[0], p[1], p[2], p[3]};
            };

            auto hcc = read(FREQ_SIZE / 2, FREQ_SIZE / 2);
            auto dxcc = read(FREQ_SIZE + FREQ_SIZE / 2, FREQ_SIZE / 2);
            auto dzcc = read(2 * FREQ_SIZE + FREQ_SIZE / 2, FREQ_SIZE / 2);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            std::cout << "FFT H(C,C)  = " << hcc[0] << "  DX=" << dxcc[0] << "  DZ=" << dzcc[0] << "\n";
        }

        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.02f, 0.03f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 proj = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.01f, 2000.0f);

        glm::vec2 camXZ(cameraPos.x, cameraPos.z);
        glm::vec2 snap = glm::floor(camXZ / PATCH_SIZE) * PATCH_SIZE;
        if (snap.x != 0.0f || snap.y != 0.0f)
        {
            cameraPos.x -= snap.x;
            cameraPos.z -= snap.y;
            worldOrigin += snap;
        }

        // water startt
        glUseProgram(progWater);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texB[FINAL_B]);
        glUniform1i(glGetUniformLocation(progWater, "uFFT"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, hdrTex);
        glUniform1i(glGetUniformLocation(progWater, "uEnvHDR"), 1);

        glUniformMatrix4fv(glGetUniformLocation(progWater, "uView"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(progWater, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));

        glUniform3f(glGetUniformLocation(progWater, "uCameraPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform1f(glGetUniformLocation(progWater, "uTime"), time);
        glUniform1f(glGetUniformLocation(progWater, "uDayNight"), dayNight);

        glUniform1f(glGetUniformLocation(progWater, "uPatchSize"), PATCH_SIZE);

        glUniform1f(glGetUniformLocation(progWater, "uHeightScale"), gHeightScale);
        glUniform1f(glGetUniformLocation(progWater, "uChoppy"), gChoppy);
        
        // swells
        GLint locSwellAmp = glGetUniformLocation(progWater, "uSwellAmp");
        if (locSwellAmp >= 0)
            glUniform1f(locSwellAmp, SWELL_AMP);
        GLint locSwellSpeed = glGetUniformLocation(progWater, "uSwellSpeed");
        if (locSwellSpeed >= 0)
            glUniform1f(locSwellSpeed, SWELL_SPEED);

        GLint locDbg = glGetUniformLocation(progWater, "uDebug");
        if (locDbg >= 0)
            glUniform1i(locDbg, shaderDebug);

        GLint locExp = glGetUniformLocation(progWater, "uEnvExposure");
        if (locExp >= 0)
            glUniform1f(locExp, 0.6f);

        GLint locMip = glGetUniformLocation(progWater, "uEnvMaxMip");
        if (locMip >= 0)
            glUniform1f(locMip, envMaxMip);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glBindVertexArray(waterVAO);

        if (!infiniteOcean)
        {
            glUniform2f(glGetUniformLocation(progWater, "uWorldOffset"), worldOrigin.x, worldOrigin.y);
            glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, (void *)0);
        }
        else
        {
            for (int tz = -oceanRadius; tz <= oceanRadius; ++tz)
                for (int tx = -oceanRadius; tx <= oceanRadius; ++tx)
                {
                    glm::vec2 tileOrigin = worldOrigin + glm::vec2((float)tx * PATCH_SIZE, (float)tz * PATCH_SIZE);
                    glUniform2f(glGetUniformLocation(progWater, "uWorldOffset"), tileOrigin.x, tileOrigin.y);
                    glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, (void *)0);
                }
        }

        glBindVertexArray(0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // sky box idk if i like the hdr maybe change later
        if (envCube)
        {
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

            glUseProgram(progSkybox);

            glm::mat4 viewNoTrans = glm::mat4(glm::mat3(view));
            glUniformMatrix4fv(glGetUniformLocation(progSkybox, "uViewNoTrans"), 1, GL_FALSE, glm::value_ptr(viewNoTrans));
            glUniformMatrix4fv(glGetUniformLocation(progSkybox, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCube);
            glUniform1i(glGetUniformLocation(progSkybox, "uEnv"), 0);
            glUniform1f(glGetUniformLocation(progSkybox, "uExposure"), 0.8f);

            glBindVertexArray(cubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);
        }

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
