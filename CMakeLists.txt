cmake_minimum_required(VERSION 3.15)
project(FFTWater LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(src)
include_directories(external/glad/include)
include_directories(external/glfw-3.4/include)
include_directories(external/glm)
include_directories(external/imgui)
include_directories(external/stb)

file(GLOB SOURCES "src/*.cpp" "src/*.h")
add_executable(FFTWater ${SOURCES})


add_library(glad STATIC external/glad/src/glad.c)
target_link_libraries(FFTWater glad)


add_subdirectory(external/glfw-3.4)
target_link_libraries(FFTWater glfw)


if (WIN32)
    target_link_libraries(FFTWater opengl32)
elseif (APPLE)
    find_library(OpenGL_LIBRARY OpenGL)
    target_link_libraries(FFTWater ${OpenGL_LIBRARY})
else()
    target_link_libraries(FFTWater GL dl pthread X11 Xrandr Xi)
endif()
