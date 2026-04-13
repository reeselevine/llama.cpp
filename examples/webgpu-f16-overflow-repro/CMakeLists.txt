cmake_minimum_required(VERSION 3.16)

project(webgpu_f16_overflow_repro LANGUAGES CXX)

set(TARGET webgpu-f16-overflow-repro)

find_package(Threads REQUIRED)

if (NOT TARGET dawn::webgpu_dawn)
    find_package(Dawn REQUIRED)
endif()

add_executable(${TARGET} main.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE dawn::webgpu_dawn)

install(TARGETS ${TARGET} RUNTIME)
