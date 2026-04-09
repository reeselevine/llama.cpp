cmake_minimum_required(VERSION 3.14)
project(webgpu_u32_read_repro LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(Dawn REQUIRED)

add_executable(webgpu-u32-read-repro-runner run-repro.cpp)
target_link_libraries(webgpu-u32-read-repro-runner PRIVATE dawn::webgpu_dawn)
target_compile_features(webgpu-u32-read-repro-runner PRIVATE cxx_std_17)
