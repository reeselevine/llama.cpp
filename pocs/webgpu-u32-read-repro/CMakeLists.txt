add_executable(extract-webgpu-u32-read-repro-shader extract-shader.cpp)
target_include_directories(extract-webgpu-u32-read-repro-shader PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-webgpu
)
