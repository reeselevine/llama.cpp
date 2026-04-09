#include "pre_wgsl.hpp"

#if __has_include("../../build-wasm/ggml/src/ggml-webgpu/generated/ggml-wgsl-shaders.hpp")
#include "../../build-wasm/ggml/src/ggml-webgpu/generated/ggml-wgsl-shaders.hpp"
#elif __has_include("../../build/ggml/src/ggml-webgpu/generated/ggml-wgsl-shaders.hpp")
#include "../../build/ggml/src/ggml-webgpu/generated/ggml-wgsl-shaders.hpp"
#else
#error "Could not find generated ggml-wgsl-shaders.hpp in build-wasm/ or build/"
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string out_path =
        argc >= 2 ? argv[1] : "pocs/webgpu-u32-read-repro/mul_mat_reg_tile_q4_K_f32_exact.wgsl";

    std::vector<std::string> defines = {
        "SRC1_INNER_TYPE=f32",
        "BYTE_HELPERS",
        "MUL_ACC_Q4_K",
        "INIT_SRC0_SHMEM_Q4_K",
        "INIT_SRC1_SHMEM_FLOAT",
        "U32_DEQUANT_HELPERS",
        "SRC0_INNER_TYPE=u32",
        "SCALAR",
        "TILE_M=8u",
        "TILE_N=8u",
        "TILE_K=32u",
        "WORKGROUP_SIZE_M=8u",
        "WORKGROUP_SIZE_N=8u",
    };

    pre_wgsl::Preprocessor pp;
    const std::string processed = pp.preprocess(wgsl_mul_mat_reg_tile, defines);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open " << out_path << "\n";
        return 1;
    }
    out << processed;
    if (!out) {
        std::cerr << "failed to write " << out_path << "\n";
        return 1;
    }

    std::cout << "wrote " << out_path << " (" << processed.size() << " bytes)\n";
    return 0;
}
