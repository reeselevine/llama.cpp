#define(VARIANTS)
[
  {
    "SHADER_NAME": "mul_mat_fast_q4_0_vec",
    "REPLS": {
      "SRC0_TYPE" : "q4_0",
      "SRC1_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
      "BLOCK_SIZE": 32,
    },
    "DECLS": ["BYTE_HELPERS", "Q4_0_T", "SRC1_F32_VEC"]
  },
  {
    "SHADER_NAME": "mul_mat_fast_q4_0",
    "REPLS": {
      "SRC0_TYPE" : "q4_0",
      "SRC1_TYPE" : "f32",
      "VEC_SIZE" : "1",
      "BLOCK_SIZE": 32,
    },
    "DECLS": ["BYTE_HELPERS", "Q4_0_T", "SRC1_F32"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(SRC1_F32_VEC)
fn zero_val_src1() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn load_src0_shmem(idx: u32) -> vec4<f32> {
    return vec4<f32>(src0_shmem[idx], src0_shmem[idx + 1], src0_shmem[idx + 2], src0_shmem[idx + 3]);
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(acc[tm][tn], acc[tm + 1][tn], acc[tm + 2][tn], acc[tm + 3][tn]);
}

fn mul_acc(src0_val: vec4<f32>, src1_val: vec4<f32>) -> f32 {
    return dot(src0_val, src1_val);
}
#enddecl(SRC1_F32_VEC)

#decl(SRC1_F32)
fn zero_val_src1() -> f32 {
    return 0.0;
}

fn load_src0_shmem(idx: u32) -> f32 {
    return src0_shmem[idx];
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

fn mul_acc(src0_val: f32, src1_val: f32) -> f32 {
    return src0_val * src1_val;
}
#enddecl(SRC1_F32)

#end(DECLS)

#define(SHADER)
enable f16;

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

// Flattened storage: for q4_0 we store blocks as contiguous f16 entries: [d, qs[0], qs[1], ..., qs[7]]
@group(0) @binding(0) var<storage, read_write> src0: array<f16>; // M rows (blocks), K columns (blocks), stored as f16
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{SRC1_TYPE}}>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

DECLS

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}


// Warning: cannot be overrides, must match values in ggml-webgpu.cpp
// must be multiple of 4 for vec4 loads
const TILE_M = 4u;
const TILE_N = 4u;

override WORKGROUP_SIZE_M: u32;
override WORKGROUP_SIZE_N: u32;
override TILE_K: u32;

override TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
override BLOCKS_K = TILE_K/{{BLOCK_SIZE}}; // the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override TILE_SRC0_Q_SHMEM = BLOCKS_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> src0_shmem: array<f32, TILE_SRC0_SHMEM>;
var<workgroup> src1_shmem: array<{{SRC1_TYPE}}, TILE_SRC1_SHMEM/{{VEC_SIZE}}>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    let wg_linear = global_id.x / TOTAL_WORKGROUP_SIZE;

    let wg_n_count = (params.n + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_per_matrix = wg_m_count * wg_n_count;

    let batch_idx = wg_linear / wg_per_matrix;

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;
    let output_col_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    var acc: array<array<f32, TILE_N>, TILE_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // Each thread works on 4 elements, so 8 threads per block
        // currently hardcoded for q4_0 with BLOCK_SIZE = 32
        for (var i = thread_id; i / 8u < TILE_SRC0_Q_SHMEM; i += TOTAL_WORKGROUP_SIZE) {
            let blck_idx = i / 8u;
            let block_offset = i % 8u;
            let shmem_idx = blck_idx * 32u + block_offset * 2u;

            let tile_m = blck_idx / BLOCKS_K;
            let block_k = blck_idx % BLOCKS_K;
            let global_m = wg_m * WORKGROUP_SIZE_M * TILE_M + tile_m;
            let global_k = k_outer/{{BLOCK_SIZE}} + block_k;
            let src0_idx = src0_batch_offset + global_m * params.stride_01 + global_k;

            if (global_m < params.m && global_k < params.k) {
                let block_idx = src0_idx;
                let base_f16 = block_idx * 9u;
                let d_f16 = src0[base_f16];
                let d = f32(d_f16);
                let q_f16 = src0[base_f16 + 1u + block_offset];
                let q_packed = bitcast<u32>(vec2(q_f16, 0.0));
                for (var j = 0u; j < 2u; j++) {
                    let q_byte = get_byte(q_packed, j);
                    let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0) * d;
                    let q_lo = (f32(q_byte & 0xF) - 8.0) * d;

                    src0_shmem[shmem_idx + j] = q_lo;
                    src0_shmem[shmem_idx + j + 16u] = q_hi;
                }
            }
        }

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_n = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_n = wg_n * WORKGROUP_SIZE_N * TILE_N + tile_n;
            let global_k = k_outer + tile_k;

            let src1_idx = src1_batch_offset + global_n * params.stride_11 + global_k;
            src1_shmem[elem_idx/{{VEC_SIZE}}] = select(
                zero_val_src1(),
                src1[src1_idx/{{VEC_SIZE}}],
                global_n < params.n && global_k < params.k);
        }

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner += {{VEC_SIZE}}) {
            var src0_tile: array<{{SRC1_TYPE}}, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
                src0_tile[tm] = load_src0_shmem(src0_idx);
            }
            for (var tn = 0u; tn < TILE_N; tn++) {
                let src1_n = local_n * TILE_N + tn;
                let src1_idx = src1_n * TILE_K + k_inner;
                let src1_vec = src1_shmem[src1_idx/{{VEC_SIZE}}];
                for (var tm = 0u; tm < TILE_M; tm++) {
                      acc[tm][tn] += mul_acc(src0_tile[tm], src1_vec);
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_row = output_row_base + tn;
        if (global_row < params.n) {
            for (var tm = 0u; tm < TILE_M; tm += {{VEC_SIZE}}) {
                let global_col = output_col_base + tm;
                if (global_col < params.m) {
                    let dst_idx = dst_batch_offset + global_row * params.m + global_col;
                    dst[dst_idx/{{VEC_SIZE}}] = store_val(acc, tn, tm);
                }
            }
        }
    }
}

#end(SHADER)
