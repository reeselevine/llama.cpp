#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "f32_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f32>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SRC0_F32_VEC", "SRC1_F32_VEC"]
  },
  {
    "SHADER_SUFFIX": "f32_f32",
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SRC0_F32", "SRC1_F32"]
  },
  {
    "SHADER_SUFFIX": "f16_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SRC0_F16_VEC", "SRC1_F32_VEC"]
  },
  {
    "SHADER_SUFFIX": "f16_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SRC0_F16", "SRC1_F32"]
  },
  {
    "SHADER_SUFFIX": "f16_f16_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f16>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SRC0_F16_VEC", "SRC1_F16_VEC"]
  },
  {
    "SHADER_SUFFIX": "f16_f16",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SRC0_F16", "SRC1_F16"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(SRC0_F32_VEC)
fn zero_val_src0() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
#enddecl(SRC0_F32_VEC)

#decl(SRC0_F32)
fn zero_val_src0() -> f32 {
    return 0.0;
}
#enddecl(SRC0_F32)

#decl(SRC0_F16_VEC)
fn zero_val_src0() -> vec4<f16> {
    return vec4<f16>(0.0, 0.0, 0.0, 0.0);
}
#enddecl(SRC0_F16_VEC)

#decl(SRC0_F16)
fn zero_val_src0() -> f16 {
    return 0.0;
}
#enddecl(SRC0_F16)

#decl(SRC1_F32_VEC)
fn zero_val_src1() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(acc[tm][tn], acc[tm + 1][tn], acc[tm + 2][tn], acc[tm + 3][tn]);
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: vec4<f32>) -> f32 {
    return dot(vec4<f32>(src0_val), src1_val);
}
#enddecl(SRC1_F32_VEC)

#decl(SRC1_F32)
fn zero_val_src1() -> f32 {
    return 0.0;
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: f32) -> f32 {
    return f32(src0_val) * src1_val;
}
#enddecl(SRC1_F32)

#decl(SRC1_F16_VEC)
fn zero_val_src1() -> vec4<f16> {
    return vec4<f16>(0.0, 0.0, 0.0, 0.0);
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(acc[tm][tn], f32(acc[tm + 1][tn]), f32(acc[tm + 2][tn]), f32(acc[tm + 3][tn]));
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: vec4<f16>) -> f32 {
    return dot(vec4<f32>(src0_val), vec4<f32>(src1_val));
}
#enddecl(SRC1_F16_VEC)

#decl(SRC1_F16)
fn zero_val_src1() -> f16 {
    return 0.0;
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: f16) -> f32 {
    return f32(src0_val) * f32(src1_val);
}
#enddecl(SRC1_F16)

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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{DST_TYPE}}>; // M rows, N columns (transposed)

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
override TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> src0_shmem: array<{{SRC0_TYPE}}, TILE_SRC0_SHMEM/{{VEC_SIZE}}>;
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

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_m = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_m = wg_m * WORKGROUP_SIZE_M * TILE_M + tile_m;
            let global_k = k_outer + tile_k;
            let src0_idx = src0_batch_offset + global_m * params.stride_01 + global_k;
            src0_shmem[elem_idx/{{VEC_SIZE}}] = select( // taking a slight performance hit to avoid oob
                zero_val_src0(),
                src0[src0_idx/{{VEC_SIZE}}],
                global_m < params.m && global_k < params.k);
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
            var src0_tile: array<{{SRC0_TYPE}}, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
                src0_tile[tm] = src0_shmem[src0_idx/{{VEC_SIZE}}];
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
