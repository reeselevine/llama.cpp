#define(VARIANTS)
[
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

#decl(SRC1_F32)
fn zero_val_src1() -> f32 {
    return 0.0;
}

fn store_src1_shmem(val: f32, idx: u32) {
    src1_shmem[idx] = val;
}

fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

fn mul_acc(src0_val: f32, src1_val: f32) -> f32 {
    return f32(src0_val) * src1_val;
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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{SRC1_TYPE}}>; // M rows, N columns (transposed)
@group(0) @binding(3) var<storage, read_write> debug: array<f32>;

@group(0) @binding(4) var<uniform> params: MulMatParams;

DECLS

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}


// Warning: cannot be overrides, must match values in ggml-webgpu.cpp
const TILE_N = 4u;
// must be multiple of 4 for vec4 loads
const TILE_M = 4u;

override WORKGROUP_SIZE_M: u32;
override WORKGROUP_SIZE_N: u32;
override TILE_K: u32;

override TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
override BLOCKS_K = max(1, TILE_K/{{BLOCK_SIZE}}); // the number of blocks we need to store at least TILE_K elements per thread. Note that since TILE_K may be less than BLOCK_SIZE, we need at least room for 1 block. Otherwise, TILE_K must be divisible by BLOCK_SIZE, or a clean fraction if it, or things will get weird.
override TILE_SRC0_Q_SHMEM = BLOCKS_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> src0_q_shmem: array<{{SRC0_TYPE}}, TILE_SRC0_Q_SHMEM>;
var<workgroup> src0_shmem: array<f32, TILE_SRC0_SHMEM>;
var<workgroup> src1_shmem: array<f32, TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    debug[0] = 42.0;

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

        // we only load new blocks of src0 when the k_outer is at a block boundary
        if (k_outer % {{BLOCK_SIZE}} == 0) {
            for (var blck_idx = thread_id; blck_idx < TILE_SRC0_SHMEM; blck_idx += TOTAL_WORKGROUP_SIZE) {
                let tile_m = blck_idx / BLOCKS_K;
                let block_k = blck_idx % BLOCKS_K;
                let global_m = wg_m * WORKGROUP_SIZE_M * TILE_M + tile_m;
                let global_k = k_outer/{{BLOCK_SIZE}} + block_k;
                let src0_idx = src0_batch_offset + global_m * params.stride_01 + global_k;
                if (global_m < params.m && global_k < params.k) {
                    if (src0_idx == 6u) {
                        debug[1] = f32(src0[src0_idx].d);
                    }
                    src0_q_shmem[blck_idx] = src0[src0_idx];
                }
            }
            workgroupBarrier();
        }

        // next, we need to dequantize the src0 block into src1-type values in shmem
        // but, only for the partition of k-values that we're currently working on
        // currently hardcoded for q4_0
        var shmem_idx = (thread_id / 8) * 32 + (thread_id % 8) * 2;
        for (var i = thread_id; shmem_idx < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE) {
            let blck_idx = i / 8;
            let block_offset = i % 8;
            shmem_idx = blck_idx * 32 + block_offset * 2;
            let block_val = src0_q_shmem[blck_idx];
            let d = f32(block_val.d);
            let q_packed = bitcast<u32>(vec2(block_val.qs[block_offset], 0.0));
            for (var j = 0u; j < 2; j++) {
                let q_byte = get_byte(q_packed, j);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;

                src0_shmem[shmem_idx + j] = q_lo;
                src0_shmem[shmem_idx + j + 16] = q_hi;
            }
            if (global_id.x == 0u && k_outer == 192 && i == 0u) {
                debug[2] = d;
                debug[3] = src0_shmem[0];
                debug[4] = src0_shmem[16];
                debug[5] = f32(blck_idx);
            }
        }

        for (var elem_idx = thread_id; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
            let tile_n = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_n = wg_n * WORKGROUP_SIZE_N * TILE_N + tile_n;
            let global_k = k_outer + tile_k;

            let src1_idx = src1_batch_offset + global_n * params.stride_11 + global_k;
            let src1_val = select(
                zero_val_src1(),
                src1[src1_idx],
                global_n < params.n && global_k < params.k);
            src1_shmem[elem_idx] = src1_val;
        }

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f32, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
                src0_tile[tm] = src0_shmem[src0_idx];
            }
            for (var tn = 0u; tn < TILE_N; tn++) {
                let src1_n = local_n * TILE_N + tn;
                let src1_idx = src1_n * TILE_K + k_inner;
                let src1_vec = src1_shmem[src1_idx];
                for (var tm = 0u; tm < TILE_M; tm++) {
                      acc[tm][tn] += mul_acc(src0_tile[tm], src1_vec);
                }
            }
        }

        workgroupBarrier();
    }

    if (global_id.x == 0u) {
        debug[6] = acc[0][0];
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_row = output_row_base + tn;
        if (global_row < params.n) {
            for (var tm = 0u; tm < TILE_M; tm ++) {
                let global_col = output_col_base + tm;
                if (global_col < params.m) {
                    let dst_idx = dst_batch_offset + global_row * params.m + global_col;
                    dst[dst_idx] = store_val(acc, tn, tm);
                }
            }
        }
    }
}

#end(SHADER)
