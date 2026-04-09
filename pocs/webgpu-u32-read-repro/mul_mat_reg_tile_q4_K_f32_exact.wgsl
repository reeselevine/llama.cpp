enable f16;
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

fn load_src0_u16_at(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_src0_u32_at(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = src0[word_idx];
    if (shift == 0u) {
        return lo;
    }
    let hi = src0[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
}

fn load_src0_f16_at(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_src0_u16_at(byte_offset));
    return f16(packed[0]);
}































fn store_shmem(val: f16, idx: u32) {
    shmem[idx] = val;
}


fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * 1; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * 1) {
        let tile_n = elem_idx / 32u;
        let tile_k = elem_idx % 32u;
        let global_n = offset_n + tile_n;
        let global_k = k_outer + tile_k;
        let src1_idx = batch_offset + global_n * params.stride_11 + global_k;
        let src1_val = select(
            f32(0.0),
            src1[src1_idx/1],
            global_n < params.n && global_k < params.k);
        store_shmem(f16(src1_val), TILE_SRC0_SHMEM + elem_idx);
    }
}









const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 144u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / 32u;
        let tile_k = elem_idx % 32u;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_src0_f16_at(block_byte_base);
        let dmin = load_src0_f16_at(block_byte_base + 2u);

        // Load packed scales
        var scale_vals: array<u32, 3>;
        for (var i: u32 = 0u; i < 3u; i++) {
            scale_vals[i] = load_src0_u32_at(block_byte_base + 4u + 4u * i);
        }

        // Map k_in_block to loop structure:
        // Outer loop over 64-element groups (alternating q_b_idx)
        // Inner loop over 2 shifts per group
        let group_of_64 = k_in_block / 64u;  // 0-3 (maps to q_b_idx)
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        var sc: u32;
        var mn: u32;

        if (is < 4u) {
            let sc_byte = get_byte(scale_vals[is / 4u], is % 4u);
            let min_byte = get_byte(scale_vals[(is + 4u) / 4u], is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(scale_vals[(is + 4u) / 4u], (is + 4u) % 4u);
            let sc_hi = get_byte(scale_vals[(is - 4u) / 4u], (is - 4u) % 4u);
            let min_hi = get_byte(scale_vals[is / 4u], is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_src0_u32_at(block_byte_base + 16u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;

        let q_val = f16(qs_val) * dl - ml;
        if (thread_id == 0u && batch_offset == 0u && offset_m == 0u && k_outer == 0u && elem_idx == 0u) {
            shmem[elem_idx] = f16(get_byte(scale_vals[0], 0u) & 63u);
        } else {
            shmem[elem_idx] = q_val;
        }
    }
}




fn store_val(acc: array<array<f32, 8u>, 8u>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

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

@group(0) @binding(0) var<storage, read_write> src0: array<u32>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<f32>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;
@group(0) @binding(4) var<storage, read_write> debug_out: array<u32>;

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / 8u;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % 8u;
}

const TOTAL_WORKGROUP_SIZE = 8u * 8u;
const TILE_SRC0_SHMEM = 32u * 8u * 8u;
const TILE_SRC1_SHMEM = 32u * 8u * 8u;

var<workgroup> shmem: array<f16, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    let wg_n_count = (params.n + 8u * 8u - 1u) / (8u * 8u);
    let wg_m_count = (params.m + 8u * 8u - 1u) / (8u * 8u);
    let wg_per_matrix = wg_m_count * wg_n_count;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    let batch_idx = wg_linear / wg_per_matrix;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_m * 8u * 8u + local_m * 8u;
    let output_col_base = wg_n * 8u * 8u + local_n * 8u;

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

    let offset_m = wg_m * 8u * 8u;
    let offset_n = wg_n * 8u * 8u;

    var acc: array<array<f32, 8u>, 8u>;
    if (wg_id.x == 0u && wg_id.y == 0u && thread_id == 0u) {
        debug_out[0] = 0xdeadbeefu;
        debug_out[1] = src0[0];
        debug_out[2] = load_src0_u16_at(0u);
        debug_out[3] = load_src0_u32_at(4u);
    }

    for (var k_outer = 0u; k_outer < params.k; k_outer += 32u) {

        // see mul_mat_decls.tmpl
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        if (k_outer == 0u && wg_id.x == 0u && wg_id.y == 0u && thread_id == 0u) {
            debug_out[4] = bitcast<u32>(f32(shmem[0]));
            debug_out[5] = bitcast<u32>(f32(shmem[TILE_SRC0_SHMEM]));
        }

        let k_end = min(32u, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f16, 8u>;
            for (var tm = 0u; tm < 8u; tm++) {
                let src0_m = local_m * 8u + tm;
                let src0_idx = k_inner + src0_m * 32u;
                src0_tile[tm] = shmem[src0_idx];
            }
            for (var tn = 0u; tn < 8u; tn++) {
                let src1_n = local_n * 8u + tn;
                let src1_idx = src1_n * 32u + k_inner;
                let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
                for (var tm = 0u; tm < 8u; tm++) {
                      acc[tm][tn] += f32(src0_tile[tm]) * f32(src1_val);
                }
            }
        }

        workgroupBarrier();
    }

    if (wg_id.x == 0u && wg_id.y == 0u && thread_id == 0u) {
        debug_out[6] = bitcast<u32>(acc[0][0]);
        debug_out[7] = bitcast<u32>(acc[0][1]);
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < 8u; tn++) {
        let global_col = output_col_base + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < 8u; tm += 1) {
                let global_row = output_row_base + tm;
                if (global_row < params.m) {
                    let dst_idx = dst_batch_offset + global_col * params.m + global_row;
                    dst[dst_idx/1] = store_val(acc, tn, tm);
                }
            }
        }
    }
}
