
enable f16;

#include "common_decls.tmpl"

#ifdef VEC

#define VEC_SIZE 4
#define DST_TYPE vec4<f32>
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(dot(SRC1_TYPE(src0_val), src1_val));
}

fn store_val(group_base: u32) -> vec4<f32> {
    return vec4<f32>(partial_sums[group_base],
                     partial_sums[group_base + THREADS_PER_OUTPUT],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 2],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 3]);
}
#endif

#ifdef SCALAR

#define VEC_SIZE 1
#define DST_TYPE f32
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

fn store_val(group_base: u32) -> f32 {
    return partial_sums[group_base];
}
#endif

#ifdef MUL_ACC_FLOAT
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * VEC_SIZE; i < tile_size; i += THREADS_PER_OUTPUT * VEC_SIZE) {
        let a = src0[(idx_base + k_outer + i) / VEC_SIZE];
        let b = shared_vector[i / VEC_SIZE];
        local_sum += inner_dot(a, b);
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q4_0

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 9u; // 1 scale + 8x4 packed weights
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_0 = src0[scale_idx + 1 + block_offset + j];
            let q_1 = src0[scale_idx + 1 + block_offset + j + 1];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0) * d;
                local_sum += q_lo * shared_vector[shmem_idx + j * 2 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 2 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q4_1

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 10u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        let m = f32(src0[scale_idx + 1u]);
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_0 = src0[scale_idx + 2u + block_offset + j];
            let q_1 = src0[scale_idx + 2u + block_offset + j + 1];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = f32((q_byte >> 4) & 0xF) * d + m;
                let q_lo = f32(q_byte & 0xF) * d + m;
                local_sum += q_lo * shared_vector[shmem_idx + j * 2 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 2 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q5_0

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 11u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        let qh0 = src0[scale_idx + 1u];
        let qh1 = src0[scale_idx + 2u];
        let qh_packed = bitcast<u32>(vec2(qh0, qh1));

        for (var j = 0u; j < 2; j++) {
            let q_0 = src0[scale_idx + 3u + block_offset + (j*2)];
            let q_1 = src0[scale_idx + 3u + block_offset + (j*2) + 1u];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));

            let j_adjusted = j + (block_offset / 2u);

            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);

                let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                let q_hi = (f32(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
                let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                let q_lo = (f32((q_byte & 0xF) | qh_lo) - 16.0) * d;

                local_sum += q_lo * shared_vector[shmem_idx + j * 4 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 4 + k + 16];
            }

        }
    }
    return local_sum;
}
#endif


#ifdef MUL_ACC_Q5_1

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 12u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        let m = src0[scale_idx + 1u];
        let qh0 = src0[scale_idx + 2u];
        let qh1 = src0[scale_idx + 3u];
        let qh_packed = bitcast<u32>(vec2(qh0, qh1));

        for (var j = 0u; j < 2; j++) {
            let q_0 = src0[scale_idx + 4u + block_offset + (j*2)];
            let q_1 = src0[scale_idx + 4u + block_offset + (j*2) + 1u];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));

            let j_adjusted = j + (block_offset / 2u);

            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);

                let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                let q_hi = f32(((q_byte >> 4) & 0xF) | qh_hi) * d + f32(m);
                let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                let q_lo = f32((q_byte & 0xF) | qh_lo) * d + f32(m);

                local_sum += q_lo * shared_vector[shmem_idx + j * 4 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 4 + k + 16];
            }

        }
    }
    return local_sum;
}
#endif


#ifdef MUL_ACC_Q8_0

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 17u;
const WEIGHTS_PER_F16 = 2u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);

        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_0 = src0[scale_idx + 1 + block_offset + j];
            let q_1 = src0[scale_idx + 1 + block_offset + j + 1];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte_i32(q_packed, k);
                let q_val = f32(q_byte) * d;
                local_sum += q_val * shared_vector[shmem_idx + j * 2 + k];
            }
        }
    }
    return local_sum;
}
#endif


#ifdef MUL_ACC_Q8_1

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 18u;
const WEIGHTS_PER_F16 = 2u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        let m = src0[scale_idx + 1u];

        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_0 = src0[scale_idx + 2u + block_offset + j];
            let q_1 = src0[scale_idx + 2u + block_offset + j + 1];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte_i32(q_packed, k);
                let q_val = f32(q_byte) * d + f32(m);
                local_sum += q_val * shared_vector[shmem_idx + j * 2 + k];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q2_K

const BLOCK_SIZE = 256;
const F16_PER_BLOCK = 42u;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig; i < tile_size; i += THREADS_PER_OUTPUT) {
        let k_global = k_outer + i;
        let block_k = k_global / BLOCK_SIZE;
        let k_in_block = k_global % BLOCK_SIZE;

        let scale_idx = (idx_base + block_k) * F16_PER_BLOCK;

        let d = f32(src0[scale_idx + 40u]);
        let dmin = f32(src0[scale_idx + 41u]);

        let block_of_32 = k_in_block / 32u;
        let pos_in_32 = k_in_block % 32u;

        let q_b_idx = (block_of_32 / 4u) * 32u;
        let shift = (block_of_32 % 4u) * 2u;
        let k = (pos_in_32 / 16u) * 16u;
        let l = pos_in_32 % 16u;

        let is = k_in_block / 16u;

        let sc_0 = src0[scale_idx + 2u * (is / 4u)];
        let sc_1 = src0[scale_idx + 2u * (is / 4u) + 1u];
        let sc_packed = bitcast<u32>(vec2(sc_0, sc_1));
        let sc = get_byte(sc_packed, is % 4u);

        let dl = d * f32(sc & 0xFu);
        let ml = dmin * f32(sc >> 4u);

        let q_idx = q_b_idx + k + l;
        let q_0 = src0[scale_idx + 8u + 2u * (q_idx / 4u)];
        let q_1 = src0[scale_idx + 8u + 2u * (q_idx / 4u) + 1u];
        let q_packed = bitcast<u32>(vec2(q_0, q_1));
        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = f32(qs_val) * dl - ml;
        local_sum += q_val * shared_vector[i];
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q3_K

const BLOCK_SIZE = 256;
const F16_PER_BLOCK = 55u;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig; i < tile_size; i += THREADS_PER_OUTPUT) {
        let k_global = k_outer + i;
        let block_k = k_global / BLOCK_SIZE;
        let k_in_block = k_global % BLOCK_SIZE;

        let scale_idx = (idx_base + block_k) * F16_PER_BLOCK;

        let d = f32(src0[scale_idx + 54u]);

        // Load and unpack scales
        let kmask1: u32 = 0x03030303u;
        let kmask2: u32 = 0x0f0f0f0fu;

        var scale_vals: array<u32, 4>;
        for (var si: u32 = 0u; si < 4u; si++) {
            let scale_0 = src0[scale_idx + 48u + (2u*si)];
            let scale_1 = src0[scale_idx + 48u + (2u*si) + 1u];
            scale_vals[si] = bitcast<u32>(vec2(scale_0, scale_1));
        }

        var tmp: u32 = scale_vals[2];
        scale_vals[2] = ((scale_vals[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        scale_vals[3] = ((scale_vals[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4u);
        scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

        // Load hmask and qs arrays
        var hmask_vals: array<u32, 8>;
        for (var hi: u32 = 0u; hi < 8u; hi++) {
            let hmask_0 = src0[scale_idx + (2u*hi)];
            let hmask_1 = src0[scale_idx + (2u*hi) + 1u];
            hmask_vals[hi] = bitcast<u32>(vec2(hmask_0, hmask_1));
        }

        var qs_vals: array<u32, 16>;
        for (var qi: u32 = 0u; qi < 16u; qi++) {
            let qs_0 = src0[scale_idx + 16u + (2u*qi)];
            let qs_1 = src0[scale_idx + 16u + (2u*qi) + 1u];
            qs_vals[qi] = bitcast<u32>(vec2(qs_0, qs_1));
        }

        let half = k_in_block / 128u;
        let pos_in_half = k_in_block % 128u;
        let shift_group = pos_in_half / 32u;
        let pos_in_32 = pos_in_half % 32u;
        let k_group = pos_in_32 / 16u;
        let l = pos_in_32 % 16u;

        let q_b_idx = half * 32u;
        let shift = shift_group * 2u;
        let k = k_group * 16u;
        let is = k_in_block / 16u;
        let m_shift = k_in_block / 32u;
        let m: u32 = 1u << m_shift;

        let sc = get_byte(scale_vals[is / 4u], is % 4u);
        let dl = d * (f32(sc) - 32.0);

        let q_idx = q_b_idx + k + l;
        let hm_idx = k + l;

        let q_byte = get_byte(qs_vals[q_idx / 4u], q_idx % 4u);
        let hmask_byte = get_byte(hmask_vals[hm_idx / 4u], hm_idx % 4u);

        let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = (f32(qs_val) - hm) * dl;
        local_sum += q_val * shared_vector[i];
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q4_K

const BLOCK_SIZE = 256;
const F16_PER_BLOCK = 72u;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig; i < tile_size; i += THREADS_PER_OUTPUT) {
        let k_global = k_outer + i;
        let block_k = k_global / BLOCK_SIZE;
        let k_in_block = k_global % BLOCK_SIZE;

        let scale_idx = (idx_base + block_k) * F16_PER_BLOCK;

        let d = f32(src0[scale_idx]);
        let dmin = f32(src0[scale_idx + 1u]);

        var scale_vals: array<u32, 3>;
        for (var si: u32 = 0u; si < 3u; si++) {
            let scale_0 = src0[scale_idx + 2u + (2u*si)];
            let scale_1 = src0[scale_idx + 2u + (2u*si) + 1u];
            scale_vals[si] = bitcast<u32>(vec2(scale_0, scale_1));
        }

        let group_of_64 = k_in_block / 64u;
        let pos_in_64 = k_in_block % 64u;
        let shift_group = pos_in_64 / 32u;
        let l = pos_in_64 % 32u;

        let q_b_idx = group_of_64 * 32u;
        let shift = shift_group * 4u;
        let is = k_in_block / 32u;

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

        let dl = d * f32(sc);
        let ml = dmin * f32(mn);

        let q_idx = q_b_idx + l;
        let q_0 = src0[scale_idx + 8u + 2u * (q_idx / 4u)];
        let q_1 = src0[scale_idx + 8u + 2u * (q_idx / 4u) + 1u];
        let q_packed = bitcast<u32>(vec2(q_0, q_1));

        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;

        let q_val = f32(qs_val) * dl - ml;
        local_sum += q_val * shared_vector[i];
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q5_K

const BLOCK_SIZE = 256;
const F16_PER_BLOCK = 88u;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig; i < tile_size; i += THREADS_PER_OUTPUT) {
        let k_global = k_outer + i;
        let block_k = k_global / BLOCK_SIZE;
        let k_in_block = k_global % BLOCK_SIZE;

        let scale_idx = (idx_base + block_k) * F16_PER_BLOCK;

        let d = f32(src0[scale_idx]);
        let dmin = f32(src0[scale_idx + 1u]);

        var scale_vals: array<u32, 3>;
        for (var si: u32 = 0u; si < 3u; si++) {
            let scale_0 = src0[scale_idx + 2u + (2u*si)];
            let scale_1 = src0[scale_idx + 2u + (2u*si) + 1u];
            scale_vals[si] = bitcast<u32>(vec2(scale_0, scale_1));
        }

        let group_of_64 = k_in_block / 64u;
        let pos_in_64 = k_in_block % 64u;
        let shift_group = pos_in_64 / 32u;
        let l = pos_in_64 % 32u;

        let q_b_idx = group_of_64 * 32u;
        let shift = shift_group * 4u;
        let is = k_in_block / 32u;
        let u_shift = k_in_block / 32u;
        let u: u32 = 1u << u_shift;

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

        let dl = d * f32(sc);
        let ml = dmin * f32(mn);

        let q_idx = q_b_idx + l;
        let q_0 = src0[scale_idx + 24u + 2u * (q_idx / 4u)];
        let q_1 = src0[scale_idx + 24u + 2u * (q_idx / 4u) + 1u];
        let q_packed = bitcast<u32>(vec2(q_0, q_1));

        let q_byte = get_byte(q_packed, q_idx % 4u);

        let qh_0 = src0[scale_idx + 8u + 2u * (l / 4u)];
        let qh_1 = src0[scale_idx + 8u + 2u * (l / 4u) + 1u];
        let qh_packed = bitcast<u32>(vec2(qh_0, qh_1));

        let qh_byte = get_byte(qh_packed, l % 4u);

        let qs_val = (q_byte >> shift) & 0xFu;
        let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);

        let q_val = (f32(qs_val) + qh_val) * dl - ml;
        local_sum += q_val * shared_vector[i];
    }
    return local_sum;
}
#endif
#ifdef MUL_ACC_Q6_K

const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 105u;

fn load_u32_at(bbase: u32, byte_offset: u32) -> u32 {
    let aligned = byte_offset & ~3u;
    let idx = bbase + aligned / 2u;
    return bitcast<u32>(vec2(src0[idx], src0[idx + 1u]));
}

fn byte_of(v: u32, b: u32) -> u32 {
    return (v >> (b * 8u)) & 0xFFu;
}

fn sbyte_of(v: u32, b: u32) -> i32 {
    let raw = i32((v >> (b * 8u)) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let tid = tig / 2u;
    let ix  = tig % 2u;
    let ip  = tid / 8u;
    let il  = tid % 8u;
    let l0  = 4u * il;
    let is  = 8u * ip + l0 / 16u;

    let y_offset   = 128u * ip + l0;
    let q_offset_l =  64u * ip + l0;
    let q_offset_h =  32u * ip + l0;

    let nb = tile_size / BLOCK_SIZE;
    let k_block_start = k_outer / BLOCK_SIZE;

    // Aligned scale byte position (is can be odd)
    let sc_base_byte = 192u + (is & ~3u);
    let sc_byte_pos  = is & 3u;

    var local_sum = 0.0;

    for (var i = ix; i < nb; i += 2u) {
        let bbase = (idx_base + k_block_start + i) * F16_PER_BLOCK;

        let d_raw = load_u32_at(bbase, 208u);
        let d = f32(bitcast<vec2<f16>>(d_raw)[0]);

        let ql1_u32  = load_u32_at(bbase, q_offset_l);
        let ql2_u32  = load_u32_at(bbase, q_offset_l + 32u);
        let qh_u32   = load_u32_at(bbase, 128u + q_offset_h);
        let sc_u32_0 = load_u32_at(bbase, sc_base_byte);
        let sc_u32_1 = load_u32_at(bbase, sc_base_byte + 4u);

        let sc0 = sbyte_of(sc_u32_0, sc_byte_pos);
        let sc2 = sbyte_of(sc_u32_0, sc_byte_pos + 2u);
        let sc4 = sbyte_of(sc_u32_1, sc_byte_pos);
        let sc6 = sbyte_of(sc_u32_1, sc_byte_pos + 2u);

        var sums = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        for (var l = 0u; l < 4u; l++) {
            let y_base = i * BLOCK_SIZE + y_offset + l;
            let yl0 = f32(shared_vector[y_base]);
            let yl1 = f32(shared_vector[y_base + 32u]);
            let yl2 = f32(shared_vector[y_base + 64u]);
            let yl3 = f32(shared_vector[y_base + 96u]);

            let q1b = byte_of(ql1_u32, l);
            let q2b = byte_of(ql2_u32, l);
            let qhb = byte_of(qh_u32,  l);

            let dq0 = f32(i32((q1b & 0x0Fu) | ((qhb & 0x03u) << 4u)) - 32);
            let dq1 = f32(i32((q2b & 0x0Fu) | ((qhb & 0x0Cu) << 2u)) - 32);
            let dq2 = f32(i32((q1b >>   4u) | ((qhb & 0x30u)       )) - 32);
            let dq3 = f32(i32((q2b >>   4u) | ((qhb & 0xC0u) >> 2u)) - 32);

            sums[0] += yl0 * dq0;
            sums[1] += yl1 * dq1;
            sums[2] += yl2 * dq2;
            sums[3] += yl3 * dq3;
        }

        local_sum += d * (sums[0] * f32(sc0) + sums[1] * f32(sc2) +
                          sums[2] * f32(sc4) + sums[3] * f32(sc6));
    }

    return local_sum;
}
#endif

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

// SRC0_TYPE and SRC1_TYPE are defined in mul_mat_decls, which is included
@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

const THREADS_PER_OUTPUT = WG_SIZE / OUTPUTS_PER_WG;

// Shared memory for collaborative loading and reduction
var<workgroup> shared_vector: array<SRC1_TYPE, TILE_K/VEC_SIZE>;  // Cache vector tile
var<workgroup> partial_sums: array<f32, WG_SIZE>;   // For reduction

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>) {
    let thread_id = local_id.x;

    // Handle batch dimensions
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    // Which of the outputs does this thread belong to?
    let thread_group = thread_id / THREADS_PER_OUTPUT;
    let thread_in_group = thread_id % THREADS_PER_OUTPUT;

    // Each workgroup computes OUTPUTS_PER_WG consecutive outputs
    let output_row = (wg_linear % output_groups) * OUTPUTS_PER_WG + thread_group;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + output_row * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    let dst_idx = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + output_row;

    var local_sum = 0.0;

    // Each thread processes multiple K elements and accumulates
    for (var k_tile = 0u; k_tile < params.k; k_tile += TILE_K) {
        let tile_size = min(TILE_K, params.k - k_tile);

        // Cooperatively load vector tile into shared memory (all threads)
        for (var i = thread_id * VEC_SIZE; i < tile_size; i += WG_SIZE * VEC_SIZE) {
            shared_vector[i / VEC_SIZE] = src1[(src1_idx_base + k_tile + i) / VEC_SIZE];
        }

        workgroupBarrier();

        if (output_row < params.m) {
            local_sum += mul_acc(thread_in_group, tile_size, src0_idx_base, k_tile);
        }

        workgroupBarrier();
    }

    // Store partial sums and reduce within each partition
    partial_sums[thread_id] = local_sum;
    workgroupBarrier();
    let group_base = thread_group * THREADS_PER_OUTPUT;
    let thread_base = group_base + thread_in_group;
    var offset: u32 = THREADS_PER_OUTPUT / 2;
    while (offset > 0) {
        if (thread_in_group < offset) {
            partial_sums[thread_base] += partial_sums[thread_base + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    // Store back to global memory
    if (output_row < params.m && thread_group % VEC_SIZE == 0 && thread_in_group == 0) {
        dst[dst_idx / VEC_SIZE] = store_val(group_base);
    }
}
