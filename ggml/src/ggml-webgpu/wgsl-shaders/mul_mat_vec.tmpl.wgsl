#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "f32_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f32>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f32_f32",
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f16_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f16>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f16",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "q4_0_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q4_0"]
  },
  {
    "SHADER_SUFFIX": "q4_1_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q4_1"]
  },
  {
    "SHADER_SUFFIX": "q5_0_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q5_0"]
  },
  {
    "SHADER_SUFFIX": "q5_1_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q5_1"]
  },
  {
    "SHADER_SUFFIX": "q8_0_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q8_0"]
  },
  {
    "SHADER_SUFFIX": "q8_1_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q8_1"]
  },
  {
    "SHADER_SUFFIX": "q2_k_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q2_K"]
  },
  {
    "SHADER_SUFFIX": "q3_k_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q3_K"]
  },
  {
    "SHADER_SUFFIX": "q4_k_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q4_K"]
  },
  {
    "SHADER_SUFFIX": "q5_k_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q5_K"]
  },
  {
    "SHADER_SUFFIX": "q6_k_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q6_K"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(VEC)
fn inner_dot(src0_val: {{SRC0_TYPE}}, src1_val: {{SRC1_TYPE}}) -> f32 {
    return f32(dot({{SRC1_TYPE}}(src0_val), src1_val));
}

fn store_val(group_base: u32) -> vec4<f32> {
    return vec4<f32>(partial_sums[group_base],
                     partial_sums[group_base + THREADS_PER_OUTPUT],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 2],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 3]);
}
#enddecl(VEC)

#decl(SCALAR)
fn inner_dot(src0_val: {{SRC0_TYPE}}, src1_val: {{SRC1_TYPE}}) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

fn store_val(group_base: u32) -> f32 {
    return partial_sums[group_base];
}
#enddecl(SCALAR)

#decl(MUL_ACC_FLOAT)

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * {{VEC_SIZE}}; i < tile_size; i += THREADS_PER_OUTPUT * {{VEC_SIZE}}) {
        let a = src0[(idx_base + k_outer + i) / {{VEC_SIZE}}];
        let b = shared_vector[i / {{VEC_SIZE}}];
        local_sum += inner_dot(a, b);
    }
    return local_sum;
}

#enddecl(MUL_ACC_FLOAT)

#decl(MUL_ACC_Q4_0)

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

#enddecl(MUL_ACC_Q4_0)

#decl(MUL_ACC_Q4_1)

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

#enddecl(MUL_ACC_Q4_1)

#decl(MUL_ACC_Q5_0)

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

#enddecl(MUL_ACC_Q5_0)


#decl(MUL_ACC_Q5_1)

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
#enddecl(MUL_ACC_Q5_1)


#decl(MUL_ACC_Q8_0)

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

#enddecl(MUL_ACC_Q8_0)


#decl(MUL_ACC_Q8_1)

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

#enddecl(MUL_ACC_Q8_1)


#decl(MUL_ACC_Q2_K)


const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 42u;
const NQ = 16u; // number of quantized elements per thread
const TILE_M = 8u;

// tig = thread_id % THREADS_PER_OUTPUT;
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;

    let block_k = k_outer / BLOCK_SIZE;
    if (block_k >= params.k / BLOCK_SIZE) {
        return 0.0;
    }

    let slice_start = k_outer;
    let slice_end   = k_outer + TILE_K;

    let src0_idx = idx_base + block_k;
    let scale_idx = src0_idx * F16_PER_BLOCK;

    let d    = src0[scale_idx + 40u];
    let dmin = src0[scale_idx + 41u];

    var is: u32 = 0u;
    var element: u32 = 0u; 

    for (var q_b_idx = 0u; q_b_idx < 64u; q_b_idx += 32u) {
        for (var shift = 0u; shift < 8u; shift += 2u) {
            for (var k = 0u; k < 32u; k += 16u) {

                let sc_0 = src0[scale_idx + 2u * (is / 4u)];
                let sc_1 = src0[scale_idx + 2u * (is / 4u) + 1u];
                let sc_packed = bitcast<u32>(vec2(sc_0, sc_1));

                let sc = get_byte(sc_packed, is % 4u);
                is++;

                let dl = f32(d) * f32(sc & 0xF);
                let ml = f32(dmin) * f32(sc >> 4);

                for (var l = 0u; l < 16u; l++) {

                    if ((element % 4u) == tig) {
                        let k_global = block_k * BLOCK_SIZE + element;

                        if (k_global >= slice_start && k_global < slice_end) {

                            let q_idx = q_b_idx + k + l;

                            let q_0 = src0[scale_idx + 8u + 2u * (q_idx / 4u)];
                            let q_1 = src0[scale_idx + 8u + 2u * (q_idx / 4u) + 1u];
                            let q_packed = bitcast<u32>(vec2(q_0, q_1));

                            let q_byte = get_byte(q_packed, q_idx % 4u);
                            let qs_val = (q_byte >> shift) & 3u;

                            let q_val = f32(qs_val) * dl - ml;

                            let tile_k = k_global - slice_start;
                            if (tile_k < TILE_K) {
                                local_sum += q_val * shared_vector[tile_k];
                            }
                        }

                    }

                    element++; 
                }
            }
        }
    }
    

    return local_sum;
}

#enddecl(MUL_ACC_Q2_K)

#decl(MUL_ACC_Q3_K)


const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 55u;
const NQ = 16u; // number of quantized elements per thread
const TILE_M = 8u;

// tig = thread_id % THREADS_PER_OUTPUT;
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;

    let block_k = k_outer / BLOCK_SIZE;
    if (block_k >= params.k / BLOCK_SIZE) {
        return 0.0;
    }

    let slice_start = k_outer;
    let slice_end   = k_outer + TILE_K;

    let src0_idx = idx_base + block_k;
    let scale_idx = src0_idx * F16_PER_BLOCK;
    
    let d    = f32(src0[scale_idx + 54u]);

    let kmask1: u32 = 0x03030303u;
    let kmask2: u32 = 0x0f0f0f0fu;
    
    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0u; i < 4u; i++) {
        let scale_0 = src0[scale_idx + 48u + (2u*i)];
        let scale_1 = src0[scale_idx + 48u + (2u*i) + 1u];
        scale_vals[i] = bitcast<u32>(vec2(scale_0, scale_1));
    }
    
    var tmp: u32 = scale_vals[2];
    scale_vals[2] = ((scale_vals[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    scale_vals[3] = ((scale_vals[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4u);
    scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

    var hmask_vals: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i++) {
        let hmask_0 = src0[scale_idx + (2u*i)];
        let hmask_1 = src0[scale_idx + (2u*i) + 1u];
        hmask_vals[i] = bitcast<u32>(vec2(hmask_0, hmask_1));
    }

    var qs_vals: array<u32, 16>;
    for (var i: u32 = 0u; i < 16u; i++) {
        let qs_0 = src0[scale_idx + 16u + (2u*i)];
        let qs_1 = src0[scale_idx + 16u + (2u*i) + 1u];
        qs_vals[i] = bitcast<u32>(vec2(qs_0, qs_1));
    }

    var is: u32 = 0u;
    var element: u32 = 0u; 
    var m: u32 = 1u;

    for (var q_b_idx = 0u; q_b_idx < 64u; q_b_idx += 32u) {
        for (var shift = 0u; shift < 8u; shift += 2u) {
            for (var k = 0u; k < 32u; k += 16u) {

                let sc = get_byte(scale_vals[is/4], is % 4u);
                is++;

                let dl = d * (f32(sc) - 32.0);

                for (var l = 0u; l < 16u; l++) {

                    if ((element % 4u) == tig) {
                        let k_global = block_k * BLOCK_SIZE + element;

                        if (k_global >= slice_start && k_global < slice_end) {

                            let q_idx = q_b_idx + k + l;
                            let hm_idx = k + l;

                            let q_byte = get_byte(qs_vals[q_idx / 4], q_idx % 4u);
                            let hmask_byte = get_byte(hmask_vals[hm_idx / 4], hm_idx % 4);

                            let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
                            let qs_val = (q_byte >> shift) & 3u;

                            let q_val = (f32(qs_val) - hm) * dl;

                            let tile_k = k_global - slice_start;

                            if (tile_k < TILE_K) {
                                local_sum += q_val * shared_vector[tile_k];
                            }
                        }

                    }

                    element++; 
                }
            }

            m <<= 1;
        }
    }
    

    return local_sum;
}

#enddecl(MUL_ACC_Q3_K)

#decl(MUL_ACC_Q4_K)

const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 72u;
const NQ = 4u;

// tig = thread_id % THREADS_PER_OUTPUT;
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;

    let block_k = k_outer / BLOCK_SIZE;
    if (block_k >= params.k / BLOCK_SIZE) {
        return 0.0;
    }

    let slice_start = k_outer;
    let slice_end   = k_outer + TILE_K;

    let src0_idx = idx_base + block_k;
    let scale_idx = src0_idx * F16_PER_BLOCK;

    let d    = f32(src0[scale_idx]);
    let dmin = f32(src0[scale_idx + 1u]);

    var is: u32 = 0u;
    var element: u32 = 0u; 

    var scale_vals: array<u32, 3>;
    for (var i: u32 = 0u; i < 3u; i++) {
        let scale_0 = src0[scale_idx + 2u + (2u*i)];
        let scale_1 = src0[scale_idx + 2u + (2u*i) + 1u];
        scale_vals[i] = bitcast<u32>(vec2(scale_0, scale_1));
    }

    for (var q_b_idx = 0u; q_b_idx < 128u; q_b_idx += 32u) {
        for (var shift = 0u; shift < 8u; shift += 4u) {
            
            var sc: u32;
            var mn: u32;

            if (is < 4u) {
                let sc_byte  = get_byte(scale_vals[is / 4u], is % 4u);
                let min_byte  = get_byte(scale_vals[(is + 4u) / 4u], is % 4u);
                sc = sc_byte & 63u;
                mn = min_byte & 63u;
            } else {
                let sc_min_lo = get_byte(scale_vals[(is + 4u) / 4u], (is + 4u) % 4u);
                let sc_hi = get_byte(scale_vals[(is - 4u) / 4u], (is - 4u) % 4u);
                let min_hi = get_byte(scale_vals[is / 4u], is % 4u);

                sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
                mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
            }

            is++;

            let dl = d    * f32(sc);
            let ml = dmin * f32(mn);

            for (var l = 0u; l < 32u; l++) {

                if ((element % NQ) == tig) {
                    let k_global = block_k * BLOCK_SIZE + element;

                    if (k_global >= slice_start && k_global < slice_end) {

                        let q_idx = q_b_idx + l;

                        let q_0 = src0[scale_idx + 8u + 2u * (q_idx / 4u)];
                        let q_1 = src0[scale_idx + 8u + 2u * (q_idx / 4u) + 1u];
                        let q_packed = bitcast<u32>(vec2(q_0, q_1));

                        let q_byte = get_byte(q_packed, q_idx % 4u);
                        let qs_val = (q_byte >> shift) & 0xFu;

                        let q_val = f32(qs_val) * dl - ml;

                        let tile_k = k_global - slice_start;
                        if (tile_k < TILE_K) {
                            local_sum += q_val * shared_vector[tile_k];
                        }
                    }
                }

                element++; 
            }
        }
    }

    return local_sum;
}

#enddecl(MUL_ACC_Q4_K)


#decl(MUL_ACC_Q5_K)


const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 88u;
const NQ = 16u; // number of quantized elements per thread
const TILE_M = 8u;

// tig = thread_id % THREADS_PER_OUTPUT;
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;

    let block_k = k_outer / BLOCK_SIZE;
    if (block_k >= params.k / BLOCK_SIZE) {
        return 0.0;
    }

    let slice_start = k_outer;
    let slice_end   = k_outer + TILE_K;

    let src0_idx = idx_base + block_k;
    let scale_idx = src0_idx * F16_PER_BLOCK;

    let d    = f32(src0[scale_idx]);
    let dmin = f32(src0[scale_idx + 1u]);

    var is: u32 = 0u;
    var element: u32 = 0u; 
    var u: u32 = 1u;

    // ---- load packed scales (3 u32) ----
    var scale_vals: array<u32, 3>;
    for (var i: u32 = 0u; i < 3u; i++) {
        let scale_0 = src0[scale_idx + 2u + (2u*i)];
        let scale_1 = src0[scale_idx + 2u + (2u*i) + 1u];
        scale_vals[i] = bitcast<u32>(vec2(scale_0, scale_1));
    }

    for (var q_b_idx = 0u; q_b_idx < 128u; q_b_idx += 32u) {
        for (var shift = 0u; shift < 8u; shift += 4u) {
            
            var sc: u32;
            var mn: u32;

            if (is < 4u) {
                let sc_byte  = get_byte(scale_vals[is / 4], is % 4);
                let min_byte  = get_byte(scale_vals[(is + 4) / 4], is % 4);
                sc = sc_byte & 63u;
                mn = min_byte & 63u;
            } else {
                let sc_min_lo = get_byte(scale_vals[(is + 4) / 4], (is + 4) % 4);
                let sc_hi = get_byte(scale_vals[(is - 4) / 4], (is - 4) % 4);
                let min_hi = get_byte(scale_vals[is / 4], is % 4);

                sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
                mn = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);

            }

            is++;

            let dl = d    * f32(sc);
            let ml = dmin * f32(mn);

            for (var l = 0u; l < 32u; l++) {

                if ((element % 4u) == tig) {
                    let k_global = block_k * BLOCK_SIZE + element;

                    if (k_global >= slice_start && k_global < slice_end) {

                        let q_idx = q_b_idx + l;

                        let q_0 = src0[scale_idx + 24u + 2u * (q_idx / 4u)];
                        let q_1 = src0[scale_idx + 24u + 2u * (q_idx / 4u) + 1u];
                        let q_packed = bitcast<u32>(vec2(q_0, q_1));

                        let q_byte = get_byte(q_packed, q_idx % 4u);


                        let qh_0 = src0[scale_idx + 8u + 2u * (l / 4u)];
                        let qh_1 = src0[scale_idx + 8u + 2u * (l / 4u) + 1u];
                        let qh_packed = bitcast<u32>(vec2(qh_0, qh_1));

                        let qh_byte = get_byte(qh_packed, l % 4u);

                        let qs_val = (q_byte >> shift) & 0xF;
                        let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);

                        let q_val = (f32(qs_val) + qh_val) * dl - ml;

                        let tile_k = k_global - slice_start;
                        if (tile_k < TILE_K) {
                          local_sum += q_val * shared_vector[tile_k];
                        }
                    }

                }

                element++; 
            }
            u <<= 1;
        }
    }
    

    return local_sum;
}

#enddecl(MUL_ACC_Q5_K)

#decl(MUL_ACC_Q6_K)

const BLOCK_SIZE = 256u;
const F16_PER_BLOCK = 105u;

// tig = thread_id % THREADS_PER_OUTPUT;  0-->3
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;

    let block_k = k_outer / BLOCK_SIZE;
    if (block_k >= params.k / BLOCK_SIZE) {
        return 0.0;
    }

    let slice_start = k_outer;
    let slice_end   = k_outer + TILE_K;

    let src0_idx = idx_base + block_k;
    let scale_idx = src0_idx * F16_PER_BLOCK;

    let d = f32(src0[scale_idx + 104u]);

    var ql_vals: array<u32, 32>;
    for (var i: u32 = 0u; i < 32u; i++) {
        ql_vals[i] = bitcast<u32>(vec2(
            src0[scale_idx + 2u*i],
            src0[scale_idx + 2u*i + 1u]
        ));
    }

    var qh_vals: array<u32, 16>;
    for (var i: u32 = 0u; i < 16u; i++) {
        qh_vals[i] = bitcast<u32>(vec2(
            src0[scale_idx + 64u + 2u*i],
            src0[scale_idx + 64u + 2u*i + 1u]
        ));
    }

    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0u; i < 4u; i++) {
        scale_vals[i] = bitcast<u32>(vec2(
            src0[scale_idx + 96u + 2u*i],
            src0[scale_idx + 96u + 2u*i + 1u]
        ));
    }

    var element: u32 = 0u;
    var qh_b_idx: u32 = 0u;
    var sc_b_idx: u32 = 0u;
    var k_base: u32 = 0u;

    for (var ql_b_idx: u32 = 0u; ql_b_idx < 128u; ql_b_idx += 64u) {
        for (var l: u32 = 0u; l < 32u; l++) {
            let ql13_b = get_byte(ql_vals[(ql_b_idx + l) / 4u], (ql_b_idx + l) % 4u);
            let ql24_b = get_byte(ql_vals[(ql_b_idx + l + 32u) / 4u], (ql_b_idx + l + 32u) % 4u);
            let qh_b = get_byte(qh_vals[(qh_b_idx + l) / 4u], (qh_b_idx + l) % 4u);

            let q1 = f32((ql13_b & 0xFu) | ((qh_b & 3u) << 4u)) - 32.0;
            let q2 = f32((ql24_b & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - 32.0;
            let q3 = f32((ql13_b >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - 32.0;
            let q4 = f32((ql24_b >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - 32.0;

            let is = l / 16u;
            let is1 = sc_b_idx + is;
            let sc1 = f32(get_byte_i32(scale_vals[is1 / 4u], is1 % 4u));
            let is2 = sc_b_idx + is + 2u;
            let sc2 = f32(get_byte_i32(scale_vals[is2 / 4u], is2 % 4u));
            let is3 = sc_b_idx + is + 4u;
            let sc3 = f32(get_byte_i32(scale_vals[is3 / 4u], is3 % 4u));
            let is4 = sc_b_idx + is + 6u;
            let sc4 = f32(get_byte_i32(scale_vals[is4 / 4u], is4 % 4u));

            // Each of the 4 threads handles every 4th element
            // q1 at element+0, q2 at element+1, q3 at element+2, q4 at element+3
            
            // q1
            if ((element % 4u) == tig) {
                let kg = block_k * BLOCK_SIZE + k_base + l;
                if (kg >= slice_start && kg < slice_end) {
                    let tile_k = kg - slice_start;
                    if (tile_k < TILE_K) {
                        local_sum += (d * sc1 * q1) * shared_vector[tile_k];
                    }
                }
            }
            element++;

            // q2
            if ((element % 4u) == tig) {
                let kg = block_k * BLOCK_SIZE + k_base + l + 32u;
                if (kg >= slice_start && kg < slice_end) {
                    let tile_k = kg - slice_start;
                    if (tile_k < TILE_K) {
                        local_sum += (d * sc2 * q2) * shared_vector[tile_k];
                    }
                }
            }
            element++;

            // q3
            if ((element % 4u) == tig) {
                let kg = block_k * BLOCK_SIZE + k_base + l + 64u;
                if (kg >= slice_start && kg < slice_end) {
                    let tile_k = kg - slice_start;
                    if (tile_k < TILE_K) {
                        local_sum += (d * sc3 * q3) * shared_vector[tile_k];
                    }
                }
            }
            element++;

            // q4
            if ((element % 4u) == tig) {
                let kg = block_k * BLOCK_SIZE + k_base + l + 96u;
                if (kg >= slice_start && kg < slice_end) {
                    let tile_k = kg - slice_start;
                    if (tile_k < TILE_K) {
                        local_sum += (d * sc4 * q4) * shared_vector[tile_k];
                    }
                }
            }
            element++;
        }

        k_base += 128u;
        qh_b_idx += 32u;
        sc_b_idx += 8u;
    }

    return local_sum;
}

#enddecl(MUL_ACC_Q6_K)



#end(DECLS)

#define(SHADER)
enable f16;

DECLS

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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // Matrix (M x K)
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // Vector (K x 1, transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{DST_TYPE}}>;  // Result vector (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

override WORKGROUP_SIZE: u32;
override TILE_K: u32;
override OUTPUTS_PER_WG: u32;
override THREADS_PER_OUTPUT = WORKGROUP_SIZE / OUTPUTS_PER_WG;

// Shared memory for collaborative loading and reduction
var<workgroup> shared_vector: array<{{SRC1_TYPE}}, TILE_K/{{VEC_SIZE}}>;  // Cache vector tile
var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE>;   // For reduction

@compute @workgroup_size(WORKGROUP_SIZE)
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
        for (var i = thread_id * {{VEC_SIZE}}; i < tile_size; i += WORKGROUP_SIZE * {{VEC_SIZE}}) {
            shared_vector[i / {{VEC_SIZE}}] = src1[(src1_idx_base + k_tile + i) / {{VEC_SIZE}}];
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
    var offset = THREADS_PER_OUTPUT / 2;
    while (offset > 0) {
        if (thread_in_group < offset) {
            partial_sums[thread_base] += partial_sums[thread_base + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    // Store back to global memory
    if (output_row < params.m && thread_group % {{VEC_SIZE}} == 0 && thread_in_group == 0) {
        dst[dst_idx / {{VEC_SIZE}}] = store_val(group_base);
    }
}
#end(SHADER)