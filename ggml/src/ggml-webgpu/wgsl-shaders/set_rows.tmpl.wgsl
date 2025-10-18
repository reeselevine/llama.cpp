#define(VARIANTS)

[
  {
    "SHADER_SUFFIX": "f16_vec",
    "REPLS": {
      "TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f16>",
      "BLOCK_SIZE": 4
    },
    "DECLS": ["F16_VEC"]
  },
  {
    "SHADER_SUFFIX": "f16",
    "REPLS": {
      "TYPE" : "f32",
      "DST_TYPE": "f16",
      "BLOCK_SIZE": 1
    },
    "DECLS": ["F16"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(F16_VEC)
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let src_vec_index = (src_base + offset) / {{BLOCK_SIZE}};
    let dst_vec_index = (dst_base + offset) / {{BLOCK_SIZE}};
    dst[dst_vec_index] = vec4<f16>(src[src_vec_index]);
}
#enddecl(F16_VEC)

#decl(F16)
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    dst[dst_base + offset] = f16(src[src_base + offset]);
}
#enddecl(F16)

#end(DECLS)

#define(SHADER)

enable f16;

DECLS

@group(0) @binding(0)
var<storage, read_write> src: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{DST_TYPE}}>;

@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src
    ne0: u32,
    n_rows: u32, // n_rows = ne1 = rows per slice
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(4)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    // Determine the total number of threads based on mode
    var max_threads: u32; 
    var i: u32;
    if {{BLOCK_SIZE}} > 1 {
        // Vectorized: one thread per vector of elements
        // # of total rows to go through * (# of threads per row)
        max_threads = (params.n_rows * params.ne2 * params.ne3) * (params.ne0 / {{BLOCK_SIZE}});
        
        // calculations are based off i being row, but when vectorized, it corresponds to a vector in a row
        // getting the row from gid
        i = gid.x / (params.ne0 / {{BLOCK_SIZE}});
    } else {
        // Non-vectorized: one thread per row
        // # of total rows in the matrix 
        max_threads = params.n_rows * params.ne2 * params.ne3;
        i = gid.x; // i corresponds to the row
    }

    if (gid.x >= max_threads) {
        return;
    }


    let i_src3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_src2 = i / params.n_rows;
    let i_src1 = i % params.n_rows;

    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

    let idx_high = (params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2) * 2;

    let idx_high_val = idx[idx_high];
    let idx_low_val = idx[idx_high + 1];

    if (idx_low_val != 0) {
        // Upper bits of index are not zero, output will be incorrect
        atomicStore(&error, 1);
        return;
    }

    let i_dst_row = params.offset_dst + idx_high_val * params.stride_dst1 + i_src2 * params.stride_dst2 + i_src3 * params.stride_dst3;
    let i_src_row = params.offset_src + i_src1 * params.stride_src1 + i_src2 * params.stride_src2 + i_src3 * params.stride_src3;

    if {{BLOCK_SIZE}} > 1 {
        // Vectorized: one thread per vector of elements

        // starts at what element of that row?
        let element_offset = (gid.x % (params.ne0 / {{BLOCK_SIZE}})) * {{BLOCK_SIZE}};
        copy_elements(i_src_row, i_dst_row, element_offset);

    } else {
        // Non-vectorized: go through each element in row, copy one by one
        for (var i: u32 = 0; i < params.ne0; i++) {
            copy_elements(i_src_row, i_dst_row, i);
        }
    }

    
}

#end(SHADER)

