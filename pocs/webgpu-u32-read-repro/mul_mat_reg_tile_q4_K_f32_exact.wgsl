fn store_val(acc: array<array<f32, 8u>, 8u>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}

@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> debug_out: array<u32>;

@compute @workgroup_size(1)
fn main() {

    debug_out[0] = 0xdeadbeefu;

    for (var k_outer = 0u; k_outer < 96; k_outer += 32u) {
        var scale_vals: array<u32, 3>;
        for (var i: u32 = 0u; i < 3u; i++) {
            scale_vals[i] = src0[i];
        }
        let is = k_outer / 32u;
        let sc = (scale_vals[is / 4u] >> ((is % 4u) * 8u)) & 0xFFu;
        debug_out[1u + is] = sc;
    }

    var acc: array<array<f32, 8u>, 8u>;

    // not necessary but avoids even potential uninitialized access
    for (var tn = 0u; tn < 8u; tn++) {
        for (var tm = 0u; tm < 8u; tm += 1) {
            acc[tm][tn] = 1.0f;;
        }
    }

    for (var tn = 0u; tn < 2u; tn ++) {
        for (var tm = 0u; tm < 4u; tm += 1) {
            let dst_idx = tn + tm;
            dst[dst_idx] = store_val(acc, tn, tm);
        }
    }
}
