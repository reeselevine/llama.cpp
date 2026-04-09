# Potential Dawn/Metal shader bug

## Issue

The shader in `mul_mat_reg_tile_q4_K_f32_exact.wgsl` reads three fixed `u32` words from a storage buffer and extracts bytes from them inside a small loop.

Expected debug output is:

- `dbg[0] = 0xdeadbeef`
- `dbg[1] = 1`
- `dbg[2] = 2`
- `dbg[3] = 3`

Observed on Apple Metal via Dawn:

- default settings: fails, `dbg[1..3] = 0`
- with `disable_robustness`: passes
- with `disable_polyfills_on_integer_div_and_mod`: passes

So either of those Dawn toggles is sufficient to make the repro pass.

## Browser Repro

Serve this directory over HTTP and open:

- `index.html`

Example:

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/`

The page prints expected and actual debug values. Testing shows this fails on Chrome on Apple Silicon, but passes on Safari.

## Native Runner

This directory also contains a small Dawn-based native runner in `run-repro.cpp`.

Build it directly from this subdirectory:

```bash
cmake -S . -B build
cmake --build build --target webgpu-u32-read-repro-runner
```

Run:

```bash
./build/webgpu-u32-read-repro-runner --shader mul_mat_reg_tile_q4_K_f32_exact.wgsl
```

Useful toggle variants:

```bash
./build/webgpu-u32-read-repro-runner --shader mul_mat_reg_tile_q4_K_f32_exact.wgsl --disable-robustness
./build/webgpu-u32-read-repro-runner --shader mul_mat_reg_tile_q4_K_f32_exact.wgsl --disable-polyfill-divmod
./build/webgpu-u32-read-repro-runner --shader mul_mat_reg_tile_q4_K_f32_exact.wgsl --disable-robustness --disable-polyfill-divmod
```

If Dawn is not discoverable via CMake:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/dawn/install
```
