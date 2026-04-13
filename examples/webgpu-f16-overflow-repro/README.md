# WebGPU f16 Overflow Repro

Small native WebGPU/Dawn repro for `f32 -> f16` overflow behavior in a compute shader.

It dispatches a WGSL shader that computes:

- `f16(exp(x))`
- `f16(exp(x) - 1.0)`

for a few carefully chosen `f32` inputs near and above the `f16` overflow threshold.

## Build

Configure llama.cpp with WebGPU enabled:

```bash
cmake -B build -DGGML_WEBGPU=ON
cmake --build build --target webgpu-f16-overflow-repro
```

## Run

```bash
./build/bin/webgpu-f16-overflow-repro
```

The program prints:

- adapter/backend information
- the input `x`
- CPU reference `exp(x)` / `exp(x)-1`
- the raw half bits produced by the shader
- the decoded half values

This directory is intentionally self-contained so it can be copied into a standalone WebGPU or Dawn bug report.
