# WebGPU U32 Read Repro

Small standalone browser repro for the WebGPU quant-buffer read issue seen in the fast `q4_K` path.

It does not depend on `llama.cpp` runtime code. The page:

- creates a synthetic 144-byte `q4_K`-like block
- uploads it as a `storage` buffer of `u32`
- runs a compute shader using the same helper style as `common_decls.tmpl`
- prints the results of direct `src0[...]`, `load_src0_u16_at(...)`, `load_src0_u32_at(...)`, and `get_byte(...)`

## Run

Serve the repo root or this directory over HTTP and open:

- `pocs/webgpu-u32-read-repro/index.html`

Example:

```bash
python3 -m http.server
```

Then open:

- `http://localhost:8000/pocs/webgpu-u32-read-repro/`

## Goal

Compare Chrome vs Safari for the same shader and uploaded data, without involving the full `llama.cpp` graph.
