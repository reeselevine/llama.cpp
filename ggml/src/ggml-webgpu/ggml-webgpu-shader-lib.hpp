#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml.h"
#include "pre_wgsl.hpp"

#include <string>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES 2
#define GGML_WEBGPU_F32_SIZE_BYTES 4
#define GGML_WEBGPU_I32_SIZE_BYTES 4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE 128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD 256u

#define GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE 512u

// helper function for replacing {{PLACEHOLDERS}}
inline void ggml_webgpu_replace_placeholder(std::string &shader_code,
                                            const std::string &key,
                                            const std::string &value) {
  std::string pattern = "{{" + key + "}}";
  size_t pos = 0;
  while ((pos = shader_code.find(pattern, pos)) != std::string::npos) {
    shader_code.replace(pos, pattern.length(), value);
    pos += value.length();
  }
}

struct ggml_webgpu_processed_shader {
  std::string wgsl;
  std::string variant;
  void *decisions;
};

// Same hash combine function as in boost
template <typename T>
inline void ggml_webgpu_hash_combine(size_t &seed, const T &value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/** FlashAttention */

struct ggml_webgpu_flash_attn_pipeline_key {
  ggml_type kv_type;
  uint32_t head_dim_qk;
  uint32_t head_dim_v;
  bool kv_direct;
  bool has_mask;
  bool has_sinks;
  bool uses_logit_softcap;

  bool operator==(const ggml_webgpu_flash_attn_pipeline_key &other) const {
    return kv_type == other.kv_type && head_dim_qk == other.head_dim_qk &&
           head_dim_v == other.head_dim_v && kv_direct == other.kv_direct &&
           has_mask == other.has_mask && has_sinks == other.has_sinks &&
           uses_logit_softcap == other.uses_logit_softcap;
  }
};

struct ggml_webgpu_flash_attn_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_flash_attn_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.kv_type);
    ggml_webgpu_hash_combine(seed, key.head_dim_qk);
    ggml_webgpu_hash_combine(seed, key.head_dim_v);
    ggml_webgpu_hash_combine(seed, key.kv_direct);
    ggml_webgpu_hash_combine(seed, key.has_mask);
    ggml_webgpu_hash_combine(seed, key.has_sinks);
    ggml_webgpu_hash_combine(seed, key.uses_logit_softcap);
    return seed;
  }
};

struct ggml_webgpu_flash_attn_shader_lib_context {
  ggml_webgpu_flash_attn_pipeline_key key;
  uint32_t sg_mat_m;
  uint32_t sg_mat_n;
  uint32_t sg_mat_k;
  size_t wg_mem_limit_bytes;
  uint32_t max_subgroup_size;
};

struct ggml_webgpu_flash_attn_shader_decisions {
  uint32_t q_tile = 0;
  uint32_t kv_tile = 0;
  uint32_t wg_size = 0;
};

// This is exposed because it's necessary in supports_op
inline size_t
ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile, uint32_t kv_tile,
                                    uint32_t head_dim_qk, uint32_t head_dim_v,
                                    bool has_mask, bool kv_direct) {
  const uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
  size_t f16_elems = 0;
  size_t f32_elems = 0;
  f16_elems += q_tile * head_dim_qk; // q_shmem
  if (!kv_direct) {
    f16_elems += kv_tile * max_head_dim; // kv_shmem
  }
  f16_elems += q_tile * head_dim_v; // o_shmem
  if (has_mask) {
    f16_elems += q_tile * kv_tile; // mask_shmem
  }
  f16_elems += q_tile * kv_tile; // inter_shmem
  f32_elems += q_tile;           // row_max_shmem
  f32_elems += q_tile;           // exp_sum_shmem
  return f16_elems * GGML_WEBGPU_F16_SIZE_BYTES +
         f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

static uint32_t ggml_webgpu_flash_attn_max_kv_tile(
    const ggml_webgpu_flash_attn_shader_lib_context &context) {
  const size_t limit_bytes = context.wg_mem_limit_bytes;
  const size_t q_tile = context.sg_mat_m;
  const size_t base_q_bytes =
      (context.key.head_dim_qk + context.key.head_dim_v) * q_tile *
          GGML_WEBGPU_F16_SIZE_BYTES +
      2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
  size_t bytes_per_kv = 0;
  if (!context.key.kv_direct) {
    bytes_per_kv += std::max(context.key.head_dim_qk, context.key.head_dim_v);
  }
  if (context.key.has_mask) {
    bytes_per_kv += q_tile;
  }
  bytes_per_kv += q_tile;
  bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
  const uint32_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
  return (max_kv_tile / context.sg_mat_n) * context.sg_mat_n;
}

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_flash_attn_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_flash_attn_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "flash_attn";

  switch (context.key.kv_type) {
  case GGML_TYPE_F32:
    defines.push_back("KV_F32");
    break;
  case GGML_TYPE_F16:
    defines.push_back("KV_F16");
    break;
  case GGML_TYPE_Q4_0:
    defines.push_back("KV_Q4_0");
    break;
  case GGML_TYPE_Q8_0:
    defines.push_back("KV_Q8_0");
    break;
  default:
    GGML_ABORT("Unsupported KV type for flash attention shader");
  }
  variant += std::string("_") + ggml_type_name(context.key.kv_type);

  if (context.key.has_mask) {
    defines.push_back("MASK");
    variant += "_mask";
  }
  if (context.key.has_sinks) {
    defines.push_back("SINKS");
    variant += "_sinks";
  }
  if (context.key.uses_logit_softcap) {
    defines.push_back("LOGIT_SOFTCAP");
    variant += "_lgsc";
  }

  if (context.key.kv_direct) {
    defines.push_back("KV_DIRECT");
    variant += "_kvdirect";
  }

  defines.push_back(std::string("HEAD_DIM_QK=") +
                    std::to_string(context.key.head_dim_qk));
  variant += std::string("_hsqk") + std::to_string(context.key.head_dim_qk);

  defines.push_back(std::string("HEAD_DIM_V=") +
                    std::to_string(context.key.head_dim_v));
  variant += std::string("_hsv") + std::to_string(context.key.head_dim_v);
  // For now these are not part of the variant name
  defines.push_back(std::string("SG_MAT_M=") +
                    std::to_string(context.sg_mat_m));
  defines.push_back(std::string("SG_MAT_N=") +
                    std::to_string(context.sg_mat_n));
  defines.push_back(std::string("SG_MAT_K=") +
                    std::to_string(context.sg_mat_k));

  // Add chosen Q/KV tile sizes
  uint32_t q_tile = context.sg_mat_m;
  uint32_t kv_tile =
      std::min(ggml_webgpu_flash_attn_max_kv_tile(context),
               context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
  if (context.key.kv_direct) {
    GGML_ASSERT(kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
    // Avoids having to use bounds-checks and decreasing performance for direct
    // KV loads
    while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
      kv_tile -= context.sg_mat_n;
    }
  }

  defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
  defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));

  // workgroup size
  uint32_t wg_size = std::max(context.max_subgroup_size,
                              GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);

  defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_flash_attn_shader_decisions *decisions =
      new ggml_webgpu_flash_attn_shader_decisions();
  decisions->q_tile = q_tile;
  decisions->kv_tile = kv_tile;
  decisions->wg_size = wg_size;
  result.decisions = decisions;
  return result;
}

/** Generic **/

struct ggml_webgpu_generic_shader_lib_context {
  int vec4;
  uint32_t max_wg_size;
};

struct ggml_webgpu_generic_shader_decisions {
  uint32_t wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_generic_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_generic_shader_lib_context &context,
    const std::string &base_variant) {
  std::vector<std::string> defines;
  std::string variant = base_variant;

  if (context.vec4) {
    defines.push_back("VEC4");
    variant += "_vec";
  }

  defines.push_back(std::string("WG_SIZE=") +
                    std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  return result;
}

/** Pad **/

struct ggml_webgpu_pad_pipeline_key {
  bool circular;

  bool operator==(const ggml_webgpu_pad_pipeline_key &other) const {
    return circular == other.circular;
  }
};

struct ggml_webgpu_pad_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_pad_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.circular);
    return seed;
  }
};

struct ggml_webgpu_pad_shader_lib_context {
  ggml_webgpu_pad_pipeline_key key;
  uint32_t max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_pad_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_pad_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "pad";

  if (context.key.circular) {
    defines.push_back("CIRCULAR");
    variant += "_circular";
  }

  defines.push_back(std::string("WG_SIZE=") +
                    std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_generic_shader_decisions *decisions =
      new ggml_webgpu_generic_shader_decisions();
  decisions->wg_size = context.max_wg_size;
  result.decisions = decisions;
  return result;
}

/** Argsort **/

struct ggml_webgpu_argsort_shader_lib_context {
  uint32_t max_wg_size;
  size_t wg_mem_limit_bytes;
  int32_t order;
};

struct ggml_webgpu_argsort_shader_decisions {
  uint32_t wg_size = 0;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_argsort_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_argsort_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "argsort";
  defines.push_back(std::string("ORDER=") + std::to_string(context.order));
  variant += std::string("_order") + std::to_string(context.order);
  uint32_t wg_size = 1;
  while (wg_size * 2 <= context.max_wg_size &&
         wg_size * GGML_WEBGPU_I32_SIZE_BYTES <=
             context.wg_mem_limit_bytes / 2) {
    wg_size *= 2;
  }
  defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_argsort_shader_decisions *decisions =
      new ggml_webgpu_argsort_shader_decisions();
  decisions->wg_size = wg_size;
  result.decisions = decisions;
  return result;
}

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_argsort_merge_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_argsort_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "argsort_merge";
  defines.push_back(std::string("ORDER=") + std::to_string(context.order));
  variant += std::string("_order") + std::to_string(context.order);
  uint32_t wg_size =
      std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
  defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_argsort_shader_decisions *decisions =
      new ggml_webgpu_argsort_shader_decisions();
  decisions->wg_size = wg_size;
  result.decisions = decisions;
  return result;
}

/** Set Rows **/

struct ggml_webgpu_set_rows_pipeline_key {
  int dst_type;
  int vec4;
  int i64_idx;

  bool operator==(const ggml_webgpu_set_rows_pipeline_key &other) const {
    return dst_type == other.dst_type && vec4 == other.vec4 &&
           i64_idx == other.i64_idx;
  }
};

struct ggml_webgpu_set_rows_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_set_rows_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.dst_type);
    ggml_webgpu_hash_combine(seed, key.vec4);
    ggml_webgpu_hash_combine(seed, key.i64_idx);
    return seed;
  }
};

struct ggml_webgpu_set_rows_shader_lib_context {
  ggml_webgpu_set_rows_pipeline_key key;
  uint32_t max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_set_rows_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_set_rows_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "set_rows";

  switch (context.key.dst_type) {
  case GGML_TYPE_F32:
    defines.push_back("DST_F32");
    variant += "_dstf32";
    break;
  case GGML_TYPE_F16:
    defines.push_back("DST_F16");
    variant += "_dstf16";
    break;
  default:
    GGML_ABORT("Unsupported dst type for set_rows shader");
  }

  if (context.key.vec4) {
    defines.push_back("VEC4");
    variant += "_vec";
  }
  if (context.key.i64_idx) {
    defines.push_back("I64_IDX");
    variant += "_i64idx";
  }

  defines.push_back(std::string("WG_SIZE=") +
                    std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_generic_shader_decisions *decisions =
      new ggml_webgpu_generic_shader_decisions();
  decisions->wg_size = context.max_wg_size;
  result.decisions = decisions;
  return result;
}

struct ggml_webgpu_unary_pipeline_key {
  int type;
  int op;
  bool is_unary; // many unary operators fall under the GGML_OP_UNARY umbrella
  bool inplace;

  bool operator==(const ggml_webgpu_unary_pipeline_key &other) const {
    return type == other.type && op == other.op && is_unary == other.is_unary &&
           inplace == other.inplace;
  }
};

struct ggml_webgpu_unary_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_unary_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.type);
    ggml_webgpu_hash_combine(seed, key.op);
    ggml_webgpu_hash_combine(seed, key.is_unary);
    ggml_webgpu_hash_combine(seed, key.inplace);
    return seed;
  }
};

struct ggml_webgpu_unary_shader_lib_context {
  ggml_webgpu_unary_pipeline_key key;
  uint32_t max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_unary_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_unary_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = context.key.is_unary
                            ? ggml_unary_op_name((ggml_unary_op)context.key.op)
                            : ggml_op_name((ggml_op)context.key.op);
  // Operation-specific behavior
  defines.push_back(variant);

  switch (context.key.type) {
  case GGML_TYPE_F32:
    defines.push_back("TYPE_F32");
    variant += "_f32";
    break;
  case GGML_TYPE_F16:
    defines.push_back("TYPE_F16");
    variant += "_f16";
    break;
  default:
    GGML_ABORT("Unsupported type for unary shader");
  }

  if (context.key.inplace) {
    defines.push_back("INPLACE");
    variant += "_inplace";
  }

  defines.push_back(std::string("WG_SIZE=") +
                    std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_generic_shader_decisions *decisions =
      new ggml_webgpu_generic_shader_decisions();
  decisions->wg_size = context.max_wg_size;
  result.decisions = decisions;
  return result;
}

/** Scale **/

struct ggml_webgpu_scale_pipeline_key {
  int inplace;

  bool operator==(const ggml_webgpu_scale_pipeline_key &other) const {
    return inplace == other.inplace;
  }
};

struct ggml_webgpu_scale_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_scale_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.inplace);
    return seed;
  }
};

struct ggml_webgpu_scale_shader_lib_context {
  ggml_webgpu_scale_pipeline_key key;
  uint32_t max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_scale_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_scale_shader_lib_context &context) {
  std::vector<std::string> defines;
  std::string variant = "scale";

  if (context.key.inplace) {
    defines.push_back("INPLACE");
    variant += "_inplace";
  }

  defines.push_back(std::string("WG_SIZE=") +
                    std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_src, defines);
  result.variant = variant;
  ggml_webgpu_generic_shader_decisions *decisions =
      new ggml_webgpu_generic_shader_decisions();
  decisions->wg_size = context.max_wg_size;
  result.decisions = decisions;
  return result;
}

/** get_rows */

struct ggml_webgpu_get_rows_pipeline_key {
  ggml_type src_type;
  int vectorized;

  bool operator==(const ggml_webgpu_get_rows_pipeline_key &other) const {
    return src_type == other.src_type && vectorized == other.vectorized;
  }
};

struct ggml_webgpu_get_rows_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_get_rows_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.src_type);
    ggml_webgpu_hash_combine(seed, key.vectorized);
    return seed;
  }
};

struct ggml_webgpu_get_rows_shader_lib_context {
  ggml_webgpu_get_rows_pipeline_key key;
  uint32_t max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_get_rows_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_get_rows_shader_lib_context &context) {

  std::vector<std::string> defines;
  std::string variant = "get_rows";

  // Determine src type string and dst type string
  const char *type_str = nullptr;
  const char *dst_type_str = nullptr;
  uint32_t block_size = 1;

  switch (context.key.src_type) {
  case GGML_TYPE_F32:
    if (context.key.vectorized) {
      defines.push_back("F32_VEC");
      type_str = "vec4<f32>";
      dst_type_str = "vec4<f32>";
      block_size = 4;
    } else {
      defines.push_back("F32");
      type_str = "f32";
      dst_type_str = "f32";
      block_size = 1;
    }
    variant += "_f32";
    break;
  case GGML_TYPE_F16:
    defines.push_back("F16");
    type_str = "f16";
    dst_type_str = "f32";
    block_size = 1;
    variant += "_f16";
    break;
  case GGML_TYPE_I32:
    defines.push_back("I32");
    type_str = "i32";
    dst_type_str = "i32";
    block_size = 1;
    variant += "_i32";
    break;
  case GGML_TYPE_Q4_0:
    type_str = "q4_0";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_0_T");
    defines.push_back("Q4_0");
    variant += "_q4_0";
    break;
  case GGML_TYPE_Q4_1:
    type_str = "q4_1";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_1_T");
    defines.push_back("Q4_1");
    variant += "_q4_1";
    break;
  case GGML_TYPE_Q5_0:
    type_str = "q5_0";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_0_T");
    defines.push_back("Q5_0");
    variant += "_q5_0";
    break;
  case GGML_TYPE_Q5_1:
    type_str = "q5_1";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_1_T");
    defines.push_back("Q5_1");
    variant += "_q5_1";
    break;
  case GGML_TYPE_Q8_0:
    type_str = "q8_0";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q8_0_T");
    defines.push_back("Q8_0");
    variant += "_q8_0";
    break;
  case GGML_TYPE_Q8_1:
    type_str = "q8_1";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q8_1_T");
    defines.push_back("Q8_1");
    variant += "_q8_1";
    break;
  case GGML_TYPE_Q2_K:
    type_str = "q2_k";
    dst_type_str = "f32";
    block_size = 256; // K-quants use 256
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q2_K_T");
    defines.push_back("Q2_K");
    variant += "_q2_k";
    break;
  case GGML_TYPE_Q3_K:
    type_str = "q3_k";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q3_K_T");
    defines.push_back("Q3_K");
    variant += "_q3_k";
    break;
  case GGML_TYPE_Q4_K:
    type_str = "q4_k";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_K_T");
    defines.push_back("Q4_K");
    defines.push_back("Q45_K_SCALE_MIN");
    variant += "_q4_k";
    break;
  case GGML_TYPE_Q5_K:
    type_str = "q5_k";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_K_T");
    defines.push_back("Q5_K");
    defines.push_back("Q45_K_SCALE_MIN");
    variant += "_q5_k";
    break;
  case GGML_TYPE_Q6_K:
    type_str = "q6_k";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q6_K_T");
    defines.push_back("Q6_K");
    variant += "_q6_k";
    break;
  case GGML_TYPE_IQ2_XXS:
    type_str = "iq2_xxs";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_XXS_T");
    defines.push_back("IQ2_XXS");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_XXS_GRID");
    variant += "_iq2_xxs";
    break;
  case GGML_TYPE_IQ2_XS:
    type_str = "iq2_xs";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_XS_T");
    defines.push_back("IQ2_XS");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_XS_GRID");
    variant += "_iq2_xs";
    break;
  case GGML_TYPE_IQ2_S:
    type_str = "iq2_s";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_S_T");
    defines.push_back("IQ2_S");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_S_GRID");
    variant += "_iq2_s";
    break;
  case GGML_TYPE_IQ3_XXS:
    type_str = "iq3_xxs";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ3_XXS_T");
    defines.push_back("IQ3_XXS");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ3_XXS_GRID");
    variant += "_iq3_xxs";
    break;
  case GGML_TYPE_IQ3_S:
    type_str = "iq3_s";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ3_S_T");
    defines.push_back("IQ3_S");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ3_S_GRID");
    variant += "_iq3_s";
    break;
  case GGML_TYPE_IQ1_S:
    type_str = "iq1_s";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ1_S_T");
    defines.push_back("IQ1_S");
    defines.push_back("IQ1_GRID");
    variant += "_iq1_s";
    break;
  case GGML_TYPE_IQ1_M:
    type_str = "iq1_m";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ1_M_T");
    defines.push_back("IQ1_M");
    defines.push_back("IQ1_GRID");
    variant += "_iq1_m";
    break;
  case GGML_TYPE_IQ4_NL:
    type_str = "iq4_nl";
    dst_type_str = "f32";
    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ4_NL_T");
    defines.push_back("IQ4_NL");
    defines.push_back("IQ4_GRID");
    variant += "_iq4_nl";
    break;
  case GGML_TYPE_IQ4_XS:
    type_str = "iq4_xs";
    dst_type_str = "f32";
    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ4_XS_T");
    defines.push_back("IQ4_XS");
    defines.push_back("IQ4_GRID");
    variant += "_iq4_xs";
    break;
  default:
    break;
  }

  // Vectorized suffix
  if (context.key.vectorized) {
    variant += "_vec";
  }

  // Manual replacement of {{...}} placeholders
  std::string shader_with_replacements = shader_src;

  ggml_webgpu_replace_placeholder(shader_with_replacements, "TYPE",
                                  type_str ? type_str : "f32");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "DST_TYPE",
                                  dst_type_str ? dst_type_str : "f32");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "BLOCK_SIZE",
                                  std::to_string(block_size));
  ggml_webgpu_replace_placeholder(shader_with_replacements, "WORKGROUP_SIZE",
                                  std::to_string(context.max_wg_size));

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_with_replacements, defines);
  result.variant = variant;

  // Create decisions structure to store workgroup size
  ggml_webgpu_generic_shader_decisions *decisions =
      new ggml_webgpu_generic_shader_decisions();
  decisions->wg_size = context.max_wg_size;
  result.decisions = decisions;

  return result;
}

/** Matrix Multiplication **/

struct ggml_webgpu_mul_mat_pipeline_key {
  ggml_type src0_type;
  ggml_type src1_type;
  int vectorized;
  int is_vec;
  int use_subgroup_matrix;
  int register_tile;

  bool operator==(const ggml_webgpu_mul_mat_pipeline_key &other) const {
    return src0_type == other.src0_type && src1_type == other.src1_type &&
           vectorized == other.vectorized && is_vec == other.is_vec &&
           use_subgroup_matrix == other.use_subgroup_matrix &&
           register_tile == other.register_tile;
  }
};

struct ggml_webgpu_mul_mat_pipeline_key_hash {
  size_t operator()(const ggml_webgpu_mul_mat_pipeline_key &key) const {
    size_t seed = 0;
    ggml_webgpu_hash_combine(seed, key.src0_type);
    ggml_webgpu_hash_combine(seed, key.src1_type);
    ggml_webgpu_hash_combine(seed, key.vectorized);
    ggml_webgpu_hash_combine(seed, key.is_vec);
    ggml_webgpu_hash_combine(seed, key.use_subgroup_matrix);
    ggml_webgpu_hash_combine(seed, key.register_tile);
    return seed;
  }
};

struct ggml_webgpu_mul_mat_shader_lib_context {
  ggml_webgpu_mul_mat_pipeline_key key;

  // For subgroup matrix paths
  uint32_t max_subgroup_size;
  uint32_t sg_mat_m;
  uint32_t sg_mat_n;
  uint32_t sg_mat_k;

  // Subgroup configuration
  uint32_t subgroup_m;
  uint32_t subgroup_n;
  uint32_t subgroup_matrix_m;
  uint32_t subgroup_matrix_n;

  // For regular tile paths
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t wg_size_m;
  uint32_t wg_size_n;

  // For vec paths
  uint32_t wg_size;
  uint32_t outputs_per_wg;
};

struct ggml_webgpu_mul_mat_shader_decisions {
  uint32_t tile_k;
  uint32_t wg_size_m;
  uint32_t wg_size_n;
  uint32_t wg_size;
  uint32_t outputs_per_wg;
  int is_vec;
  int use_subgroup_matrix;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_mul_mat_shader(
    pre_wgsl::Preprocessor &preprocessor, const char *shader_src,
    const ggml_webgpu_mul_mat_shader_lib_context &context) {

  std::vector<std::string> defines;
  std::string variant = "mul_mat";

  // Determine base variant name based on kernel type
  if (context.key.is_vec) {
    variant = "mul_mat_vec";
  } else if (context.key.use_subgroup_matrix) {
    variant = "mul_mat_subgroup_matrix";
  } else if (context.key.register_tile) {
    variant = "mul_mat_reg_tile";
  }

  // Determine src0/src1 type strings
  const char *src0_type_str = nullptr;
  const char *src1_type_str = nullptr;
  const char *dst_type_str = nullptr;
  const char *shmem_type_str = nullptr;

  uint32_t block_size = 1;

  bool is_fast_path = context.key.is_vec || context.key.use_subgroup_matrix ||
                      context.key.register_tile;

  // Map src1 type
  switch (context.key.src1_type) {
  case GGML_TYPE_F32:
    src1_type_str = context.key.vectorized ? "vec4<f32>" : "f32";
    dst_type_str = context.key.vectorized ? "vec4<f32>" : "f32";
    break;
  case GGML_TYPE_F16:
    src1_type_str = context.key.vectorized ? "vec4<f16>" : "f16";
    dst_type_str = context.key.vectorized ? "vec4<f32>" : "f32";
    break;
  default:
    break;
  }

  // same for all types
  shmem_type_str = context.key.vectorized ? "vec4<f16>" : "f16";

  switch (context.key.src0_type) {
  case GGML_TYPE_F32:
    src0_type_str = context.key.vectorized ? "vec4<f32>" : "f32";

    block_size = 1;

    defines.push_back("FLOAT");
    defines.push_back("MUL_ACC_FLOAT");
    defines.push_back("INIT_SRC0_SHMEM_FLOAT");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_f32";
    break;

  case GGML_TYPE_F16:
    src0_type_str = context.key.vectorized ? "vec4<f16>" : "f16";

    block_size = 1;

    defines.push_back("FLOAT");
    defines.push_back("MUL_ACC_FLOAT");
    defines.push_back("INIT_SRC0_SHMEM_FLOAT");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_f16";
    break;

  case GGML_TYPE_Q4_0:
    src0_type_str = "q4_0";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_0_T");

    defines.push_back("Q4_0");
    defines.push_back("MUL_ACC_Q4_0");
    defines.push_back("INIT_SRC0_SHMEM_Q4_0");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q4_0";
    break;

  case GGML_TYPE_Q4_1:
    src0_type_str = "q4_1";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_1_T");

    defines.push_back("Q4_1");
    defines.push_back("MUL_ACC_Q4_1");
    defines.push_back("INIT_SRC0_SHMEM_Q4_1");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q4_1";
    break;

  case GGML_TYPE_Q5_0:
    src0_type_str = "q5_0";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_0_T");

    defines.push_back("Q5_0");
    defines.push_back("MUL_ACC_Q5_0");
    defines.push_back("INIT_SRC0_SHMEM_Q5_0");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q5_0";
    break;

  case GGML_TYPE_Q5_1:
    src0_type_str = "q5_1";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_1_T");

    defines.push_back("Q5_1");
    defines.push_back("MUL_ACC_Q5_1");
    defines.push_back("INIT_SRC0_SHMEM_Q5_1");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q5_1";
    break;

  case GGML_TYPE_Q8_0:
    src0_type_str = "q8_0";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q8_0_T");

    defines.push_back("Q8_0");
    defines.push_back("MUL_ACC_Q8_0");
    defines.push_back("INIT_SRC0_SHMEM_Q8_0");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q8_0";
    break;

  case GGML_TYPE_Q8_1:
    src0_type_str = "q8_1";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q8_1_T");

    defines.push_back("Q8_1");
    defines.push_back("MUL_ACC_Q8_1");
    defines.push_back("INIT_SRC0_SHMEM_Q8_1");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q8_1";
    break;

  case GGML_TYPE_Q2_K:
    src0_type_str = "q2_k";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q2_K_T");

    defines.push_back("Q2_K");
    defines.push_back("MUL_ACC_Q2_K");
    defines.push_back("INIT_SRC0_SHMEM_Q2_K");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q2_k";
    break;

  case GGML_TYPE_Q3_K:
    src0_type_str = "q3_k";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q3_K_T");

    defines.push_back("Q3_K");
    defines.push_back("MUL_ACC_Q3_K");
    defines.push_back("INIT_SRC0_SHMEM_Q3_K");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q3_k";
    break;

  case GGML_TYPE_Q4_K:
    src0_type_str = "q4_k";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q4_K_T");
    defines.push_back("Q45_K_SCALE_MIN");

    defines.push_back("Q4_K");
    defines.push_back("MUL_ACC_Q4_K");
    defines.push_back("INIT_SRC0_SHMEM_Q4_K");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q4_k";
    break;

  case GGML_TYPE_Q5_K:
    src0_type_str = "q5_k";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q5_K_T");
    defines.push_back("Q45_K_SCALE_MIN");

    defines.push_back("Q5_K");
    defines.push_back("MUL_ACC_Q5_K");
    defines.push_back("INIT_SRC0_SHMEM_Q5_K");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q5_k";
    break;

  case GGML_TYPE_Q6_K:
    src0_type_str = "q6_k";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("Q6_K_T");

    defines.push_back("Q6_K");
    defines.push_back("MUL_ACC_Q6_K");
    defines.push_back("INIT_SRC0_SHMEM_Q6_K");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_q6_k";
    break;

  case GGML_TYPE_IQ2_XXS:
    src0_type_str = "iq2_xxs";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_XXS_T");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_XXS_GRID");

    defines.push_back("IQ2_XXS");
    defines.push_back("MUL_ACC_IQ2_XXS");
    defines.push_back("INIT_SRC0_SHMEM_IQ2_XXS");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq2_xxs";
    break;

  case GGML_TYPE_IQ2_XS:
    src0_type_str = "iq2_xs";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_XS_T");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_XS_GRID");

    defines.push_back("IQ2_XS");
    defines.push_back("MUL_ACC_IQ2_XS");
    defines.push_back("INIT_SRC0_SHMEM_IQ2_XS");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq2_xs";
    break;

  case GGML_TYPE_IQ2_S:
    src0_type_str = "iq2_s";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ2_S_T");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ2_S_GRID");

    defines.push_back("IQ2_S");
    defines.push_back("MUL_ACC_IQ2_S");
    defines.push_back("INIT_SRC0_SHMEM_IQ2_S");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq2_s";
    break;

  case GGML_TYPE_IQ3_XXS:
    src0_type_str = "iq3_xxs";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ3_XXS_T");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ3_XXS_GRID");

    defines.push_back("IQ3_XXS");
    defines.push_back("MUL_ACC_IQ3_XXS");
    defines.push_back("INIT_SRC0_SHMEM_IQ3_XXS");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq3_xxs";
    break;

  case GGML_TYPE_IQ3_S:
    src0_type_str = "iq3_s";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ3_S_T");
    defines.push_back("IQ23_TABLES");
    defines.push_back("IQ3_S_GRID");

    defines.push_back("IQ3_S");
    defines.push_back("MUL_ACC_IQ3_S");
    defines.push_back("INIT_SRC0_SHMEM_IQ3_S");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq3_s";
    break;

  case GGML_TYPE_IQ1_S:
    src0_type_str = "iq1_s";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ1_S_T");
    defines.push_back("IQ1_GRID");

    defines.push_back("IQ1_S");
    defines.push_back("MUL_ACC_IQ1_S");
    defines.push_back("INIT_SRC0_SHMEM_IQ1_S");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq1_s";
    break;

  case GGML_TYPE_IQ1_M:
    src0_type_str = "iq1_m";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ1_M_T");
    defines.push_back("IQ1_GRID");

    defines.push_back("IQ1_M");
    defines.push_back("MUL_ACC_IQ1_M");
    defines.push_back("INIT_SRC0_SHMEM_IQ1_M");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq1_m";
    break;

  case GGML_TYPE_IQ4_NL:
    src0_type_str = "iq4_nl";

    block_size = 32;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ4_NL_T");
    defines.push_back("IQ4_GRID");

    defines.push_back("IQ4_NL");
    defines.push_back("MUL_ACC_IQ4_NL");
    defines.push_back("INIT_SRC0_SHMEM_IQ4_NL");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq4_nl";
    break;

  case GGML_TYPE_IQ4_XS:
    src0_type_str = "iq4_xs";

    block_size = 256;
    defines.push_back("BYTE_HELPERS");
    defines.push_back("IQ4_XS_T");
    defines.push_back("IQ4_GRID");

    defines.push_back("IQ4_XS");
    defines.push_back("MUL_ACC_IQ4_XS");
    defines.push_back("INIT_SRC0_SHMEM_IQ4_XS");
    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

    variant += "_iq4_xs";
    break;

  default:
    break;
  }

  //   if (context.key.is_vec) {
  //     // if quantized type and vectorized, need to use f16 instead of the
  //     // quantized type
  //     if (context.key.src0_type != GGML_TYPE_F32 &&
  //         context.key.src0_type != GGML_TYPE_F16) {
  //       printf("DEBUG: Overriding src0_type_str from '%s' to 'f16' for "
  //              "is_vec=true\n",
  //              src0_type_str);
  //       src0_type_str = "f16";
  //     }
  //   }

  // Add VEC/SCALAR defines
  //   if (is_fast_path) {
  //     defines.push_back(context.key.vectorized ? "VEC" : "SCALAR");
  //     if (!context.key.is_vec) {
  //       defines.push_back(context.key.vectorized ? "SHMEM_VEC" :
  //       "SHMEM_SCALAR");
  //     }
  //   }

  // Add VEC/SCALAR defines
  if (is_fast_path) {
    // if quantized type and vectorized, need to use f16 instead of the
    // quantized type
    if (context.key.src0_type != GGML_TYPE_F32 &&
        context.key.src0_type != GGML_TYPE_F16) {
      src0_type_str = "f16";
    }

    if (context.key.is_vec) {
      defines.push_back(context.key.vectorized ? "VEC" : "SCALAR");
    } else {
      // mul_mat_reg_tile and mul_mat_vec need to add normal and shmem versions
      defines.push_back(context.key.vectorized ? "VEC" : "SCALAR");
      defines.push_back(context.key.vectorized ? "SHMEM_VEC" : "SHMEM_SCALAR");
    }
  }

  // Append src1 type
  variant += std::string("_") +
             (context.key.src1_type == GGML_TYPE_F32 ? "f32" : "f16");

  // printf("DEBUG: After appending src1 type: variant='%s'\n",
  // variant.c_str());

  // Vectorized suffix
  if (context.key.vectorized) {
    variant += "_vec";
  }

  // Manual replacement of {{...}} placeholders before preprocessing
  std::string shader_with_replacements = shader_src;

  // Replace {{placeholders}}

  ggml_webgpu_replace_placeholder(
      shader_with_replacements, "VEC_SIZE",
      std::to_string(context.key.vectorized ? 4 : 1));
  ggml_webgpu_replace_placeholder(shader_with_replacements, "SRC0_TYPE",
                                  src0_type_str ? src0_type_str : "f32");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "SRC1_TYPE",
                                  src1_type_str ? src1_type_str : "f32");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "DST_TYPE",
                                  dst_type_str ? dst_type_str : "f32");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "SHMEM_TYPE",
                                  shmem_type_str ? shmem_type_str : "f16");
  ggml_webgpu_replace_placeholder(shader_with_replacements, "WEBGPU_TILE_M",
                                  std::to_string(context.tile_m));
  ggml_webgpu_replace_placeholder(shader_with_replacements, "WEBGPU_TILE_N",
                                  std::to_string(context.tile_n));
  ggml_webgpu_replace_placeholder(shader_with_replacements, "BLOCK_SIZE",
                                  std::to_string(block_size));

  // Replace {{placeholders}} that only come up in mul_mat_subgroup_matrix
  if (context.key.use_subgroup_matrix) {
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_MAX_SUBGROUP_SIZE",
                                    std::to_string(context.max_subgroup_size));
    ggml_webgpu_replace_placeholder(shader_with_replacements, "WEBGPU_TILE_K",
                                    std::to_string(context.tile_k));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SUBGROUP_M",
                                    std::to_string(context.subgroup_m));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SUBGROUP_N",
                                    std::to_string(context.subgroup_n));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SUBGROUP_MATRIX_M",
                                    std::to_string(context.subgroup_matrix_m));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SUBGROUP_MATRIX_N",
                                    std::to_string(context.subgroup_matrix_n));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SG_MAT_M_SIZE",
                                    std::to_string(context.sg_mat_m));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SG_MAT_N_SIZE",
                                    std::to_string(context.sg_mat_n));
    ggml_webgpu_replace_placeholder(shader_with_replacements,
                                    "WEBGPU_SG_MAT_K_SIZE",
                                    std::to_string(context.sg_mat_k));
  }

  ggml_webgpu_processed_shader result;
  result.wgsl = preprocessor.preprocess(shader_with_replacements, defines);
  result.variant = variant;

  //   if (context.key.src0_type >= GGML_TYPE_Q4_0) {
  //     std::string path_suffix =
  //         context.key.is_vec
  //             ? "vec"
  //             : (context.key.use_subgroup_matrix
  //                    ? "subgroup"
  //                    : (context.key.register_tile ? "tile" : "nonfast"));
  //     std::string filename = "/tmp/" +
  //                            std::string(ggml_type_name(context.key.src0_type))
  //                            +
  //                            "_" + path_suffix + "_debug.wgsl";
  //     std::ofstream out(filename);
  //     out << result.wgsl;
  //     out.close();
  //   }

  ggml_webgpu_mul_mat_shader_decisions *decisions =
      new ggml_webgpu_mul_mat_shader_decisions();
  decisions->tile_k = context.tile_k;
  decisions->wg_size_m = context.wg_size_m;
  decisions->wg_size_n = context.wg_size_n;
  decisions->wg_size = context.wg_size;
  decisions->outputs_per_wg = context.outputs_per_wg;
  decisions->is_vec = context.key.is_vec;
  decisions->use_subgroup_matrix = context.key.use_subgroup_matrix;
  result.decisions = decisions;

  return result;
}

#endif // GGML_WEBGPU_SHADER_LIB_HPP
