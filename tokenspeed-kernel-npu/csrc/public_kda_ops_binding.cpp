/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Copyright (c) 2026 LightSeek Foundation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from the public vLLM-Ascend KDA Torch adapters.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <tuple>
#include <vector>

#include <third_party/op-plugin/op_plugin/utils/op_api_common_base.h>

void GetPublicKdaApiFunc(
    const char* api_name,
    const char* workspace_api_name,
    void*& op_api,
    void*& workspace_api);

#define GetApiFunc GetPublicKdaApiFunc
#define EXEC_NPU_CMD EXEC_NPU_CMD_EXT

namespace {

void* custom_op_api_handle() {
  static void* handle = [] {
    const char* paths = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    TORCH_CHECK(paths != nullptr, "ASCEND_CUSTOM_OPP_PATH is not set");
    std::string remaining(paths);
    while (!remaining.empty()) {
      const size_t separator = remaining.find(':');
      const std::string root = remaining.substr(0, separator);
      if (!root.empty()) {
        const std::string library = root + "/op_api/lib/libcust_opapi.so";
        if (void* candidate = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL)) {
          return candidate;
        }
      }
      if (separator == std::string::npos) {
        break;
      }
      remaining.erase(0, separator + 1);
    }
    TORCH_CHECK(
        false,
        "cannot load libcust_opapi.so from ASCEND_CUSTOM_OPP_PATH: ",
        dlerror());
  }();
  return handle;
}

}  // namespace

void GetPublicKdaApiFunc(
    const char* api_name,
    const char* workspace_api_name,
    void*& op_api,
    void*& workspace_api) {
  if (op_api != nullptr && workspace_api != nullptr) {
    return;
  }
  void* handle = custom_op_api_handle();
  op_api = dlsym(handle, api_name);
  workspace_api = dlsym(handle, workspace_api_name);
  TORCH_CHECK(
      op_api != nullptr && workspace_api != nullptr,
      api_name,
      " or ",
      workspace_api_name,
      " is missing from libcust_opapi.so");
}

namespace tokenspeed_npu_public_kda {

at::Tensor recurrent_kda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& gate,
    const at::Tensor& beta,
    at::Tensor& initial_state,
    const at::Tensor& cu_seqlens,
    const at::Tensor& state_indices,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    double scale,
    bool use_qk_l2norm_in_kernel,
    bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    bool safe_gate,
    double lower_bound) {
  const bool is_tnd = query.dim() == 3;
  TORCH_CHECK(
      (is_tnd && key.dim() == 3 && value.dim() == 3 && gate.dim() == 3 && beta.dim() == 2) ||
          (!is_tnd && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
           gate.dim() == 4 && beta.dim() == 3),
      "recurrent_kda: inconsistent TND or BSND tensor ranks");
  TORCH_CHECK(query.sizes() == key.sizes(), "recurrent_kda: q/k shapes must match");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
          value.scalar_type() == at::kBFloat16,
      "recurrent_kda: q/k/v must be bfloat16");
  TORCH_CHECK(
      (gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kBFloat16 ||
       gate.scalar_type() == at::kHalf) &&
          (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16 ||
           beta.scalar_type() == at::kHalf),
      "recurrent_kda: gate and beta must be float32, bfloat16 or float16");
  TORCH_CHECK(
      key.device() == query.device() && value.device() == query.device() &&
          gate.device() == query.device() && beta.device() == query.device() &&
          initial_state.device() == query.device(),
      "recurrent_kda: q/k/v/gate/beta/state must share a device");
  TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2 &&
          (cu_seqlens.scalar_type() == at::kInt || cu_seqlens.scalar_type() == at::kLong) &&
          cu_seqlens.device() == query.device(),
      "recurrent_kda: cu_seqlens must be a device int32/int64 vector");

  const int64_t batch = is_tnd ? 1 : query.size(0);
  const int64_t tokens = is_tnd ? query.size(0) : query.size(1);
  const int64_t total_tokens = batch * tokens;
  const int64_t sequences = cu_seqlens.size(0) - 1;
  const int64_t heads = is_tnd ? query.size(1) : query.size(2);
  const int64_t key_dim = query.size(-1);
  const int64_t value_heads = is_tnd ? value.size(1) : value.size(2);
  const int64_t value_dim = value.size(-1);
  TORCH_CHECK(
      total_tokens > 0 && heads > 0 && value_heads >= heads && value_heads % heads == 0,
      "recurrent_kda: invalid token/head dimensions");
  TORCH_CHECK(
      key_dim == 128 && (value_dim == 128 || value_dim == 256),
      "recurrent_kda: K must be 128 and V must be 128 or 256");
  TORCH_CHECK(
      (is_tnd && value.size(0) == total_tokens && gate.size(0) == total_tokens &&
       beta.size(0) == total_tokens && gate.size(1) == value_heads &&
       gate.size(2) == key_dim && beta.size(1) == value_heads) ||
          (!is_tnd && value.size(0) == batch && value.size(1) == tokens &&
           gate.size(0) == batch && gate.size(1) == tokens &&
           gate.size(2) == value_heads && gate.size(3) == key_dim &&
           beta.size(0) == batch && beta.size(1) == tokens &&
           beta.size(2) == value_heads),
      "recurrent_kda: value/gate/beta shapes do not match the layout");
  const bool packed_indices = state_indices.dim() == 1 && state_indices.numel() >= total_tokens;
  const bool speculative_indices =
      state_indices.dim() == 2 && state_indices.size(0) == sequences && state_indices.size(1) > 0;
  TORCH_CHECK(
      (state_indices.scalar_type() == at::kInt || state_indices.scalar_type() == at::kLong) &&
          (packed_indices || speculative_indices) && state_indices.device() == query.device(),
      "recurrent_kda: state_indices must be packed [T] or [sequence,max_step]");
  TORCH_CHECK(
      initial_state.dim() == 4 && initial_state.size(0) >= 1 &&
          initial_state.size(1) == value_heads && initial_state.size(2) == key_dim &&
          initial_state.size(3) == value_dim,
      "recurrent_kda: state must be a non-empty [capacity,HV,K,V] pool");
  TORCH_CHECK(
      initial_state.scalar_type() == at::kFloat || initial_state.scalar_type() == at::kBFloat16,
      "recurrent_kda: state must be float32 or bfloat16");
  TORCH_CHECK(
      a_log.scalar_type() == at::kFloat && a_log.dim() == 1 &&
          a_log.numel() == value_heads && a_log.device() == query.device(),
      "recurrent_kda: A_log must be float32 [HV] on the query device");
  TORCH_CHECK(
      dt_bias.scalar_type() == at::kFloat &&
          ((dt_bias.dim() == 1 && dt_bias.numel() == value_heads * key_dim) ||
           (dt_bias.dim() == 2 && dt_bias.size(0) == value_heads &&
            dt_bias.size(1) == key_dim)) &&
          dt_bias.device() == query.device(),
      "recurrent_kda: dt_bias must be float32 [HV*K] or [HV,K]");
  if (num_accepted_tokens.has_value() && num_accepted_tokens->defined()) {
    TORCH_CHECK(
        num_accepted_tokens->dim() == 1 && num_accepted_tokens->size(0) == sequences &&
            (num_accepted_tokens->scalar_type() == at::kInt ||
             num_accepted_tokens->scalar_type() == at::kLong) &&
            num_accepted_tokens->device() == query.device(),
        "recurrent_kda: num_accepted_tokens must be [sequence] int32/int64");
  }
  TORCH_CHECK(
      !safe_gate || (lower_bound >= -5.0 && lower_bound < 0.0),
      "recurrent_kda: lower_bound must be in [-5,0) for safe gate");

  at::Tensor output = at::empty_like(value);
  at::Tensor final_state = initial_state;
  const at::Tensor accepted = num_accepted_tokens.value_or(at::Tensor());
  const char* layout = is_tnd ? "TND" : "BSND";
  const bool output_final_state = false;
  const bool inplace_final_state = true;
  const bool state_v_first = false;
  EXEC_NPU_CMD(
      aclnnRecurrentKda,
      query,
      key,
      value,
      gate,
      beta,
      initial_state,
      cu_seqlens,
      state_indices,
      a_log,
      dt_bias,
      accepted,
      layout,
      scale,
      output_final_state,
      inplace_final_state,
      use_qk_l2norm_in_kernel,
      use_gate_in_kernel,
      use_beta_sigmoid_in_kernel,
      allow_neg_eigval,
      safe_gate,
      lower_bound,
      state_v_first,
      output,
      final_state);
  return output;
}

at::Tensor kda_gate_cumsum(
    const at::Tensor& gate,
    int64_t chunk_size,
    const c10::optional<at::Tensor>& a_log,
    const c10::optional<at::Tensor>& dt_bias,
    c10::optional<at::IntArrayRef> cu_seqlens,
    bool use_gate_in_kernel,
    bool safe_gate,
    double lower_bound,
    c10::string_view layout) {
  TORCH_CHECK(
      gate.dim() == 3 || gate.dim() == 4,
      "kda_gate_cumsum: gate must be rank 3 or 4");
  TORCH_CHECK(
      chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
      "kda_gate_cumsum: chunk size must be 32, 64, or 128");
  TORCH_CHECK(
      !safe_gate || use_gate_in_kernel,
      "kda_gate_cumsum: safe gate requires the raw-gate path");
  std::string layout_string(layout.data(), layout.size());
  TORCH_CHECK(
      layout_string == "BSND" || layout_string == "BNSD" || layout_string == "TND" ||
          layout_string == "NTD",
      "kda_gate_cumsum: invalid layout");
  at::Tensor output = at::empty(gate.sizes(), gate.options().dtype(at::kFloat));
  char* layout_ptr = const_cast<char*>(layout_string.c_str());
  EXEC_NPU_CMD(
      aclnnKdaGateCumsum,
      gate,
      a_log,
      dt_bias,
      cu_seqlens,
      chunk_size,
      use_gate_in_kernel,
      safe_gate,
      lower_bound,
      layout_ptr,
      output);
  return output;
}

struct ChunkShape {
  int64_t tokens;
  int64_t heads;
  int64_t value_heads;
  int64_t key_dim;
  int64_t value_dim;
  int64_t sequences;
  int64_t total_chunks;
};

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

ChunkShape chunk_shape(
    const at::Tensor& q,
    const at::Tensor& v,
    int64_t chunk_size,
    const std::string& layout,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices) {
  const bool rank_three = layout == "TND" || layout == "NTD";
  TORCH_CHECK(
      q.dim() == (rank_three ? 3 : 4) && v.dim() == q.dim(),
      "chunk_kda_fwd: tensor rank does not match layout");
  const bool internal = layout == "BNSD" || layout == "NTD";
  const int64_t batch = rank_three ? 1 : q.size(0);
  const int64_t tokens = layout == "TND"   ? q.size(0)
      : layout == "NTD"                    ? q.size(1)
      : layout == "BNSD"                   ? q.size(2)
                                            : q.size(1);
  const int64_t heads = layout == "TND"   ? q.size(1)
      : layout == "NTD"                  ? q.size(0)
      : layout == "BNSD"                 ? q.size(1)
                                          : q.size(2);
  const int64_t value_heads = layout == "TND"   ? v.size(1)
      : layout == "NTD"                        ? v.size(0)
      : internal                               ? v.size(1)
                                               : v.size(2);
  int64_t sequences = batch;
  int64_t total_chunks = batch * ceil_div(tokens, chunk_size);
  if (cu_seqlens.has_value()) {
    const auto values = cu_seqlens.value();
    TORCH_CHECK(
        values.size() >= 2 && values[0] == 0 && values[values.size() - 1] == tokens,
        "chunk_kda_fwd: cu_seqlens must start at zero and end at token count");
    sequences = static_cast<int64_t>(values.size()) - 1;
    total_chunks = 0;
    for (size_t index = 0; index + 1 < values.size(); ++index) {
      TORCH_CHECK(
          values[index] <= values[index + 1],
          "chunk_kda_fwd: cu_seqlens must be monotonic");
      total_chunks += ceil_div(values[index + 1] - values[index], chunk_size);
    }
  }
  if (chunk_indices.has_value()) {
    TORCH_CHECK(
        chunk_indices->size() % 2 == 0,
        "chunk_kda_fwd: chunk_indices must contain sequence/chunk pairs");
    total_chunks = static_cast<int64_t>(chunk_indices->size()) / 2;
  }
  TORCH_CHECK(total_chunks > 0, "chunk_kda_fwd: at least one chunk is required");
  return {
      tokens,
      heads,
      value_heads,
      q.size(-1),
      v.size(-1),
      sequences,
      total_chunks};
}

std::tuple<at::Tensor, at::Tensor> chunk_kda_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& gate_cumsum,
    const at::Tensor& beta,
    double scale,
    int64_t chunk_size,
    c10::string_view layout,
    const c10::optional<at::Tensor>& initial_state,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices) {
  TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd: q/k shapes must match");
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf,
      "chunk_kda_fwd: q/k/v must be bfloat16 or float16");
  TORCH_CHECK(
      k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
      "chunk_kda_fwd: q/k/v dtypes must match");
  TORCH_CHECK(
      chunk_size == 64 || chunk_size == 128,
      "chunk_kda_fwd: public split path supports chunk size 64 or 128");
  std::string layout_string(layout.data(), layout.size());
  TORCH_CHECK(
      layout_string == "BSND" || layout_string == "BNSD" || layout_string == "TND" ||
          layout_string == "NTD",
      "chunk_kda_fwd: invalid layout");
  const ChunkShape shape =
      chunk_shape(q, v, chunk_size, layout_string, cu_seqlens, chunk_indices);
  TORCH_CHECK(
      shape.heads > 0 && shape.value_heads >= shape.heads &&
          shape.value_heads % shape.heads == 0,
      "chunk_kda_fwd: invalid head relationship");
  TORCH_CHECK(
      shape.key_dim % 16 == 0 && shape.value_dim % 16 == 0 && shape.value_dim <= 256,
      "chunk_kda_fwd: unsupported key/value dimensions");

  at::Tensor output = at::empty_like(v);
  at::Tensor final_state = at::empty(
      {shape.sequences, shape.value_heads, shape.key_dim, shape.value_dim},
      q.options().dtype(at::kFloat));
  at::Tensor empty = at::empty({0}, q.options());
  char* layout_ptr = const_cast<char*>(layout_string.c_str());
  const bool output_final_state = true;
  EXEC_NPU_CMD(
      aclnnChunkKdaFwd,
      q,
      k,
      v,
      gate_cumsum,
      beta,
      initial_state,
      cu_seqlens,
      chunk_indices,
      layout_ptr,
      scale,
      chunk_size,
      output_final_state,
      shape.total_chunks,
      output,
      final_state,
      empty,
      empty,
      empty,
      empty,
      empty,
      empty,
      empty,
      empty);
  return {output, final_state};
}

at::Tensor recurrent_kda_meta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& value,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    double,
    bool,
    bool,
    bool,
    bool,
    bool,
    double) {
  return at::empty_like(value);
}

at::Tensor kda_gate_cumsum_meta(
    const at::Tensor& gate,
    int64_t,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    c10::optional<at::IntArrayRef>,
    bool,
    bool,
    double,
    c10::string_view) {
  return at::empty(gate.sizes(), gate.options().dtype(at::kFloat));
}

std::tuple<at::Tensor, at::Tensor> chunk_kda_fwd_meta(
    const at::Tensor& q,
    const at::Tensor&,
    const at::Tensor& v,
    const at::Tensor&,
    const at::Tensor&,
    double,
    int64_t chunk_size,
    c10::string_view layout,
    const c10::optional<at::Tensor>&,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices) {
  std::string layout_string(layout.data(), layout.size());
  const ChunkShape shape =
      chunk_shape(q, v, chunk_size, layout_string, cu_seqlens, chunk_indices);
  return {
      at::empty_like(v),
      at::empty(
          {shape.sequences, shape.value_heads, shape.key_dim, shape.value_dim},
          q.options().dtype(at::kFloat))};
}

}  // namespace tokenspeed_npu_public_kda

TORCH_LIBRARY(tokenspeed_npu_public_kda, ops) {
  ops.def(
      "recurrent_kda(Tensor query, Tensor key, Tensor value, Tensor gate, Tensor beta, "
      "Tensor(a!) initial_state, Tensor cu_seqlens, Tensor state_indices, Tensor a_log, "
      "Tensor dt_bias, *, Tensor? num_accepted_tokens=None, "
      "float scale=0.08838834764831845, bool use_qk_l2norm_in_kernel=True, "
      "bool use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=False, "
      "bool allow_neg_eigval=False, bool safe_gate=True, float lower_bound=-5.0) -> Tensor");
  ops.def(
      "kda_gate_cumsum(Tensor gate, int chunk_size, *, Tensor? a_log=None, "
      "Tensor? dt_bias=None, int[]? cu_seqlens=None, bool use_gate_in_kernel=False, "
      "bool safe_gate=False, float lower_bound=-5.0, str layout=\"BSND\") -> Tensor");
  ops.def(
      "chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor gate_cumsum, Tensor beta, "
      "float scale, int chunk_size, str layout=\"BSND\", *, Tensor? initial_state=None, "
      "int[]? cu_seqlens=None, int[]? chunk_indices=None) -> "
      "(Tensor output, Tensor final_state)");
}

TORCH_LIBRARY_IMPL(tokenspeed_npu_public_kda, PrivateUse1, ops) {
  ops.impl("recurrent_kda", &tokenspeed_npu_public_kda::recurrent_kda);
  ops.impl("kda_gate_cumsum", &tokenspeed_npu_public_kda::kda_gate_cumsum);
  ops.impl("chunk_kda_fwd", &tokenspeed_npu_public_kda::chunk_kda_fwd);
}

TORCH_LIBRARY_IMPL(tokenspeed_npu_public_kda, Meta, ops) {
  ops.impl("recurrent_kda", &tokenspeed_npu_public_kda::recurrent_kda_meta);
  ops.impl("kda_gate_cumsum", &tokenspeed_npu_public_kda::kda_gate_cumsum_meta);
  ops.impl("chunk_kda_fwd", &tokenspeed_npu_public_kda::chunk_kda_fwd_meta);
}
