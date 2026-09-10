/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace YAML {
class Node;
}  // namespace YAML

#include <holoscan/holoscan_core_export.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/operators/inference/fast_path_eligibility.hpp>

#include <holoinfer/holoinfer.hpp>

namespace holoscan::ops {

/// @brief Model name -> single string value map (e.g. model path, backend name).
using InferenceDataMap = std::map<std::string, std::string>;

/// @brief Model name -> ordered list of tensor names map (e.g. per-model input/output tensors).
using InferenceDataVecMap = std::map<std::string, std::vector<std::string>>;

/// @brief Compile-time-authored configuration for InferenceOp.
///
/// Configuration is supplied as one immutable aggregate when the Operator is instantiated on the
/// authoring Graph (`graph.op<InferenceOp>("name", InferenceOpParams{...})`).
/// InferenceOp::setup() reads `params()` to dynamically
/// declare one Input<Tensor>/Output<Tensor> port per configured tensor name (see inference.cpp).
struct InferenceOpParams {
  /// Backend to use for inference. Only "trt" is supported by this TensorRT-only module.
  std::string backend{"trt"};
  /// Optional per-model backend override map. Every value must be "trt".
  InferenceDataMap backend_map;
  /// Map of model name -> path to the model (ONNX source or TensorRT engine file).
  InferenceDataMap model_path_map;
  /// Map of model name -> ordered list of input tensor names.
  InferenceDataVecMap input_map;
  /// Map of model name -> ordered list of output tensor names.
  InferenceDataVecMap output_map;
  /// Per-model TensorRT optimization profile. Map of model_name -> ordered list of "min,opt,max"
  /// triples (one entry per dynamic dim of one input, in TRT input order). Empty vector for a
  /// model means "use HoloInfer's default {1,1,1} profile" (existing behavior). This is the
  /// authoring escape hatch for models whose ONNX declares static input shapes that mismatch
  /// the default profile (e.g. an input with a fixed non-batch dimension baked into the export).
  InferenceDataVecMap trt_opt_profile;
  /// Whether `model_path_map` entries are pre-built TensorRT engine files.
  bool is_engine_path{false};
  /// Whether received input tensors are device-resident.
  bool input_on_cuda{true};
  /// Whether HoloInfer's output buffers should be device-resident.
  bool output_on_cuda{true};
  /// Whether emitted output tensors should be device-resident.
  bool transmit_on_cuda{true};
  /// Whether input tensor dimensions may change between compute() calls.
  bool dynamic_input_dims{false};
  /// Reserved for a possible future dynamic-output-dims feature; not yet consumed (HoloInfer's
  /// output dimensions are re-queried every compute() call whenever `dynamic_input_dims` is set,
  /// which already covers the common case).
  bool output_dynamic_dims{false};
  /// Enable TensorRT FP16 engine building.
  bool enable_fp16{false};
  /// Enable CUDA Graphs capture in the TensorRT backend. When a capture cannot be protected or
  /// fails, the fallback ladder degrades (and reports) rather than failing — see the fallback
  /// ladder documentation for the runtime rungs.
  bool enable_cuda_graphs{true};
  /// DLA core index to build the engine on; -1 disables DLA.
  int32_t dla_core{-1};
  /// @brief Per-model DLA core override. Map of model name -> DLA core index (as a string; e.g.
  /// `{"detector": "0", "classifier": "1"}`). The value `"-1"` means "run this model on the GPU"
  /// while other models honor the aggregate-level `dla_core`. When a model is absent from this
  /// map, it falls back to the aggregate's `dla_core`. Empty (the default) => every model uses
  /// `dla_core` uniformly, matching pre-existing behavior. Use this for heterogeneous
  /// multi-model deployments where different models target different DLA cores or a mix of DLA
  /// and GPU execution.
  InferenceDataMap dla_core_map;
  /// If DLA is enabled, allow GPU fallback for layers that cannot run on DLA.
  bool dla_gpu_fallback{true};
  /// Run models in parallel (multi-model configurations only).
  bool parallel_inference{true};
  /// @brief TensorRT engine-build ceiling on batch dimension. Enforced during ONNX->engine
  /// conversion: if any optimization profile authored via `trt_opt_profile` requests a batch
  /// value larger than this ceiling, engine building fails with a clear error. Only consulted
  /// when a new engine is built from ONNX; ignored for pre-built engines
  /// (`is_engine_path=true`). Raise when your model or optimization profile needs batches larger
  /// than 256 (the HoloInfer default).
  int32_t max_batch_size{256};
  /// @brief TensorRT engine-build workspace memory ceiling in bytes. Passed to TRT as
  /// `kWORKSPACE` memory-pool limit during engine construction; tactic selection is bounded by
  /// this budget, so smaller values yield potentially slower engines but lower peak build-time
  /// GPU memory. Only consulted when a new engine is built from ONNX; ignored for pre-built
  /// engines (`is_engine_path=true`). Default is 10 GB (10'000'000'000 bytes), matching
  /// HoloInfer's historical default. Lower to constrain build memory on smaller GPUs; raise
  /// only if you observe workspace-limited tactics blocking your best engine plan.
  std::size_t max_memory{10'000'000'000ULL};
  /// CUDA stream pool (include/holoscan/cuda_execution/cuda_stream_pool.hpp). Consumed by
  /// InferenceOp::start(): when present, the pool's acquire() API supplies the dedicated
  /// per-backend stream(s) used by the fast dispatch paths (kFast: one stream; kParFast/
  /// kDynParFast: one distinct stream per model). ExecutionContext::cuda_stream() remains the
  /// stream used for the data-movement copies InferenceOp performs directly and is the stream all
  /// per-model streams synchronize with via CUDA events. When null, HoloInfer backends fall back
  /// to their own internally-created stream and kParFast/kDynParFast demote to kSeqFast/
  /// kDynSeqFast with an explicit warning (fallback ladder case B_STREAM_POOL_INSUFFICIENT).
  std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool;
  /// CUDA green-context pool (include/holoscan/cuda_execution/cuda_green_context_pool.hpp).
  /// Consumed by InferenceOp::start(): when present, the pool's default partition
  /// CUcontext/SM-count are threaded through to HoloInfer (InferenceSpecs::build_cuda_context_ /
  /// build_sm_count_) so TensorRT engine building and tactic selection are constrained to that SM
  /// partition for every configured model. There is no per-model partition map on this aggregate,
  /// so all models share the pool's default partition. When null, TensorRT engine building uses
  /// the full GPU. The pool is also what unlocks the top rung of the fallback ladder (green-context
  /// isolation of concurrent CUDA-Graph capture); a pool with fewer partitions than models
  /// degrades with an explicit warning (case C_GREEN_CONTEXT_PARTITIONS_INSUFFICIENT).
  std::shared_ptr<holoscan::CudaGreenContextPool> cuda_green_context_pool;

  /// @brief Loads an `InferenceOpParams` aggregate from a YAML file.
  ///
  /// The YAML schema is 1:1 with the fields of this struct:
  ///
  /// ```yaml
  /// model_path_map: {model_a: /path/to/model_a.onnx}
  /// backend: trt                      # default: "trt"
  /// backend_map: {model_a: trt}       # optional per-model override
  /// input_map: {model_a: [input_tensor]}
  /// output_map: {model_a: [output_tensor]}
  /// trt_opt_profile: {model_a: ["1,4,8"]}  # optional; "min,opt,max" per dynamic dim
  /// is_engine_path: false
  /// input_on_cuda: true
  /// output_on_cuda: true
  /// transmit_on_cuda: true
  /// enable_cuda_graphs: true
  /// enable_fp16: false
  /// dla_core: -1
  /// dla_core_map: {model_a: "0", model_b: "-1"}  # optional per-model DLA core override
  /// dla_gpu_fallback: true
  /// parallel_inference: false
  /// max_batch_size: 256                          # TRT engine-build batch ceiling
  /// max_memory: 10000000000                      # TRT engine-build workspace budget (bytes)
  /// dynamic_input_dims: false
  /// output_dynamic_dims: false
  /// # Optional: constructs a real CudaStreamPool when present.
  /// cuda_stream_pool:
  ///   max_size: 4
  /// # Optional: constructs a real CudaGreenContextPool when present.
  /// cuda_green_context_pool:
  ///   num_partitions: 2
  ///   device_id: 0
  ///   sms_per_partition: [8, 8]   # required when num_partitions >= 2; length must equal
  ///                               # num_partitions; each entry is the SM count for that partition
  ///                               # and must be a multiple of the device's structured_sm_quantum.
  /// ```
  ///
  /// Every top-level key is optional except `model_path_map`, `input_map`, and `output_map`;
  /// absent scalar/bool keys fall back to the member's in-class default shown above. The
  /// `cuda_stream_pool` and `cuda_green_context_pool` sub-maps are the only keys that construct
  /// real resource objects (`holoscan::CudaStreamPool` / `holoscan::CudaGreenContextPool`,
  /// respectively) rather than populating a plain-data field; when a sub-map is absent, the
  /// corresponding `shared_ptr` member stays null, matching a default-constructed
  /// `InferenceOpParams`.
  ///
  /// @param path Path to a YAML file matching the schema above.
  /// @return The populated `InferenceOpParams`.
  /// @throws std::runtime_error if the file cannot be read/parsed, or a required key
  ///   (`model_path_map`, `input_map`, `output_map`) is missing.
  [[nodiscard]] static InferenceOpParams from_yaml(std::string_view path);

  /// @brief Load an `InferenceOpParams` aggregate from an already-parsed YAML sub-node.
  ///
  /// Use this when a single YAML file configures multiple operators — each operator's
  /// parameters live under a named key at the file's root. `from_yaml_node()` reads that
  /// subtree and produces the aggregate; `config_dir` is the directory that relative
  /// `model_path_map` paths resolve against (typically the parent of the YAML file that
  /// `node` was loaded from).
  ///
  /// ```yaml
  /// # multi_op_app.yaml — three operators in one file
  /// primary_inference:                 # →
  /// InferenceOpParams::from_yaml_node(root["primary_inference"], dir) model_path_map: {model_a:
  /// models/model_a.onnx} input_map:  {model_a: [tensor_in]} output_map: {model_a: [logits]}
  ///   enable_cuda_graphs: true
  /// secondary_inference:               # → a second InferenceOpParams
  ///   model_path_map: {model_b: models/model_b.onnx}
  ///   input_map:  {model_b: [logits]}
  ///   output_map: {model_b: [prediction]}
  /// other_operator:                    # → some other operator's own from_yaml_node
  ///   ...
  /// ```
  ///
  /// The subtree schema below the named key is identical to what `from_yaml()` accepts at the
  /// root — the same required + optional keys and the same behavior for absent keys.
  ///
  /// `from_yaml(path)` is now a thin wrapper: it loads the file, uses the file's parent as
  /// `config_dir`, and calls `from_yaml_node(root, parent)`. Existing single-op YAMLs (where
  /// the file's root is the InferenceOp's section) keep working unchanged.
  ///
  /// @param node YAML node containing the InferenceOp parameter subtree (as documented above).
  /// @param config_dir Directory used to resolve relative `model_path_map` paths. Absolute paths
  ///   pass through unchanged.
  /// @return The populated `InferenceOpParams`.
  /// @throws std::runtime_error if a required key (`model_path_map`, `input_map`, `output_map`)
  ///   is missing from `node`.
  [[nodiscard]] static InferenceOpParams from_yaml_node(const YAML::Node& node,
                                                        const std::filesystem::path& config_dir);
};

/// @brief Partial `HOLOSCAN_PARAM_SCHEMA` for `InferenceOpParams`.
///
/// Known limitation: `holoscan::field<>` (parameter_schema.hpp) has no `std::map` field kind, so
/// `InferenceDataMap`/`InferenceDataVecMap` (which back `model_path_map`, `input_map`,
/// `output_map`, `backend_map` — the fields that determine `InferenceOp`'s port topology) and the
/// `shared_ptr`-held resources (`cuda_stream_pool`, `cuda_green_context_pool`) cannot
/// be listed below. This schema is therefore declared `.partial()`, covering only the
/// scalar/bool/string fields, so `param_schema_availability<InferenceOpParams>()` reports
/// `kDeclared` rather than `kOpaqueLocal`.
HOLOSCAN_PARAM_SCHEMA(
    InferenceOpParams, holoscan::param_schema("nvidia.holoscan.ops.InferenceOpParams", 1).partial(),
    holoscan::field<&InferenceOpParams::backend>("backend"),
    holoscan::field<&InferenceOpParams::is_engine_path>("is_engine_path"),
    holoscan::field<&InferenceOpParams::input_on_cuda>("input_on_cuda"),
    holoscan::field<&InferenceOpParams::output_on_cuda>("output_on_cuda"),
    holoscan::field<&InferenceOpParams::transmit_on_cuda>("transmit_on_cuda"),
    holoscan::field<&InferenceOpParams::dynamic_input_dims>("dynamic_input_dims"),
    holoscan::field<&InferenceOpParams::output_dynamic_dims>("output_dynamic_dims"),
    holoscan::field<&InferenceOpParams::enable_fp16>("enable_fp16"),
    holoscan::field<&InferenceOpParams::enable_cuda_graphs>("enable_cuda_graphs"),
    holoscan::field<&InferenceOpParams::dla_core>("dla_core"),
    holoscan::field<&InferenceOpParams::parallel_inference>("parallel_inference"),
    holoscan::field<&InferenceOpParams::max_batch_size>("max_batch_size"),
    holoscan::field<&InferenceOpParams::max_memory>("max_memory"));

/// @brief Inference Operator class to perform single/multi model TensorRT inference.
///
/// This is a TensorRT-only build of InferenceOp, built on the Holoscan Operator/port model. See
/// InferenceOpParams for the retained configuration surface.
///
/// ==Ports==
///
/// This operator declares one `Input<Tensor>` per *external* input tensor name found in
/// `input_map` and one `Output<Tensor>` per *external* output tensor name found in `output_map`,
/// named after the tensor. Ports are declared dynamically in setup() from the params() aggregate.
///
/// A tensor name that is both an `output_map` value (for one model) and an `input_map` value (for
/// another model) is a "chained" tensor -- an internal handoff between models managed entirely by
/// HoloInfer's own execution plan (see the fallback ladder's `H_CHAINED_MODELS` case below), not an
/// external port. No `Input<Tensor>`/`Output<Tensor>` port is declared for it: declaring both would
/// be rejected by the compiler (`kDuplicateName`, since the same tensor name would otherwise back
/// both an input and an output port on this operator).
///
/// ==Dispatch selection and the fallback ladder==
///
/// start() auto-selects the fastest safe dispatch path for the configuration (see
/// fast_path_eligibility.hpp for the seven DispatchKind values and the eligibility rule) and then
/// runs it through the fallback ladder in fallback_ladder.hpp
/// (`inference_eligibility::resolve_dispatch_with_fallback()`). The ladder degrades one documented
/// rung at a time — green-context isolation -> mutex-based capture protection -> parallel without
/// CUDA graphs ->
/// sequential fast path -> single-model fast path -> kStandard — until it reaches a configuration
/// that is guaranteed to run. **start() never refuses to start because of a dispatch/capture
/// configuration**, and no degradation is silent: each rung emits an INFO or WARN block of the form
///
/// ```
/// InferenceOp/HoloInfer: fallback triggered — <REASON_KEY>
///     requested: <dispatch> + <capture protection>
///     using:     <dispatch> + <capture protection>
///     reason:    <why>
///     fix:       <user-side suggestion, or "none">
/// ```
///
/// naming which primary path was rejected, what replaced it, why, and how to get the faster path
/// back. Runtime-only degradations raised inside HoloInfer (a failed green-context push, a failed
/// CUDA-graph capture setup or replay, a blocking stream swapped out from under CUDA Graphs) use
/// the same format. `modules/inference/README.md`, section "Fallback ladder", documents all
/// fallback triggers, what causes them, and how to avoid each one.
class HOLOSCAN_CORE_EXPORT InferenceOp : public holoscan::Operator<InferenceOpParams> {
 public:
  InferenceOp() = delete;

  /// @brief Construct InferenceOp with its final `InferenceOpParams` already bound.
  ///
  /// @param params The operator's final configuration.
  /// @throws RuntimeError if `params` fails aggregate validation.
  explicit InferenceOp(InferenceOpParams params);

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  /// @brief Trigger only once every configured input tensor has an aligned head sample.
  ///
  [[nodiscard]] Contract contract() const override;

  /// @brief Look up one of the dynamically-declared input ports by tensor name (see the class
  /// doc comment: one `Input<Tensor>` per *external* `input_map` tensor name).
  ///
  /// @param tensor_name Input tensor name (an `input_map` value).
  /// @return Pointer to the port, or nullptr if `tensor_name` is not configured, or is an internal
  ///   chained tensor with no external port (see the class doc comment).
  [[nodiscard]] Input<Tensor>* input_port(const std::string& tensor_name) const {
    auto it = input_port_map_.find(tensor_name);
    return it == input_port_map_.end() ? nullptr : it->second;
  }

  /// @brief Output-port counterpart of input_port(); one `Output<Tensor>` per *external*
  /// `output_map` tensor name (nullptr for an internal chained tensor).
  [[nodiscard]] Output<Tensor>* output_port(const std::string& tensor_name) const {
    auto it = output_port_map_.find(tensor_name);
    return it == output_port_map_.end() ? nullptr : it->second;
  }

 private:
  /// Pointer to inference context.
  std::unique_ptr<holoscan::inference::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications.
  std::shared_ptr<holoscan::inference::InferenceSpecs> inference_specs_;

  /// @brief External input tensor names (consumed but not produced by any configured model) --
  /// i.e. excluding internal/chained tensors. One `Input<Tensor>` port is declared per entry.
  std::vector<std::string> model_inputs_;
  /// @brief External output tensor names (produced but not consumed by any configured model) --
  /// i.e. excluding internal/chained tensors. One `Output<Tensor>` port is declared per entry.
  std::vector<std::string> model_outputs_;

  /// Operator identifier, used in reporting.
  const std::string module_{"Inference Operator"};

  /// Stable storage for dynamically-declared ports (deque keeps element addresses stable across
  /// growth, which OperatorSpec::input()/output() requires).
  std::deque<Input<Tensor>> input_ports_;
  std::deque<Output<Tensor>> output_ports_;

  /// Name -> port lookup used by the glue layer in holoinfer_utils.
  std::map<std::string, Input<Tensor>*> input_port_map_;
  std::map<std::string, Output<Tensor>*> output_port_map_;

  bool validate_tensor_dimensions_ = true;

  /// @brief Cached single-model name used by the single-model fast paths (kFast / kDynFast).
  std::string fast_single_model_name_;

  /// @brief Dispatch kind auto-selected in start() from the operator's configuration. See
  /// fast_path_eligibility.hpp for the selection rule.
  inference_eligibility::DispatchKind dispatch_kind_ =
      inference_eligibility::DispatchKind::kStandard;

  /// @brief Streams leased from `cuda_stream_pool` for the lifetime of the operator. Pre-sized in
  /// start() to `n_streams_needed` (1 for kFast, N for kParFast/kDynParFast) and slotted into by
  /// index from `allocate_cuda_stream`'s callback (see inference.cpp). HoloInfer may invoke that
  /// callback from multiple threads for kParFast/kDynParFast, so slots are claimed via
  /// `stream_lease_idx_` rather than via `push_back`, which would race concurrent callers.
  /// Each slot is `std::optional` because `StreamLease` cannot be safely default-constructed —
  /// a default-constructed lease has no owning pool and its destructor would trip on release.
  /// Leases are returned to the pool in stop()/destruction.
  std::vector<std::optional<holoscan::CudaStreamPool::StreamLease>> stream_leases_;

  /// @brief Next free index into `stream_leases_`, claimed with `fetch_add` by the
  /// `allocate_cuda_stream` callback so concurrent HoloInfer-side callers each get a distinct,
  /// non-overlapping slot without a lock (see M1 in inference.cpp::start()).
  std::atomic<size_t> stream_lease_idx_{0};
};

}  // namespace holoscan::ops
