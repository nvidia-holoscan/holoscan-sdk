/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/inference/inference.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/operators/inference/fallback_ladder.hpp>
#include <holoscan/utils/holoinfer_utils.hpp>

#include <holoinfer/holoinfer_utils.hpp>

namespace holoscan::ops {

namespace HoloInfer = holoscan::inference;

namespace {

// Collect the unique, deterministically ordered set of tensor names referenced as values across
// a model -> [tensor names] map (std::map already iterates by sorted model name).
std::vector<std::string> unique_tensor_names(const InferenceDataVecMap& model_to_tensors) {
  std::vector<std::string> ordered;
  std::set<std::string> seen;
  for ([[maybe_unused]] const auto& [model_name, tensors] : model_to_tensors) {
    for (const auto& tensor_name : tensors) {
      if (seen.insert(tensor_name).second) {
        ordered.push_back(tensor_name);
      }
    }
  }
  return ordered;
}

}  // namespace

InferenceOp::InferenceOp(InferenceOpParams params) {
  auto bound = holoscan::detail::GraphOperatorAccess::bind(*this, std::move(params));
  if (!bound) {
    throw std::runtime_error("InferenceOp: invalid InferenceOpParams: " + bound.error().message);
  }

  // Derive the external-input/output tensor-name lists and the stable port dequeues here at
  // construction time, so contract() and setup() no longer depend on each other's execution
  // order (a fresh instance reached via a non-setup()-first replay path could otherwise
  // observe an empty contract). setup() only *declares* the already-computed ports.
  const InferenceOpParams& config = this->params();

  std::set<std::string> produced;
  std::set<std::string> consumed;
  for ([[maybe_unused]] const auto& [model_name, tensors] : config.output_map) {
    produced.insert(tensors.begin(), tensors.end());
  }
  for ([[maybe_unused]] const auto& [model_name, tensors] : config.input_map) {
    consumed.insert(tensors.begin(), tensors.end());
  }

  for (const auto& name : unique_tensor_names(config.input_map)) {
    if (produced.find(name) == produced.end()) {
      model_inputs_.push_back(name);
    }
  }
  for (const auto& name : unique_tensor_names(config.output_map)) {
    if (consumed.find(name) == consumed.end()) {
      model_outputs_.push_back(name);
    }
  }

  for (const auto& name : model_inputs_) {
    input_ports_.emplace_back(name);
    input_port_map_[name] = &input_ports_.back();
  }
  for (const auto& name : model_outputs_) {
    output_ports_.emplace_back(name);
    output_port_map_[name] = &output_ports_.back();
  }
}

namespace {

// InferenceOp does not know per-tensor rank/dtype at setup() time (models are not loaded until
// start()), so only the memory-kind fact is declared here. This is the minimum fact the
// compiler needs to provision a CUDA compute stream for a CUDA-device Tensor port (see
// RUNTIME_PORT_BINDING_UNAVAILABLE in run_session.cpp::prepare_tensor_stream_access). Rank/dtype
// are intentionally left unconstrained (std::nullopt); TensorRepresentation treats every absent
// member as unconstrained.
[[nodiscard]] TensorInputSpec memory_kind_input_spec(bool on_cuda) noexcept {
  return TensorInputSpec{
      .representation =
          TensorRepresentation{
              .memory_kind = on_cuda ? MemoryKind::kCudaDevice : MemoryKind::kHost,
          },
  };
}

constexpr std::uint32_t kMaxOutputTensorRank = 8U;
// TODO(follow-up MR): the 256 MiB cap is stamped on every output tensor uniformly, which
// over-provisions small outputs (e.g. classifier logits) and silently rejects outputs larger
// than 256 MiB (e.g. high-resolution segmentation masks). Replace with a configurable
// per-model override in InferenceOpParams (default 256 MiB), and eventually derive from
// model metadata once engine build can be reordered before compile.
constexpr std::size_t kMaxOutputTensorByteSpan = 256UL * 1024UL * 1024UL;

// Runtime-pool output spec carrying only the known memory-kind fact plus the conservative-large
// rank/byte-span bound above. Rank/dtype remain unconstrained in the representation for the same
// reason as memory_kind_input_spec(); TensorOutputSpecBuilder derives the compiler's
// TensorPoolBounds directly from this spec's bounds when storage is kRuntimePool.
[[nodiscard]] TensorOutputSpec memory_kind_output_spec(bool on_cuda) noexcept {
  return TensorOutputSpec{
      .representation =
          TensorRepresentation{
              .memory_kind = on_cuda ? MemoryKind::kCudaDevice : MemoryKind::kHost,
          },
      .bounds =
          TensorBounds{.max_rank = kMaxOutputTensorRank, .max_byte_span = kMaxOutputTensorByteSpan},
      .storage = TensorOutputStorage::kRuntimePool,
  };
}

}  // namespace

void InferenceOp::setup(OperatorSpec& spec) {
  const InferenceOpParams& config = params();

  // Ports and the model_inputs_/model_outputs_ name lists are populated in the constructor
  // (D1). setup() only declares each pre-built port on `spec`; it holds no name-derivation or
  // container-mutation logic and is therefore idempotent — a repeated call will redeclare the
  // same ports without appending duplicates (which would otherwise trip kDuplicateName).
  const TensorInputSpec input_spec = memory_kind_input_spec(config.input_on_cuda);
  // The pool declares the memory kind of the *emitted* tensor, which is `transmit_on_cuda`
  // (B2/B3): the compute path allocates from this pool and emits through it.
  const TensorOutputSpec output_spec = memory_kind_output_spec(config.transmit_on_cuda);

  for (size_t i = 0; i < model_inputs_.size(); ++i) {
    spec.input(input_ports_.at(i), model_inputs_[i]).expects_tensor(input_spec);
  }
  for (size_t i = 0; i < model_outputs_.size(); ++i) {
    spec.output(output_ports_.at(i), model_outputs_[i]).produces_tensor(output_spec);
  }
}

Contract InferenceOp::contract() const {
  Contract result;
  if (input_ports_.size() >= 2U) {
    std::vector<PortRef> refs;
    refs.reserve(input_ports_.size());
    for (const Input<Tensor>& port : input_ports_) {
      refs.emplace_back(port);
    }
    result.trigger(OnAligned{.ports = PortList{refs}, .window = std::chrono::nanoseconds{0}});
    for (const Input<Tensor>& port : input_ports_) {
      result.input(port, Exact{});
    }
  } else if (input_ports_.size() == 1U) {
    result.trigger(OnEach{input_ports_.front()});
  }
  return result;
}

void InferenceOp::start() {
  const InferenceOpParams& config = params();
  // Re-arm the one-shot dimension-check latch so a stop()/start() cycle re-validates input
  // dimensions on the first tick after restart (D2).
  validate_tensor_dimensions_ = true;
  try {
    std::vector<std::string> external_inputs;
    std::vector<std::string> external_outputs;
    std::vector<std::string> transmit_outputs;
    auto status = HoloInfer::setup_inference_io(config.input_map,
                                                config.output_map,
                                                external_inputs,
                                                external_outputs,
                                                transmit_outputs,
                                                {});
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    status = HoloInfer::inference_validity_check(config.model_path_map,
                                                 config.input_map,
                                                 config.output_map,
                                                 external_inputs,
                                                 external_outputs);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    // Initialize the CUDA green-context pool BEFORE the fallback ladder queries num_partitions().
    // num_partitions() returns green_contexts_.size(), which is 0 until initialize_pool() runs,
    // so without this early init the ladder would always see 0 partitions and fall back even
    // when a valid pool was supplied. On init failure we leave the pool untouched: num_partitions()
    // stays at 0, the ladder falls back, and TensorRT engine building uses all GPU SMs.
    bool green_context_pool_initialized = false;
    if (config.cuda_green_context_pool) {
      auto init_status = config.cuda_green_context_pool->initialize_pool();
      if (!init_status) {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: cuda_green_context_pool initialization failed ({}); TensorRT engine "
            "will be built with all GPU SMs and the fallback ladder will not see isolated "
            "partitions.",
            init_status.error().message);
      } else {
        green_context_pool_initialized = true;
      }
    }

    // ── Auto-detect the fastest safe dispatch path (TRT-only), then run the fallback ladder
    const size_t n_models = config.model_path_map.size();
    inference_eligibility::FallbackContext fallback_ctx;
    fallback_ctx.model_path_map = config.model_path_map;
    fallback_ctx.input_map = config.input_map;
    fallback_ctx.output_map = config.output_map;
    fallback_ctx.backend = config.backend;
    fallback_ctx.backend_map = config.backend_map;
    fallback_ctx.input_on_cuda = config.input_on_cuda;
    fallback_ctx.dynamic_input_dims = config.dynamic_input_dims;
    fallback_ctx.parallel_inference = config.parallel_inference;
    fallback_ctx.enable_cuda_graphs = config.enable_cuda_graphs;
    fallback_ctx.stream_pool_present = static_cast<bool>(config.cuda_stream_pool);
    fallback_ctx.stream_pool_max_size =
        config.cuda_stream_pool ? static_cast<uint32_t>(config.cuda_stream_pool->capacity()) : 0;
    fallback_ctx.green_context_pool_present = static_cast<bool>(config.cuda_green_context_pool);
    fallback_ctx.green_context_num_partitions =
        config.cuda_green_context_pool
            ? static_cast<uint32_t>(config.cuda_green_context_pool->num_partitions())
            : 0;

    const auto fallback_decision =
        inference_eligibility::resolve_dispatch_with_fallback(fallback_ctx);
    inference_eligibility::emit_fallback_decision(fallback_decision);

    dispatch_kind_ = fallback_decision.final_choice;
    fast_single_model_name_ = fallback_decision.fast_single_model_name;
    const bool enable_cuda_graphs = fallback_decision.enable_cuda_graphs;

    switch (dispatch_kind_) {
      case inference_eligibility::DispatchKind::kFast:
        HOLOSCAN_LOG_INFO("InferenceOp: fast path auto-selected (kFast) for single TRT model '{}'.",
                          fast_single_model_name_);
        break;
      case inference_eligibility::DispatchKind::kDynFast:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: fast path auto-selected (kDynFast) for single TRT model '{}'.",
            fast_single_model_name_);
        break;
      case inference_eligibility::DispatchKind::kSeqFast:
        HOLOSCAN_LOG_INFO("InferenceOp: fast path auto-selected (kSeqFast) for {} TRT models.",
                          n_models);
        break;
      case inference_eligibility::DispatchKind::kDynSeqFast:
        HOLOSCAN_LOG_INFO("InferenceOp: fast path auto-selected (kDynSeqFast) for {} TRT models.",
                          n_models);
        break;
      case inference_eligibility::DispatchKind::kParFast:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: fast path auto-selected (kParFast) for {} TRT models (parallel).",
            n_models);
        break;
      case inference_eligibility::DispatchKind::kDynParFast:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: fast path auto-selected (kDynParFast) for {} TRT models (parallel).",
            n_models);
        break;
      case inference_eligibility::DispatchKind::kStandard:
      default:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: dispatch kind kStandard selected (fast paths not eligible).");
        break;
    }

    // CUDA stream pool wiring (see InferenceOpParams::cuda_stream_pool). When present,
    // HoloInfer calls this callback once per model to obtain a dedicated stream for kParFast /
    // kDynParFast (must be distinct per call; see infer_manager_setup.cpp). HoloInfer may invoke
    // this callback concurrently from multiple threads (one per model) for the parallel fast
    // paths, so stream_leases_ is pre-sized here and each call claims its own slot via an atomic
    // fetch_add rather than mutating the vector's size, which would race (M1: see
    // stream_lease_idx_'s doc comment in inference.hpp).
    const size_t n_streams_needed = n_models > 0 ? n_models : 1;
    stream_leases_.clear();
    stream_leases_.resize(n_streams_needed);
    stream_lease_idx_.store(0, std::memory_order_relaxed);

    std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream;
    if (config.cuda_stream_pool) {
      auto init_status = config.cuda_stream_pool->initialize_pool();
      if (!init_status) {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: cuda_stream_pool initialization failed ({}); fast-path dispatch will "
            "use internally-created streams.",
            init_status.error().message);
      } else {
        auto pool = config.cuda_stream_pool;
        allocate_cuda_stream =
            [this, pool, n_streams_needed](int32_t /*device_id*/) -> cudaStream_t {
          auto leased = pool->acquire();
          if (!leased) {
            HOLOSCAN_LOG_WARN("InferenceOp: cuda_stream_pool exhausted ({})",
                              leased.error().message);
            return nullptr;
          }
          // Each concurrent caller claims a distinct, never-reused slot. stream_leases_ was
          // sized to n_streams_needed above and is never resized again for the life of this
          // operator, so writing to a claimed index is data-race free: no two callers can ever
          // receive the same index, and no read of stream_leases_ (stop()/destruction) happens
          // concurrently with these writes.
          const size_t slot = stream_lease_idx_.fetch_add(1, std::memory_order_relaxed);
          if (slot >= n_streams_needed) {
            HOLOSCAN_LOG_WARN(
                "InferenceOp: cuda_stream_pool allocation callback invoked more times ({}) than "
                "the {} stream(s) reserved at start(); returning no stream for this call.",
                slot + 1,
                n_streams_needed);
            return nullptr;
          }
          const cudaStream_t handle = leased.value().native_handle();
          stream_leases_.at(slot).emplace(std::move(leased.value()));
          return handle;
        };
      }
    }

    inference_specs_ = std::make_shared<HoloInfer::InferenceSpecs>(
        config.backend,             // backend
        config.backend_map,         // backend_map
        config.model_path_map,      // model_path_map
        config.input_map,           // input_map
        config.output_map,          // output_map
        config.dla_core_map,        // dla_core_map (per-model DLA core override)
        config.trt_opt_profile,     // trt_opt_profile
        config.dynamic_input_dims,  // dynamic_input_dims
        config.is_engine_path,      // is_engine_path
        config.parallel_inference,  // parallel_proc
        config.enable_fp16,         // use_fp16
        config.input_on_cuda,       // cuda_buffer_in
        config.output_on_cuda,      // cuda_buffer_out
        enable_cuda_graphs,         // use_cuda_graphs (resolved by fallback ladder)
        config.dla_core,            // dla_core
        config.dla_gpu_fallback,    // dla_gpu_fallback
        false,                      // gpu_resident_inference (not exposed by InferenceOpParams)
        allocate_cuda_stream);      // allocate_cuda_stream callback

    // TensorRT engine-build tuning (backward-compatible: defaults preserved).
    inference_specs_->max_batch_size_ = config.max_batch_size;
    inference_specs_->max_memory_ = config.max_memory;

    // Enable HoloInfer's manager-side multi-model fast-path caches (per-model parallel streams +
    // completion events) whenever a multi-model fast path was chosen.
    const bool multi_model_fast =
        dispatch_kind_ == inference_eligibility::DispatchKind::kSeqFast ||
        dispatch_kind_ == inference_eligibility::DispatchKind::kParFast ||
        dispatch_kind_ == inference_eligibility::DispatchKind::kDynSeqFast ||
        dispatch_kind_ == inference_eligibility::DispatchKind::kDynParFast;
    inference_specs_->fast_multi_model_ = multi_model_fast;

    // CUDA green-context pool wiring (see InferenceOpParams::cuda_green_context_pool).
    // When present AND successfully initialized above (pre-ladder), thread the pool's default
    // partition CUcontext/SM-count through to HoloInfer so TensorRT engine building is constrained
    // to that SM partition, and record every partition's CUcontext for the runtime dispatch layer.
    // Degrades silently to full-GPU engine building when the pool is null, init failed, or the
    // default partition cannot be resolved.
    if (green_context_pool_initialized) {
      auto default_index = config.cuda_green_context_pool->default_context_index();
      if (default_index) {
        auto ctx = config.cuda_green_context_pool->cuda_context(default_index.value());
        auto sms = config.cuda_green_context_pool->partition_sms(default_index.value());
        if (ctx && sms) {
          inference_specs_->build_cuda_context_ = ctx.value();
          inference_specs_->build_sm_count_ = static_cast<int32_t>(sms.value());
          HOLOSCAN_LOG_INFO(
              "InferenceOp: TensorRT engine will be built within green context partition {} "
              "({} SMs)",
              default_index.value(),
              inference_specs_->build_sm_count_);
        } else {
          HOLOSCAN_LOG_WARN(
              "InferenceOp: could not resolve CUcontext/SM count for the default green context "
              "partition; TensorRT engine will be built with all GPU SMs.");
        }
      } else {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: could not resolve a default green context partition; TensorRT engine "
            "will be built with all GPU SMs.");
      }

      // Runtime dispatch wiring: resolve every materialized partition's CUcontext so
      // ManagerInfer can bind each multi-model fast-path model (index m) to partition
      // m % num_partitions() and actually push that context around TrtInfer::
      // do_inference_on_stream()'s enqueueV3 call (see infer/trt/core.cpp). Without this,
      // decide_capture_protection() reports kGreenContextIsolated but every model still
      // enqueues into the default/shared CUDA context — safety would come only from the
      // capture_serialization_mutex_ net, defeating the purpose of supplying a pool.
      const auto num_partitions =
          static_cast<uint32_t>(config.cuda_green_context_pool->num_partitions());
      std::vector<CUcontext> partition_contexts;
      partition_contexts.reserve(num_partitions);
      for (uint32_t p = 0; p < num_partitions; ++p) {
        auto part_ctx = config.cuda_green_context_pool->cuda_context(p);
        if (!part_ctx) {
          HOLOSCAN_LOG_WARN(
              "InferenceOp: could not resolve CUcontext for green context partition {}; "
              "runtime dispatch binding for this partition will be skipped.",
              p);
          partition_contexts.push_back(nullptr);
          continue;
        }
        partition_contexts.push_back(part_ctx.value());
      }
      inference_specs_->green_context_partition_contexts_ = std::move(partition_contexts);
    }

    // ── Auto-safety guard: CUDA-Graph capture protection for kParFast/kDynParFast ──
    // See fast_path_eligibility.hpp::decide_capture_protection() for the full rationale.
    // Concurrent CUDA-Graph capture across InferenceOp instances sharing a CUDA context can
    // invalidate an in-progress capture on another stream. kParFast is protected by runtime
    // mutex-based capture serialization (ManagerInfer::capture_serialization_mutex_); kDynParFast
    // has no setup-time warm-up path (dynamic shapes), so it is safe only when a
    // CudaGreenContextPool provides context isolation.
    switch (fallback_decision.final_protection) {
      case inference_eligibility::CaptureProtectionOutcome::kGreenContextIsolated:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: concurrent CUDA-Graph capture protected by CudaGreenContextPool ({} "
            "partitions).",
            fallback_ctx.green_context_num_partitions);
        break;
      case inference_eligibility::CaptureProtectionOutcome::kMutexProtected:
        HOLOSCAN_LOG_INFO(
            "InferenceOp: concurrent CUDA-Graph capture protected by runtime mutex-based "
            "serialization (static shapes).");
        break;
      case inference_eligibility::CaptureProtectionOutcome::kNotApplicable:
      case inference_eligibility::CaptureProtectionOutcome::kRefuse:
      default:
        break;
    }
    // Thread the decision through to ManagerInfer: only skip capture_serialization_mutex_ when
    // context isolation is the actual reason this configuration is safe, not merely because a
    // pool happens to be present (e.g. kMutexProtected still needs the mutex net).
    inference_specs_->green_context_isolated_ =
        (fallback_decision.final_protection ==
         inference_eligibility::CaptureProtectionOutcome::kGreenContextIsolated);

    holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();
    status = holoscan_infer_context_->set_inference_params(inference_specs_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Start, Parameters setup, " + status.get_message());
    }
    HOLOSCAN_LOG_INFO("Inference Specifications created");
  } catch (const std::bad_alloc& e) {
    HOLOSCAN_LOG_ERROR("{}: memory allocation failed during start(): {}", module_, e.what());
    throw;
  } catch (const std::runtime_error& e) {
    HOLOSCAN_LOG_ERROR("{}: runtime error during start(): {}", module_, e.what());
    throw;
  } catch (...) {
    HOLOSCAN_LOG_ERROR("{}: unknown exception during start()", module_);
    throw;
  }
}

void InferenceOp::stop() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
  stream_leases_.clear();
}

expected<void, Error> InferenceOp::compute(ExecutionContext& context) {
  // Compute callbacks read the pinned per-activation view via ExecutionContext, not the
  // operator-stored params() aggregate (B1). Do not retain this reference across a
  // callback boundary.
  const InferenceOpParams& config = context.params<InferenceOpParams>();
  const cudaStream_t cuda_stream = context.cuda_stream().get();

  auto extracted = holoscan::utils::get_data_per_model(input_port_map_,
                                                       inference_specs_->data_per_tensor_,
                                                       inference_specs_->dims_per_tensor_,
                                                       config.input_on_cuda,
                                                       module_,
                                                       cuda_stream);
  if (!extracted) {
    return make_unexpected(std::move(extracted).error());
  }

  if (validate_tensor_dimensions_ && !config.dynamic_input_dims) {
    validate_tensor_dimensions_ = false;
    auto model_in_dims_map = holoscan_infer_context_->get_input_dimensions();
    auto dim_status = HoloInfer::tensor_dimension_check(
        config.input_map, model_in_dims_map, inference_specs_->dims_per_tensor_, model_inputs_);
    if (dim_status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("{}: {}", module_, dim_status.get_message());
      return make_unexpected(Error{ErrorCode::kFailure});
    }
  }

  // HOLOINFER_TIME_INFERENCE_COMPUTE is a developer-only "grep me and rebuild" flag for
  // per-tick inference timing; it is deliberately not exposed as a CMake option so it never
  // ships enabled. To use it: add
  // `target_compile_definitions(holoscan_op_inference PRIVATE HOLOINFER_TIME_INFERENCE_COMPUTE)`
  // to modules/inference/operators/CMakeLists.txt (or pass
  // `-DHOLOINFER_TIME_INFERENCE_COMPUTE` via CXX flags) and rebuild.
#ifdef HOLOINFER_TIME_INFERENCE_COMPUTE
  HoloInfer::TimePoint s_time;
  HoloInfer::TimePoint e_time;
  HoloInfer::timer_init(s_time);
#endif

  // Dispatch to the streamlined entry point auto-selected in start().
  HoloInfer::InferStatus status;
  switch (dispatch_kind_) {
    case inference_eligibility::DispatchKind::kFast:
      status = holoscan_infer_context_->execute_inference_fast(
          inference_specs_, cuda_stream, fast_single_model_name_);
      break;
    case inference_eligibility::DispatchKind::kSeqFast:
      status = holoscan_infer_context_->execute_inference_seq_fast(inference_specs_, cuda_stream);
      break;
    case inference_eligibility::DispatchKind::kParFast:
      status = holoscan_infer_context_->execute_inference_par_fast(inference_specs_, cuda_stream);
      break;
    case inference_eligibility::DispatchKind::kDynFast:
      status = holoscan_infer_context_->execute_inference_dyn_fast(
          inference_specs_, cuda_stream, fast_single_model_name_);
      break;
    case inference_eligibility::DispatchKind::kDynSeqFast:
      status =
          holoscan_infer_context_->execute_inference_dyn_seq_fast(inference_specs_, cuda_stream);
      break;
    case inference_eligibility::DispatchKind::kDynParFast:
      status =
          holoscan_infer_context_->execute_inference_dyn_par_fast(inference_specs_, cuda_stream);
      break;
    case inference_eligibility::DispatchKind::kStandard:
    default:
      status = holoscan_infer_context_->execute_inference(inference_specs_, cuda_stream);
      break;
  }
#ifdef HOLOINFER_TIME_INFERENCE_COMPUTE
  HoloInfer::timer_init(e_time);
  HoloInfer::timer_check(s_time, e_time, "Inference Operator: Inference execution");
#endif

  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    HOLOSCAN_LOG_ERROR("{}: {}", module_, status.get_message());
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  auto model_out_dims_map = holoscan_infer_context_->get_output_dimensions();

  auto transmitted = holoscan::utils::transmit_data_per_model(config.output_map,
                                                              output_port_map_,
                                                              inference_specs_->output_per_model_,
                                                              model_out_dims_map,
                                                              config.output_on_cuda,
                                                              config.transmit_on_cuda,
                                                              module_,
                                                              context);
  if (!transmitted) {
    return make_unexpected(std::move(transmitted).error());
  }

  return {};
}

}  // namespace holoscan::ops
