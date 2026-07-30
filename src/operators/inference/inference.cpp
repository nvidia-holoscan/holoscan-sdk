/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/inference/fast_path_eligibility.hpp>
#include <holoscan/operators/inference/inference.hpp>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_green_context.hpp>
#include <holoscan/operators/inference/codecs.hpp>
#include <holoscan/utils/holoinfer_utils.hpp>

#include <holoinfer_utils.hpp>

/**
 * Custom YAML parser for DataMap class
 */
template <>
struct YAML::convert<holoscan::ops::InferenceOp::DataMap> {
  static Node encode(const holoscan::ops::InferenceOp::DataMap& datamap) {
    Node node;
    auto mappings = datamap.get_map();
    for (const auto& [key, value] : mappings) {
      node[key] = value;
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceOp::DataMap& datamap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }
    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::string value = it->second.as<std::string>();
        datamap.insert(key, std::move(value));
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

/**
 * Custom YAML parser for DataVecMap class
 */
template <>
struct YAML::convert<holoscan::ops::InferenceOp::DataVecMap> {
  static Node encode(const holoscan::ops::InferenceOp::DataVecMap& datavmap) {
    Node node;
    auto mappings = datavmap.get_map();
    for (const auto& [key, vec_of_values] : mappings) {
      for (const auto& value : vec_of_values)
        node[key].push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::InferenceOp::DataVecMap& datavmap) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();

        switch (it->second.Type()) {
          case YAML::NodeType::Scalar: {  // For backward compatibility v0.5 and lower
            HOLOSCAN_LOG_WARN("Values for model {} not in a vector form.", key);
            HOLOSCAN_LOG_INFO(
                "HoloInfer in Holoscan SDK 0.6 onwards expects tensor names for models in a "
                "vector "
                "form in the parameter set.");
            HOLOSCAN_LOG_INFO(
                "Converting input tensor names for model {} to vector form for backward "
                "compatibility.",
                key);
            HOLOSCAN_LOG_WARN("Single I/O per model supported in backward compatibility mode.");
            std::string value = it->second.as<std::string>();
            datavmap.insert(key, {std::move(value)});
          } break;
          case YAML::NodeType::Sequence: {
            std::vector<std::string> value = it->second.as<std::vector<std::string>>();
            datavmap.insert(key, value);
          } break;
          default: {
            HOLOSCAN_LOG_ERROR("Unsupported entry in parameter set for model {}", key);
            return false;
          }
        }
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

namespace holoscan::ops {

void InferenceOp::setup(OperatorSpec& spec) {
  register_converter<DataMap>();
  register_converter<DataVecMap>();
  spec.input<std::vector<gxf::Entity>>("receivers", IOSpec::kAnySize);
  spec.input<std::any>("model_activation_specs").condition(ConditionType::kNone);

  spec.output<gxf::Entity>("transmitter");

  spec.param(backend_, "backend", "Supported backend", "backend", {});
  spec.param(backend_map_, "backend_map", "Supported backend map", "", DataMap());
  spec.param(model_path_map_,
             "model_path_map",
             "Model Keyword with File Path",
             "Path to ONNX model to be loaded.",
             DataMap());
  spec.param(device_map_,
             "device_map",
             "Model Keyword with associated device",
             "Device ID on which model will do inference.",
             DataMap());
  spec.param(dla_core_map_,
             "dla_core_map",
             "Model Keyword with associated DLA core index",
             "The DLA core index on which model will do inference, starts at 0. Set to -1 to "
             "disable DLA.",
             DataMap());
  spec.param(temporal_map_,
             "temporal_map",
             "Model Keyword with associated frame execution delay",
             "Frame delay for model inference.",
             DataMap());
  spec.param(activation_map_,
             "activation_map",
             "Model Keyword with associated model inference activation",
             "Activation of model inference (1 = active, 0 = inactive).",
             DataMap());
  // [Parameter rename] `input_map` is the new preferred YAML key; `pre_processor_map` is
  // retained as a backward-compat alias. When both are set, `input_map` takes precedence.
  spec.param(pre_processor_map_,
             "pre_processor_map",
             "Pre-processor setting per model (LEGACY name, prefer 'input_map')",
             "Map of model name → list of input tensor names. Backward-compat alias.",
             DataVecMap());
  spec.param(input_map_,
             "input_map",
             "Input tensor(s) per model",
             "Map of model name → list of input tensor names. Preferred name; supersedes "
             "the legacy 'pre_processor_map' when both are set.",
             DataVecMap());
  // [Parameter rename] `output_map` is the new preferred YAML key; `inference_map` is
  // retained as a backward-compat alias. When both are set, `output_map` takes precedence.
  spec.param(inference_map_,
             "inference_map",
             "Inferred tensor per model (LEGACY name, prefer 'output_map')",
             "Map of model name → list of output tensor names. Backward-compat alias.",
             DataVecMap());
  spec.param(output_map_,
             "output_map",
             "Output tensor(s) per model",
             "Map of model name → list of output tensor names. Preferred name; supersedes "
             "the legacy 'inference_map' when both are set.",
             DataVecMap());
  spec.param(in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {});
  spec.param(out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {});
  spec.param(trt_opt_profile_,
             "trt_opt_profile",
             "TensorRT Opt Profile",
             "Optimization profile for input tensors",
             DataVecMap());
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(infer_on_cpu_, "infer_on_cpu", "Inference on CPU", "Use CPU.", false);
  spec.param(is_engine_path_, "is_engine_path", "Input path is engine file", "", false);

  spec.param(enable_fp16_, "enable_fp16", "Use fp16", "Use fp16.", false);
  spec.param(enable_cuda_graphs_,
             "enable_cuda_graphs",
             "Use CUDA graphs",
             "Enable usage of CUDA Graphs for backends which support it.",
             true);

  spec.param(dla_core_,
             "dla_core",
             "DLA core index",
             "The DLA core index to execute the engine on, starts at 0. Set to -1 (the default) to "
             "disable DLA.",
             -1);

  spec.param(
      dla_gpu_fallback_,
      "dla_gpu_fallback",
      "Enable DLA GPU fallback",
      "If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is "
      "disabled, engine creation will fail if a layer cannot executed on DLA.",
      true);

  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", true);
  spec.param(output_on_cuda_, "output_on_cuda", "Output buffer on CUDA", "", true);
  spec.param(transmit_on_cuda_, "transmit_on_cuda", "Transmit message on CUDA", "", true);
  spec.param(dynamic_input_dims_, "dynamic_input_dims", "Dynamic Input dimensions", "", false);

  spec.param(parallel_inference_, "parallel_inference", "Parallel inference", "", true);
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             ParameterFlag::kOptional);
}

void InferenceOp::initialize() {
  register_converter<DataMap>();
  register_converter<DataVecMap>();
  holoscan::gxf::GXFExecutor::register_codec<std::vector<ActivationSpec>>(
      "std::vector<holoscan::ops::InferenceOp::InputSpec>", true);
  Operator::initialize();
}

void InferenceOp::start() {
  try {
    // [Parameter rename backward compat] `input_map` / `output_map` are the preferred
    // YAML keys; `pre_processor_map` / `inference_map` are retained as legacy aliases.
    // When both are set the NEW name wins; when only the legacy is set, we fold it
    // into the new-name Parameter so the rest of start() reads a single source of truth.
    if (!input_map_.get().get_map().empty()) {
      if (!pre_processor_map_.get().get_map().empty()) {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: both 'input_map' and legacy 'pre_processor_map' set. "
            "Using 'input_map'; ignoring 'pre_processor_map'.");
      }
      pre_processor_map_ = input_map_.get();
    } else if (!pre_processor_map_.get().get_map().empty()) {
      HOLOSCAN_LOG_INFO(
          "InferenceOp: legacy YAML key 'pre_processor_map' detected. Prefer 'input_map' "
          "in new configs; both are supported.");
    }
    if (!output_map_.get().get_map().empty()) {
      if (!inference_map_.get().get_map().empty()) {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: both 'output_map' and legacy 'inference_map' set. "
            "Using 'output_map'; ignoring 'inference_map'.");
      }
      inference_map_ = output_map_.get();
    } else if (!inference_map_.get().get_map().empty()) {
      HOLOSCAN_LOG_INFO(
          "InferenceOp: legacy YAML key 'inference_map' detected. Prefer 'output_map' "
          "in new configs; both are supported.");
    }

    auto status = HoloInfer::setup_inference_io(pre_processor_map_.get().get_map(),
                                                inference_map_.get().get_map(),
                                                model_inputs_,
                                                model_outputs_,
                                                transmit_outputs_,
                                                out_tensor_names_.get());
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    //  Check for the validity of parameters from configuration
    status = HoloInfer::inference_validity_check(model_path_map_.get().get_map(),
                                                 pre_processor_map_.get().get_map(),
                                                 inference_map_.get().get_map(),
                                                 model_inputs_,
                                                 model_outputs_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Parameter Validation failed: " + status.get_message());
    }

    // Use ExecutionContext::allocate_cuda_stream to allocate CUDA streams for inference
    // backends. This delegates to CudaObjectHandler, which discovers the stream pool by:
    //   1. Looking for a parameter named "cuda_stream_pool"
    //   2. Scanning operator resources for any CudaStreamPool (type-based lookup)
    //   3. Creating a default pool if neither is found
    //
    // The previous approach (cuda_stream_pool_.try_get()) only worked when the pool was
    // passed as a named Arg("cuda_stream_pool", pool). When passed positionally via
    // make_operator (e.g. make_operator<InferenceOp>("name", pool)), the pool is added to
    // the operator's resources_ map but the cuda_stream_pool_ Parameter is not set, so
    // try_get() would return false and no stream callback would be configured.
    //
    // Behavior change: when no CudaStreamPool is explicitly provided, CudaObjectHandler
    // creates a default pool (capacity 1), so inference backends will now receive a
    // Holoscan-managed stream rather than creating their own unmanaged streams.
    //
    // The lambda adapts ExecutionContext::allocate_cuda_stream's (string name) ->
    // expected<cudaStream_t> interface to HoloInfer's (int32_t device_id) -> cudaStream_t
    // callback signature. Unique stream names ensure each backend gets its own dedicated
    // stream, and device_from_stream verifies device compatibility.
    std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream;
    auto exec_ctx = execution_context();
    if (exec_ctx) {
      auto counter = std::make_shared<int>(0);
      allocate_cuda_stream = [exec_ctx = std::move(exec_ctx),
                              counter = std::move(counter)](int32_t device_id) -> cudaStream_t {
        auto stream_name = "inference_" + std::to_string((*counter)++);
        auto maybe_stream = exec_ctx->allocate_cuda_stream(stream_name);
        if (!maybe_stream) {
          throw std::runtime_error(std::string("Failed to allocate CUDA stream: ") +
                                   maybe_stream.error().what());
        }
        auto stream = maybe_stream.value();
        // Verify the allocated stream is on the requested device.
        // device_from_stream only works with Holoscan-managed streams, so skip
        // the check for the default stream (which should not occur in practice).
        if (stream != cudaStreamDefault) {
          auto maybe_dev = exec_ctx->device_from_stream(stream);
          if (!maybe_dev) {
            // Should not happen: stream was just allocated by this ExecutionContext.
            // Log and continue without device verification.
            HOLOSCAN_LOG_ERROR("Failed to query device for allocated CUDA stream '{}': {}",
                               stream_name,
                               maybe_dev.error().what());
          } else if (maybe_dev.value() != device_id) {
            return nullptr;
          }
        }
        return stream;
      };
    }

    // Create inference specification structure
    inference_specs_ =
        std::make_shared<HoloInfer::InferenceSpecs>(backend_.get(),
                                                    backend_map_.get().get_map(),
                                                    model_path_map_.get().get_map(),
                                                    pre_processor_map_.get().get_map(),
                                                    inference_map_.get().get_map(),
                                                    device_map_.get().get_map(),
                                                    dla_core_map_.get().get_map(),
                                                    temporal_map_.get().get_map(),
                                                    activation_map_.get().get_map(),
                                                    trt_opt_profile_.get().get_map(),
                                                    dynamic_input_dims_.get(),
                                                    is_engine_path_.get(),
                                                    infer_on_cpu_.get(),
                                                    parallel_inference_.get(),
                                                    enable_fp16_.get(),
                                                    input_on_cuda_.get(),
                                                    output_on_cuda_.get(),
                                                    enable_cuda_graphs_.get(),
                                                    dla_core_.get(),
                                                    dla_gpu_fallback_.get(),
                                                    false,
                                                    allocate_cuda_stream);
    // ── Auto-detect the fastest safe dispatch path (TRT-only) ──
    // Fast paths are enabled automatically when every configured model uses the
    // "trt" backend AND none of the following blockers is present:
    //   - activation_map / temporal_map
    //   - any device_map entry routing a model off the default data-transfer GPU
    //   - input_on_cuda=false / output_on_cuda=false / transmit_on_cuda=false
    //     (do_inference_on_stream cannot host<->device stage buffers)
    //   - chained models where one model consumes another's output
    //     (the fast paths cannot enforce producer-before-consumer ordering)
    // All other configurations run the canonical dispatch path. There is no
    // user-facing opt-in — the operator never runs slower than it needs to for
    // a given config.
    {
      const auto& path_map = model_path_map_.get().get_map();
      const auto& act_map = activation_map_.get().get_map();
      const auto& tmp_map = temporal_map_.get().get_map();
      const auto& dev_map = device_map_.get().get_map();
      const auto& pre_map = pre_processor_map_.get().get_map();
      const auto& inf_map = inference_map_.get().get_map();
      const bool has_activation = !act_map.empty();
      const bool has_temporal = !tmp_map.empty();

      // Multi-GPU blocker: any explicit device_map entry that routes a model to
      // a device other than the operator's default data-transfer GPU (0). See
      // fast_path_eligibility.hpp for the exact rule.
      const bool has_off_gpu_dt_assignment =
          inference_eligibility::has_off_gpu_dt_assignment(dev_map);

      const bool has_dyn_dims = dynamic_input_dims_.get();
      const bool has_parallel = parallel_inference_.get();

      // Data-buffer blockers: fast paths (do_inference_on_stream) do not
      // perform host<->device staging on input/output buffers. Any of the
      // *_on_cuda flags being false demands the canonical path.
      const bool needs_host_input = !input_on_cuda_.get();
      const bool needs_host_output = !output_on_cuda_.get();
      const bool needs_host_transmit = !transmit_on_cuda_.get();

      // Model-dependency blocker: chained models cannot use the multi-model
      // fast paths — see fast_path_eligibility.hpp for the exact rule.
      const bool has_chained_models_flag =
          inference_eligibility::has_chained_models(pre_map, inf_map);

      const bool static_blockers = has_activation || has_temporal || has_off_gpu_dt_assignment ||
                                   needs_host_input || needs_host_output || needs_host_transmit ||
                                   has_chained_models_flag;

      // TRT-only gate. If backend_map_ is populated, every entry must be "trt";
      // otherwise fall back to the scalar backend_ string.
      const std::string bk = backend_.get();
      const auto& bmap = backend_map_.get().get_map();
      bool all_trt = false;
      if (!bmap.empty()) {
        all_trt = true;
        for (const auto& [_, b] : bmap) {
          if (b != "trt") {
            all_trt = false;
            break;
          }
        }
      } else {
        all_trt = (bk == "trt");
      }

      const size_t n_models = path_map.size();
      const bool eligible = all_trt && !static_blockers && n_models >= 1;

      if (!eligible) {
        dispatch_kind_ = DispatchKind::kStandard;
      } else if (n_models == 1) {
        fast_single_model_name_ = path_map.begin()->first;
        dispatch_kind_ = has_dyn_dims ? DispatchKind::kDynFast : DispatchKind::kFast;
        HOLOSCAN_LOG_INFO("InferenceOp: fast path auto-selected ({}) for single TRT model '{}'.",
                          has_dyn_dims ? "kDynFast" : "kFast",
                          fast_single_model_name_);
      } else if (has_parallel) {
        dispatch_kind_ = has_dyn_dims ? DispatchKind::kDynParFast : DispatchKind::kParFast;
        HOLOSCAN_LOG_INFO("InferenceOp: fast path auto-selected ({}) for {} TRT models (parallel).",
                          has_dyn_dims ? "kDynParFast" : "kParFast",
                          n_models);
      } else {
        dispatch_kind_ = has_dyn_dims ? DispatchKind::kDynSeqFast : DispatchKind::kSeqFast;
        HOLOSCAN_LOG_INFO(
            "InferenceOp: fast path auto-selected ({}) for {} TRT models (sequential).",
            has_dyn_dims ? "kDynSeqFast" : "kSeqFast",
            n_models);
      }

      // Enable the manager-side multi-model fast-path caches whenever a
      // multi-model fast path was chosen. The internal spec flag is retained;
      // only the user-facing parameter has been removed.
      const bool multi_model_fast = dispatch_kind_ == DispatchKind::kSeqFast ||
                                    dispatch_kind_ == DispatchKind::kParFast ||
                                    dispatch_kind_ == DispatchKind::kDynSeqFast ||
                                    dispatch_kind_ == DispatchKind::kDynParFast;
      inference_specs_->fast_multi_model_ = multi_model_fast;
    }
    HOLOSCAN_LOG_INFO("Inference Specifications created");

    // If a CudaGreenContext resource is present, thread its CUcontext and SM count through
    // to HoloInfer so TRT engine building is constrained to the partition's SMs.
    std::shared_ptr<holoscan::CudaGreenContext> selected_gc;
    for (auto& [_, resource] : resources()) {
      auto gc_resource = std::dynamic_pointer_cast<holoscan::CudaGreenContext>(resource);
      if (!gc_resource || !gc_resource->get()) {
        continue;
      }
      if (selected_gc) {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: multiple CudaGreenContext resources attached; "
            "using '{}', ignoring '{}'",
            selected_gc->name(),
            gc_resource->name());
        continue;
      }
      selected_gc = gc_resource;
    }
    if (selected_gc) {
      auto* gxf_gc = selected_gc->get();  // nvidia::gxf::CudaGreenContext*
      auto maybe_ctx = gxf_gc->cudaContext();
      if (maybe_ctx) {
        inference_specs_->build_cuda_context_ = maybe_ctx.value();
      } else {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: could not obtain CUcontext from CudaGreenContext '{}'; "
            "TRT engine will be built with all GPU SMs.",
            selected_gc->name());
      }
      auto* pool = gxf_gc->cudaGreenContextPool();
      if (pool) {
        auto maybe_sms = pool->getPartitionSms(gxf_gc->index());
        if (maybe_sms) {
          inference_specs_->build_sm_count_ = static_cast<int32_t>(maybe_sms.value());
          HOLOSCAN_LOG_INFO(
              "InferenceOp: TRT engine will be built within green context partition {} ({} SMs)",
              gxf_gc->index(),
              inference_specs_->build_sm_count_);
        } else {
          HOLOSCAN_LOG_WARN(
              "InferenceOp: could not obtain SM count from CudaGreenContext '{}'; "
              "TRT engine will be built with all GPU SMs.",
              selected_gc->name());
        }
      } else {
        HOLOSCAN_LOG_WARN(
            "InferenceOp: could not obtain CudaGreenContextPool from CudaGreenContext '{}'; "
            "TRT engine will be built with all GPU SMs.",
            selected_gc->name());
      }
    }

    // Create holoscan inference context
    holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();

    // Set and transfer inference specification to inference context
    // inference specifications are updated with memory allocations
    status = holoscan_infer_context_->set_inference_params(inference_specs_);
    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Start, Parameters setup, " + status.get_message());
    }
    HOLOSCAN_LOG_INFO("Inference context setup complete");

    cached_input_on_cuda_ = input_on_cuda_.get();
    cached_output_on_cuda_ = output_on_cuda_.get();
    cached_transmit_on_cuda_ = transmit_on_cuda_.get();
    cached_dynamic_input_dims_ = dynamic_input_dims_.get();
  } catch (const std::bad_alloc& b_) {
    HoloInfer::raise_error(module_, "Start, Memory allocation, Message: " + std::string(b_.what()));
  } catch (const std::runtime_error& rt_) {
    HOLOSCAN_LOG_ERROR(rt_.what());
    throw;
  } catch (...) {
    HoloInfer::raise_error(module_, "Start, Unknown exception");
  }
}

void InferenceOp::stop() {
  // Reset the transmit cache to release the persistent GXF entity reference while the context
  // is still valid.
  transmit_cache_ = {};
  inference_specs_.reset();
  holoscan_infer_context_.reset();
}

void InferenceOp::compute(InputContext& op_input, OutputContext& op_output,
                          ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  auto cont = context.context();
  try {
    // Get activation maps spec from user, if any, build filters on inference specs
    const auto act_map_specs_message =
        op_input.receive<std::vector<ActivationSpec>>("model_activation_specs");
    std::vector<ActivationSpec> maybe_act_map_specs;

    // Update activation_map by specs
    if (act_map_specs_message) {
      maybe_act_map_specs = act_map_specs_message.value();
    }

    // Extract relevant data from input GXF Receivers, and update inference specifications
    // (cuda_stream will be set by get_data_per_model).
    //
    // Route through the cached variant when the input shape is static. The first
    // call falls through to the canonical extractor and populates `extract_cache_`; on every
    // subsequent compute() call the cached helper skips the dtype/storage validation, the dims
    // rebuild, and the per-tensor data_per_tensor_ map lookup. When dynamic_input_dims_ is true we
    // call the original extractor unchanged because dims can change between compute() calls.
    cudaStream_t cuda_stream{};
    gxf_result_t stat;

    if (cached_dynamic_input_dims_) {
      stat = holoscan::utils::get_data_per_model(op_input,
                                                 model_inputs_,
                                                 inference_specs_->data_per_tensor_,
                                                 inference_specs_->dims_per_tensor_,
                                                 cached_input_on_cuda_,
                                                 module_,
                                                 cuda_stream);
    } else {
      stat = holoscan::utils::get_data_per_model_cached(op_input,
                                                        model_inputs_,
                                                        inference_specs_->data_per_tensor_,
                                                        inference_specs_->dims_per_tensor_,
                                                        cached_input_on_cuda_,
                                                        module_,
                                                        cuda_stream,
                                                        extract_cache_);
    }

    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Compute, Data extraction");
    }

    // Transmit this stream on the output port if needed
    if (cuda_stream != cudaStreamDefault && cached_output_on_cuda_ &&
        cuda_stream != last_set_transmit_stream_) {
      HOLOSCAN_LOG_TRACE("InferenceOp: forwarding CUDA stream from receivers input to output");
      op_output.set_cuda_stream(cuda_stream, "transmitter");
      last_set_transmit_stream_ = cuda_stream;
    }

    // check for tensor validity the first time
    if (validate_tensor_dimensions_ && !cached_dynamic_input_dims_) {
      validate_tensor_dimensions_ = false;
      auto model_in_dims_map = holoscan_infer_context_->get_input_dimensions();

      auto dim_status = HoloInfer::tensor_dimension_check(pre_processor_map_.get().get_map(),
                                                          model_in_dims_map,
                                                          inference_specs_->dims_per_tensor_,
                                                          model_inputs_);
      if (dim_status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
        HoloInfer::raise_error(module_,
                               "Compute, Inference execution, " + dim_status.get_message());
      }
    }
    //  Execute inference and populate output buffer in inference specifications
    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    // Set activation map for inference specifications
    stat = holoscan::utils::set_activation_per_model(
        inference_specs_, activation_map_.get().get_map(), maybe_act_map_specs, module_);
    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Compute, Inference activation specification");
    }
    // Dispatch to the streamlined entry point auto-selected in start().
    HoloInfer::InferStatus status;
    switch (dispatch_kind_) {
      case DispatchKind::kFast:
        status = holoscan_infer_context_->execute_inference_fast(
            inference_specs_, cuda_stream, fast_single_model_name_);
        break;
      case DispatchKind::kSeqFast:
        status = holoscan_infer_context_->execute_inference_seq_fast(inference_specs_, cuda_stream);
        break;
      case DispatchKind::kParFast:
        status = holoscan_infer_context_->execute_inference_par_fast(inference_specs_, cuda_stream);
        break;
      case DispatchKind::kDynFast:
        status = holoscan_infer_context_->execute_inference_dyn_fast(
            inference_specs_, cuda_stream, fast_single_model_name_);
        break;
      case DispatchKind::kDynSeqFast:
        status =
            holoscan_infer_context_->execute_inference_dyn_seq_fast(inference_specs_, cuda_stream);
        break;
      case DispatchKind::kDynParFast:
        status =
            holoscan_infer_context_->execute_inference_dyn_par_fast(inference_specs_, cuda_stream);
        break;
      case DispatchKind::kStandard:
      default:
        status = holoscan_infer_context_->execute_inference(inference_specs_, cuda_stream);
        break;
    }
    HoloInfer::timer_init(e_time);
    HoloInfer::timer_check(s_time, e_time, "Inference Operator: Inference execution");

    if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
      status.display_message();
      HoloInfer::raise_error(module_, "Compute, Inference execution, " + status.get_message());
    }
    if (holoscan::log_level() <= holoscan::LogLevel::DEBUG) {
      HOLOSCAN_LOG_DEBUG(status.get_message());
    }

    if (cached_dynamic_input_dims_ || !cached_output_dims_valid_) {
      cached_output_dims_ = holoscan_infer_context_->get_output_dimensions();
      cached_output_dims_valid_ = true;
    }
    auto model_out_dims_map = cached_output_dims_;

    // Transmit output buffers via a single GXF transmitter.
    stat = holoscan::utils::transmit_data_per_model(cont,
                                                    inference_map_.get().get_map(),
                                                    inference_specs_->output_per_model_,
                                                    op_output,
                                                    transmit_outputs_,
                                                    model_out_dims_map,
                                                    cached_output_on_cuda_,
                                                    cached_transmit_on_cuda_,
                                                    allocator.value(),
                                                    module_,
                                                    cuda_stream,
                                                    transmit_cache_);

    if (stat != GXF_SUCCESS) {
      HoloInfer::raise_error(module_, "Compute, Data Transmission");
    }
  } catch (const std::runtime_error& r_) {
    HoloInfer::raise_error(module_,
                           "Compute, Inference execution, Message->" + std::string(r_.what()));
  } catch (...) {
    HoloInfer::raise_error(module_, "Compute, unknown exception");
  }
}

}  // namespace holoscan::ops
