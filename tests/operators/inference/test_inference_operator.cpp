/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer_utils.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/fast_path_eligibility.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/utils/cuda_macros.hpp>

// Manager-internal helpers (from modules/holoinfer/src/manager) — surfaced via
// target_include_directories on this test target.
#include "tensor_lookup.hpp"

#include "../../utils/holoinfer_backend_test_utils.hpp"

// Test tensor dimensions BATCH_SIZE x TENSOR_SIZE x TENSOR_SIZE
constexpr int TENSOR_SIZE = 256;
constexpr int BATCH_SIZE = 1;

namespace holoscan::ops {

class TensorGeneratorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorGeneratorOp)

  TensorGeneratorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Allocator");
    spec.param(fill_value_, "fill_value", "Fill value", "Fill value", 0.f);
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto entity = holoscan::gxf::Entity::New(&context);

    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    auto tensor =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("tensor").value();
    nvidia::gxf::Shape shape({BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE});
    tensor->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    std::vector<float> tensor_data;
    size_t tensor_size = BATCH_SIZE * TENSOR_SIZE * TENSOR_SIZE;
    tensor_data.resize(tensor_size);
    for (int i = 0; i < tensor_size; i++) {
      tensor_data[i] = fill_value_ + static_cast<float>(i);
    }

    HOLOSCAN_CUDA_CALL(cudaMemcpy(tensor->pointer(),
                                  tensor_data.data(),
                                  tensor_size * sizeof(float),
                                  cudaMemcpyHostToDevice));

    op_output.emit(entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<float> fill_value_;
};

class ResultCheckerOp : public holoscan::Operator {
 public:
  explicit ResultCheckerOp(bool enable_green_context)
      : enable_green_context_(enable_green_context) {}

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");

    if (enable_green_context_) {
      int version = 0;
      CUresult result = cuDriverGetVersion(&version);
      if (result == CUDA_SUCCESS) {
        result = cuGetProcAddress("cuStreamGetGreenCtx",
                                  reinterpret_cast<void**>(&fnCuStreamGetGreenCtx),
                                  version,
                                  0,
                                  nullptr);
        if (result != CUDA_SUCCESS) {
          const char* error_string;
          cuGetErrorString(result, &error_string);
          HOLOSCAN_LOG_WARN(
              "Failed (error code: {}) getting cuStreamGetGreenCtx() from CUDA driver {}",
              error_string,
              version);
        }
      }
    }
  }

  static void read_data(holoscan::gxf::Entity& entity, const std::string& name,
                        std::vector<float>& out) {
    size_t tensor_size = BATCH_SIZE * TENSOR_SIZE * TENSOR_SIZE;
    out.resize(tensor_size, 0.F);
    auto tensor = entity.get<holoscan::Tensor>(name.c_str());
    if (!tensor) {
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", name));
    }
    HOLOSCAN_CUDA_CALL(cudaMemcpy(
        out.data(), tensor->data(), tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_message = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_message) {
      std::string err_msg =
          fmt::format("No input message received by inference operator on port 'receivers': {}",
                      maybe_message.error().what());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    auto streams = op_input.receive_cuda_streams("input");
    if (streams.empty()) {
      throw std::runtime_error("No CUDA stream found in message");
    }
    cudaStream_t stream = nullptr;
    if (streams.size() > 0 && streams[0].has_value()) {
      stream = streams[0].value();
      cudaStreamSynchronize(stream);
      // Check if the cuda stream is associated with a green context
      if (fnCuStreamGetGreenCtx != nullptr) {
        CUgreenCtx green_context;
        fnCuStreamGetGreenCtx(stream, &green_context);
        if (green_context == nullptr) {
          throw std::runtime_error("Stream is expected to be associated with a green context");
        }
      }
    }

    // Data validation
    std::vector<float> data;
    size_t tensor_size = BATCH_SIZE * TENSOR_SIZE * TENSOR_SIZE;
    read_data(maybe_message.value(), "tensor", data);
    for (int i = 0; i < tensor_size; i++) {
      ASSERT_EQ(data[i], static_cast<float>(i));
    }

    HOLOSCAN_LOG_INFO("Inference result verified");
  }

 private:
  bool enable_green_context_;
  CUresult (*fnCuStreamGetGreenCtx)(CUstream, CUgreenCtx*) = nullptr;
};

}  // namespace holoscan::ops

// Test app for inference operator using flow from tensor generator to inference to result checker
// to verify the inference(identity model) result.
// The identity model is a simple model that takes a tensor as input and returns the same tensor.
// The test tensor has a dimension of BATCH_SIZE x TENSOR_SIZE x TENSOR_SIZE.
// Supports testing both one and two inference ops through the test_two parameter.
// Testing of two inference ops uses the EventBasedScheduler to verify running two pipelines
// parallelly.
class InferenceOpTestApp : public holoscan::Application {
 public:
  explicit InferenceOpTestApp(std::string backend, std::string model_path,
                              bool enable_green_context, bool test_two = false)
      : backend_(backend),
        model_path_(model_path),
        enable_green_context_(enable_green_context),
        test_two_(test_two) {}

  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("pool");

    // Create CUDA stream pools
    std::shared_ptr<CudaGreenContextPool> cuda_green_context_pool = nullptr;
    std::shared_ptr<CudaGreenContext> cuda_green_context1 = nullptr;
    std::shared_ptr<CudaGreenContext> cuda_green_context2 = nullptr;

    if (enable_green_context_) {
      std::vector<uint32_t> partitions = {4, 4};
      cuda_green_context_pool = make_resource<CudaGreenContextPool>(
          "cuda_green_context_pool", 0, 0, partitions.size(), partitions);
      cuda_green_context1 =
          make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 0);
      if (test_two_) {
        cuda_green_context2 =
            make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 1);
      }
    }

    auto cuda_stream_pool1 = make_resource<CudaStreamPool>(
        "cuda_stream_pool1", 0, cudaStreamNonBlocking, 0, 1, 5, cuda_green_context1);
    auto cuda_stream_pool2 =
        test_two_ ? make_resource<CudaStreamPool>(
                        "cuda_stream_pool2", 0, cudaStreamNonBlocking, 0, 1, 5, cuda_green_context2)
                  : nullptr;

    auto tensor_generator_op = make_operator<ops::TensorGeneratorOp>(
        "tensor_generator", Arg("allocator") = allocator, make_condition<CountCondition>(10));

    ops::InferenceOp::DataMap model_path_map1;
    ops::InferenceOp::DataMap model_path_map2;
    std::string model_path = "tests/operators/inference/models/" + model_path_;
    model_path_map1.insert("first", model_path);
    if (test_two_) {
      model_path_map2.insert("second", model_path);
    }
    std::vector<int> in_tensor_dimensions = {BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE};

    // First inference operator
    auto infer_op1 =
        make_operator<ops::InferenceOp>("infer1",
                                        from_config("inference"),
                                        Arg("backend") = backend_,
                                        Arg("model_path_map") = std::move(model_path_map1),
                                        Arg("allocator") = allocator,
                                        Arg("in_tensor_dimensions") = in_tensor_dimensions,
                                        cuda_stream_pool1);

    // Second inference operator (only if testing two inference ops in parallel)
    std::shared_ptr<ops::InferenceOp> infer_op2 = nullptr;
    if (test_two_) {
      ops::InferenceOp::DataVecMap pre_processor_map2;
      ops::InferenceOp::DataVecMap inference_map2;
      pre_processor_map2.insert("second", {"tensor"});
      inference_map2.insert("second", {"tensor"});

      infer_op2 = make_operator<ops::InferenceOp>(
          "infer2",
          Arg("backend") = backend_,
          Arg("model_path_map") = std::move(model_path_map2),
          Arg("allocator") = allocator,
          Arg("in_tensor_names") = std::vector<std::string>{"tensor"},
          Arg("out_tensor_names") = std::vector<std::string>{"tensor"},
          Arg("parallel_inference") = true,
          Arg("infer_on_cpu") = false,
          Arg("enable_fp16") = false,
          Arg("enable_cuda_graphs") = true,
          Arg("input_on_cuda") = true,
          Arg("output_on_cuda") = true,
          Arg("transmit_on_cuda") = true,
          Arg("in_tensor_dimensions") = in_tensor_dimensions,
          Arg("pre_processor_map") = std::move(pre_processor_map2),
          Arg("inference_map") = std::move(inference_map2),
          cuda_stream_pool2);
    }

    auto result_checker_op1 =
        make_operator<ops::ResultCheckerOp>("checker1", enable_green_context_);
    auto result_checker_op2 =
        test_two_ ? make_operator<ops::ResultCheckerOp>("checker2", enable_green_context_)
                  : nullptr;

    // Add flows
    add_flow(tensor_generator_op, infer_op1, {{"output", "receivers"}});
    add_flow(infer_op1, result_checker_op1, {{"transmitter", "input"}});

    if (test_two_) {
      add_flow(tensor_generator_op, infer_op2, {{"output", "receivers"}});
      add_flow(infer_op2, result_checker_op2, {{"transmitter", "input"}});
    }
  }

 private:
  std::string backend_;
  std::string model_path_;
  bool enable_green_context_;
  bool test_two_;
};

class InferenceOpTestFixture
    : public ::testing::TestWithParam<std::tuple<std::string, std::string, bool, bool>> {};

TEST_P(InferenceOpTestFixture, InferenceOpTestApp) {
  using namespace holoscan;

  auto& [backend, model, enable_green_context, test_two] = GetParam();

  if (backend == "onnxrt") {
    HOLOSCAN_TEST_SKIP_IF_ONNX_RUNTIME_BACKEND_DISABLED();
  }

  // Skip torch tests if torch CUDA is unavailable or SM-incompatible
  if (backend == "torch") {
#if defined(HOLOINFER_TORCH_ENABLED)
    if (!inference::is_torch_cuda_available()) {
      GTEST_SKIP() << "Torch CUDA unavailable";
    }
    if (!inference::is_torch_cuda_sm_compatible()) {
      GTEST_SKIP() << "Torch CUDA SM incompatible";
    }
#else
    GTEST_SKIP() << "Torch backend not enabled";
#endif
  }

  HOLOSCAN_LOG_INFO("backend = {}", backend);
  HOLOSCAN_LOG_INFO("model = {}", model);
  HOLOSCAN_LOG_INFO("enable_green_context = {}", enable_green_context);
  HOLOSCAN_LOG_INFO("test_two = {}", test_two);

  std::filesystem::path config_path = "tests/operators/inference/inference.yaml";

  auto app = make_application<InferenceOpTestApp>(backend, model, enable_green_context, test_two);
  app->config(config_path);

  // Use EventBasedScheduler if testing two inference ops
  if (test_two) {
    auto scheduler = app->make_scheduler<holoscan::EventBasedScheduler>(
        "event_based_scheduler",
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(2)),
        holoscan::Arg("stop_on_deadlock", true),
        holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(500)),
        holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000)));
    app->scheduler(scheduler);
  }

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(log_output.find("Inference result verified") != std::string::npos);
}

// ============================================================================
// Multi-model fast-track test
//
// Validates that InferenceOp auto-selects a multi-model fast path for a
// multi-model TRT pipeline and runs it end-to-end. Catches:
//   - dispatcher auto-selection (kSeqFast or kParFast) — log line is asserted
//   - manager-side cache population (all_params_ / all_ctx_ / all_indata_ / all_outdata_)
//   - prewarm correctness (par-fast path only)
//   - end-to-end pipeline completion (counted via a simple sink that just receives entities)
// ============================================================================
class FastTrackCountingSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FastTrackCountingSinkOp)
  FastTrackCountingSinkOp() = default;
  void setup(holoscan::OperatorSpec& spec) override { spec.input<std::any>("in"); }
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto maybe = op_input.receive<std::any>("in");
    if (maybe) {
      ++count_;
    }
  }
  int count() const { return count_; }

 private:
  int count_ = 0;
};

class InferenceOpFastTrackApp : public holoscan::Application {
 public:
  explicit InferenceOpFastTrackApp(bool parallel) : parallel_(parallel) {}
  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("pool");
    // Pool sized to cover par-fast's N=2 distinct streams + ambient operator overhead.
    auto stream_pool =
        make_resource<CudaStreamPool>("stream_pool", 0, cudaStreamNonBlocking, 0, 1, 16);

    auto gen = make_operator<ops::TensorGeneratorOp>(
        "tensor_generator", Arg("allocator") = allocator, make_condition<CountCondition>(5));

    // Two-model InferenceOp (same identity model file, two distinct model_name keys).
    // The source emits a single tensor named "tensor" that both models consume.
    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map, inference_map;
    const std::string model_path = "tests/operators/inference/models/identity_model.onnx";
    model_path_map.insert("model_a", model_path);
    model_path_map.insert("model_b", model_path);
    pre_processor_map.insert("model_a", {"tensor"});
    pre_processor_map.insert("model_b", {"tensor"});
    inference_map.insert("model_a", {"tensor_a"});
    inference_map.insert("model_b", {"tensor_b"});

    std::vector<int> in_tensor_dimensions = {BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE};
    auto infer = make_operator<holoscan::ops::InferenceOp>(
        "infer_fast_multi_model",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map") = model_path_map,
        Arg("pre_processor_map") = pre_processor_map,
        Arg("inference_map") = inference_map,
        Arg("in_tensor_names") = std::vector<std::string>{"tensor"},
        Arg("out_tensor_names") = std::vector<std::string>{"tensor_a", "tensor_b"},
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("infer_on_cpu") = false,
        Arg("enable_fp16") = false,
        Arg("enable_cuda_graphs") = true,
        Arg("parallel_inference") = parallel_,
        // Fast path is auto-selected for TRT models; no opt-in parameter.
        Arg("in_tensor_dimensions") = in_tensor_dimensions,
        Arg("allocator") = allocator,
        Arg("cuda_stream_pool") = stream_pool);

    sink_ = make_operator<FastTrackCountingSinkOp>("sink");
    add_flow(gen, infer, {{"output", "receivers"}});
    add_flow(infer, sink_, {{"transmitter", "in"}});
  }
  int sink_count() const { return sink_ ? sink_->count() : 0; }

 private:
  bool parallel_;
  std::shared_ptr<FastTrackCountingSinkOp> sink_;
};

TEST(InferenceOpFastTrack, SequentialDispatch) {
  using namespace holoscan;
  testing::internal::CaptureStderr();
  auto app = make_application<InferenceOpFastTrackApp>(false);  // sequential
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  // Dispatcher auto-selected multi-model seq-fast (kSeqFast)
  EXPECT_NE(log.find("fast path auto-selected (kSeqFast)"), std::string::npos)
      << "kSeqFast dispatch not auto-selected. Log tail:\n"
      << log;
  // Pipeline executed end-to-end at least once
  EXPECT_GT(app->sink_count(), 0)
      << "InferenceOp multi-model fast path produced no output entities. Log tail:\n"
      << log;
}

TEST(InferenceOpFastTrack, ParallelDispatch) {
  using namespace holoscan;
  testing::internal::CaptureStderr();
  auto app = make_application<InferenceOpFastTrackApp>(true);  // parallel
  app->run();
  std::string log = testing::internal::GetCapturedStderr();
  // Either kParFast was auto-selected, or it was safely demoted to seq-fast at
  // the manager layer. Both are acceptable correctness outcomes; we just need
  // the pipeline to complete.
  const bool par_fast_on = log.find("fast path auto-selected (kParFast)") != std::string::npos;
  const bool demoted_par = log.find("par-fast demoted") != std::string::npos;
  const bool seq_fast_on = log.find("fast path auto-selected (kSeqFast)") != std::string::npos;
  EXPECT_TRUE(par_fast_on || demoted_par || seq_fast_on)
      << "Expected par-fast / seq-fast auto-selection or par-fast demotion log. Log tail:\n"
      << log;
  EXPECT_GT(app->sink_count(), 0)
      << "InferenceOp multi-model parallel fast path produced no output entities. Log tail:\n"
      << log;
}

INSTANTIATE_TEST_CASE_P(InferenceOpTestApp, InferenceOpTestFixture,
                        ::testing::Values(
                            // Single inference op
                            std::make_tuple("onnxrt", "identity_model.onnx", false, false),
                            std::make_tuple("onnxrt", "identity_model.onnx", true, false),
                            std::make_tuple("trt", "identity_model.onnx", false, false),
                            std::make_tuple("trt", "identity_model.onnx", true, false),
                            std::make_tuple("torch", "identity_model.pt", false, false),
                            std::make_tuple("torch", "identity_model.pt", true, false),
                            // Two inference ops
                            std::make_tuple("onnxrt", "identity_model.onnx", false, true),
                            std::make_tuple("onnxrt", "identity_model.onnx", true, true),
                            std::make_tuple("trt", "identity_model.onnx", false, true),
                            std::make_tuple("trt", "identity_model.onnx", true, true),
                            std::make_tuple("torch", "identity_model.pt", false, true),
                            std::make_tuple("torch", "identity_model.pt", true, true)));

// ============================================================================
// Correctness regressions for the auto-selected fast paths.
//
// The identity model returns its input unchanged, so the output on any frame
// must equal the input written for that same frame. To detect stale output
// (or a missing H2D / D2H copy), each frame emits a UNIQUE fill pattern —
// `fill_value + frame_index * FRAME_STRIDE + i`. Any implementation that skips
// a per-frame copy will surface as a numerical mismatch on frame >= 2.
//
// ============================================================================

namespace holoscan::ops {

// Emits a distinct fill pattern per compute() call so stale-output bugs
// surface as a numerical mismatch on frame >= 2. Storage type (device or
// host) is selectable so the same generator drives input_on_cuda={true,false}
// pipelines.
class FrameCountingTensorGeneratorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameCountingTensorGeneratorOp)
  FrameCountingTensorGeneratorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Allocator");
    spec.param(on_cuda_, "on_cuda", "Emit device tensor", "true → kDevice, false → kHost", true);
    spec.param(tensor_name_, "tensor_name", "Tensor name", "", std::string("tensor"));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto entity = holoscan::gxf::Entity::New(&context);
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity)
                      .add<nvidia::gxf::Tensor>(tensor_name_.get().c_str())
                      .value();
    nvidia::gxf::Shape shape({BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE});
    const auto storage = on_cuda_.get() ? nvidia::gxf::MemoryStorageType::kDevice
                                        : nvidia::gxf::MemoryStorageType::kHost;
    tensor->reshape<float>(shape, storage, allocator.value());

    const size_t n = BATCH_SIZE * TENSOR_SIZE * TENSOR_SIZE;
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
      data[i] = static_cast<float>(frame_index_) * kFrameStride + static_cast<float>(i);
    }
    if (on_cuda_.get()) {
      HOLOSCAN_CUDA_CALL(
          cudaMemcpy(tensor->pointer(), data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      std::memcpy(tensor->pointer(), data.data(), n * sizeof(float));
    }
    ++frame_index_;
    op_output.emit(entity, "output");
  }

  static constexpr float kFrameStride = 1e6F;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<bool> on_cuda_;
  Parameter<std::string> tensor_name_;
  int frame_index_ = 0;
};

// Reads named tensor(s) from each incoming entity and verifies element `i`
// equals `frame_index * kFrameStride + i` (identity-model output).
class NumericalValidationSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NumericalValidationSinkOp)
  NumericalValidationSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    spec.param(
        tensor_names_, "tensor_names", "Tensors to validate", "", std::vector<std::string>{});
    spec.param(input_on_cuda_, "input_on_cuda", "Input on CUDA", "", true);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto maybe_message = op_input.receive<holoscan::gxf::Entity>("in");
    if (!maybe_message) {
      last_error_ = "no message received";
      failures_ += 1;
      return;
    }
    auto entity = maybe_message.value();
    const size_t n = BATCH_SIZE * TENSOR_SIZE * TENSOR_SIZE;
    std::vector<float> data(n, 0.F);

    // Wait for producer stream so device memory is safe to read here.
    auto streams = op_input.receive_cuda_streams("in");
    if (!streams.empty() && streams[0].has_value()) {
      cudaStreamSynchronize(streams[0].value());
    }

    for (const auto& name : tensor_names_.get()) {
      auto tensor = entity.get<holoscan::Tensor>(name.c_str());
      if (!tensor) {
        last_error_ = "tensor missing: " + name;
        failures_ += 1;
        return;
      }
      if (input_on_cuda_.get()) {
        HOLOSCAN_CUDA_CALL(
            cudaMemcpy(data.data(), tensor->data(), n * sizeof(float), cudaMemcpyDeviceToHost));
      } else {
        std::memcpy(data.data(), tensor->data(), n * sizeof(float));
      }
      const float expected0 =
          static_cast<float>(frame_index_) * FrameCountingTensorGeneratorOp::kFrameStride;
      // Spot-check first and last elements + a middle sample. Full-buffer check
      // would explode test wall-time on hosts without SIMD memcmp.
      constexpr float kTol = 1e-3F;
      const float e0 = expected0;
      const float eN = expected0 + static_cast<float>(n - 1);
      const float eMid = expected0 + static_cast<float>(n / 2);
      if (std::abs(data[0] - e0) > kTol || std::abs(data[n - 1] - eN) > kTol ||
          std::abs(data[n / 2] - eMid) > kTol) {
        last_error_ = fmt::format(
            "numerical mismatch on tensor '{}' frame={} data[0]={} expected0={} "
            "data[n-1]={} expectedN={}",
            name,
            frame_index_,
            data[0],
            e0,
            data[n - 1],
            eN);
        failures_ += 1;
        return;
      }
    }
    ++frame_index_;
    ++frames_ok_;
  }

  int frames_ok() const { return frames_ok_; }
  int failures() const { return failures_; }
  const std::string& last_error() const { return last_error_; }

 private:
  Parameter<std::vector<std::string>> tensor_names_;
  Parameter<bool> input_on_cuda_;
  int frame_index_ = 0;
  int frames_ok_ = 0;
  int failures_ = 0;
  std::string last_error_;
};

}  // namespace holoscan::ops

// Parametrized correctness app driving one or two inference models with
// configurable CUDA/host buffer flags and dispatch mode.
struct FastPathConfig {
  int n_models;   // 1 or 2
  bool parallel;  // parallel_inference
  bool input_on_cuda;
  bool output_on_cuda;
  bool transmit_on_cuda;
  std::string label;  // gtest name
};

class InferenceOpCorrectnessApp : public holoscan::Application {
 public:
  explicit InferenceOpCorrectnessApp(FastPathConfig c) : c_(std::move(c)) {}

  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("pool");
    auto stream_pool =
        make_resource<CudaStreamPool>("stream_pool", 0, cudaStreamNonBlocking, 0, 1, 32);

    auto gen = make_operator<holoscan::ops::FrameCountingTensorGeneratorOp>(
        "generator",
        Arg("allocator") = allocator,
        Arg("on_cuda") = c_.input_on_cuda,
        Arg("tensor_name") = std::string("tensor"),
        make_condition<CountCondition>(kFrames));

    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map, inference_map;
    const std::string model_path = "tests/operators/inference/models/identity_model.onnx";
    std::vector<std::string> out_names;
    if (c_.n_models == 1) {
      model_path_map.insert("model_a", model_path);
      pre_processor_map.insert("model_a", {"tensor"});
      inference_map.insert("model_a", {"tensor_a"});
      out_names = {"tensor_a"};
    } else {
      model_path_map.insert("model_a", model_path);
      model_path_map.insert("model_b", model_path);
      pre_processor_map.insert("model_a", {"tensor"});
      pre_processor_map.insert("model_b", {"tensor"});
      inference_map.insert("model_a", {"tensor_a"});
      inference_map.insert("model_b", {"tensor_b"});
      out_names = {"tensor_a", "tensor_b"};
    }

    std::vector<int> dims = {BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE};
    auto infer = make_operator<holoscan::ops::InferenceOp>(
        "infer",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map") = model_path_map,
        Arg("pre_processor_map") = pre_processor_map,
        Arg("inference_map") = inference_map,
        Arg("in_tensor_names") = std::vector<std::string>{"tensor"},
        Arg("out_tensor_names") = out_names,
        Arg("input_on_cuda") = c_.input_on_cuda,
        Arg("output_on_cuda") = c_.output_on_cuda,
        Arg("transmit_on_cuda") = c_.transmit_on_cuda,
        Arg("infer_on_cpu") = false,
        Arg("enable_fp16") = false,
        Arg("enable_cuda_graphs") = true,
        Arg("parallel_inference") = c_.parallel,
        Arg("in_tensor_dimensions") = dims,
        Arg("allocator") = allocator,
        Arg("cuda_stream_pool") = stream_pool);

    sink_ = make_operator<holoscan::ops::NumericalValidationSinkOp>(
        "sink", Arg("tensor_names") = out_names, Arg("input_on_cuda") = c_.transmit_on_cuda);
    add_flow(gen, infer, {{"output", "receivers"}});
    add_flow(infer, sink_, {{"transmitter", "in"}});
  }

  int frames_ok() const { return sink_ ? sink_->frames_ok() : 0; }
  int failures() const { return sink_ ? sink_->failures() : 0; }
  std::string last_error() const { return sink_ ? sink_->last_error() : ""; }

  static constexpr int kFrames = 4;  // >= 2 so stale-output bugs are catchable

 private:
  FastPathConfig c_;
  std::shared_ptr<holoscan::ops::NumericalValidationSinkOp> sink_;
};

class InferenceOpFastPathCorrectness : public ::testing::TestWithParam<FastPathConfig> {};

TEST_P(InferenceOpFastPathCorrectness, IdentityModelReturnsInputPerFrame) {
  const auto cfg = GetParam();
  testing::internal::CaptureStderr();
  auto app = holoscan::make_application<InferenceOpCorrectnessApp>(cfg);
  app->run();
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(app->failures(), 0) << "Config '" << cfg.label
                                << "' produced numerical mismatch or missing tensors.\n"
                                << "  last_error: " << app->last_error() << "\n"
                                << "  Log tail:\n"
                                << log;
  EXPECT_GE(app->frames_ok(), InferenceOpCorrectnessApp::kFrames)
      << "Config '" << cfg.label << "' did not deliver all " << InferenceOpCorrectnessApp::kFrames
      << " frames (frames_ok=" << app->frames_ok() << ").\n"
      << "  Log tail:\n"
      << log;
}

INSTANTIATE_TEST_CASE_P(
    FastPathConfigMatrix, InferenceOpFastPathCorrectness,
    ::testing::Values(
        // (cached-extractor storage recheck): single model, host input.
        FastPathConfig{1, false, /*in*/ false, /*out*/ true, /*tx*/ true, "single_host_input"},
        // (do_inference_on_stream H2D copy): two models, host input.
        FastPathConfig{2, false, /*in*/ false, /*out*/ true, /*tx*/ true, "two_host_input_seq"},
        FastPathConfig{2, true, /*in*/ false, /*out*/ true, /*tx*/ true, "two_host_input_par"},
        // (do_inference_on_stream D2H copy): two models, host output.
        FastPathConfig{2, false, /*in*/ true, /*out*/ false, /*tx*/ false, "two_host_output_seq"},
        FastPathConfig{2, true, /*in*/ true, /*out*/ false, /*tx*/ false, "two_host_output_par"},
        // Baselines: all-CUDA path (should still be exercised).
        FastPathConfig{1, false, true, true, true, "single_all_cuda"},
        FastPathConfig{2, false, true, true, true, "two_all_cuda_seq"},
        FastPathConfig{2, true, true, true, true, "two_all_cuda_par"}),
    [](const ::testing::TestParamInfo<FastPathConfig>& info) { return info.param.label; });

// ============================================================================
// Chained-model regression.
//
// Two models where model_b consumes model_a's output. Model names are chosen
// so std::map iteration ("a", "b") is fortunate to be correct; we additionally
// register a REVERSED case with "z_producer" / "a_consumer" to force a wrong
// map order and prove the canonical path's execution_plan_ walk honours
// dependency ordering.
// ============================================================================

class InferenceOpDependentModelsApp : public holoscan::Application {
 public:
  InferenceOpDependentModelsApp(bool reverse_map_order, bool parallel)
      : reverse_map_order_(reverse_map_order), parallel_(parallel) {}

  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("pool");
    auto stream_pool =
        make_resource<CudaStreamPool>("stream_pool", 0, cudaStreamNonBlocking, 0, 1, 32);
    auto gen = make_operator<holoscan::ops::FrameCountingTensorGeneratorOp>(
        "generator",
        Arg("allocator") = allocator,
        Arg("on_cuda") = true,
        Arg("tensor_name") = std::string("tensor"),
        make_condition<CountCondition>(kFrames));

    // producer: input tensor "tensor",       output "intermediate"
    // consumer: input tensor "intermediate", output "final"
    // Map key order is what std::map iterates by. When reverse_map_order_ is
    // true, we name producer "z_producer" and consumer "a_consumer", forcing
    // std::map insertion/iteration to visit consumer first.
    const std::string producer_key = reverse_map_order_ ? "z_producer" : "producer";
    const std::string consumer_key = reverse_map_order_ ? "a_consumer" : "consumer";
    const std::string model_path = "tests/operators/inference/models/identity_model.onnx";

    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map, inference_map;
    model_path_map.insert(producer_key, model_path);
    model_path_map.insert(consumer_key, model_path);
    pre_processor_map.insert(producer_key, {"tensor"});
    pre_processor_map.insert(consumer_key, {"intermediate"});
    inference_map.insert(producer_key, {"intermediate"});
    inference_map.insert(consumer_key, {"final"});

    std::vector<int> dims = {BATCH_SIZE, TENSOR_SIZE, TENSOR_SIZE};
    auto infer = make_operator<holoscan::ops::InferenceOp>(
        "infer",
        Arg("backend") = std::string("trt"),
        Arg("model_path_map") = model_path_map,
        Arg("pre_processor_map") = pre_processor_map,
        Arg("inference_map") = inference_map,
        Arg("in_tensor_names") = std::vector<std::string>{"tensor"},
        Arg("out_tensor_names") = std::vector<std::string>{"final"},
        Arg("input_on_cuda") = true,
        Arg("output_on_cuda") = true,
        Arg("transmit_on_cuda") = true,
        Arg("parallel_inference") = parallel_,
        Arg("infer_on_cpu") = false,
        Arg("enable_fp16") = false,
        Arg("enable_cuda_graphs") = true,
        Arg("in_tensor_dimensions") = dims,
        Arg("allocator") = allocator,
        Arg("cuda_stream_pool") = stream_pool);

    sink_ = make_operator<holoscan::ops::NumericalValidationSinkOp>(
        "sink",
        Arg("tensor_names") = std::vector<std::string>{"final"},
        Arg("input_on_cuda") = true);
    add_flow(gen, infer, {{"output", "receivers"}});
    add_flow(infer, sink_, {{"transmitter", "in"}});
  }

  int frames_ok() const { return sink_ ? sink_->frames_ok() : 0; }
  int failures() const { return sink_ ? sink_->failures() : 0; }
  std::string last_error() const { return sink_ ? sink_->last_error() : ""; }

  static constexpr int kFrames = 4;

 private:
  bool reverse_map_order_;
  bool parallel_;
  std::shared_ptr<holoscan::ops::NumericalValidationSinkOp> sink_;
};

TEST(InferenceOpDependentModels, SequentialMapOrderCorrect) {
  auto app = holoscan::make_application<InferenceOpDependentModelsApp>(
      /*reverse_map_order=*/false, /*parallel=*/false);
  app->run();
  EXPECT_EQ(app->failures(), 0) << "final tensor mismatch: " << app->last_error();
  EXPECT_GE(app->frames_ok(), InferenceOpDependentModelsApp::kFrames);
}

TEST(InferenceOpDependentModels, SequentialReversedMapOrder) {
  // With reversed keys, std::map iterates consumer BEFORE producer. The
  // eligibility gate now routes chained configs to kStandard, which walks
  // execution_plan_ level-by-level — the consumer must still see the
  // producer's output because canonical dispatch enforces ordering.
  auto app = holoscan::make_application<InferenceOpDependentModelsApp>(
      /*reverse_map_order=*/true, /*parallel=*/false);
  app->run();
  EXPECT_EQ(app->failures(), 0) << "reversed-map-order final mismatch: " << app->last_error();
  EXPECT_GE(app->frames_ok(), InferenceOpDependentModelsApp::kFrames);
}

TEST(InferenceOpDependentModels, ParallelReversedMapOrder) {
  auto app = holoscan::make_application<InferenceOpDependentModelsApp>(
      /*reverse_map_order=*/true, /*parallel=*/true);
  app->run();
  EXPECT_EQ(app->failures(), 0) << "parallel + reversed-map-order final mismatch: "
                                << app->last_error();
  EXPECT_GE(app->frames_ok(), InferenceOpDependentModelsApp::kFrames);
}

// ============================================================================
// Pure-function unit tests (no GPU, no scheduler, no operators).
// ============================================================================

TEST(TensorLookupHelper, FoundInPrimary) {
  std::map<std::string, int> primary{{"a", 1}, {"b", 2}};
  std::map<std::string, int> fallback{{"a", 99}};
  const int* out = nullptr;
  EXPECT_TRUE(holoscan::inference::lookup_tensor_buffer(primary, fallback, "a", &out));
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(*out, 1) << "Primary map should win on hit — fallback ignored";
}

TEST(TensorLookupHelper, AbsentFromPrimaryFoundInFallback) {
  std::map<std::string, int> primary{{"a", 1}};
  std::map<std::string, int> fallback{{"b", 42}};
  const int* out = nullptr;
  EXPECT_TRUE(holoscan::inference::lookup_tensor_buffer(primary, fallback, "b", &out));
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(*out, 42);
}

TEST(TensorLookupHelper, AbsentFromBoth) {
  std::map<std::string, int> primary{{"a", 1}};
  std::map<std::string, int> fallback{{"b", 2}};
  const int* out = nullptr;
  EXPECT_FALSE(holoscan::inference::lookup_tensor_buffer(primary, fallback, "missing", &out));
  EXPECT_EQ(out, nullptr) << "out pointer must remain unchanged on miss";
}

TEST(TensorLookupHelper, EmptyPrimaryEmptyFallback) {
  std::map<std::string, int> primary;
  std::map<std::string, int> fallback;
  const int* out = nullptr;
  EXPECT_FALSE(holoscan::inference::lookup_tensor_buffer(primary, fallback, "x", &out));
}

TEST(HasOffGpuDtAssignment, EmptyMapReturnsFalse) {
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({}));
}

TEST(HasOffGpuDtAssignment, SingleModelOnDeviceZero) {
  // {model: "0"} → same as gpu-dt → not an off-device assignment.
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({{"model", "0"}}));
}

TEST(HasOffGpuDtAssignment, SingleModelOnDeviceOneIsOffGpuDt) {
  // {model: "1"} routes to GPU 1 while gpu-dt is still 0. This MUST
  // block the fast path even though the map has only one entry.
  EXPECT_TRUE(holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({{"model", "1"}}));
}

TEST(HasOffGpuDtAssignment, GpuDtEntryIgnored) {
  // The reserved key "gpu-dt" identifies the transfer device itself, not a
  // model assignment. It should never count as an off-device assignment.
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({{"gpu-dt", "1"}}));
}

TEST(HasOffGpuDtAssignment, MixedModelsOneOnDeviceOne) {
  // {a: "0", b: "1"} — b lands off gpu-dt → block.
  EXPECT_TRUE(
      holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({{"a", "0"}, {"b", "1"}}));
}

TEST(HasOffGpuDtAssignment, EmptyValueIgnored) {
  // Guard against odd YAML that produced an empty value — treat as "unknown"
  // rather than off-device.
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_off_gpu_dt_assignment({{"model", ""}}));
}

TEST(HasChainedModels, NoOverlap) {
  std::map<std::string, std::vector<std::string>> pre{{"a", {"input_a"}}, {"b", {"input_b"}}};
  std::map<std::string, std::vector<std::string>> inf{{"a", {"out_a"}}, {"b", {"out_b"}}};
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_chained_models(pre, inf));
}

TEST(HasChainedModels, ConsumerReadsProducerOutput) {
  // producer emits "intermediate"; consumer reads "intermediate" → chained.
  std::map<std::string, std::vector<std::string>> pre{{"producer", {"raw"}},
                                                      {"consumer", {"intermediate"}}};
  std::map<std::string, std::vector<std::string>> inf{{"producer", {"intermediate"}},
                                                      {"consumer", {"final"}}};
  EXPECT_TRUE(holoscan::ops::inference_eligibility::has_chained_models(pre, inf));
}

TEST(HasChainedModels, ReverseKeyOrderStillDetected) {
  // Key order shouldn't affect detection; set-based scan is order-free.
  std::map<std::string, std::vector<std::string>> pre{{"a_consumer", {"intermediate"}},
                                                      {"z_producer", {"raw"}}};
  std::map<std::string, std::vector<std::string>> inf{{"a_consumer", {"final"}},
                                                      {"z_producer", {"intermediate"}}};
  EXPECT_TRUE(holoscan::ops::inference_eligibility::has_chained_models(pre, inf));
}

TEST(HasChainedModels, EmptyMapsReturnFalse) {
  EXPECT_FALSE(holoscan::ops::inference_eligibility::has_chained_models({}, {}));
}

TEST(HasChainedModels, ModelReadingItsOwnOutputAlsoCountsAsChained) {
  // Contrived — a model with output tensor also appearing in its own input list
  // is technically a cycle. The helper does not distinguish this from a valid
  // consumer chain; both must disqualify fast dispatch.
  std::map<std::string, std::vector<std::string>> pre{{"m", {"t"}}};
  std::map<std::string, std::vector<std::string>> inf{{"m", {"t"}}};
  EXPECT_TRUE(holoscan::ops::inference_eligibility::has_chained_models(pre, inf));
}
