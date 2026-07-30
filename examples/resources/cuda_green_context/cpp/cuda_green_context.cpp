/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/resources/gxf/cuda_green_context_pool.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include "test_kernel.cu.hpp"

constexpr int kMinCudaDriverVersion = 12040;
constexpr const char* kHsdkFaqUrl = "https://docs.nvidia.com/holoscan/sdk-user-guide/hsdk_faq.html";

static std::optional<std::vector<uint32_t>> green_context_partitions_for_current_arch() {
  struct utsname os_info{};
  if (uname(&os_info) != 0) {
    return std::nullopt;
  }
  std::string arch(static_cast<const char*>(os_info.machine));
  if (arch == "x86_64" || arch == "amd64") {
    return std::vector<uint32_t>{8, 8};
  }
  if (arch == "aarch64" || arch == "arm64") {
    // For reference, Jetson Orin AGX has 16 SMs,
    //                Jetson Orin Nano has 8 SMs,
    //                Jetson AGX Thor has 20 SMs (Blackwell sm_110).
    return std::vector<uint32_t>{4, 4};
  }
  return std::nullopt;
}

// Example requires a certain minimum number of GPU Streaming Multiprocessors (SMs) to run
static int required_sm_count_for_green_context(const std::vector<uint32_t>& partitions) {
  int total = 0;
  for (const auto partition_sm_count : partitions) {
    total += static_cast<int>(partition_sm_count);
  }
  return total;
}

// Gets the current CUDA Driver API, such as "12040" for 12.4
static std::optional<int> detect_cuda_driver_version() {
  int version = 0;
  if (cudaDriverGetVersion(&version) != cudaSuccess) {
    return std::nullopt;
  }
  return version;
}

// Green Context APIs are introduced in CUDA Driver 12.4
static bool green_context_supported_by_cuda_driver() {
  auto version = detect_cuda_driver_version();
  return version.has_value() && version.value() >= kMinCudaDriverVersion;
}

// Detects the number of GPU Streaming Multiprocessors (SMs) available on the device
static std::optional<int> detect_device_multiprocessor_count() {
  int sm_count = 0;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
    return std::nullopt;
  }
  return sm_count;
}

// Returns the compute capability major version for CUDA device 0 (same meaning as
// cudaDeviceProp::major), or nullopt on failure.
static std::optional<int> detect_device_compute_capability_major() {
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
    return std::nullopt;
  }
  return major;
}

// Typical minimum SM block size for Green Context pools by architecture. The sample uses this
// for Fragment::add_default_green_context_pool(..., min_sm_size) together with sms_per_partition
// so the pool matches the driver's SM grouping.
//
// \param major Compute capability major: 7 Volta/Turing, 8 Ampere, 9 Hopper, 10+ Blackwell+.
// \return Value suitable for Holoscan min_sm_size (GXF min_sm_count). Unknown majors default to 2.
static uint32_t green_context_min_sm_size_for_device_major(int major) {
  if (major == 7) {
    return 2;
  }
  if (major == 8 || major == 9) {
    return 4;
  }
  if (major >= 10) {
    return 8;
  }
  return 2;
}

// Pick the largest min_sm_size for which the CUDA driver accepts ``partitions``.
//
// The architecture-default heuristic above is intentionally coarse and may not match every GPU's
// actual SM-grouping granularity.  For example, on IGX Thor (compute capability 11.x,
// sm_count=20) the Blackwell-class default of 8 is too large for the example's aarch64
// partitioning of {4, 4} even though the driver itself accepts the smaller min_sm_size=4
// grouping just fine.
//
// This helper probes candidate values starting at the architecture default and halving down to
// 2, returning the largest value for which CudaGreenContextPool::is_partitioning_supported
// succeeds.  An arithmetic gate skips candidates that would produce a degenerate per-partition
// resource count of 0.  Returns std::nullopt when no candidate works.
static std::optional<uint32_t> green_context_resolve_min_sm_size(
    int32_t dev_id, std::optional<int> cc_major, const std::vector<uint32_t>& partitions) {
  auto sm_count = detect_device_multiprocessor_count();
  if (!sm_count.has_value()) {
    return std::nullopt;
  }
  uint32_t total = 0;
  for (auto p : partitions) {
    total += p;
  }
  if (static_cast<int>(total) > sm_count.value()) {
    return std::nullopt;
  }

  uint32_t starting =
      cc_major.has_value() ? green_context_min_sm_size_for_device_major(cc_major.value()) : 2;
  for (uint32_t candidate = starting; candidate >= 2; candidate /= 2) {
    bool arithmetic_ok = true;
    for (auto p : partitions) {
      if (p < candidate || (p % candidate) != 0) {
        arithmetic_ok = false;
        break;
      }
    }
    uint32_t remainder = static_cast<uint32_t>(sm_count.value()) - total;
    if (arithmetic_ok && remainder != 0 && (remainder % candidate) != 0) {
      arithmetic_ok = false;
    }
    if (arithmetic_ok &&
        holoscan::CudaGreenContextPool::is_partitioning_supported(dev_id, candidate, partitions)) {
      return candidate;
    }
  }
  return std::nullopt;
}

namespace holoscan::ops {
class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out");
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CUDA Stream Pool",
               "CUDA Stream Pool",
               std::shared_ptr<CudaStreamPool>(nullptr));
  }

  // NOLINTBEGIN(readability-function-cognitive-complexity)
  void start() override {
    size_t tensor_bytes = sizeof(float) * tensor_size_ * tensor_size_;
    // Allocate CUDA memory
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&input1_, tensor_bytes), "cudaMalloc failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&input2_, tensor_bytes), "cudaMalloc failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&output_, tensor_bytes), "cudaMalloc failed!");
    // Create CUDA events for timing
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventCreate(&start_event_), "cudaEventCreate failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventCreate(&stop_event_), "cudaEventCreate failed!");
  }

  void stop() override {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(start_event_), "cudaEventDestroy failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(stop_event_), "cudaEventDestroy failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(input1_), "cudaFree failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(input2_), "cudaFree failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(output_), "cudaFree failed!");
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = index_++;
    if (stream_ == nullptr) {
      auto maybe_new_stream = context.allocate_cuda_stream("tx_stream");
      if (maybe_new_stream) {
        stream_ = maybe_new_stream.value();
      }
    }
    cudaStream_t stream = stream_;

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(start_event_, stream),
                                   "cudaEventRecord failed!");
    asyncLaunchMatrixMultiplyKernel(input1_, input2_, output_, tensor_size_, stream);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(stop_event_, stream), "cudaEventRecord failed!");

    op_output.emit(value, "out");
  };

 private:
  int index_ = 1;
  int tensor_size_ = 256;
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t stop_event_ = nullptr;
  cudaStream_t stream_ = nullptr;
  float* input1_ = nullptr;
  float* input2_ = nullptr;
  float* output_ = nullptr;

  Parameter<std::shared_ptr<CudaGreenContext>> cuda_green_context_;
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
};
class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    if (!value_vector.empty()) {
      HOLOSCAN_LOG_INFO("Rx message value: {}", value_vector[0]);
    }
  };

 private:
  int count_ = 1;
};

}  // namespace holoscan::ops

class SampleCudaStreamPoolApp : public holoscan::Application {
 public:
  // Override of the resolved Green Context min_sm_size, set by main() via
  // set_green_context_min_sm_size(). When unset, compose() probes for a value the driver accepts.
  void set_green_context_min_sm_size(uint32_t min_sm_size) { min_sm_size_override_ = min_sm_size; }

  void compose() override {
    using namespace holoscan;

    // Create a cuda green context
    auto partitions = green_context_partitions_for_current_arch();
    if (!partitions.has_value()) {
      throw std::runtime_error("Unsupported platform architecture for Green Context sample");
    }

    // Create a green context pool which will be used as the default green context pool for the
    // current fragment
    uint32_t min_sm_size = 0;
    if (min_sm_size_override_.has_value()) {
      min_sm_size = min_sm_size_override_.value();
    } else {
      const auto cc_major = detect_device_compute_capability_major();
      auto resolved = green_context_resolve_min_sm_size(0, cc_major, partitions.value());
      if (!resolved.has_value()) {
        throw std::runtime_error(
            "Green Context partitioning is not supported on this GPU; the application launcher "
            "should have skipped before reaching compose()");
      }
      min_sm_size = resolved.value();
    }
    const auto cuda_green_context_pool =
        add_default_green_context_pool(0, std::move(partitions.value()), -1, min_sm_size);

    // Use green context 0 from the provided green context pool
    const auto cuda_green_context1 =
        make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 0, "tx1");

    const auto cuda_stream_pool1 = make_resource<CudaStreamPool>(
        "cuda_stream_pool", 0, 0, 0, 1, 5, cuda_green_context1, "tx1");
    // Define the tx and rx operators, allowing the tx operator to execute 10 times
    auto tx1 =
        make_operator<ops::PingTxOp>("tx1", make_condition<CountCondition>(10), cuda_stream_pool1);

    auto rx1 = make_operator<ops::PingRxOp>("rx1");

    // Create a thread pool with two threads
    auto pool1 = make_thread_pool("pool1", 2);
    // can assign operators individually to this thread pool (setting pinning to true)
    pool1->add(tx1, true);
    pool1->add(rx1, true);
    add_flow(tx1, rx1);

    // Use a specific green context identified by "index"
    auto cuda_green_context2 =
        make_resource<CudaGreenContext>("cuda_green_context", cuda_green_context_pool, 1, "tx2");
    auto cuda_stream_pool2 = make_resource<CudaStreamPool>(
        "cuda_stream_pool2", 0, 0, 0, 1, 5, cuda_green_context2, "tx2");
    auto tx2 =
        make_operator<ops::PingTxOp>("tx2", make_condition<CountCondition>(15), cuda_stream_pool2);
    auto rx2 = make_operator<ops::PingRxOp>("rx2");

    // Create a thread pool with two threads
    auto pool2 = make_thread_pool("pool2", 2);
    pool2->add(tx2, true);
    pool2->add(rx2, true);
    add_flow(tx2, rx2);

    // Use the default green context from the provided green context pool
    auto tx3 = make_operator<ops::PingTxOp>(
        "tx3", make_condition<CountCondition>(20), cuda_green_context_pool);
    auto rx3 = make_operator<ops::PingRxOp>("rx3");

    // Create a thread pool with two threads
    auto pool3 = make_thread_pool("pool3", 2);
    pool3->add(tx3, true);
    pool3->add(rx3, true);
    add_flow(tx3, rx3);

    // Use the fragment default green context pool without providing a green context pool
    auto tx4 = make_operator<ops::PingTxOp>("tx4", make_condition<CountCondition>(25));
    auto rx4 = make_operator<ops::PingRxOp>("rx4");

    // Create a thread pool with two threads
    auto pool4 = make_thread_pool("pool4", 2);
    pool4->add(tx4, true);
    pool4->add(rx4, true);
    add_flow(tx4, rx4);
  }

 private:
  std::optional<uint32_t> min_sm_size_override_{};
};

// CTest skip return code when Green Context is not available.
constexpr int kSkipReturnCode = 77;

int main() {
  if (!green_context_supported_by_cuda_driver()) {
    auto version = detect_cuda_driver_version();
    std::cerr << "Green Context requires CUDA Driver API >= 12.4 (cudaDriverGetVersion >= "
              << kMinCudaDriverVersion << ", detected: "
              << (version.has_value() ? std::to_string(version.value()) : "unknown") << "). See "
              << kHsdkFaqUrl << '\n';
    return kSkipReturnCode;
  }

  auto partitions = green_context_partitions_for_current_arch();
  if (!partitions.has_value()) {
    std::cerr << "Green Context sample is not configured for this architecture."
              << " See " << kHsdkFaqUrl << '\n';
    return kSkipReturnCode;
  }

  const int required_sm_count = required_sm_count_for_green_context(partitions.value());
  auto sm_count = detect_device_multiprocessor_count();
  if (!sm_count.has_value() || sm_count.value() < required_sm_count) {
    std::cerr << "Green Context requires at least " << required_sm_count
              << " SMs for this sample's partitioning (detected: "
              << (sm_count.has_value() ? std::to_string(sm_count.value()) : "unknown") << "). See "
              << kHsdkFaqUrl << '\n';
    return kSkipReturnCode;
  }

  auto cc_major = detect_device_compute_capability_major();
  auto resolved_min_sm = green_context_resolve_min_sm_size(0, cc_major, partitions.value());
  if (!resolved_min_sm.has_value()) {
    std::cerr << "Green Context partitioning is not supported on this GPU (sm_count="
              << sm_count.value() << ", partitions=[";
    for (size_t i = 0; i < partitions.value().size(); ++i) {
      if (i > 0)
        std::cerr << ", ";
      std::cerr << partitions.value()[i];
    }
    std::cerr << "]). See " << kHsdkFaqUrl << '\n';
    return kSkipReturnCode;
  }

  auto app = holoscan::make_application<SampleCudaStreamPoolApp>();
  app->set_green_context_min_sm_size(resolved_min_sm.value());

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));

  app->run();

  return 0;
}
