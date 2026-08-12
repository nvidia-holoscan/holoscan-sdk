/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "descriptor_frame_multi_order_traits.hpp"
#include "runtime/runtime_worker.hpp"

namespace {

bool cuda_scenario_skipped = false;

using namespace std::chrono_literals;

using Native = app3p::MultiDescriptorFrameT;
using Fields = app3p::MultiDescriptorFrameTensorFields;

constexpr std::string_view kInstanceEnvironment = "HOLOSCAN_TEST_MULTI_TENSOR_INSTANCE";
constexpr std::uint32_t kDefaultQueueDepth = 2U;
constexpr holoscan::DeviceId kDevice{0};
constexpr std::array<std::uint16_t, 3U> kFieldIds{1U, 3U, 5U};
constexpr std::array<std::uint64_t, 3U> kFieldBounds{64U, 32U, 32U};
constexpr std::array<holoscan::MemoryKind, 3U> kFieldMemory{
    holoscan::MemoryKind::kHost,
    holoscan::MemoryKind::kCudaDevice,
    holoscan::MemoryKind::kPinnedHost,
};
constexpr std::array<std::int64_t, 2U> kFirstShape{2, 3};
constexpr std::array<std::int64_t, 1U> kSecondShape{4};
constexpr std::array<std::int64_t, 2U> kThirdShape{1, 2};
constexpr std::array<std::int64_t, 3U> kInvalidRankShape{1, 2, 3};
constexpr std::array<std::int64_t, 2U> kInvalidExtentShape{2, -1};
constexpr std::array<std::int64_t, 2U> kOverBoundShape{5, 5};
constexpr std::array<std::int64_t, 2U> kInvalidByteStrides{12, -4};
constexpr std::array<float, 6U> kFirstValues{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
constexpr std::array<std::uint8_t, 4U> kSecondValues{3U, 5U, 8U, 13U};
constexpr std::array<std::int16_t, 2U> kThirdValues{21, 34};

enum class SourceMode : std::uint8_t {
  kValid,
  kMissingStagingContract,
  kMixedCarriage,
  kOverBound,
  kLoanBundle,
  kLoanBundleRestart,
  kInvalidLoanBundle,
};

enum class Topology : std::uint8_t {
  kProcessLocal,
  kCrossProcess,
  kCrossProcessFanout,
};

struct ProbeState {
  static void reset() noexcept {
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
    consumer_ready.store(0U, std::memory_order_release);
    source_released.store(false, std::memory_order_release);
    emitted.store(0U, std::memory_order_release);
    received.store(0U, std::memory_order_release);
    over_bound_rejections.store(0U, std::memory_order_release);
    invalid_bundle_rejections.store(0U, std::memory_order_release);
  }

  static void notify() noexcept { ready.notify_all(); }

  [[nodiscard]] static bool wait_for(const std::atomic<std::size_t>& counter,
                                     const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [&counter, expected] {
      return counter.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
  static inline std::atomic<std::size_t> consumer_ready{};
  static inline std::atomic<bool> source_released{};
  static inline std::atomic<std::size_t> emitted{};
  static inline std::atomic<std::size_t> received{};
  static inline std::atomic<std::size_t> over_bound_rejections{};
  static inline std::atomic<std::size_t> invalid_bundle_rejections{};
};

[[nodiscard]] bool check(const bool condition, const std::string_view message) {
  if (!condition) {
    HOLOSCAN_LOG_ERROR("{}", message);
  }
  return condition;
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    const std::string port =
        diagnostic.source.port ? std::to_string(diagnostic.source.port.value()) : "<none>";
    HOLOSCAN_LOG_ERROR("{} vertex={} port={} service={} required={} provided={}",
                       diagnostic.token,
                       diagnostic.source.vertex.value,
                       port,
                       diagnostic.service,
                       diagnostic.required,
                       diagnostic.provided);
  }
}

[[nodiscard]] bool cuda_available() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > kDevice.value) {
    return true;
  }
  cuda_scenario_skipped = true;
  HOLOSCAN_LOG_INFO("SKIP: multi-Tensor runtime validation requires CUDA device 0");
  return false;
}

class FixtureTensorMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t) override {
    return {};
  }

  bool set_deallocation_stream(cudaStream_t) override { return true; }
};

[[nodiscard]] std::shared_ptr<holoscan::Tensor> make_tensor(
    void* data, const std::initializer_list<std::int64_t> shape, const DLDevice device,
    const DLDataType dtype) {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  if (device.device_type == kDLCUDA) {
    context->memory_reference = std::make_shared<FixtureTensorMemoryReference>();
  }
  context->dl_shape.assign(shape);
  context->tensor.dl_tensor.data = data;
  context->tensor.dl_tensor.device = device;
  context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0U;
  return std::make_shared<holoscan::Tensor>(std::move(context));
}

class TensorFixtureStorage {
 public:
  [[nodiscard]] bool initialize() {
    if (cudaSetDevice(kDevice.value) != cudaSuccess ||
        cudaMalloc(&device_data_, kSecondValues.size()) != cudaSuccess ||
        cudaMemcpy(
            device_data_, kSecondValues.data(), kSecondValues.size(), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        // FixtureTensorMemoryReference advertises no outstanding producer work. Complete the
        // default-stream initialization before the Tensor can reach a nonblocking consumer stream.
        cudaStreamSynchronize(nullptr) != cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void**>(&pinned_data_),
                       pinned_values_.size() * sizeof(pinned_values_.front())) != cudaSuccess) {
      release();
      return false;
    }
    std::copy(pinned_values_.begin(), pinned_values_.end(), pinned_data_);
    first = make_tensor(first_values_.data(),
                        {2, 3},
                        DLDevice{.device_type = kDLCPU, .device_id = 0},
                        DLDataType{kDLFloat, 32U, 1U});
    second = make_tensor(device_data_,
                         {4},
                         DLDevice{.device_type = kDLCUDA, .device_id = kDevice.value},
                         DLDataType{kDLUInt, 8U, 1U});
    third = make_tensor(pinned_data_,
                        {1, 2},
                        DLDevice{.device_type = kDLCUDAHost, .device_id = 0},
                        DLDataType{kDLInt, 16U, 1U});
    return true;
  }

  void release() noexcept {
    first.reset();
    second.reset();
    third.reset();
    if (pinned_data_ != nullptr) {
      static_cast<void>(cudaFreeHost(pinned_data_));
      pinned_data_ = nullptr;
    }
    if (device_data_ != nullptr) {
      static_cast<void>(cudaFree(device_data_));
      device_data_ = nullptr;
    }
  }

  ~TensorFixtureStorage() { release(); }

  std::shared_ptr<holoscan::Tensor> first;
  std::shared_ptr<holoscan::Tensor> second;
  std::shared_ptr<holoscan::Tensor> third;

 private:
  std::array<float, 6U> first_values_{kFirstValues};
  std::array<std::int16_t, 2U> pinned_values_{kThirdValues};
  void* device_data_{};
  std::int16_t* pinned_data_{};
};

[[nodiscard]] Native make_native_metadata() {
  Native value{};
  value.sequence = 42U;
  value.label = "heterogeneous three-Tensor frame";
  value.exposure = 2.5F;
  value.header = std::make_unique<app3p::FrameHeaderT>();
  value.header->source = "stereo-camera";
  value.header->frame_id = 17U;
  return value;
}

[[nodiscard]] Native make_native(const TensorFixtureStorage& storage) {
  Native value = make_native_metadata();
  value.first = storage.first;
  value.second = storage.second;
  value.third = storage.third;
  return value;
}

[[nodiscard]] auto make_loan_requests() {
  // Deliberately author a valid noncanonical order; Runtime must normalize by Tensor index.
  return std::array{
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::third,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kThirdShape,
                  .dtype = DLDataType{kDLInt, 16U, 1U},
              },
      },
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::first,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kFirstShape,
                  .dtype = DLDataType{kDLFloat, 32U, 1U},
              },
      },
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::second,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kSecondShape,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
              },
      },
  };
}

[[nodiscard]] auto make_shape_requests() {
  // Deliberately author a valid noncanonical order; Runtime must normalize by Tensor index.
  return std::array{
      holoscan::tensor_shape(Fields::third, kThirdShape),
      holoscan::tensor_shape(Fields::first, kFirstShape),
      holoscan::tensor_shape(Fields::second, kSecondShape),
  };
}

// Emits one heterogeneous generated three-Tensor value using the selected validation or loan mode.
// Representative tests: multi_tensor_schema_contracts, multi_tensor_schema_bundle-cross-process.
template <SourceMode Mode>
class MultiTensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    const holoscan::TensorPortLayout first_layout{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 2U,
    };
    const holoscan::TensorPortLayout second_layout{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    const holoscan::TensorPortLayout third_layout{
        .memory_kind = holoscan::MemoryKind::kPinnedHost,
        .dtype = DLDataType{kDLInt, 16U, 1U},
        .rank = 2U,
    };
    constexpr std::uint64_t first_bound = Mode == SourceMode::kOverBound ? 8U : kFieldBounds[0];

    auto& output_builder = spec.output(output, "frames").max_emits_per_compute(1U);
    if constexpr (Mode == SourceMode::kMixedCarriage) {
      output_builder.tensor_field(Fields::first)
          .produces_tensor(first_layout)
          .tensor_allocation(
              holoscan::TensorAllocationBounds{.max_rank = 2U, .max_byte_span = first_bound})
          .payload_options(holoscan::TensorFieldPayloadOptions{
              .max_inline_bytes = 8U,
              .max_tensor_byte_span = first_bound,
              .interface_tensor_emit_policy =
                  holoscan::InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
          });
    } else {
      output_builder.tensor_field(Fields::first)
          .produces_tensor(first_layout)
          .tensor_allocation(
              holoscan::TensorAllocationBounds{.max_rank = 2U, .max_byte_span = first_bound})
          .payload_options(holoscan::TensorFieldPayloadOptions{
              .max_tensor_byte_span = first_bound,
              .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
          });
    }
    if constexpr (Mode == SourceMode::kMissingStagingContract) {
      output_builder.tensor_field(Fields::second)
          .produces_tensor(second_layout)
          .tensor_allocation(
              holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kFieldBounds[1]});
    } else {
      output_builder.tensor_field(Fields::second)
          .produces_tensor(second_layout)
          .tensor_allocation(
              holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kFieldBounds[1]})
          .payload_options(holoscan::TensorFieldPayloadOptions{
              .max_tensor_byte_span = kFieldBounds[1],
              .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
          });
    }
    output_builder.tensor_field(Fields::third)
        .produces_tensor(third_layout)
        .tensor_allocation(
            holoscan::TensorAllocationBounds{.max_rank = 2U, .max_byte_span = kFieldBounds[2]})
        .payload_options(holoscan::TensorFieldPayloadOptions{
            .max_tensor_byte_span = kFieldBounds[2],
            .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  void start() override {
    initialized_ = storage_.initialize();
    ProbeState::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  void stop() override { storage_.release(); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (!initialized_) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported,
                                                       "CUDA fixture initialization failed"});
    }
    if (completed_) {
      return {};
    }
    if (!ProbeState::source_released.load(std::memory_order_acquire)) {
      return {};
    }
    if constexpr (Mode == SourceMode::kLoanBundleRestart) {
      if (!restart_requested_) {
        auto restart = context.request_restart(
            holoscan::LifecycleRestartOptions{.from_stage = holoscan::LifecycleStage::kStart});
        if (!restart) {
          return holoscan::make_unexpected(std::move(restart.error()));
        }
        restart_requested_ = true;
        return {};
      }
    }
    holoscan::expected<void, holoscan::Error> emitted;
    if constexpr (Mode == SourceMode::kInvalidLoanBundle) {
      auto requests = make_loan_requests();
      const auto missing = output.allocate_tensors(
          std::span<const holoscan::TensorFieldLoanRequest<Native>>{requests}.first(2U));
      requests[2].field = Fields::first;
      const auto duplicate = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[2].field = holoscan::TensorFieldSelector<Native>{99U, 2U};
      const auto unknown = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.dtype.bits = 0U;
      const auto invalid_dtype = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kInvalidRankShape;
      const auto invalid_rank = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kInvalidExtentShape;
      const auto invalid_shape = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.byte_strides =
          holoscan::TensorByteStridesView{std::span<const std::int64_t>{kInvalidByteStrides}};
      const auto invalid_strides = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kOverBoundShape;
      const auto over_bound = output.allocate_tensors(requests);
      const auto rejected_as_invalid = [](const auto& result, const std::string_view kind) {
        if (!result && result.error().code == holoscan::ErrorCode::kInvalidArgument) {
          return true;
        }
        if (!result) {
          HOLOSCAN_LOG_ERROR("bundle request was not rejected as invalid: {} code={} message={}",
                             kind,
                             holoscan::error_code_name(result.error().code),
                             result.error().message);
        } else {
          HOLOSCAN_LOG_ERROR("bundle request was not rejected as invalid: {}", kind);
        }
        return false;
      };
      // Key assertion: malformed field inventories and Tensor views fail before reserving a
      // bundle, then one valid noncanonical request set can still use the same finite pools.
      if (!rejected_as_invalid(missing, "missing") ||
          !rejected_as_invalid(duplicate, "duplicate") ||
          !rejected_as_invalid(unknown, "unknown") ||
          !rejected_as_invalid(invalid_dtype, "dtype") ||
          !rejected_as_invalid(invalid_rank, "rank") ||
          !rejected_as_invalid(invalid_shape, "shape") ||
          !rejected_as_invalid(invalid_strides, "strides") ||
          !rejected_as_invalid(over_bound, "byte span")) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "invalid bundle request set was admitted"});
      }
      auto failed_callback_bundle = output.allocate_tensors(make_shape_requests());
      if (!failed_callback_bundle) {
        return holoscan::make_unexpected(std::move(failed_callback_bundle).error());
      }
      std::array<std::size_t, 3U> failed_callback_counts{};
      auto callback_failure = failed_callback_bundle->write_all(
          holoscan::host_field(
              Fields::first,
              [&](const holoscan::TensorOutputWriteGuard&) { ++failed_callback_counts[0]; }),
          holoscan::device_field(Fields::second,
                                 context.cuda_stream(),
                                 [&](const holoscan::TensorOutputWriteGuard&)
                                     -> holoscan::expected<void, holoscan::Error> {
                                   ++failed_callback_counts[1];
                                   return holoscan::make_unexpected(holoscan::Error{
                                       holoscan::ErrorCode::kFailure, "injected callback failure"});
                                 }),
          holoscan::host_field(Fields::third, [&](const holoscan::TensorOutputWriteGuard&) {
            ++failed_callback_counts[2];
          }));
      if (callback_failure || callback_failure.error().code != holoscan::ErrorCode::kFailure ||
          failed_callback_counts != std::array<std::size_t, 3U>{1U, 1U, 0U}) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "batch-write callback failure was not terminal"});
      }
      auto partial_emit =
          output.emit_tensors(std::move(*failed_callback_bundle), make_native(storage_));
      if (partial_emit || partial_emit.error().code != holoscan::ErrorCode::kInvalidArgument ||
          partial_emit.error().message != "TENSOR_SCHEMA_BUNDLE_FIELD_PREFLIGHT_FAILED") {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "partially committed bundle was publishable"});
      }
      auto preflight_bundle = output.allocate_tensors(make_shape_requests());
      if (!preflight_bundle) {
        return holoscan::make_unexpected(std::move(preflight_bundle).error());
      }
      std::size_t callback_count = 0U;
      auto duplicate_write = preflight_bundle->write_all(
          holoscan::host_field(Fields::first,
                               [&](const holoscan::TensorOutputWriteGuard&) { ++callback_count; }),
          holoscan::host_field(Fields::first,
                               [&](const holoscan::TensorOutputWriteGuard&) { ++callback_count; }),
          holoscan::device_field(
              Fields::second, context.cuda_stream(), [&](const holoscan::TensorOutputWriteGuard&) {
                ++callback_count;
              }));
      if (duplicate_write ||
          duplicate_write.error().code != holoscan::ErrorCode::kInvalidArgument ||
          callback_count != 0U) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "batch-write preflight invoked a callback"});
      }
      auto written = write_loan_bundle(context, *preflight_bundle);
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      ProbeState::invalid_bundle_rejections.fetch_add(1U, std::memory_order_acq_rel);
      ProbeState::notify();
      emitted = output.emit_tensors(std::move(*preflight_bundle), make_native(storage_));
    } else {
      if constexpr (Mode == SourceMode::kLoanBundle || Mode == SourceMode::kLoanBundleRestart) {
        emitted = emit_loan_bundle(context);
      } else {
        emitted = output.emit(make_native(storage_));
      }
    }
    if constexpr (Mode == SourceMode::kOverBound) {
      if (!emitted && emitted.error().code == holoscan::ErrorCode::kInvalidArgument) {
        completed_ = true;
        ProbeState::over_bound_rejections.fetch_add(1U, std::memory_order_acq_rel);
        ProbeState::notify();
        return {};
      }
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "over-bound field was not rejected as invalid"});
    }
    if (!emitted && (emitted.error().code == holoscan::ErrorCode::kNotReady ||
                     emitted.error().code == holoscan::ErrorCode::kBackpressured)) {
      return {};
    }
    if (!emitted) {
      HOLOSCAN_LOG_ERROR("multi-Tensor emit failed: {}: {}",
                         holoscan::error_code_name(emitted.error().code),
                         emitted.error().message);
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    completed_ = true;
    ProbeState::emitted.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Output<Native> output;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> write_loan_bundle(
      holoscan::ExecutionContext& context, holoscan::TensorSchemaOutputLoan<Native>& bundle) {
    return bundle.write_all(
        holoscan::host_field(Fields::first,
                             [](const holoscan::TensorOutputWriteGuard& writer) {
                               std::ranges::copy(kFirstValues, writer.data_as<float>());
                             }),
        holoscan::device_field(Fields::second,
                               context.cuda_stream(),
                               [&](const holoscan::TensorOutputWriteGuard& writer)
                                   -> holoscan::expected<void, holoscan::Error> {
                                 if (cudaMemcpyAsync(writer.data(),
                                                     kSecondValues.data(),
                                                     kSecondValues.size(),
                                                     cudaMemcpyHostToDevice,
                                                     context.cuda_stream().get()) != cudaSuccess) {
                                   return holoscan::make_unexpected(
                                       holoscan::Error{holoscan::ErrorCode::kFailure});
                                 }
                                 return {};
                               }),
        holoscan::host_field(Fields::third, [](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(kThirdValues, writer.data_as<std::int16_t>());
        }));
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_loan_bundle(
      holoscan::ExecutionContext& context) {
    const auto requests = make_shape_requests();
    auto bundle = output.allocate_tensors(requests);
    if (!bundle) {
      return holoscan::make_unexpected(std::move(bundle.error()));
    }
    auto written = write_loan_bundle(context, *bundle);
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    return output.emit_tensors(std::move(*bundle), make_native(storage_));
  }

  TensorFixtureStorage storage_;
  bool initialized_{};
  bool completed_{};
  bool restart_requested_{};
};

// Receives and validates the exact metadata, placement, shape, and contents of all three fields.
// Representative tests: multi_tensor_schema_process-local, multi_tensor_schema_bundle-fanout.
class MultiTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "frames").queue_depth(kDefaultQueueDepth);
    input_builder.tensor_field(Fields::first)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kHost,
            .dtype = DLDataType{kDLFloat, 32U, 1U},
            .rank = 2U,
        });
    input_builder.tensor_field(Fields::second)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kCudaDevice,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        });
    input_builder.tensor_field(Fields::third)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kPinnedHost,
            .dtype = DLDataType{kDLInt, 16U, 1U},
            .rank = 2U,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override {
    ProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      HOLOSCAN_LOG_ERROR("multi-Tensor sink receive failed: {}: {}",
                         holoscan::error_code_name(sample.error().code),
                         sample.error().message);
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const Native& value = sample->data;
    const auto dtype_matches = [](const holoscan::Tensor& tensor, const DLDataType expected) {
      const DLDataType actual = tensor.dtype();
      return actual.code == expected.code && actual.bits == expected.bits &&
             actual.lanes == expected.lanes;
    };
    if (value.first == nullptr || value.second == nullptr || value.third == nullptr ||
        value.header == nullptr || value.sequence != 42U ||
        value.label != "heterogeneous three-Tensor frame" || value.exposure != 2.5F ||
        value.header->source != "stereo-camera" || value.header->frame_id != 17U ||
        value.first->device().device_type != kDLCPU ||
        value.second->device().device_type != kDLCUDA ||
        value.second->device().device_id != kDevice.value ||
        value.third->device().device_type != kDLCUDAHost || value.first->ndim() != 2 ||
        value.second->ndim() != 1 || value.third->ndim() != 2 ||
        !std::ranges::equal(value.first->shape_span(), kFirstShape) ||
        !std::ranges::equal(value.second->shape_span(), kSecondShape) ||
        !std::ranges::equal(value.third->shape_span(), kThirdShape) ||
        !dtype_matches(*value.first, DLDataType{kDLFloat, 32U, 1U}) ||
        !dtype_matches(*value.second, DLDataType{kDLUInt, 8U, 1U}) ||
        !dtype_matches(*value.third, DLDataType{kDLInt, 16U, 1U})) {
      HOLOSCAN_LOG_ERROR("multi-Tensor sink metadata validation failed");
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "multi-Tensor metadata mismatch"});
    }
    auto first_ready = value.first->synchronize_for_host();
    auto third_ready = value.third->synchronize_for_host();
    const auto* first = value.first->data_as<float>();
    const auto* third = value.third->data_as<std::int16_t>();
    std::array<std::uint8_t, 4U> second{};
    const cudaStream_t stream = context.cuda_stream().get();
    cudaError_t copy_result = cudaErrorInvalidValue;
    cudaError_t synchronize_result = cudaErrorInvalidValue;
    if (first_ready && third_ready && first != nullptr && third != nullptr && stream != nullptr) {
      copy_result = cudaMemcpyAsync(
          second.data(), value.second->data(), second.size(), cudaMemcpyDeviceToHost, stream);
      if (copy_result == cudaSuccess) {
        synchronize_result = cudaStreamSynchronize(stream);
      }
    }
    if (!first_ready || !third_ready || first == nullptr || third == nullptr || stream == nullptr ||
        copy_result != cudaSuccess || synchronize_result != cudaSuccess || first[0] != 1.0F ||
        first[5] != 6.0F || second != std::array<std::uint8_t, 4U>{3U, 5U, 8U, 13U} ||
        third[0] != 21 || third[1] != 34) {
      HOLOSCAN_LOG_ERROR(
          "multi-Tensor sink data/readiness validation failed: first_ready={} third_ready={} "
          "first={} third={} stream={} device_data={} readiness_contract={} copy={} synchronize={}",
          static_cast<bool>(first_ready),
          static_cast<bool>(third_ready),
          static_cast<const void*>(first),
          static_cast<const void*>(third),
          static_cast<const void*>(stream),
          value.second->data(),
          value.second->has_readiness_stream_contract(),
          cudaGetErrorName(copy_result),
          cudaGetErrorName(synchronize_result));
      if (first != nullptr) {
        HOLOSCAN_LOG_ERROR("multi-Tensor sink first values: {}, {}", first[0], first[5]);
      }
      HOLOSCAN_LOG_ERROR("multi-Tensor sink second values: {}, {}, {}, {}",
                         static_cast<unsigned int>(second[0]),
                         static_cast<unsigned int>(second[1]),
                         static_cast<unsigned int>(second[2]),
                         static_cast<unsigned int>(second[3]));
      if (third != nullptr) {
        HOLOSCAN_LOG_ERROR("multi-Tensor sink third values: {}, {}", third[0], third[1]);
      }
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "multi-Tensor data mismatch"});
    }
    ProbeState::received.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Input<Native> input;
};

template <holoscan::MemoryKind Kind, typename Builder>
void configure_inline_output(Builder& builder) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  builder.tensor_field(Fields::first)
      .produces_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLFloat, 32U, 1U},
          .rank = 2U,
      })
      .tensor_allocation(
          holoscan::TensorAllocationBounds{.max_rank = 2U, .max_byte_span = kFieldBounds[0]})
      .payload_options(holoscan::TensorFieldPayloadOptions{
          .max_tensor_byte_span = kFieldBounds[0],
          .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      });
  builder.tensor_field(Fields::second)
      .produces_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
          .rank = 1U,
      })
      .tensor_allocation(
          holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kFieldBounds[1]})
      .payload_options(holoscan::TensorFieldPayloadOptions{
          .max_tensor_byte_span = kFieldBounds[1],
          .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      });
  builder.tensor_field(Fields::third)
      .produces_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLInt, 16U, 1U},
          .rank = 2U,
      })
      .tensor_allocation(
          holoscan::TensorAllocationBounds{.max_rank = 2U, .max_byte_span = kFieldBounds[2]})
      .payload_options(holoscan::TensorFieldPayloadOptions{
          .max_tensor_byte_span = kFieldBounds[2],
          .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
      });
}

template <holoscan::MemoryKind Kind, typename Builder>
void configure_inline_input(Builder& builder) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  builder.tensor_field(Fields::first)
      .expects_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLFloat, 32U, 1U},
          .rank = 2U,
      });
  builder.tensor_field(Fields::second)
      .expects_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
          .rank = 1U,
      });
  builder.tensor_field(Fields::third)
      .expects_tensor(holoscan::TensorPortLayout{
          .memory_kind = Kind,
          .dtype = DLDataType{kDLInt, 16U, 1U},
          .rank = 2U,
      });
}

template <holoscan::MemoryKind Kind, typename Stream>
// The test fixture intentionally keeps the transactional three-field write and validation in one
// helper, and takes the owning value by value so it can move it into the emitted sample.
// NOLINTBEGIN(readability-function-cognitive-complexity, performance-unnecessary-value-param)
[[nodiscard]] holoscan::expected<void, holoscan::Error> emit_inline_bundle(
    holoscan::Output<Native>& output, Native value, Stream stream, const bool relay) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  const auto requests = make_shape_requests();
  auto bundle = output.allocate_tensors(requests);
  if (!bundle) {
    return holoscan::make_unexpected(std::move(bundle).error());
  }
  holoscan::expected<void, holoscan::Error> first_written;
  holoscan::expected<void, holoscan::Error> second_written;
  holoscan::expected<void, holoscan::Error> third_written;
  if constexpr (Kind == holoscan::MemoryKind::kHost) {
    const auto* first_source =
        relay && value.first != nullptr ? value.first->data_as<float>() : kFirstValues.data();
    const auto* second_source = relay && value.second != nullptr
                                    ? value.second->data_as<std::uint8_t>()
                                    : kSecondValues.data();
    const auto* third_source = relay && value.third != nullptr
                                   ? value.third->data_as<std::int16_t>()
                                   : kThirdValues.data();
    first_written = bundle->write_host_and_commit(
        Fields::first, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{first_source, kFirstValues.size()}, writer.data_as<float>());
        });
    second_written = bundle->write_host_and_commit(
        Fields::second, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{second_source, kSecondValues.size()},
                            writer.data_as<std::uint8_t>());
        });
    third_written = bundle->write_host_and_commit(
        Fields::third, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{third_source, kThirdValues.size()},
                            writer.data_as<std::int16_t>());
        });
  } else {
    if (stream.get() == nullptr ||
        (relay && (value.first == nullptr || value.second == nullptr || value.third == nullptr))) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline CUDA stream is unavailable"});
    }
    const auto copy_kind = relay ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    first_written = bundle->write_and_commit(
        Fields::first,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.first->data() : kFirstValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kFirstValues.size() * sizeof(kFirstValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
    second_written = bundle->write_and_commit(
        Fields::second,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.second->data() : kSecondValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kSecondValues.size() * sizeof(kSecondValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
    third_written = bundle->write_and_commit(
        Fields::third,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.third->data() : kThirdValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kThirdValues.size() * sizeof(kThirdValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
  }
  if (!first_written || !second_written || !third_written) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "inline bundle field write failed"});
  }
  return output.emit_tensors(std::move(*bundle), std::move(value));
}
// NOLINTEND(readability-function-cognitive-complexity, performance-unnecessary-value-param)

// Publishes one generated three-field descriptor schema from uniformly placed field pools.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& builder = spec.output(output, "frames").max_emits_per_compute(1U);
    configure_inline_output<Kind>(builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  void start() override {
    ProbeState::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (completed_ || !ProbeState::source_released.load(std::memory_order_acquire)) {
      return {};
    }
    auto emitted =
        emit_inline_bundle<Kind>(output, make_native_metadata(), context.cuda_stream(), false);
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    completed_ = true;
    ProbeState::emitted.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Output<Native> output;

 private:
  bool completed_{};
};

// Relays a generated multi-Tensor value through a second independent set of field pools.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "in").queue_depth(kDefaultQueueDepth);
    configure_inline_input<Kind>(input_builder);
    auto& output_builder = spec.output(output, "out").max_emits_per_compute(1U);
    configure_inline_output<Kind>(output_builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return emit_inline_bundle<Kind>(output, std::move(sample->data), context.cuda_stream(), true);
  }

  holoscan::Input<Native> input;
  holoscan::Output<Native> output;
};

// Validates the final uniformly placed fields after the generated schema crosses two inline edges.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& builder = spec.input(input, "frames").queue_depth(kDefaultQueueDepth);
    configure_inline_input<Kind>(builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override {
    ProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const Native& value = sample->data;
    std::array<float, kFirstValues.size()> first_values{};
    std::array<std::uint8_t, kSecondValues.size()> second_values{};
    std::array<std::int16_t, kThirdValues.size()> third_values{};
    if (value.first == nullptr || value.second == nullptr || value.third == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline Tensor field is missing"});
    }
    if constexpr (Kind == holoscan::MemoryKind::kHost) {
      const auto* first = value.first->data_as<float>();
      const auto* second = value.second->data_as<std::uint8_t>();
      const auto* third = value.third->data_as<std::int16_t>();
      if (first == nullptr || second == nullptr || third == nullptr) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "inline host field is unavailable"});
      }
      std::ranges::copy(std::span{first, first_values.size()}, first_values.begin());
      std::ranges::copy(std::span{second, second_values.size()}, second_values.begin());
      std::ranges::copy(std::span{third, third_values.size()}, third_values.begin());
    } else {
      if (context.cuda_stream().get() == nullptr ||
          cudaMemcpyAsync(first_values.data(),
                          value.first->data(),
                          first_values.size() * sizeof(first_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaMemcpyAsync(second_values.data(),
                          value.second->data(),
                          second_values.size() * sizeof(second_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaMemcpyAsync(third_values.data(),
                          value.third->data(),
                          third_values.size() * sizeof(third_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "inline CUDA readback failed"});
      }
    }
    constexpr auto expected_device_type = Kind == holoscan::MemoryKind::kHost ? kDLCPU : kDLCUDA;
    const bool device_ids_match = Kind == holoscan::MemoryKind::kHost
                                      ? value.first->device().device_id == 0 &&
                                            value.second->device().device_id == 0 &&
                                            value.third->device().device_id == 0
                                      : value.first->device().device_id == kDevice.value &&
                                            value.second->device().device_id == kDevice.value &&
                                            value.third->device().device_id == kDevice.value;
    if (value.header == nullptr || value.sequence != 42U ||
        value.label != "heterogeneous three-Tensor frame" || value.exposure != 2.5F ||
        value.header->source != "stereo-camera" || value.header->frame_id != 17U ||
        value.first->device().device_type != expected_device_type ||
        value.second->device().device_type != expected_device_type ||
        value.third->device().device_type != expected_device_type || !device_ids_match ||
        !std::ranges::equal(kFirstValues, first_values) ||
        !std::ranges::equal(kSecondValues, second_values) ||
        !std::ranges::equal(kThirdValues, third_values)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline multi-Tensor data mismatch"});
    }
    ProbeState::received.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Input<Native> input;
};

// Reports that one isolated consumer worker has started before producer publication is released.
// Representative tests: multi_tensor_schema_cross-process, multi_tensor_schema_fanout.
class ConsumerReadyProbe final : public holoscan::Operator<> {
 public:
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (!reported_) {
      reported_ = true;
      ProbeState::consumer_ready.fetch_add(1U, std::memory_order_acq_rel);
      ProbeState::notify();
    }
    return {};
  }

 private:
  bool reported_{};
};

[[nodiscard]] std::string graph_name(const Topology topology, const SourceMode mode) {
  std::string result{"multi-tensor-schema-"};
  switch (topology) {
    case Topology::kProcessLocal:
      result += "local";
      break;
    case Topology::kCrossProcess:
      result += "cross";
      break;
    case Topology::kCrossProcessFanout:
      result += "fanout";
      break;
  }
  result += "-" + std::to_string(static_cast<unsigned int>(mode));
  if (const char* const instance = std::getenv(kInstanceEnvironment.data()); instance != nullptr) {
    result += "-" + std::string{instance};
  }
  return result;
}

//   source(MultiTensorSource)
//      +--[frames]--> first-sink(MultiTensorSink)
//      `--[frames, fanout only]--> second-sink(MultiTensorSink)
//   first-consumer-ready(ConsumerReadyProbe)    (isolated in first consumer process)
//   second-consumer-ready(ConsumerReadyProbe)   (fanout only; isolated in second consumer process)
//
template <SourceMode Mode>
[[nodiscard]] holoscan::ExecutionPlan compile_plan(const Topology topology,
                                                   const std::uint32_t queue_depth) {
  holoscan::Graph graph{graph_name(topology, Mode)};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto source = graph.op<MultiTensorSource<Mode>>("source");
  auto first = graph.op<MultiTensorSink>("first-sink");
  auto first_ready = graph.op<ConsumerReadyProbe>("first-consumer-ready");
  graph.add_flow(
      source->output, first->input, holoscan::ConnectionOptions{.queue_depth = queue_depth});
  std::optional<holoscan::OperatorHandle<MultiTensorSink>> second;
  std::optional<holoscan::OperatorHandle<ConsumerReadyProbe>> second_ready;
  if (topology == Topology::kCrossProcessFanout) {
    second = graph.op<MultiTensorSink>("second-sink");
    second_ready = graph.op<ConsumerReadyProbe>("second-consumer-ready");
    graph.add_flow(
        source->output, (*second)->input, holoscan::ConnectionOptions{.queue_depth = queue_depth});
  }
  if constexpr (Mode == SourceMode::kLoanBundleRestart) {
    graph.lifecycle().allow_self_restart(source);
  }
  graph.partition("producer").add(source);
  graph.partition("first-consumer").add(first);
  graph.partition("first-consumer-ready").add(first_ready);
  if (second.has_value()) {
    graph.partition("second-consumer").add(*second);
    graph.partition("second-consumer-ready").add(*second_ready);
  }

  holoscan::CompileOptions options;
  if constexpr (Mode == SourceMode::kLoanBundleRestart) {
    options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
        .max_pending_requests = 4U,
        .max_pending_transactions = 4U,
        .max_receipts = 8U,
        .max_fenced_operators = 4U,
        .max_candidate_generations = 4U,
        .max_events = 64U,
        .max_audit_records = 8U,
        .max_quiesce_records = 8U,
        .max_rollback_records = 8U,
    };
  }
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "frames",
      .device = kDevice,
  });
  if (topology != Topology::kProcessLocal) {
    holoscan::detail::ProcessPlacementBinding process_placement;
    const std::array source_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "source"},
    };
    const std::array first_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "first-sink"},
        holoscan::detail::GraphObjectSelector{.graph_path = "first-consumer-ready"},
    };
    process_placement
        .bind_process_role("producer",
                           holoscan::detail::ProcessPlacement{
                               .members = source_selector,
                               .launch_group = "producer-process",
                           })
        .bind_process_role("first-consumer",
                           holoscan::detail::ProcessPlacement{
                               .members = first_selector,
                               .launch_group = "first-consumer-process",
                           });
    if (second.has_value()) {
      const std::array second_selector{
          holoscan::detail::GraphObjectSelector{.graph_path = "second-sink"},
          holoscan::detail::GraphObjectSelector{.graph_path = "second-consumer-ready"},
      };
      process_placement.bind_process_role("second-consumer",
                                          holoscan::detail::ProcessPlacement{
                                              .members = second_selector,
                                              .launch_group = "second-consumer-process",
                                          });
    }
    return holoscan::detail::compile_with_process_placement(
        graph, std::move(options), std::move(process_placement));
  }
  return holoscan::compile(graph, std::move(options));
}

//   source(InlineMultiTensorSource)
//      |
//      v
//   relay(InlineMultiTensorRelay)
//      |
//      v
//   sink(InlineMultiTensorSink)
//
template <holoscan::MemoryKind Kind>
[[nodiscard]] holoscan::ExecutionPlan compile_inline_partition_plan() {
  holoscan::Graph graph{"multi-tensor-schema-inline-partition"};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto source = graph.op<InlineMultiTensorSource<Kind>>("source");
  auto relay = graph.op<InlineMultiTensorRelay<Kind>>("relay");
  auto sink = graph.op<InlineMultiTensorSink<Kind>>("sink");
  graph.add_flow(
      source->output, relay->input, holoscan::ConnectionOptions{.queue_depth = kDefaultQueueDepth});
  graph.add_flow(
      relay->output, sink->input, holoscan::ConnectionOptions{.queue_depth = kDefaultQueueDepth});
  holoscan::CompileOptions options;
  if constexpr (Kind == holoscan::MemoryKind::kCudaDevice) {
    options.deployment
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "source",
            .output_port = "frames",
            .device = kDevice,
        })
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "relay",
            .output_port = "out",
            .device = kDevice,
        });
  }
  return holoscan::compile(graph, std::move(options));
}

[[nodiscard]] const holoscan::detail::ExecutionPlanState* plan_state(
    const holoscan::ExecutionPlan& plan) {
  return holoscan::detail::ExecutionPlanAccess::state(plan);
}

[[nodiscard]] bool field_contract_matches(
    const holoscan::detail::TensorFieldServiceStorageContract& field, const std::size_t index,
    const bool cross_process) {
  const std::array<DLDataType, 3U> dtypes{
      DLDataType{kDLFloat, 32U, 1U},
      DLDataType{kDLUInt, 8U, 1U},
      DLDataType{kDLInt, 16U, 1U},
  };
  constexpr std::array<std::uint32_t, 3U> ranks{2U, 1U, 2U};
  return field.field_id == kFieldIds[index] && field.tensor_index == index &&
         field.max_byte_span == kFieldBounds[index] && field.dtype.has_value() &&
         field.dtype->code == dtypes[index].code && field.dtype->bits == dtypes[index].bits &&
         field.dtype->lanes == dtypes[index].lanes && field.rank == ranks[index] &&
         field.memory_kind == kFieldMemory[index] &&
         field.device ==
             (index == 1U || (cross_process && index == 2U) ? kDevice : holoscan::DeviceId{}) &&
         field.tensor_emit_policy == holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded;
}

[[nodiscard]] bool exact_capacity(const holoscan::ExecutionPlan& plan, const Topology topology,
                                  const std::uint32_t queue_depth) {
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  const std::uint32_t consumer_count = topology == Topology::kCrossProcessFanout ? 2U : 1U;
  const bool cross_process = topology != Topology::kProcessLocal;
  const std::uint32_t producer = 1U;
  const std::uint32_t delivery = queue_depth * consumer_count;
  const std::uint32_t transport_retention = cross_process ? delivery : 0U;
  const std::uint32_t retirement = 1U;
  const std::uint32_t slots_per_field = producer + delivery + transport_retention + retirement;
  const std::uint32_t contributor_count = cross_process ? 4U : 3U;
  bool valid = true;
  const auto allocations = plan.tensor_field_allocations();
  valid &= check(allocations.size() == kFieldIds.size(),
                 "plan did not retain exactly three field pools");
  std::uint32_t total_slots = 0U;
  for (std::size_t index = 0U; index < allocations.size(); ++index) {
    const auto& allocation = allocations[index];
    valid &= check(
        allocation.field_id == kFieldIds[index] && allocation.tensor_index == index &&
            allocation.bounds.max_byte_span == kFieldBounds[index] &&
            allocation.memory_kind == kFieldMemory[index] &&
            allocation.device ==
                (index == 1U || (cross_process && index == 2U) ? kDevice : holoscan::DeviceId{}) &&
            allocation.slot_count == slots_per_field &&
            allocation.aggregate_byte_capacity == slots_per_field * kFieldBounds[index],
        "field allocation does not match the hand-derived contract");
    total_slots += allocation.slot_count;
    const std::array expected{
        std::pair{std::string{".producer"}, producer},
        std::pair{std::string{".delivery"}, delivery},
        std::pair{std::string{".transport-retention"}, transport_retention},
        std::pair{std::string{".retention"}, 0U},
        std::pair{std::string{".retirement"}, retirement},
    };
    std::uint32_t observed_count = 0U;
    for (const auto& [suffix, maximum] : expected) {
      const auto id = allocation.pool_id + suffix;
      const auto found =
          std::find_if(plan.pool_capacity_contributors().begin(),
                       plan.pool_capacity_contributors().end(),
                       [&](const auto& contributor) { return contributor.contributor_id == id; });
      if (maximum == 0U) {
        valid &= check(found == plan.pool_capacity_contributors().end(),
                       "zero-capacity contributor was materialized");
      } else {
        ++observed_count;
        valid &= check(found != plan.pool_capacity_contributors().end() &&
                           found->maximum_distinct_slots == maximum,
                       "field contributor does not match the hand-derived maximum");
      }
    }
    valid &= check(observed_count == contributor_count,
                   "field contributor count does not match the hand derivation");
  }
  valid &= check(total_slots == slots_per_field * kFieldIds.size(),
                 "aggregate field buffer count does not match the hand derivation");
  valid &= check(plan.pool_capacity_contributors().size() == contributor_count * kFieldIds.size(),
                 "plan retained an unexpected capacity contributor");

  const auto* state = plan_state(plan);
  valid &= check(state != nullptr, "accepted plan has no private execution state");
  if (state == nullptr) {
    return false;
  }
  const auto validate_services = [&](const auto& services) {
    valid &= check(services.size() == 1U, "plan did not coalesce one expected Tensor service");
    if (services.size() != 1U) {
      return;
    }
    valid &= check(services.front().tensor_fields.size() == kFieldIds.size(),
                   "Tensor service did not retain three field contracts");
    for (std::size_t index = 0U; index < services.front().tensor_fields.size(); ++index) {
      valid &=
          check(field_contract_matches(services.front().tensor_fields[index], index, cross_process),
                "Tensor service field contract mismatch");
    }
    valid &= check(services.front().max_subscribers == consumer_count,
                   "Tensor service fanout bound mismatch");
  };
  if (cross_process) {
    validate_services(state->cross_process_channel_services);
  } else {
    validate_services(state->process_local_channel_services);
  }
  if (!valid) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
  }
  return valid;
}

[[nodiscard]] bool has_diagnostic(const holoscan::ExecutionPlan& plan,
                                  const std::string_view token) {
  return std::ranges::any_of(plan.diagnostics(),
                             [token](const auto& diagnostic) { return diagnostic.token == token; });
}

// REQ: OPERATOR-036
// REQ: CHAN-123
// REQ: CHAN-125
// REQ: CHAN-126
[[nodiscard]] bool run_contract_validation(const std::uint32_t queue_depth) {
  // Key setup: compile the same generated three-field schema as local, cross-process, and
  // coalesced cross-process fanout plans with literal per-field bounds.
  const auto local = compile_plan<SourceMode::kValid>(Topology::kProcessLocal, queue_depth);
  const auto cross = compile_plan<SourceMode::kValid>(Topology::kCrossProcess, queue_depth);
  const auto fanout = compile_plan<SourceMode::kValid>(Topology::kCrossProcessFanout, queue_depth);
  const auto missing =
      compile_plan<SourceMode::kMissingStagingContract>(Topology::kCrossProcess, queue_depth);
  const auto mixed = compile_plan<SourceMode::kMixedCarriage>(Topology::kCrossProcess, queue_depth);
  if (missing.ok() || !has_diagnostic(missing, "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED")) {
    HOLOSCAN_LOG_ERROR("{}", missing.json());
  }
  if (mixed.ok() || !has_diagnostic(mixed, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED")) {
    HOLOSCAN_LOG_ERROR("{}", mixed.json());
  }

  // Key assertion: exact producer/delivery/transport/cache/retirement sums hold per field, while
  // incomplete staging and mixed descriptor/inline carriage fail closed.
  return exact_capacity(local, Topology::kProcessLocal, queue_depth) &&
         exact_capacity(cross, Topology::kCrossProcess, queue_depth) &&
         exact_capacity(fanout, Topology::kCrossProcessFanout, queue_depth) &&
         check(!missing.ok() && has_diagnostic(missing, "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED"),
               "missing per-field staging contract was admitted") &&
         check(!mixed.ok() && has_diagnostic(mixed, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"),
               "mixed descriptor/inline carriage was admitted");
}

// REQ: CHAN-125
// REQ: CHAN-126
// REQ: CHAN-127
[[nodiscard]] bool run_process_local() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: execute one heterogeneous generated value over direct process-local retention.
  const auto plan = compile_plan<SourceMode::kValid>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!exact_capacity(plan, Topology::kProcessLocal, kDefaultQueueDepth)) {
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool completed = started && ProbeState::wait_for(ProbeState::received, 1U) &&
                         ProbeState::emitted.load(std::memory_order_acquire) == 1U;
  session.request_stop();
  session.wait();
  // Key assertion: all three Tensor fields and the non-Tensor header arrive exactly once.
  const bool valid = completed && session.termination() == holoscan::RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    HOLOSCAN_LOG_ERROR("probe source_starts={} sink_starts={} emitted={} received={}",
                       ProbeState::source_starts.load(),
                       ProbeState::sink_starts.load(),
                       ProbeState::emitted.load(),
                       ProbeState::received.load());
    print_diagnostics(session);
  }
  return check(valid, "process-local multi-Tensor execution failed");
}

template <holoscan::MemoryKind Kind>
[[nodiscard]] bool run_inline_partition() {
  if constexpr (Kind == holoscan::MemoryKind::kCudaDevice) {
    if (!cuda_available()) {
      return true;
    }
  }
  ProbeState::reset();
  // Key setup: compile a three-Operator chain using an application-owned generated schema with
  // three independently pooled Tensor fields on each internal output.
  const auto plan = compile_inline_partition_plan<Kind>();
  const auto* state = plan_state(plan);
  const bool device_resources_match =
      Kind == holoscan::MemoryKind::kHost
          ? state != nullptr && state->tensor_device_resources.empty()
          : state != nullptr && state->tensor_device_resources.size() == 1U &&
                state->tensor_device_resources.front().device == kDevice &&
                state->tensor_device_resources.front().fused_linear_edges.size() == 2U;
  if (!plan.ok() || state == nullptr || !state->inline_partition_execution ||
      state->inline_partition_execution->edges.size() != 2U ||
      state->tensor_field_allocations.size() != 6U || !device_resources_match) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return check(false, "multi-Tensor schema chain was not admitted for inline execution");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U) &&
                       ProbeState::wait_for(ProbeState::sink_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool completed = started && ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: the generated multi-Tensor chain runs through the authoritative inline plan
  // without a worker-lane fallback and uses the one shared CUDA stream only when requested.
  const bool valid = completed && ProbeState::emitted.load(std::memory_order_acquire) == 1U &&
                     ProbeState::received.load(std::memory_order_acquire) == 1U &&
                     session.termination() == holoscan::RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    print_diagnostics(session);
  }
  return check(valid, "inline multi-Tensor schema execution failed");
}

[[nodiscard]] bool run_over_bound() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: compile a valid field contract whose first 24-byte Tensor exceeds its 8-byte bound.
  const auto plan =
      compile_plan<SourceMode::kOverBound>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool rejected = started && ProbeState::wait_for(ProbeState::over_bound_rejections, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: publication returns kInvalidArgument before any consumer receives the value.
  return check(rejected && ProbeState::received.load(std::memory_order_acquire) == 0U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "over-bound multi-Tensor field did not fail closed");
}

// REQ: OPERATOR-037
// REQ: CHAN-127
[[nodiscard]] bool run_invalid_bundle() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: issue malformed inventory and Tensor-view transactions against active finite pools.
  const auto plan =
      compile_plan<SourceMode::kInvalidLoanBundle>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool rejected = started && ProbeState::wait_for(ProbeState::invalid_bundle_rejections, 1U);
  const bool recovered = rejected && ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: every malformed set fails without consuming capacity, then a valid bundle
  // reaches the sink from the same pools.
  return check(recovered && ProbeState::emitted.load(std::memory_order_acquire) == 1U &&
                   ProbeState::received.load(std::memory_order_acquire) == 1U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "invalid multi-Tensor bundle requests did not recover capacity");
}

// REQ: OPERATOR-037
// REQ: CHAN-127
[[nodiscard]] bool run_bundle_restart() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: restart the source from Start before its first three-field bundle publication.
  const auto plan =
      compile_plan<SourceMode::kLoanBundleRestart>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool first_start = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(first_start, std::memory_order_release);
  ProbeState::notify();
  const bool restarted = first_start && ProbeState::wait_for(ProbeState::source_starts, 2U);
  const bool received = restarted && ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: every field pool rebinds under the new generation and one complete bundle
  // reaches the sink without a stale child binding or partial publication.
  return check(received && ProbeState::emitted.load(std::memory_order_acquire) == 1U &&
                   ProbeState::received.load(std::memory_order_acquire) == 1U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "multi-Tensor bundle did not recover across source restart");
}

#if defined(__linux__)
[[nodiscard]] bool write_byte(const int descriptor, const char value) {
  return write(descriptor, &value, sizeof(value)) == sizeof(value);
}

template <SourceMode Mode>
[[nodiscard]] bool run_consumer_worker(const int result_descriptor, const bool fanout,
                                       const std::size_t target_index) {
  ProbeState::reset();
  // Key setup: reconstruct the full plan after exec and materialize only this consumer group.
  const Topology topology = fanout ? Topology::kCrossProcessFanout : Topology::kCrossProcess;
  const auto plan = compile_plan<Mode>(topology, kDefaultQueueDepth);
  const auto* state = plan_state(plan);
  bool valid = plan.ok() && state != nullptr && state->cross_process_channel_services.size() == 1U;
  if (!valid) {
    static_cast<void>(write_byte(result_descriptor, 0));
    return false;
  }
  const auto& service = state->cross_process_channel_services.front();
  valid = target_index < service.target_process_groups.size();
  if (!valid) {
    static_cast<void>(write_byte(result_descriptor, 0));
    return false;
  }
  holoscan::RunSession consumer =
      holoscan::detail::run_async_worker(plan, service.target_process_groups[target_index]);
  const bool started = ProbeState::wait_for(ProbeState::sink_starts, 1U) &&
                       ProbeState::wait_for(ProbeState::consumer_ready, 1U) &&
                       ProbeState::source_starts.load(std::memory_order_acquire) == 0U;
  const bool ready_written = write_byte(result_descriptor, started ? 1 : 0);
  const bool received = started && ready_written && ProbeState::wait_for(ProbeState::received, 1U);
  consumer.request_stop();
  consumer.wait();
  // Key assertion: this isolated consumer imports every field and no producer Operator.
  const bool child_valid = received && ProbeState::received.load(std::memory_order_acquire) == 1U &&
                           ProbeState::source_starts.load(std::memory_order_acquire) == 0U &&
                           consumer.termination() == holoscan::RunTermination::kStopped &&
                           consumer.diagnostics().empty();
  if (!child_valid) {
    print_diagnostics(consumer);
    HOLOSCAN_LOG_ERROR("{}", plan.json());
  }
  return write_byte(result_descriptor, child_valid ? 1 : 0) && child_valid;
}

[[nodiscard]] bool read_byte(const int descriptor, char& value) {
  return read(descriptor, &value, sizeof(value)) == sizeof(value);
}

[[nodiscard]] bool reap_child(const pid_t child) {
  int status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

template <SourceMode Mode>
// Process orchestration is intentionally linear so each fork/exec cleanup path remains visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool run_cross_process(const bool fanout) {
  if (!cuda_available()) {
    return true;
  }
  const std::string instance = std::to_string(static_cast<std::uint64_t>(getpid()));
  if (setenv(kInstanceEnvironment.data(), instance.c_str(), 1) != 0) {
    return check(false, "failed to set multi-Tensor service instance");
  }
  ProbeState::reset();
  const Topology topology = fanout ? Topology::kCrossProcessFanout : Topology::kCrossProcess;
  // Key setup: exec one independently compiled consumer per fanout arm before starting producer.
  const auto plan = compile_plan<Mode>(topology, kDefaultQueueDepth);
  if (!exact_capacity(plan, topology, kDefaultQueueDepth)) {
    return false;
  }
  const auto* state = plan_state(plan);
  if (state == nullptr || state->cross_process_channel_services.size() != 1U) {
    return check(false, "cross-process plan has no unique multi-Tensor service");
  }
  const auto& service = state->cross_process_channel_services.front();
  const std::size_t consumer_count = fanout ? 2U : 1U;
  if (service.target_process_groups.size() != consumer_count) {
    return check(false, "cross-process plan has incorrect consumer-group geometry");
  }

  std::array<std::array<int, 2U>, 2U> pipes{};
  std::array<pid_t, 2U> children{};
  bool valid = true;
  for (std::size_t index = 0U; index < consumer_count; ++index) {
    if (pipe(pipes[index].data()) != 0) {
      valid = false;
      break;
    }
    const pid_t child = fork();
    if (child < 0) {
      valid = false;
      break;
    }
    if (child == 0) {
      close(pipes[index][0]);
      const std::string descriptor = std::to_string(pipes[index][1]);
      const std::string target = std::to_string(index);
      execl("/proc/self/exe",
            "multi_tensor_schema_runtime_validation",
            "consumer",
            descriptor.c_str(),
            fanout ? "fanout" : "cross",
            target.c_str(),
            Mode == SourceMode::kLoanBundle ? "bundle" : "owning",
            static_cast<char*>(nullptr));
      std::_Exit(127);
    }
    children[index] = child;
    close(pipes[index][1]);
  }
  for (std::size_t index = 0U; valid && index < consumer_count; ++index) {
    char ready = 0;
    valid &= read_byte(pipes[index][0], ready) && ready == 1;
  }

  if (valid) {
    holoscan::RunSession producer =
        holoscan::detail::run_async_worker(plan, service.source_process_group);
    const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
    // The source gate keeps its first publication behind producer-endpoint creation. Allow the
    // already-running consumer endpoints to complete discovery before releasing that one
    // non-retryable fanout publication.
    std::this_thread::sleep_for(300ms);
    ProbeState::source_released.store(started, std::memory_order_release);
    ProbeState::notify();
    const bool emitted = started && ProbeState::wait_for(ProbeState::emitted, 1U) &&
                         ProbeState::sink_starts.load(std::memory_order_acquire) == 0U;
    for (std::size_t index = 0U; index < consumer_count; ++index) {
      char result = 0;
      valid &= read_byte(pipes[index][0], result) && result == 1;
    }
    producer.request_stop();
    producer.wait();
    // Key assertion: one producer publication is imported exactly once by every process arm.
    valid &= emitted && producer.termination() == holoscan::RunTermination::kStopped &&
             producer.diagnostics().empty();
  }
  for (std::size_t index = 0U; index < consumer_count; ++index) {
    if (pipes[index][0] != 0) {
      close(pipes[index][0]);
    }
    if (children[index] > 0) {
      valid &= reap_child(children[index]);
    }
  }
  return check(valid, "cross-process multi-Tensor execution failed");
}
#else
template <SourceMode Mode>
[[nodiscard]] bool run_cross_process(bool) {
  // Cross-address-space validation uses the Linux process API.
  return true;
}
#endif

[[nodiscard]] std::optional<std::uint32_t> parse_uint(const std::string_view value) {
  const std::string owned{value};
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(owned.c_str(), &end, 10);
  if (end == owned.c_str() || *end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] std::optional<std::uint32_t> parse_positive(const std::string_view value) {
  const auto parsed = parse_uint(value);
  return parsed && *parsed != 0U ? parsed : std::nullopt;
}

}  // namespace

// Scenario dispatch is intentionally centralized in this standalone validation executable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(const int argc, char** argv) {
  if (argc < 2) {
    return 2;
  }
  const std::string_view scenario{argv[1]};
  const auto scenario_result = [](const bool success) {
    return cuda_scenario_skipped ? 77 : (success ? 0 : 1);
  };
  if (scenario == "contracts" && argc == 2) {
    return scenario_result(run_contract_validation(kDefaultQueueDepth));
  }
  if (scenario == "capacity" && argc == 3) {
    const auto depth = parse_positive(argv[2]);
    return scenario_result(depth && run_contract_validation(*depth));
  }
  if (scenario == "process-local" && argc == 2) {
    return scenario_result(run_process_local());
  }
  if (scenario == "inline-partition" && argc == 2) {
    return scenario_result(run_inline_partition<holoscan::MemoryKind::kHost>());
  }
  if (scenario == "inline-partition-cuda" && argc == 2) {
    return scenario_result(run_inline_partition<holoscan::MemoryKind::kCudaDevice>());
  }
  if (scenario == "over-bound" && argc == 2) {
    return scenario_result(run_over_bound());
  }
  if (scenario == "bundle-invalid" && argc == 2) {
    return scenario_result(run_invalid_bundle());
  }
  if (scenario == "bundle-restart" && argc == 2) {
    return scenario_result(run_bundle_restart());
  }
  if (scenario == "cross-process" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kValid>(false));
  }
  if (scenario == "fanout" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kValid>(true));
  }
  if (scenario == "bundle-cross-process" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kLoanBundle>(false));
  }
  if (scenario == "bundle-fanout" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kLoanBundle>(true));
  }
#if defined(__linux__)
  if (scenario == "consumer" && argc == 6) {
    const auto descriptor = parse_positive(argv[2]);
    const auto target = parse_uint(argv[4]);
    if (!descriptor || !target) {
      return 1;
    }
    const bool fanout = std::string_view(argv[3]) == "fanout";
    const bool valid =
        std::string_view(argv[5]) == "bundle"
            ? run_consumer_worker<SourceMode::kLoanBundle>(
                  static_cast<int>(*descriptor), fanout, static_cast<std::size_t>(*target))
            : run_consumer_worker<SourceMode::kValid>(
                  static_cast<int>(*descriptor), fanout, static_cast<std::size_t>(*target));
    return valid ? 0 : 1;
  }
#endif
  return 2;
}
