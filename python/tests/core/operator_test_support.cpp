/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <unistd.h>
#include <holomq/context_builder.hpp>
#include "../../../src/channel/holomq_tensor.hpp"

#include <holoscan/authoring/register_graph.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/params/register_operator.hpp>

#include "../../holoscan/timeout.hpp"
#include "phase_counter_support.hpp"
#include "pipeline_test_operators.hpp"
#include "pod_test_operators.hpp"

namespace {

// Manufacture a real imported HoloMQ lease for CuPy stream-exchange regressions.
class HoloMQTensorImportProbe {
 public:
  explicit HoloMQTensorImportProbe(bool pinned_host) {
    if (cudaFree(nullptr) != cudaSuccess) {
      throw std::runtime_error("HoloMQ probe CUDA context initialization failed");
    }
    static std::atomic_uint64_t next{};
    socket_ = "/tmp/hsdk_py_hmq_" + std::to_string(getpid()) + "_" +
              std::to_string(next.fetch_add(1U)) + ".sock";
    auto context =
        holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_).create();
    if (!context) {
      throw std::runtime_error("HoloMQ probe context creation failed");
    }
    auto& active_context = context_.emplace(std::move(*context));
    auto pool = pinned_host ? active_context.create_host_pinned_buffer_pool(
                                  64U, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                            : active_context.create_device_buffer_pool(64U, 1U);
    if (!pool) {
      throw std::runtime_error("HoloMQ probe pool creation failed");
    }
    pool_.emplace(std::move(*pool));
    if (cudaStreamCreateWithFlags(&producer_stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("HoloMQ probe producer stream creation failed");
    }
  }
  ~HoloMQTensorImportProbe() {
    static_cast<void>(::unlink(socket_.c_str()));
    // CUDA releases a destroyed stream once its pending work, which a test gate may hold, is done.
    static_cast<void>(cudaStreamDestroy(producer_stream_));
  }
  // A CUDA writer fills the slot on the producer stream and records the readiness fence after
  // the fill, as a producer in another process would. A host writer finishes the fill first and
  // records no fence.
  holoscan::Tensor import_tensor(bool cuda_writer) {
    if (!pool_.has_value() || !context_.has_value()) {
      throw std::runtime_error("HoloMQ probe is not initialized");
    }
    auto owner = pool_->acquire();
    if (!owner || (cuda_writer ? cudaMemsetAsync(owner->ptr(), 7, 64U, producer_stream_)
                               : cudaMemset(owner->ptr(), 7, 64U)) != cudaSuccess) {
      throw std::runtime_error("HoloMQ probe slot is not available");
    }
    auto handle = cuda_writer ? owner->export_handle(producer_stream_)
                              : owner->export_handle_unsynchronized();
    if (!handle) {
      throw std::runtime_error("HoloMQ probe export failed");
    }
    // Preserve the mapping without holding the slot, so a failed retirement assertion
    // cannot turn cleanup of its still-pending reader into a sticky GPU fault.
    if (!cached_mapping_) {
      auto cached = context_->import_buffer_unsynchronized(*handle);
      if (!cached) {
        throw std::runtime_error("HoloMQ probe mapping import failed");
      }
      auto& mapping = cached_mapping_.emplace(std::move(*cached));
      mapping.release();
    }
    auto imported = context_->import_buffer_unsynchronized(*handle);
    if (!imported) {
      throw std::runtime_error("HoloMQ probe import failed");
    }
    const std::array<std::int64_t, 1U> shape{64};
    auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
        shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, std::move(*imported));
    if (!tensor) {
      throw std::runtime_error("HoloMQ probe Tensor construction failed");
    }
    return std::move(*tensor);
  }
  bool slot_available() { return pool_.has_value() && pool_->acquire().has_value(); }
  [[nodiscard]] std::uintptr_t producer_stream() const noexcept {
    return reinterpret_cast<std::uintptr_t>(producer_stream_);
  }

 private:
  cudaStream_t producer_stream_{};
  std::string socket_;
  std::optional<holomq::Context> context_;
  std::optional<holomq::BufferPool> pool_;
  std::optional<holomq::ConstBuffer> cached_mapping_;
};

// A GPU memory wait leaves CUDA's host-callback thread free to retire other streams.
class CUDATensorValueGate {
 public:
  CUDATensorValueGate() {
    auto status =
        cudaHostAlloc(reinterpret_cast<void**>(&word_), sizeof(*word_), cudaHostAllocMapped);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cudaHostGetDevicePointer(&device_word_, word_, 0U);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      static_cast<void>(cudaFreeHost(word_));
      throw std::runtime_error(cudaGetErrorString(status));
    }
    *word_ = 0U;
  }
  CUDATensorValueGate(const CUDATensorValueGate&) = delete;
  CUDATensorValueGate& operator=(const CUDATensorValueGate&) = delete;
  ~CUDATensorValueGate() {
    release();
    if (queued_ && (!recorded_ || cudaEventSynchronize(event_) != cudaSuccess)) {
      return;  // Failed completion must retain memory still potentially read by the GPU.
    }
    static_cast<void>(cudaEventDestroy(event_));
    static_cast<void>(cudaFreeHost(word_));
  }
  void block(std::uintptr_t pointer) {
    if (queued_) {
      throw std::runtime_error("GPU memory gate accepts one wait");
    }
    const auto stream = reinterpret_cast<cudaStream_t>(pointer);
    if (cuStreamWaitValue32(
            stream, reinterpret_cast<CUdeviceptr>(device_word_), 1U, CU_STREAM_WAIT_VALUE_EQ) !=
        CUDA_SUCCESS) {
      throw std::runtime_error("GPU memory gate wait enqueue failed");
    }
    queued_ = true;
    const auto status = cudaEventRecord(event_, stream);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    recorded_ = true;
  }
  void release() noexcept {
    std::atomic_ref<std::uint32_t>(*word_).store(1U, std::memory_order_release);
  }

 private:
  std::uint32_t* word_{};
  void* device_word_{};
  cudaEvent_t event_{};
  bool queued_{};
  bool recorded_{};
};

// Keep a CUDA stream pending deterministically without running any Python or CUDA API in the
// host function. The callback retains its state, so failed assertions can release the gate safely.
class CUDATensorGate {
 public:
  CUDATensorGate() : state_(std::make_shared<State>()) {}
  ~CUDATensorGate() { release(); }

  void block(std::uintptr_t pointer) {
    auto callback_state = std::make_unique<std::shared_ptr<State>>(state_);
    const auto status = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(pointer),
        [](void* data) {
          const std::unique_ptr<std::shared_ptr<State>> owner{
              static_cast<std::shared_ptr<State>*>(data)};
          auto& state = **owner;
          std::unique_lock lock{state.mutex};
          state.started = true;
          state.changed.notify_all();
          state.changed.wait(lock, [&] { return state.released; });
        },
        callback_state.get());
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    // The accepted CUDA host callback now owns the state passed through callback_state.get().
    static_cast<void>(callback_state.release());  // NOLINT(bugprone-unused-return-value)
  }

  bool wait_started(double timeout) {
    const nanobind::gil_scoped_release release;
    std::unique_lock lock{state_->mutex};
    return state_->changed.wait_for(
        lock, std::chrono::duration<double>{timeout}, [&] { return state_->started; });
  }

  void release() noexcept {
    const std::lock_guard lock{state_->mutex};
    state_->released = true;
    state_->changed.notify_all();
  }

 private:
  struct State {
    std::mutex mutex;
    std::condition_variable changed;
    bool started{};
    bool released{};
  };
  std::shared_ptr<State> state_;
};

}  // namespace

namespace {
struct TemporalTensorProbe {
  std::optional<CUDATensorGate> gate{};
  void* result{};
  std::atomic<std::size_t> callbacks{};
  std::atomic<std::uintptr_t> stream{};
};
TemporalTensorProbe temporal_tensor_probe;
}  // namespace

namespace holoscan::python::test {

struct PythonChannelClientPod {
  std::uint32_t value{};
};

}  // namespace holoscan::python::test

namespace holoscan {

template <>
struct ChannelTypeIdentity<python::test::PythonChannelClientPod> {
  static constexpr std::string_view name = "holoscan.test.PythonChannelClientPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<python::test::PythonChannelClientPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &python::test::PythonChannelClientPod::value,
                           offsetof(python::test::PythonChannelClientPod, value)));
};

}  // namespace holoscan

namespace holoscan::python::test {
namespace {

// The operator submits its own native restart; Python only observes the canonical receipt.
class SelfRestartProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "in");
    spec.output(output_, "out");
  }
  void start() override { ++starts_; }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input_.receive();
    if (!sample)
      return make_unexpected(std::move(sample).error());
    if (sample->data == 1U) {
      auto requested = context.request_restart({.from_stage = LifecycleStage::kStart});
      if (!requested)
        return make_unexpected(std::move(requested).error());
      const std::lock_guard lock{receipt_mutex_};
      receipt_ = *requested;
      receipt_changed_.notify_all();
    }
    return output_.emit(sample->data * starts_);
  }
  static LifecycleRestartReceipt receipt(std::chrono::nanoseconds timeout) {
    std::unique_lock lock{receipt_mutex_};
    receipt_changed_.wait_for(lock, timeout, [] { return receipt_.has_value(); });
    if (!receipt_)
      throw RuntimeError{Error{ErrorCode::kTimeout}};
    return *receipt_;
  }
  static void register_for_test() {
    const auto result = authoring::authoring_registry().register_operator(
        authoring::detail::make_registered_operator<SelfRestartProbe>(
            "holoscan.test.SelfRestartProbe"));
    if (!result)
      throw RuntimeError{result.error()};
    const std::lock_guard lock{receipt_mutex_};
    receipt_.reset();
  }

 private:
  Input<std::uint32_t> input_;
  Output<std::uint32_t> output_;
  std::uint32_t starts_{};
  inline static std::mutex receipt_mutex_;
  inline static std::condition_variable receipt_changed_;
  inline static std::optional<LifecycleRestartReceipt> receipt_;
};

// Notifications leave the finite Tensor source idle between captures, so restart cannot
// discard a queued output and accidentally invalidate the native Retain completion proof.
class TensorRestartSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "out")
        .produces_tensor({.representation = {.memory_kind = MemoryKind::kHost,
                                             .dtype = DLDataType{kDLUInt, 8, 1},
                                             .rank = 1U},
                          .bounds = tensor_bounds(4U),
                          .storage = TensorOutputStorage::kRuntimePool});
    spec.notification_source(step_, "step").capacity(1U).sender_reference_capacity(2U);
    spec.lifecycle()
        .stage(LifecycleStage::kAllocate, &TensorRestartSource::allocate)
        .stage(LifecycleStage::kArm, &TensorRestartSource::arm);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = step_}).complete_after(ComputeCalls{.limit = 2U});
    return result;
  }
  LifecycleStatus allocate(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(step_);
    if (!sender) {
      context.report_error(sender.error());
      return LifecycleStatus::kFatalFailure;
    }
    const std::lock_guard lock{mutex_};
    sender_ = std::move(*sender);
    ++arms_;
    changed_.notify_all();
    return LifecycleStatus::kOk;
  }
  void stop() override {
    const std::lock_guard lock{mutex_};
    sender_ = {};
  }
  expected<void, Error> compute(ExecutionContext&) override {
    const std::array<std::int64_t, 1U> shape{4};
    auto loan = output_.allocate_tensor(shape);
    if (!loan)
      return make_unexpected(std::move(loan).error());
    auto filled = loan->write_host_and_commit([&](const auto& writer) {
      std::memset(writer.data(), static_cast<int>(11U * (next_ + 1U)), writer.byte_size());
    });
    if (!filled)
      return filled;
    auto emitted = output_.emit(std::move(*loan));
    if (emitted)
      ++next_;
    return emitted;
  }
  static void register_for_test() {
    static std::once_flag registered;
    std::call_once(registered, [] {
      const auto result = authoring::authoring_registry().register_operator(
          authoring::detail::make_registered_operator<TensorRestartSource>(
              "holoscan.test.TensorRestartSource"));
      if (!result)
        throw RuntimeError{result.error()};
    });
    const std::lock_guard lock{mutex_};
    sender_ = {};
    arms_ = 0U;
  }
  static bool post_for_test(std::size_t generation, std::chrono::nanoseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    NotificationSender sender;
    {
      std::unique_lock lock{mutex_};
      if (!changed_.wait_until(lock, deadline, [&] { return arms_ >= generation; }))
        return false;
      sender = sender_;
    }
    // ARM supplies the sender before START admits notifications. Retry only native
    // non-admission outcomes, sharing the caller's original bounded deadline.
    auto posted = sender.post();
    while (!posted &&
           (posted.error().code == ErrorCode::kNotReady ||
            posted.error().code == ErrorCode::kBackpressured) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
      posted = sender.post();
    }
    if (!posted)
      throw RuntimeError{posted.error()};
    return true;
  }

 private:
  Output<Tensor> output_;
  NotificationSource step_;
  std::size_t next_{};
  inline static std::mutex mutex_;
  inline static std::condition_variable changed_;
  inline static NotificationSender sender_;
  inline static std::size_t arms_{};
};

[[nodiscard]] nanobind::tuple tensor_client_catalogs(const std::string& endpoint_name,
                                                     const std::string& memory_kind_name) {
  const MemoryKind memory_kind = [&] {
    if (memory_kind_name == "host") {
      return MemoryKind::kHost;
    }
    if (memory_kind_name == "pinned_host") {
      return MemoryKind::kPinnedHost;
    }
    if (memory_kind_name == "cuda_device") {
      return MemoryKind::kCudaDevice;
    }
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "TEST_TENSOR_MEMORY_KIND_INVALID"}};
  }();
  const std::array<std::int64_t, 2U> max_shape{2, 4};
  const TensorBoundaryContract tensor{
      .representation =
          TensorRepresentation{
              .memory_kind = memory_kind,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .rank = 2U,
              .strides = TensorStridePolicy::kContiguousOnly,
          },
      .bounds = tensor_bounds(max_shape),
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
          },
  };
  const TopicOptions import_options{
      .tensor = tensor,
      .bounds =
          EndpointBounds{
              .publisher = PublisherEndpointBounds{.max_loaned_samples = 1U},
          },
  };
  const TopicOptions export_options{
      .tensor = tensor,
      .bounds =
          EndpointBounds{
              .subscriber = SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U},
          },
  };
  auto import_builder = ChannelEndpointCatalog::builder("python-tensor-client-import");
  auto export_builder = ChannelEndpointCatalog::builder("python-tensor-client-export");
  const std::string uri = "holomq://" + endpoint_name;
  if (auto added = import_builder.import_topic<Tensor>(endpoint_name, uri, import_options);
      !added) {
    throw RuntimeError{std::move(added).error()};
  }
  if (auto added = export_builder.export_topic<Tensor>(endpoint_name, uri, export_options);
      !added) {
    throw RuntimeError{std::move(added).error()};
  }
  auto import_catalog = std::move(import_builder).build();
  if (!import_catalog) {
    throw RuntimeError{std::move(import_catalog).error()};
  }
  auto export_catalog = std::move(export_builder).build();
  if (!export_catalog) {
    throw RuntimeError{std::move(export_catalog).error()};
  }
  return nanobind::make_tuple(import_catalog->json(), export_catalog->json());
}

[[nodiscard]] std::string pod_client_catalog(const std::string& endpoint_name) {
  auto builder = ChannelEndpointCatalog::builder("python-pod-client-import");
  const std::string uri = "holomq://" + endpoint_name;
  if (auto added = builder.import_topic<PythonChannelClientPod>(endpoint_name, uri); !added) {
    throw RuntimeError{std::move(added).error()};
  }
  auto catalog = std::move(builder).build();
  if (!catalog) {
    throw RuntimeError{std::move(catalog).error()};
  }
  return catalog->json();
}

// Provides an incompatible input for test_authoring_rejections_preserve_core_tokens.
class TensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// Exercises imported CuPy ownership in a mixed Python/C++ CUDA graph.
class CUDATensorForward : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "in").queue_depth(1U);
    spec.output(output, "out")
        .produces_tensor(TensorOutputSpec{
            .representation = TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
            .bounds = TensorBounds{.max_rank = 8U, .max_byte_span = 1024U * 1024U * 1024U},
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if (cudaLaunchHostFunc(context.cuda_stream().get(), [](void*) {}, nullptr) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return output.emit(std::move(*sample));
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Reuse and release a Tensor from the preceding activation on the native compute stream.
class CUDATensorTemporal final : public CUDATensorForward {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if (previous_) {
      auto& gate = temporal_tensor_probe.gate;
      if (!gate.has_value()) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      gate->block(reinterpret_cast<std::uintptr_t>(context.cuda_stream().get()));
      if (cudaMemcpyAsync(temporal_tensor_probe.result,
                          previous_->data(),
                          static_cast<std::size_t>(previous_->nbytes()),
                          cudaMemcpyDeviceToDevice,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      previous_.reset();
    } else {
      previous_.emplace(sample->data);
    }
    temporal_tensor_probe.stream.store(
        reinterpret_cast<std::uintptr_t>(context.cuda_stream().get()), std::memory_order_relaxed);
    temporal_tensor_probe.callbacks.fetch_add(1U, std::memory_order_release);
    return output.emit(std::move(*sample));
  }

 private:
  std::optional<Tensor> previous_;
};

enum class NativeCompletionRole : std::size_t { kCopyThenRead, kFirstConsumer, kSecondConsumer };

struct NativeCompletionProbe {
  std::optional<CUDATensorGate> gate{};
  void* result{};
  void* host_copy{};
  const void* first{};
  cudaStream_t shared_stream{};
  std::size_t bytes{};
  std::mutex mutex;
  std::condition_variable changed;
  bool source_advanced{};
  bool second_admitted{};
  bool first_completed{};
  std::array<std::atomic<std::size_t>, 3U> callbacks{};
};

NativeCompletionProbe& native_completion_probe() {
  static NativeCompletionProbe probe;
  return probe;
}

// These native callbacks deliberately coalesce multiple admissions/completions on one stream.
// Copies take an early completion snapshot; runtime read-lease release must snapshot again.
// REQ: TENSOR-STORAGE-010
template <NativeCompletionRole Role>
class CUDATensorCompletionSink final : public CUDATensorForward {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    auto& probe = native_completion_probe();
    if (sample->data.data() != probe.first) {
      probe.callbacks[static_cast<std::size_t>(Role)].fetch_add(1U, std::memory_order_release);
      return {};
    }
    {
      std::unique_lock lock{probe.mutex};
      if (probe.shared_stream != nullptr && probe.shared_stream != context.cuda_stream().get()) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      probe.shared_stream = context.cuda_stream().get();
      if constexpr (Role == NativeCompletionRole::kSecondConsumer) {
        // Admission has already registered this second use before either completion.
        probe.second_admitted = true;
        probe.changed.notify_all();
      }
      if (!probe.changed.wait_for(lock, std::chrono::seconds{5}, [&] {
            if (!probe.source_advanced) {
              return false;
            }
            if constexpr (Role == NativeCompletionRole::kFirstConsumer) {
              return probe.second_admitted;
            }
            if constexpr (Role == NativeCompletionRole::kSecondConsumer) {
              return probe.first_completed;
            }
            return true;
          })) {
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
    }
    if constexpr (Role != NativeCompletionRole::kSecondConsumer) {
      auto copied =
          sample->data.copy_to_host_async(probe.host_copy, probe.bytes, context.cuda_stream());
      if (!copied) {
        return copied;
      }
      if constexpr (Role == NativeCompletionRole::kFirstConsumer) {
        const std::lock_guard lock{probe.mutex};
        probe.first_completed = true;
        probe.changed.notify_all();
      }
    }
    if constexpr (Role != NativeCompletionRole::kFirstConsumer) {
      auto& gate = probe.gate;
      if (!gate.has_value()) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      gate->block(reinterpret_cast<std::uintptr_t>(context.cuda_stream().get()));
      if (cudaMemcpyAsync(probe.result,
                          sample->data.data(),
                          probe.bytes,
                          cudaMemcpyDeviceToDevice,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    probe.callbacks[static_cast<std::size_t>(Role)].fetch_add(1U, std::memory_order_release);
    return {};
  }
};

// Supplies healthy rows for alias-collision and reserved-name discovery tests.
// Representative test: test_operator_aliases_protect_static_and_reserved_module_names.
class DiscoveryOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

struct AttributeCollisionParams {
  std::uint32_t shared{1U};
};

HOLOSCAN_PARAM_SCHEMA(AttributeCollisionParams,
                      param_schema("holoscan.test.AttributeCollisionParams", 1),
                      field<&AttributeCollisionParams::shared>("shared"));

// Exercises fixed-property and port-before-parameter precedence in generated provider stubs.
class AttributeCollisionOperator final : public Operator<AttributeCollisionParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(name_port, "name");
    spec.output(ports_port, "ports");
    spec.output(shared_port, "shared");
    spec.output(keyword_port, "class");
    spec.output(literal_underscore_port, "class_");
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> name_port;
  Output<std::uint32_t> ports_port;
  Output<std::uint32_t> shared_port;
  Output<std::uint32_t> keyword_port;
  Output<std::uint32_t> literal_underscore_port;
};

// Throws during setup so discovery can prove that one invalid row is isolated.
// Representative test: test_registered_operator_discovery_isolates_failing_rows.
class FailingDiscoveryOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {
    throw RuntimeError{Error{ErrorCode::kFailure, "TEST_OPERATOR_DISCOVERY_FAILED"}};
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct ConfigDetectorParams {
  std::uint32_t threshold{4U};
  float gain{1.0F};
  std::uint32_t window{16U};
};

HOLOSCAN_PARAM_SCHEMA(
    ConfigDetectorParams, param_schema("holoscan.test.ConfigDetectorParams", 1),
    field<&ConfigDetectorParams::threshold>("threshold").range(1U, 64U).alias("legacy_threshold"),
    field<&ConfigDetectorParams::gain>("gain").range(0.0F, 10.0F),
    field<&ConfigDetectorParams::window>("window").range(1U, 64U));

// Supplies the config-bound sink in test_sparse_config_patch_attaches_to_graph_with_provenance.
class ConfigDetector final : public Operator<ConfigDetectorParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "in"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

enum class ParameterProbeMode : std::uint8_t {
  kDisabled = 0,
  kAutomatic = 2,
  kManual = 7,
};

struct ParameterProbeSensor {
  std::uint32_t channels{1U};
};

HOLOSCAN_PARAM_SCHEMA(ParameterProbeSensor, param_schema("holoscan.test.ParameterProbeSensor", 1),
                      field<&ParameterProbeSensor::channels>("channels").range(1U, 4U));

}  // namespace
}  // namespace holoscan::python::test

namespace holoscan::config {

template <>
struct ConfigCodec<python::test::ParameterProbeSensor>
    : params::RequiredParamSchemaCodec<python::test::ParameterProbeSensor> {};

}  // namespace holoscan::config

namespace holoscan::python::test {
namespace {

[[nodiscard]] bool parameter_probe_even(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

struct ParameterProbeParams {
  bool enabled{true};
  std::int32_t offset{};
  ParameterProbeMode mode{ParameterProbeMode::kAutomatic};
  double gain{1.0};
  std::string label{"default"};
  std::chrono::nanoseconds timeout{std::chrono::milliseconds{1}};
  std::uint32_t even{2U};
  std::vector<std::int32_t> levels{1};
  std::vector<std::string> labels{"default"};
  std::optional<std::uint32_t> optional_count{7U};
  std::vector<ParameterProbeMode> modes{ParameterProbeMode::kDisabled};
  ParameterProbeSensor sensor{};
  std::optional<ParameterProbeSensor> backup_sensor{};
  std::string secret{"unset"};
  std::string internal_note{"internal"};
  std::string transient_note{"transient"};
  std::filesystem::path output_path{};
};

HOLOSCAN_PARAM_SCHEMA(
    ParameterProbeParams, param_schema("holoscan.test.ParameterProbeParams", 1),
    field<&ParameterProbeParams::enabled>("enabled"),
    field<&ParameterProbeParams::offset>("offset").range(-4, 4),
    field<&ParameterProbeParams::mode>("mode").enum_values(
        {param_enum_value("disabled", ParameterProbeMode::kDisabled),
         param_enum_value("automatic", ParameterProbeMode::kAutomatic),
         param_enum_value("manual", ParameterProbeMode::kManual)}),
    field<&ParameterProbeParams::gain>("gain").range(0.0, 2.0),
    field<&ParameterProbeParams::label>("label")
        .encoding(ParamStringEncoding::kUtf8)
        .max_utf8_bytes(12U),
    field<&ParameterProbeParams::timeout>("timeout").range(std::chrono::nanoseconds{1},
                                                           std::chrono::seconds{10}),
    field<&ParameterProbeParams::even>("even").validator<parameter_probe_even>(
        "holoscan.test.validator.parameter-probe-even.v1"),
    field<&ParameterProbeParams::levels>("levels").max_elements(4U),
    field<&ParameterProbeParams::labels>("labels")
        .element_encoding(ParamStringEncoding::kUtf8)
        .max_element_utf8_bytes(8U)
        .max_elements(3U),
    field<&ParameterProbeParams::optional_count>("optional_count"),
    field<&ParameterProbeParams::modes>("modes")
        .enum_values({param_enum_value("disabled", ParameterProbeMode::kDisabled),
                      param_enum_value("automatic", ParameterProbeMode::kAutomatic),
                      param_enum_value("manual", ParameterProbeMode::kManual)})
        .max_elements(4U),
    field<&ParameterProbeParams::sensor>("sensor").nested<ParameterProbeSensor>(),
    field<&ParameterProbeParams::backup_sensor>("backup_sensor").nested<ParameterProbeSensor>(),
    field<&ParameterProbeParams::secret>("secret").sensitive(),
    field<&ParameterProbeParams::internal_note>("internal_note").internal(),
    field<&ParameterProbeParams::transient_note>("transient_note").transient(),
    field<&ParameterProbeParams::output_path>("output_path"));

// Exposes all supported parameter shapes to
// test_kwargs_cover_schema_value_kinds_and_redact_sensitive_values.
class ParameterProbe final : public Operator<ParameterProbeParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

struct RequiredRegistryParams {
  std::uint32_t required_count{7U};
};

HOLOSCAN_PARAM_SCHEMA(RequiredRegistryParams,
                      param_schema("holoscan.test.RequiredRegistryParams", 1),
                      field<&RequiredRegistryParams::required_count>("required_count")
                          .required_on_load()
                          .range(1U, 64U));

// Keeps a valid native default while requiring every language-facing construction path to supply
// the field explicitly.
class RequiredRegistrySource final : public Operator<RequiredRegistryParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

enum class RuntimeControlMode : std::uint8_t {
  kAutomatic = 0,
  kManual = 1,
};

struct RuntimeControlSensor {
  std::uint32_t channels{1U};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeControlSensor, param_schema("holoscan.test.RuntimeControlSensor", 1),
                      field<&RuntimeControlSensor::channels>("channels").range(1U, 4U));

struct RuntimeControlParams {
  std::int32_t scalar{1};
  std::chrono::nanoseconds interval{std::chrono::milliseconds{1}};
  std::vector<std::int32_t> runtime_levels{1};
  std::optional<std::uint32_t> runtime_optional_count{};
  RuntimeControlSensor runtime_sensor{};
  std::string runtime_label{"initial"};
  std::vector<std::int32_t> levels{1};
  std::optional<std::uint32_t> optional_count{};
  RuntimeControlMode mode{RuntimeControlMode::kAutomatic};
  std::vector<RuntimeControlMode> modes{RuntimeControlMode::kAutomatic};
  RuntimeControlSensor sensor{};
  std::optional<RuntimeControlSensor> backup_sensor{};
  std::string label{"initial"};
  std::filesystem::path output_path{};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeControlParams, param_schema("holoscan.test.RuntimeControlParams", 1),
                      field<&RuntimeControlParams::scalar>("scalar")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .range(-10, 10),
                      field<&RuntimeControlParams::interval>("interval")
                          .runtime_mutable()
                          .range(std::chrono::microseconds{1}, std::chrono::seconds{1}),
                      field<&RuntimeControlParams::runtime_levels>("runtime_levels")
                          .runtime_mutable()
                          .max_elements(4U),
                      field<&RuntimeControlParams::runtime_optional_count>("runtime_optional_count")
                          .runtime_mutable(),
                      field<&RuntimeControlParams::runtime_sensor>("runtime_sensor")
                          .runtime_mutable()
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::runtime_label>("runtime_label")
                          .runtime_mutable()
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16U),
                      field<&RuntimeControlParams::levels>("levels")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .max_elements(4U),
                      field<&RuntimeControlParams::optional_count>("optional_count")
                          .lifecycle_mutable(LifecycleStage::kConfigure),
                      field<&RuntimeControlParams::mode>("mode").runtime_mutable().enum_values(
                          {param_enum_value("automatic", RuntimeControlMode::kAutomatic),
                           param_enum_value("manual", RuntimeControlMode::kManual)}),
                      field<&RuntimeControlParams::modes>("modes")
                          .runtime_mutable()
                          .enum_values({param_enum_value("automatic",
                                                         RuntimeControlMode::kAutomatic),
                                        param_enum_value("manual", RuntimeControlMode::kManual)})
                          .max_elements(4U),
                      field<&RuntimeControlParams::sensor>("sensor")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::backup_sensor>("backup_sensor")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::label>("label")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16U),
                      field<&RuntimeControlParams::output_path>("output_path").runtime_mutable());

// Records fixed summaries of mixed C++ parameter generations without allocating in compute.
// Representative test: test_live_parameter_and_lifecycle_control_of_registered_cpp_operator.
struct RuntimeControlObservation {
  static void reset() noexcept {
    activation_count.store(0U, std::memory_order_relaxed);
    generation.store(0U, std::memory_order_relaxed);
    scalar.store(0, std::memory_order_relaxed);
    interval_ns.store(0, std::memory_order_relaxed);
    level_count.store(0U, std::memory_order_relaxed);
    level_sum.store(0, std::memory_order_relaxed);
    optional_count.store(-1, std::memory_order_relaxed);
    mode.store(0U, std::memory_order_relaxed);
    mode_count.store(0U, std::memory_order_relaxed);
    sensor_channels.store(0U, std::memory_order_relaxed);
    backup_channels.store(-1, std::memory_order_relaxed);
    label_is_live.store(false, std::memory_order_relaxed);
    start_count.store(0U, std::memory_order_relaxed);
  }

  static void record(const RuntimeControlParams& params, ParamGeneration observed_generation) {
    std::int32_t observed_level_sum{};
    for (const std::int32_t level : params.levels) {
      observed_level_sum += level;
    }
    {
      const std::lock_guard lock{mutex};
      scalar.store(params.scalar, std::memory_order_relaxed);
      interval_ns.store(params.interval.count(), std::memory_order_relaxed);
      level_count.store(params.levels.size(), std::memory_order_relaxed);
      level_sum.store(observed_level_sum, std::memory_order_relaxed);
      optional_count.store(
          params.optional_count ? static_cast<std::int64_t>(*params.optional_count) : -1,
          std::memory_order_relaxed);
      mode.store(static_cast<std::uint8_t>(params.mode), std::memory_order_relaxed);
      mode_count.store(params.modes.size(), std::memory_order_relaxed);
      sensor_channels.store(params.sensor.channels, std::memory_order_relaxed);
      backup_channels.store(
          params.backup_sensor ? static_cast<std::int64_t>(params.backup_sensor->channels) : -1,
          std::memory_order_relaxed);
      label_is_live.store(params.label == "live", std::memory_order_relaxed);
      activation_count.fetch_add(1U, std::memory_order_relaxed);
      generation.store(observed_generation, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_generation(ParamGeneration expected,
                                                std::chrono::nanoseconds timeout) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, timeout, [expected] {
      return activation_count.load(std::memory_order_relaxed) != 0U &&
             generation.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint64_t> activation_count{};
  static inline std::atomic<ParamGeneration> generation{};
  static inline std::atomic<std::int32_t> scalar{};
  static inline std::atomic<std::int64_t> interval_ns{};
  static inline std::atomic<std::size_t> level_count{};
  static inline std::atomic<std::int32_t> level_sum{};
  static inline std::atomic<std::int64_t> optional_count{-1};
  static inline std::atomic<std::uint8_t> mode{};
  static inline std::atomic<std::size_t> mode_count{};
  static inline std::atomic<std::uint32_t> sensor_channels{};
  static inline std::atomic<std::int64_t> backup_channels{-1};
  static inline std::atomic_bool label_is_live{};
  static inline std::atomic<std::uint32_t> start_count{};
};

// Provides a clock-driven registered C++ target for Python runtime-control integration tests.
// Representative test: test_live_parameter_and_lifecycle_control_of_registered_cpp_operator.
class RuntimeControlProbe final : public Operator<RuntimeControlParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &RuntimeControlProbe::on_start);
  }

  [[nodiscard]] Contract contract() const override {
    using namespace std::chrono_literals;
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RuntimeControlObservation::record(context.params<RuntimeControlParams>(),
                                      context.params_generation());
    return {};
  }

  LifecycleStatus on_start(LifecycleContext&) noexcept {
    RuntimeControlObservation::start_count.fetch_add(1U, std::memory_order_relaxed);
    return LifecycleStatus::kOk;
  }
};

// Non-installed cross-language START delay/failure fixture.
struct StartupProbeParams {
  std::uint32_t delay_ms{};
  bool fail{};
};

HOLOSCAN_PARAM_SCHEMA(StartupProbeParams, param_schema("holoscan.test.StartupProbeParams", 1),
                      field<&StartupProbeParams::delay_ms>("delay_ms").range(0U, 5000U),
                      field<&StartupProbeParams::fail>("fail"));

class StartupProbe final : public Operator<StartupProbeParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &StartupProbe::on_start);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus on_start(LifecycleContext& context) noexcept {
    const auto& params = context.params<StartupProbeParams>();
    std::this_thread::sleep_for(std::chrono::milliseconds{params.delay_ms});
    return params.fail ? LifecycleStatus::kFatalFailure : LifecycleStatus::kOk;
  }
};

struct ManualRowSourceParams {
  std::uint32_t limit{4U};
};

HOLOSCAN_PARAM_SCHEMA(ManualRowSourceParams, param_schema("holoscan.test.ManualRowSourceParams", 1),
                      field<&ManualRowSourceParams::limit>("limit").range(1U, 1'000'000U));

// Exercises manual registration of a non-default-constructible operator in
// test_manual_rows_accept_python_keyword_parameters.
class ManualRowSource final : public Operator<> {
 public:
  explicit ManualRowSource(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;

 private:
  std::uint32_t limit_{};
};

struct ManualRowSourceThunks final : authoring::ManualOperatorRegistrationThunks<ManualRowSource> {
  static authoring::OpaqueOperatorRef add_to_graph(
      Graph& graph, std::string_view name, const authoring::MaterializedParameters* params) {
    const ManualRowSourceParams defaults{};
    if (params != nullptr &&
        (params->type() == nullptr ||
         params->type()->local_type_token !=
             param_type_descriptor<ManualRowSourceParams>().local_type_token)) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    const auto* values =
        params == nullptr ? &defaults : static_cast<const ManualRowSourceParams*>(params->data());
    return make_operator_ref(graph.op<ManualRowSource>(name, values->limit));
  }

  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw RuntimeError{
        Error{ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }
};

expected<authoring::OpaqueOperatorRef, Error> add_manual_row_from_assignments(
    Graph& graph, std::string_view name,
    std::span<const authoring::ParameterAssignment> assignments) {
  auto materialized =
      authoring::materialize_parameters(param_type_descriptor<ManualRowSourceParams>(),
                                        param_schema_descriptor<ManualRowSourceParams>(),
                                        assignments);
  if (!materialized) {
    return make_unexpected(std::move(materialized).error());
  }
  return ManualRowSourceThunks::invoke_add(
      [&] { return ManualRowSourceThunks::add_to_graph(graph, name, &*materialized); });
}

// Declares before validating values so that
// test_registered_operator_failure_after_declaration_is_atomic observes registry rollback.
expected<authoring::OpaqueOperatorRef, Error> add_manual_row_validating_after_declaration(
    Graph& graph, std::string_view name,
    std::span<const authoring::ParameterAssignment> assignments) {
  auto declared = ManualRowSourceThunks::invoke_add(
      [&] { return ManualRowSourceThunks::add_to_graph(graph, name, nullptr); });
  if (!declared) {
    return declared;
  }
  auto materialized =
      authoring::materialize_parameters(param_type_descriptor<ManualRowSourceParams>(),
                                        param_schema_descriptor<ManualRowSourceParams>(),
                                        assignments);
  if (!materialized) {
    return make_unexpected(std::move(materialized).error());
  }
  return declared;
}

authoring::RegisteredOperatorType make_manual_row(std::string_view name, bool expose_descriptor) {
  using Thunks = ManualRowSourceThunks;
  return authoring::RegisteredOperatorType{
      .name = name,
      .param_type = &param_type_descriptor<ManualRowSourceParams>(),
      .param_schema =
          expose_descriptor ? param_schema_descriptor<ManualRowSourceParams>() : nullptr,
      .parameter_introspection = expose_descriptor
                                     ? authoring::OperatorParameterIntrospection::kConstructorReplay
                                     : authoring::OperatorParameterIntrospection::kPassThrough,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &add_manual_row_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  };
}

template <std::derived_from<OperatorBase> Op>
authoring::RegisteredOperatorType make_configurable_operator(std::string_view name) {
  return authoring::detail::make_registered_operator<Op>(
      name, &authoring::detail::OperatorConfigBindingThunk<Op>::bind);
}

template <std::derived_from<Graph> G>
void ensure_test_graph_registered(std::string_view name) {
  auto& registry = authoring::authoring_registry();
  if (registry.graph_type(name)) {
    return;
  }
  auto registered = registry.register_graph(authoring::detail::make_registered_graph<G>(name));
  if (!registered) {
    throw RuntimeError{registered.error()};
  }
}

template <std::derived_from<OperatorBase> Op>
void ensure_test_operator_registered(std::string_view name) {
  auto& registry = authoring::authoring_registry();
  if (registry.operator_type(name)) {
    return;
  }
  auto registered =
      registry.register_operator(authoring::detail::make_registered_operator<Op>(name));
  if (!registered) {
    throw RuntimeError{registered.error()};
  }
}

void register_conditional_graph_for_test() {
  ensure_test_graph_registered<ConditionalInterfaceGraph>(
      "holoscan.test.ConditionalInterfaceGraph");
}

void register_failing_graph_for_test() {
  ensure_test_graph_registered<FailingDiscoveryGraph>("holoscan.test.FailingDiscoveryGraph");
}

void register_scale_graph_alias_a_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.test.a.CollisionGraph");
}

void register_scale_graph_alias_b_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.test.b.CollisionGraph");
}

void register_reserved_graph_alias_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.test.GraphType");
}

void register_helper_graph_alias_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.test._aliases");
}

void register_failing_operator_for_test() {
  ensure_test_operator_registered<FailingDiscoveryOperator>(
      "holoscan.test.FailingDiscoveryOperator");
}

void register_operator_aliases_for_test() {
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.test.a.CollisionOperator");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.test.b.CollisionOperator");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.test.OperatorHandle");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.test.OperatorType");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.test.class");
}

void register_runtime_control_probe_for_test() {
  ensure_test_operator_registered<RuntimeControlProbe>("holoscan.test.RuntimeControlProbe");
}

bool wait_for_runtime_control_generation(std::uint64_t generation, double timeout_seconds) {
  const auto timeout = detail::seconds_to_timeout(timeout_seconds);
  const nanobind::gil_scoped_release release;
  return RuntimeControlObservation::wait_for_generation(generation, timeout);
}

nanobind::dict runtime_control_observation() {
  nanobind::dict result;
  result["activation_count"] =
      RuntimeControlObservation::activation_count.load(std::memory_order_relaxed);
  result["generation"] = RuntimeControlObservation::generation.load(std::memory_order_acquire);
  result["scalar"] = RuntimeControlObservation::scalar.load(std::memory_order_relaxed);
  result["interval_ns"] = RuntimeControlObservation::interval_ns.load(std::memory_order_relaxed);
  result["level_count"] = RuntimeControlObservation::level_count.load(std::memory_order_relaxed);
  result["level_sum"] = RuntimeControlObservation::level_sum.load(std::memory_order_relaxed);
  result["optional_count"] =
      RuntimeControlObservation::optional_count.load(std::memory_order_relaxed);
  result["mode"] = RuntimeControlObservation::mode.load(std::memory_order_relaxed);
  result["mode_count"] = RuntimeControlObservation::mode_count.load(std::memory_order_relaxed);
  result["sensor_channels"] =
      RuntimeControlObservation::sensor_channels.load(std::memory_order_relaxed);
  result["backup_channels"] =
      RuntimeControlObservation::backup_channels.load(std::memory_order_relaxed);
  result["label_is_live"] =
      RuntimeControlObservation::label_is_live.load(std::memory_order_relaxed);
  result["start_count"] = RuntimeControlObservation::start_count.load(std::memory_order_relaxed);
  return result;
}

// Keep the fixture catalog registrations and their error propagation together so Python test
// startup admits the complete set of related operators and graphs in a visible order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> register_test_operator_catalog() {
  try {
    auto& registry = authoring::authoring_registry();
    const auto register_row = [&](authoring::RegisteredOperatorType row) -> expected<void, Error> {
      auto registered = registry.register_operator(std::move(row));
      if (!registered) {
        return make_unexpected(std::move(registered).error());
      }
      return {};
    };
    const auto register_graph = [&](authoring::RegisteredGraphType row) -> expected<void, Error> {
      auto registered = registry.register_graph(std::move(row));
      if (!registered) {
        return make_unexpected(std::move(registered).error());
      }
      return {};
    };

    if (auto result = register_row(authoring::detail::make_registered_operator<CUDATensorTemporal>(
            "holoscan.test.CUDATensorTemporal"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<
                                   CUDATensorCompletionSink<NativeCompletionRole::kCopyThenRead>>(
            "holoscan.test.CUDATensorCopyThenRead"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<
                                   CUDATensorCompletionSink<NativeCompletionRole::kFirstConsumer>>(
            "holoscan.test.CUDATensorFirstCompletion"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<
                                   CUDATensorCompletionSink<NativeCompletionRole::kSecondConsumer>>(
            "holoscan.test.CUDATensorSecondCompletion"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<CUDATensorForward>(
            "holoscan.test.CUDATensorForward"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<CounterSource>(
            "holoscan.test.CounterSource"));
        !result) {
      return result;
    }
    if (auto result = registry.register_payload(
            authoring::detail::make_registered_payload<TestPose>("holoscan.test.Pose"));
        !result) {
      return result;
    }
    if (auto result = registry.register_payload(
            authoring::detail::make_registered_payload<TestByteRecord>("holoscan.test.ByteRecord"));
        !result) {
      return result;
    }
    if (auto result = registry.register_payload(
            authoring::detail::make_registered_payload<TestSingleton>("holoscan.test.Singleton"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<TestPoseSource>(
            "holoscan.test.PoseSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<TestPoseSink>("holoscan.test.PoseSink"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<AttributeCollisionOperator>(
                "holoscan.test.AttributeCollisionOperator"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<DoubleIt>("holoscan.test.DoubleIt"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<TopicCameraSource>(
            "holoscan.test.TopicCameraSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<TopicWindowFusion>(
            "holoscan.test.TopicWindowFusion"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<Sink>("holoscan.test.Sink"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<TickSource>("holoscan.test.TickSource"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<SparseTimestampedSource>(
                "holoscan.test.SparseTimestampedSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<CompletingSink>(
            "holoscan.test.CompletingSink"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<TensorSink>("holoscan.test.TensorSink"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<StructuredFailureSource>(
                "holoscan.test.StructuredFailureSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            make_configurable_operator<ConfigDetector>("holoscan.test.ConfigDetector"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<StartupProbe>(
            "holoscan.test.StartupProbe"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            make_configurable_operator<ParameterProbe>("holoscan.test.ParameterProbe"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<RequiredRegistrySource>(
                "holoscan.test.RequiredRegistrySource"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(make_manual_row("holoscan.test.ManualRowSourceWithDescriptor", true));
        !result) {
      return result;
    }
    if (auto result =
            register_row(make_manual_row("holoscan.test.ManualRowSourceWithoutDescriptor", false));
        !result) {
      return result;
    }
    auto validating_row =
        make_manual_row("holoscan.test.ManualRowValidatesAfterDeclaration", false);
    validating_row.add_to_graph_from_assignments = &add_manual_row_validating_after_declaration;
    if (auto result = register_row(std::move(validating_row)); !result) {
      return result;
    }
    if (auto result = register_graph(authoring::detail::make_registered_graph<ScalePairGraph>(
            "holoscan.test.ScalePairGraph"));
        !result) {
      return result;
    }
    return register_graph(authoring::detail::make_registered_graph<WrappedScaleGraph>(
        "holoscan.test.WrappedScaleGraph"));
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "AUTHORING_TEST_CATALOG_REGISTRATION_FAILED"});
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "AUTHORING_TEST_CATALOG_REGISTRATION_FAILED"});
  }
}

// Supplies every supplemental field to test_diagnostic_detail_is_typed_read_only_and_complete.
Diagnostic structured_diagnostic_fixture() {
  return Diagnostic{
      .id = DiagnosticId{19U},
      .severity = DiagnosticSeverity::kError,
      .token = "TEST_STRUCTURED_DIAGNOSTIC",
      .source = DiagnosticSource{.graph_generation = GraphGeneration{7U},
                                 .graph = GraphId{2U},
                                 .vertex = VertexId{3U},
                                 .port = PortId{4U}},
      .cause_id = DiagnosticId{5U},
      .detail =
          DiagnosticDetail{
              .operation = DiagnosticOperation::kSetup,
              .error = Error{ErrorCode::kInvalidArgument, "primary fixture cause"},
              .stage = LifecycleStage::kWarmUp,
              .phase = LifecycleTransitionPhase::kRestartForward,
              .transaction_id = 41U,
              .role = DiagnosticFailureRole::kRollback,
              .operator_path = "source",
              .port_name = "out",
              .related_source = DiagnosticSource{.graph_generation = GraphGeneration{7U},
                                                 .graph = GraphId{2U},
                                                 .vertex = VertexId{6U},
                                                 .port = PortId{8U}},
              .related_operator_path = "sink",
              .related_port_name = "in",
              .mismatches = {DiagnosticMismatch{
                  .field = "dtype", .provided = "float32", .required = "uint8"}},
              .causes = {Error{ErrorCode::kNotFound, "nested fixture cause"}},
              .primary_diagnostic_index = 2U,
              .truncated = true,
          },
  };
}

}  // namespace
}  // namespace holoscan::python::test

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(holoscan_test_support, module) {
  holoscan::python_test_support::register_phase_counter_support(module);
  nanobind::class_<HoloMQTensorImportProbe>(module, "_HoloMQTensorImportProbe")
      .def(nanobind::init<bool>(), nanobind::arg("pinned_host") = false)
      .def("import_tensor",
           &HoloMQTensorImportProbe::import_tensor,
           nanobind::arg("cuda_writer") = false)
      .def("slot_available", &HoloMQTensorImportProbe::slot_available)
      .def_prop_ro("producer_stream", &HoloMQTensorImportProbe::producer_stream);
  nanobind::class_<CUDATensorValueGate>(module, "_CUDATensorValueGate")
      .def(nanobind::init<>())
      .def("block", &CUDATensorValueGate::block)
      .def("release", &CUDATensorValueGate::release);
  nanobind::class_<CUDATensorGate>(module, "_CUDATensorGate")
      .def(nanobind::init<>())
      .def("block", &CUDATensorGate::block)
      .def("wait_started", &CUDATensorGate::wait_started)
      .def("release", &CUDATensorGate::release);
  module.def("_configure_temporal_tensor_probe", [](CUDATensorGate& gate, std::uintptr_t result) {
    temporal_tensor_probe.gate.emplace(gate);
    temporal_tensor_probe.result = reinterpret_cast<void*>(result);
    temporal_tensor_probe.callbacks.store(0U, std::memory_order_relaxed);
    temporal_tensor_probe.stream.store(0U, std::memory_order_relaxed);
  });
  module.def("_temporal_tensor_probe_status", [] {
    const auto callbacks = temporal_tensor_probe.callbacks.load(std::memory_order_acquire);
    return nanobind::make_tuple(callbacks,
                                temporal_tensor_probe.stream.load(std::memory_order_relaxed));
  });
  module.def("_configure_native_completion_probe",
             [](CUDATensorGate& gate,
                std::uintptr_t result,
                std::uintptr_t host_copy,
                std::uintptr_t first,
                std::size_t bytes) {
               auto& probe = holoscan::python::test::native_completion_probe();
               const std::lock_guard lock{probe.mutex};
               probe.gate.emplace(gate);
               probe.result = reinterpret_cast<void*>(result);
               probe.host_copy = reinterpret_cast<void*>(host_copy);
               probe.first = reinterpret_cast<const void*>(first);
               probe.bytes = bytes;
               probe.shared_stream = nullptr;
               probe.source_advanced = false;
               probe.second_admitted = false;
               probe.first_completed = false;
               for (auto& count : probe.callbacks) {
                 count.store(0U, std::memory_order_relaxed);
               }
             });
  module.def("_mark_native_completion_source_advanced", [] {
    auto& probe = holoscan::python::test::native_completion_probe();
    const std::lock_guard lock{probe.mutex};
    probe.source_advanced = true;
    probe.changed.notify_all();
  });
  module.def("_native_completion_callback_counts", [] {
    const auto& counts = holoscan::python::test::native_completion_probe().callbacks;
    return nanobind::make_tuple(counts[0].load(std::memory_order_acquire),
                                counts[1].load(std::memory_order_acquire),
                                counts[2].load(std::memory_order_acquire));
  });
  module.def("_copy_tensor_to_host_async",
             [](const holoscan::Tensor& tensor,
                std::uintptr_t destination,
                std::size_t bytes,
                std::uintptr_t stream) {
               auto copied = tensor.copy_to_host_async(reinterpret_cast<void*>(destination),
                                                       bytes,
                                                       reinterpret_cast<cudaStream_t>(stream));
               if (!copied) {
                 throw holoscan::RuntimeError{copied.error()};
               }
             });
  module.doc() = "Private Operator and reusable-Graph support for Holoscan Python tests.";
  auto registered = holoscan::python::test::register_test_operator_catalog();
  if (!registered) {
    throw holoscan::RuntimeError{registered.error()};
  }
  module.def("_reset_pipeline_observation", &holoscan::python::test::PipelineObservation::reset);
  module.def(
      "_wait_for_pipeline_observation",
      [](std::size_t expected, double timeout) {
        const auto duration = holoscan::python::detail::seconds_to_timeout(timeout);
        const nanobind::gil_scoped_release release;
        return holoscan::python::test::PipelineObservation::wait_for(expected, duration);
      },
      nanobind::arg("expected"),
      nanobind::arg("timeout"));
  module.def("_pipeline_observation_count", &holoscan::python::test::PipelineObservation::count);
  module.def("_reset_test_pose", &holoscan::python::test::TestPoseObservation::reset);
  module.def("_last_test_pose", []() -> nanobind::object {
    const auto pose = holoscan::python::test::TestPoseObservation::last();
    if (!pose) {
      return nanobind::none();
    }
    nanobind::dict result;
    result["sequence"] = pose->sequence;
    result["position"] =
        nanobind::make_tuple(pose->position[0], pose->position[1], pose->position[2]);
    result["confidence"] = pose->confidence;
    result["quadrant"] = pose->quadrant;
    result["flags"] = nanobind::make_tuple(pose->flags[0], pose->flags[1]);
    return result;
  });
  module.def("_pipeline_observation_last_value",
             &holoscan::python::test::PipelineObservation::last_value);
  module.def("_structured_diagnostic_fixture", [] {
    return nanobind::cast(holoscan::python::test::structured_diagnostic_fixture());
  });
  module.def("_tensor_client_catalogs",
             &holoscan::python::test::tensor_client_catalogs,
             nanobind::arg("endpoint_name"),
             nanobind::arg("memory_kind"));
  module.def("_pod_client_catalog",
             &holoscan::python::test::pod_client_catalog,
             nanobind::arg("endpoint_name"));
  module.def("_register_conditional_graph_for_test",
             &holoscan::python::test::register_conditional_graph_for_test);
  module.def("_register_failing_graph_for_test",
             &holoscan::python::test::register_failing_graph_for_test);
  module.def("_register_scale_graph_alias_a_for_test",
             &holoscan::python::test::register_scale_graph_alias_a_for_test);
  module.def("_register_scale_graph_alias_b_for_test",
             &holoscan::python::test::register_scale_graph_alias_b_for_test);
  module.def("_register_reserved_graph_alias_for_test",
             &holoscan::python::test::register_reserved_graph_alias_for_test);
  module.def("_register_helper_graph_alias_for_test",
             &holoscan::python::test::register_helper_graph_alias_for_test);
  module.def("_register_failing_operator_for_test",
             &holoscan::python::test::register_failing_operator_for_test);
  module.def("_register_operator_aliases_for_test",
             &holoscan::python::test::register_operator_aliases_for_test);
  module.def("_register_self_restart_probe_for_test",
             &holoscan::python::test::SelfRestartProbe::register_for_test);
  module.def(
      "_self_restart_receipt",
      [](double timeout) {
        const auto duration = holoscan::python::detail::seconds_to_timeout(timeout);
        const nanobind::gil_scoped_release release;
        return holoscan::python::test::SelfRestartProbe::receipt(duration);
      },
      nanobind::arg("timeout") = 0.0);
  module.def("_register_tensor_restart_source_for_test",
             &holoscan::python::test::TensorRestartSource::register_for_test);
  module.def(
      "_post_tensor_restart_source",
      [](std::size_t generation, double timeout) {
        const auto duration = holoscan::python::detail::seconds_to_timeout(timeout);
        const nanobind::gil_scoped_release release;
        return holoscan::python::test::TensorRestartSource::post_for_test(generation, duration);
      },
      nanobind::arg("generation"),
      nanobind::arg("timeout"));
  module.def("_register_runtime_control_probe_for_test",
             &holoscan::python::test::register_runtime_control_probe_for_test);
  module.def("_reset_runtime_control_observation",
             &holoscan::python::test::RuntimeControlObservation::reset);
  module.def("_wait_for_runtime_control_generation",
             &holoscan::python::test::wait_for_runtime_control_generation,
             nanobind::arg("generation"),
             nanobind::arg("timeout"));
  module.def("_runtime_control_observation", &holoscan::python::test::runtime_control_observation);
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
