/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/threading/thread_placement.hpp>

#include "../common/transport_benchmark_common.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"

namespace holoscan::benchmarking {

constexpr std::uint64_t kMaximumSamples = 10'000'000U;
constexpr std::size_t kMaximumTensorBytes = 1U << 30U;
constexpr DLDataType kTensorByteDtype{kDLUInt, 8U, 1U};

enum class TensorStorage { kPageableHost, kPinnedHost, kDevice };
enum class SupportThreadRole : std::uint8_t {
  kController = 0,
  kChannelPump = 1,
  kNotificationDriver = 2,
};

struct SupportThreadBinding {
  SupportThreadRole role{SupportThreadRole::kController};
  int requested_cpu{-1};
  bool required{};
  bool forbidden{};
};

struct SupportThreadEvidence {
  SupportThreadRole role{SupportThreadRole::kController};
  int requested_cpu{-1};
  bool required{};
  bool forbidden{};
  bool entered{};
  bool affinity_supported{};
  bool thread_identity_is_native{};
  bool native_tid_syscall_unavailable{};
  std::optional<std::uint64_t> native_tid;
  std::vector<std::uint32_t> observed_affinity;
  ErrorCode error_code{ErrorCode::kSuccess};
  std::string error_message;
};

/// Preserve a positive process-local thread identity and its provenance.
struct BenchmarkThreadIdentity {
  std::uint64_t value{};
  bool is_native{};
  bool native_tid_syscall_unavailable{};
};

/// Return a positive process-local identity for the calling thread and its provenance.
///
/// Linux reports the kernel TID needed by external tracing tools when the syscall succeeds. A
/// failed syscall and other platforms retain a stable hash of `std::thread::id`, which is
/// sufficient for same-process callback/thread comparisons.
[[nodiscard]] inline BenchmarkThreadIdentity benchmark_thread_identity() noexcept {
  bool native_tid_syscall_unavailable = false;
#if defined(__linux__)
  const auto native_tid = static_cast<std::int64_t>(syscall(SYS_gettid));
  if (native_tid > 0) {
    return {
        .value = static_cast<std::uint64_t>(native_tid),
        .is_native = true,
        .native_tid_syscall_unavailable = false,
    };
  }
  native_tid_syscall_unavailable = true;
#endif
  const auto hashed =
      static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  return {
      .value = hashed == 0U ? 1U : hashed,
      .is_native = false,
      .native_tid_syscall_unavailable = native_tid_syscall_unavailable,
  };
}

[[nodiscard]] inline const char* tensor_storage_name(const TensorStorage storage) noexcept {
  switch (storage) {
    case TensorStorage::kPageableHost:
      return "pageable_host";
    case TensorStorage::kPinnedHost:
      return "pinned_host";
    case TensorStorage::kDevice:
      return "device";
  }
  return "unknown";
}

[[nodiscard]] inline MemoryKind tensor_memory_kind(const TensorStorage storage) noexcept {
  switch (storage) {
    case TensorStorage::kPageableHost:
      return MemoryKind::kHost;
    case TensorStorage::kPinnedHost:
      return MemoryKind::kPinnedHost;
    case TensorStorage::kDevice:
      return MemoryKind::kCudaDevice;
  }
  return MemoryKind::kUnknown;
}

[[nodiscard]] inline MemoryStorageType tensor_memory_storage_type(
    const TensorStorage storage) noexcept {
  switch (storage) {
    case TensorStorage::kPageableHost:
      return MemoryStorageType::kSystem;
    case TensorStorage::kPinnedHost:
      return MemoryStorageType::kHost;
    case TensorStorage::kDevice:
      return MemoryStorageType::kDevice;
  }
  return MemoryStorageType::kSystem;
}

/// Benchmark-only owner for one immutable Tensor backing initialized before measurement.
class PreinitializedTensorStorage final : public TensorMemoryReference {
 public:
  [[nodiscard]] static expected<std::shared_ptr<PreinitializedTensorStorage>, Error> create(
      const std::size_t byte_count, const TensorStorage storage, const int device,
      const std::uint8_t fill, const std::size_t max_fanout_release_streams = 1U) {
    try {
      auto result = std::shared_ptr<PreinitializedTensorStorage>{new PreinitializedTensorStorage(
          max_fanout_release_streams, storage == TensorStorage::kDevice)};
      if (storage == TensorStorage::kDevice && cudaSetDevice(device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kNotReady, "failed to select Tensor device"});
      }
      auto allocated = result->buffer_.resize(
          std::make_shared<UnboundedAllocator>(), byte_count, tensor_memory_storage_type(storage));
      if (!allocated) {
        return make_unexpected(std::move(allocated.error()));
      }
      if (storage == TensorStorage::kDevice) {
        if (cudaMemset(result->buffer_.pointer(), static_cast<int>(fill), byte_count) !=
            cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure, "device Tensor initialization failed"});
        }
      } else {
        std::memset(result->buffer_.pointer(), static_cast<int>(fill), byte_count);
      }
      return result;
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }

  [[nodiscard]] void* data() const noexcept { return buffer_.pointer(); }
  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(buffer_.size());
  }

  bool set_deallocation_stream(cudaStream_t stream) override { return buffer_.set_stream(stream); }

  bool set_readiness_stream(cudaStream_t stream) override {
    readiness_stream_.store(stream, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return has_readiness_stream_contract_;
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_stream_.load(std::memory_order_acquire);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    const cudaStream_t producer_stream = readiness_stream();
    if (producer_stream == nullptr || producer_stream == consumer_stream) {
      return {};
    }
    if (cudaStreamSynchronize(producer_stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "Tensor readiness synchronization failed"});
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    const cudaStream_t producer_stream = readiness_stream();
    if (producer_stream != nullptr && cudaStreamSynchronize(producer_stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "Tensor host synchronization failed"});
    }
    return {};
  }

 private:
  explicit PreinitializedTensorStorage(const std::size_t max_fanout_release_streams,
                                       const bool has_readiness_stream_contract)
      : buffer_(MemoryBuffer::Options{
            .max_fanout_release_streams = max_fanout_release_streams,
        }),
        has_readiness_stream_contract_(has_readiness_stream_contract) {}

  MemoryBuffer buffer_;
  bool has_readiness_stream_contract_{false};
  std::atomic<cudaStream_t> readiness_stream_{nullptr};
};

[[nodiscard]] inline expected<std::shared_ptr<Tensor>, Error> make_preinitialized_byte_tensor(
    const std::size_t byte_count, const TensorStorage storage, const int device,
    const std::uint8_t fill, std::shared_ptr<PreinitializedTensorStorage>& storage_owner,
    const std::size_t max_fanout_release_streams = 1U) {
  auto owner = PreinitializedTensorStorage::create(
      byte_count, storage, device, fill, max_fanout_release_streams);
  if (!owner) {
    return make_unexpected(std::move(owner.error()));
  }
  try {
    auto context = std::make_shared<DLManagedTensorContext>();
    context->dl_shape = {static_cast<std::int64_t>(byte_count)};
    context->tensor.dl_tensor.data = (*owner)->data();
    context->tensor.dl_tensor.device = DLDevice{
        storage == TensorStorage::kDevice
            ? kDLCUDA
            : (storage == TensorStorage::kPinnedHost ? kDLCUDAHost : kDLCPU),
        storage == TensorStorage::kDevice ? device : 0,
    };
    context->tensor.dl_tensor.ndim = 1;
    context->tensor.dl_tensor.dtype = kTensorByteDtype;
    context->tensor.dl_tensor.shape = context->dl_shape.data();
    context->tensor.dl_tensor.strides = nullptr;
    context->tensor.dl_tensor.byte_offset = 0U;
    context->memory_reference = *owner;
    auto result = std::make_shared<Tensor>(std::move(context));
    storage_owner = std::move(*owner);
    return result;
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

[[nodiscard]] inline bool tensor_device_matches(const Tensor& tensor, const TensorStorage storage,
                                                const int device) noexcept {
  const DLDevice actual = tensor.device();
  switch (storage) {
    case TensorStorage::kPageableHost:
      return actual.device_type == kDLCPU && actual.device_id == 0;
    case TensorStorage::kPinnedHost:
      return actual.device_type == kDLCUDAHost && actual.device_id == 0;
    case TensorStorage::kDevice:
      return actual.device_type == kDLCUDA && actual.device_id == device;
  }
  return false;
}

[[nodiscard]] inline bool tensor_descriptor_matches(const Tensor& tensor,
                                                    const std::size_t byte_size,
                                                    const TensorStorage storage, const int device,
                                                    const bool require_read_only) noexcept {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const DLDataType dtype = tensor.dtype();
  return shape.size() == 1U && shape.front() == static_cast<std::int64_t>(byte_size) &&
         dtype.code == kTensorByteDtype.code && dtype.bits == kTensorByteDtype.bits &&
         dtype.lanes == kTensorByteDtype.lanes && tensor.nbytes() == byte_size &&
         tensor.data() != nullptr && (!require_read_only || tensor.is_read_only()) &&
         tensor_device_matches(tensor, storage, device);
}

[[nodiscard]] inline bool tensor_edge_bytes_match(const Tensor& tensor, const std::uint8_t expected,
                                                  const TensorStorage storage) noexcept {
  if (tensor.nbytes() == 0U || tensor.data() == nullptr) {
    return false;
  }
  const auto synchronized = tensor.synchronize_for_host();
  if (!synchronized) {
    return false;
  }
  std::array<std::uint8_t, 2U> observed{};
  const auto* bytes = static_cast<const std::uint8_t*>(tensor.data());
  if (storage == TensorStorage::kDevice) {
    if (cudaMemcpy(&observed[0], bytes, 1U, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&observed[1], bytes + tensor.nbytes() - 1U, 1U, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return false;
    }
  } else {
    observed = {bytes[0], bytes[tensor.nbytes() - 1U]};
  }
  return observed[0] == expected && observed[1] == expected;
}

[[nodiscard]] inline ThreadPlacement make_placement(const int cpu, const bool busy_poll) {
  ThreadPlacement placement{.wake_mode =
                                busy_poll ? WakeMode::kBusyPoll : WakeMode::kSleepingWaitset};
  if (cpu >= 0) {
    placement.affinity = CPUAffinity{{static_cast<std::uint32_t>(cpu)}};
  }
  return placement;
}

[[nodiscard]] inline const char* support_thread_role_name(const SupportThreadRole role) noexcept {
  switch (role) {
    case SupportThreadRole::kController:
      return "controller";
    case SupportThreadRole::kChannelPump:
      return "channel_pump";
    case SupportThreadRole::kNotificationDriver:
      return "notification_driver";
  }
  return "unknown";
}

class BenchmarkSupportThreadControl final : public detail::RuntimeSupportThreadControl {
 public:
  explicit BenchmarkSupportThreadControl(std::vector<SupportThreadBinding> bindings) {
    evidence_.reserve(bindings.size());
    for (const SupportThreadBinding& binding : bindings) {
      evidence_.push_back(SupportThreadEvidence{
          .role = binding.role,
          .requested_cpu = binding.requested_cpu,
          .required = binding.required,
          .forbidden = binding.forbidden,
      });
    }
  }

  [[nodiscard]] expected<void, Error> enter(const detail::RuntimeSupportThreadRole role) override {
    switch (role) {
      case detail::RuntimeSupportThreadRole::kController:
        return enter_role(SupportThreadRole::kController);
      case detail::RuntimeSupportThreadRole::kChannelPump:
        return enter_role(SupportThreadRole::kChannelPump);
    }
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "unknown Runtime support-thread role"});
  }

  [[nodiscard]] expected<void, Error> enter_notification_driver() {
    return enter_role(SupportThreadRole::kNotificationDriver);
  }

  [[nodiscard]] std::vector<SupportThreadEvidence> evidence() const {
    const std::lock_guard lock(mutex_);
    return evidence_;
  }

  [[nodiscard]] bool required_roles_entered() const {
    const std::lock_guard lock(mutex_);
    return std::all_of(evidence_.begin(), evidence_.end(), [](const auto& evidence) {
      return (!evidence.forbidden || !evidence.entered) &&
             (!evidence.required ||
              (evidence.entered && evidence.error_code == ErrorCode::kSuccess));
    });
  }

 private:
  [[nodiscard]] static std::optional<std::size_t> role_index(
      const std::span<const SupportThreadEvidence> evidence,
      const SupportThreadRole role) noexcept {
    const auto found = std::find_if(
        evidence.begin(), evidence.end(), [role](const auto& item) { return item.role == role; });
    if (found == evidence.end()) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(evidence.begin(), found));
  }

  [[nodiscard]] expected<void, Error> enter_role(const SupportThreadRole role) {
    SupportThreadEvidence observed;
    std::size_t index{};
    {
      const std::lock_guard lock(mutex_);
      const auto found = role_index(evidence_, role);
      if (!found.has_value()) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "support-thread role was not configured"});
      }
      index = *found;
      observed = evidence_[index];
    }
    observed.entered = true;
    observed.affinity_supported = false;
    const BenchmarkThreadIdentity identity = benchmark_thread_identity();
    observed.thread_identity_is_native = identity.is_native;
    observed.native_tid_syscall_unavailable = identity.native_tid_syscall_unavailable;
    observed.native_tid = identity.value;
    observed.observed_affinity.clear();
    observed.error_code = ErrorCode::kSuccess;
    observed.error_message.clear();
#if defined(__linux__)
    observed.affinity_supported = true;
    if (observed.requested_cpu >= CPU_SETSIZE) {
      return record_error(
          index,
          std::move(observed),
          Error{ErrorCode::kInvalidArgument, "support-thread CPU exceeds CPU_SETSIZE"});
    }
    if (observed.requested_cpu >= 0) {
      cpu_set_t requested;
      CPU_ZERO(&requested);
      CPU_SET(observed.requested_cpu, &requested);
      const int status = pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested);
      if (status != 0) {
        return record_error(
            index,
            std::move(observed),
            Error{status == EINVAL ? ErrorCode::kInvalidArgument : ErrorCode::kFailure,
                  "pthread_setaffinity_np failed with status " + std::to_string(status)});
      }
    }
    cpu_set_t current;
    CPU_ZERO(&current);
    const int status = pthread_getaffinity_np(pthread_self(), sizeof(current), &current);
    if (status != 0) {
      return record_error(
          index,
          std::move(observed),
          Error{ErrorCode::kFailure,
                "pthread_getaffinity_np failed with status " + std::to_string(status)});
    }
    for (std::uint32_t cpu = 0U; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(static_cast<int>(cpu), &current)) {
        observed.observed_affinity.push_back(cpu);
      }
    }
    if (observed.requested_cpu >= 0 && (observed.observed_affinity.size() != 1U ||
                                        observed.observed_affinity.front() !=
                                            static_cast<std::uint32_t>(observed.requested_cpu))) {
      return record_error(
          index,
          std::move(observed),
          Error{ErrorCode::kIncompatible, "support-thread affinity readback mismatch"});
    }
#else
    if (observed.requested_cpu >= 0) {
      return record_error(
          index,
          std::move(observed),
          Error{ErrorCode::kNotSupported, "support-thread affinity is supported only on Linux"});
    }
#endif
    {
      const std::lock_guard lock(mutex_);
      evidence_[index] = std::move(observed);
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> record_error(const std::size_t index,
                                                   SupportThreadEvidence observed, Error error) {
    observed.error_code = error.code;
    observed.error_message = error.message;
    {
      const std::lock_guard lock(mutex_);
      evidence_[index] = std::move(observed);
    }
    return make_unexpected(std::move(error));
  }

  mutable std::mutex mutex_;
  std::vector<SupportThreadEvidence> evidence_;
};

inline void emit_optional_cpu(std::ostream& output, const int cpu) {
  if (cpu < 0) {
    output << "null";
  } else {
    output << cpu;
  }
}

inline void emit_cpu_list(std::ostream& output, const std::span<const int> cpus) {
  output << '[';
  for (std::size_t index = 0U; index < cpus.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    emit_optional_cpu(output, cpus[index]);
  }
  output << ']';
}

inline void emit_support_thread_evidence(std::ostream& output,
                                         const BenchmarkSupportThreadControl& control) {
  output << '[';
  bool first = true;
  for (const SupportThreadEvidence& evidence : control.evidence()) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"role\":\"" << support_thread_role_name(evidence.role) << "\",\"requested_cpu\":";
    emit_optional_cpu(output, evidence.requested_cpu);
    output << ",\"entered\":" << (evidence.entered ? "true" : "false")
           << ",\"affinity_supported\":" << (evidence.affinity_supported ? "true" : "false")
           << ",\"thread_identity_kind\":\""
           << (evidence.entered
                   ? (evidence.thread_identity_is_native ? "native_tid" : "std_thread_id_hash")
                   : "none")
           << "\",\"native_tid_syscall_unavailable\":"
           << (evidence.native_tid_syscall_unavailable ? "true" : "false") << ",\"native_tid\":";
    if (evidence.native_tid.has_value()) {
      output << *evidence.native_tid;
    } else {
      output << "null";
    }
    output << ",\"observed_affinity\":[";
    for (std::size_t index = 0U; index < evidence.observed_affinity.size(); ++index) {
      if (index != 0U) {
        output << ',';
      }
      output << evidence.observed_affinity[index];
    }
    output << "],\"error\":";
    if (evidence.error_code == ErrorCode::kSuccess) {
      output << "null";
    } else {
      output << "{\"code\":\"" << error_code_name(evidence.error_code) << "\",\"message\":\""
             << json_escape(evidence.error_message) << "\"}";
    }
    output << '}';
  }
  output << ']';
}

[[nodiscard]] constexpr std::string_view diagnostic_severity_name(
    const DiagnosticSeverity severity) noexcept {
  switch (severity) {
    case DiagnosticSeverity::kWarning:
      return "warning";
    case DiagnosticSeverity::kError:
      return "error";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view diagnostic_operation_name(
    const DiagnosticOperation operation) noexcept {
  switch (operation) {
    case DiagnosticOperation::kUnknown:
      return "unknown";
    case DiagnosticOperation::kCompose:
      return "compose";
    case DiagnosticOperation::kConstruct:
      return "construct";
    case DiagnosticOperation::kSetup:
      return "setup";
    case DiagnosticOperation::kContract:
      return "contract";
    case DiagnosticOperation::kLifecycle:
      return "lifecycle";
    case DiagnosticOperation::kCompute:
      return "compute";
    case DiagnosticOperation::kRuntime:
      return "runtime";
    case DiagnosticOperation::kCompile:
      return "compile";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view diagnostic_failure_role_name(
    const DiagnosticFailureRole role) noexcept {
  switch (role) {
    case DiagnosticFailureRole::kPrimary:
      return "primary";
    case DiagnosticFailureRole::kRollback:
      return "rollback";
    case DiagnosticFailureRole::kRestoration:
      return "restoration";
    case DiagnosticFailureRole::kTeardown:
      return "teardown";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view lifecycle_stage_name(const LifecycleStage stage) noexcept {
  switch (stage) {
    case LifecycleStage::kConfigure:
      return "configure";
    case LifecycleStage::kDiscover:
      return "discover";
    case LifecycleStage::kAllocate:
      return "allocate";
    case LifecycleStage::kWarmUp:
      return "warm_up";
    case LifecycleStage::kArm:
      return "arm";
    case LifecycleStage::kStart:
      return "start";
    case LifecycleStage::kStop:
      return "stop";
    case LifecycleStage::kRelease:
      return "release";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view lifecycle_transition_phase_name(
    const LifecycleTransitionPhase phase) noexcept {
  switch (phase) {
    case LifecycleTransitionPhase::kNone:
      return "none";
    case LifecycleTransitionPhase::kStartupForward:
      return "startup_forward";
    case LifecycleTransitionPhase::kPreRestartTeardown:
      return "pre_restart_teardown";
    case LifecycleTransitionPhase::kRestartForward:
      return "restart_forward";
    case LifecycleTransitionPhase::kForwardRollback:
      return "forward_rollback";
    case LifecycleTransitionPhase::kRestoreOld:
      return "restore_old";
    case LifecycleTransitionPhase::kShutdownTeardown:
      return "shutdown_teardown";
  }
  return "unknown";
}

inline void emit_diagnostic_source(std::ostream& output, const DiagnosticSource& source) {
  output << "{\"graph_generation\":" << source.graph_generation
         << ",\"graph\":" << source.graph.value << ",\"vertex\":" << source.vertex.value
         << ",\"port\":" << source.port.value() << '}';
}

inline void emit_diagnostic_error(std::ostream& output, const Error& error) {
  output << "{\"code\":\"" << error_code_name(error.code)
         << "\",\"numeric_code\":" << static_cast<std::uint32_t>(error.code) << ",\"message\":\""
         << json_escape(error.message) << "\"}";
}

/// Emit complete retained Runtime diagnostic evidence after a benchmark run has terminated.
///
/// This deliberately lives in result serialization rather than Runtime instrumentation. It adds
/// no work, allocation, or branch to publication, delivery, scheduling, or callback hot paths.
inline void emit_runtime_diagnostics(std::ostream& output,
                                     const std::span<const Diagnostic> diagnostics) {
  output << '[';
  for (std::size_t index = 0U; index < diagnostics.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const Diagnostic& diagnostic = diagnostics[index];
    output << "{\"id\":" << diagnostic.id.value << ",\"severity\":\""
           << diagnostic_severity_name(diagnostic.severity) << "\",\"token\":\""
           << json_escape(diagnostic.token) << "\",\"source\":";
    emit_diagnostic_source(output, diagnostic.source);
    output << ",\"service\":\"" << json_escape(diagnostic.service) << "\",\"endpoint\":\""
           << json_escape(diagnostic.endpoint) << "\",\"json_pointer\":\""
           << json_escape(diagnostic.json_pointer) << "\",\"required\":\""
           << json_escape(diagnostic.required) << "\",\"provided\":\""
           << json_escape(diagnostic.provided) << "\",\"cause_id\":";
    if (diagnostic.cause_id.has_value()) {
      output << diagnostic.cause_id->value;
    } else {
      output << "null";
    }
    output << ",\"detail\":";
    if (!diagnostic.detail.has_value()) {
      output << "null}";
      continue;
    }

    const DiagnosticDetail& detail = *diagnostic.detail;
    output << "{\"operation\":\"" << diagnostic_operation_name(detail.operation) << "\",\"error\":";
    emit_diagnostic_error(output, detail.error);
    output << ",\"stage\":";
    if (detail.stage.has_value()) {
      output << '\"' << lifecycle_stage_name(*detail.stage) << '\"';
    } else {
      output << "null";
    }
    output << ",\"phase\":\"" << lifecycle_transition_phase_name(detail.phase)
           << "\",\"transaction_id\":" << detail.transaction_id << ",\"role\":\""
           << diagnostic_failure_role_name(detail.role) << "\",\"operator_path\":\""
           << json_escape(detail.operator_path) << "\",\"port_name\":\""
           << json_escape(detail.port_name) << "\",\"related_source\":";
    emit_diagnostic_source(output, detail.related_source);
    output << ",\"related_operator_path\":\"" << json_escape(detail.related_operator_path)
           << "\",\"related_port_name\":\"" << json_escape(detail.related_port_name)
           << "\",\"mismatches\":[";
    for (std::size_t mismatch_index = 0U; mismatch_index < detail.mismatches.size();
         ++mismatch_index) {
      if (mismatch_index != 0U) {
        output << ',';
      }
      const DiagnosticMismatch& mismatch = detail.mismatches[mismatch_index];
      output << "{\"field\":\"" << json_escape(mismatch.field) << "\",\"provided\":\""
             << json_escape(mismatch.provided) << "\",\"required\":\""
             << json_escape(mismatch.required) << "\"}";
    }
    output << "],\"causes\":[";
    for (std::size_t cause_index = 0U; cause_index < detail.causes.size(); ++cause_index) {
      if (cause_index != 0U) {
        output << ',';
      }
      emit_diagnostic_error(output, detail.causes[cause_index]);
    }
    output << "],\"primary_diagnostic_index\":";
    if (detail.primary_diagnostic_index.has_value()) {
      output << *detail.primary_diagnostic_index;
    } else {
      output << "null";
    }
    output << ",\"truncated\":" << (detail.truncated ? "true" : "false") << "}}";
  }
  output << ']';
}

[[nodiscard]] constexpr std::string_view runtime_thread_usage_role_name(
    const detail::RuntimeThreadUsageRole role) noexcept {
  switch (role) {
    case detail::RuntimeThreadUsageRole::kController:
      return "controller";
    case detail::RuntimeThreadUsageRole::kExecutionLane:
      return "execution_lanes";
    case detail::RuntimeThreadUsageRole::kChannelPump:
      return "channel_pump";
    case detail::RuntimeThreadUsageRole::kNotificationDriver:
      return "notification_driver";
    case detail::RuntimeThreadUsageRole::kCount:
      break;
  }
  return "unknown";
}

/// Emit opt-in Runtime-role CPU and context-switch evidence after observed threads stop.
inline void emit_runtime_thread_usage(std::ostream& output,
                                      const detail::RuntimePhaseCounters* counters) {
  output << "  \"runtime_thread_usage\":{";
  if (counters == nullptr) {
    output << "\"enabled\":false},\n";
    return;
  }
  const auto usage = counters->thread_usage_snapshot();
  detail::RuntimeThreadUsageRecord aggregate;
  const auto add = [](std::uint64_t& target, const std::uint64_t value) {
    target = value > std::numeric_limits<std::uint64_t>::max() - target
                 ? std::numeric_limits<std::uint64_t>::max()
                 : target + value;
  };
  for (const auto& record : usage) {
    add(aggregate.samples, record.samples);
    add(aggregate.user_cpu_ns, record.user_cpu_ns);
    add(aggregate.system_cpu_ns, record.system_cpu_ns);
    add(aggregate.voluntary_context_switches, record.voluntary_context_switches);
    add(aggregate.involuntary_context_switches, record.involuntary_context_switches);
    add(aggregate.minor_page_faults, record.minor_page_faults);
    add(aggregate.major_page_faults, record.major_page_faults);
  }
  output << "\"enabled\":true,\"supported\":"
         << (detail::runtime_thread_usage_detail::kSupported ? "true" : "false")
         << ",\"scope\":\"whole_runtime_thread_lifetime_including_startup_and_teardown\""
         << ",\"roles\":[";
  for (std::size_t index = 0U; index < usage.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const auto& record = usage[index];
    output << "{\"role\":\"" << runtime_thread_usage_role_name(record.role)
           << "\",\"supported\":" << (record.supported ? "true" : "false")
           << ",\"samples\":" << record.samples << ",\"user_cpu_ns\":" << record.user_cpu_ns
           << ",\"system_cpu_ns\":" << record.system_cpu_ns
           << ",\"voluntary_context_switches\":" << record.voluntary_context_switches
           << ",\"involuntary_context_switches\":" << record.involuntary_context_switches
           << ",\"minor_page_faults\":" << record.minor_page_faults
           << ",\"major_page_faults\":" << record.major_page_faults << '}';
  }
  output << "],\"aggregate\":{\"scope\":\"sum_of_runtime_roles\""
         << ",\"samples\":" << aggregate.samples << ",\"user_cpu_ns\":" << aggregate.user_cpu_ns
         << ",\"system_cpu_ns\":" << aggregate.system_cpu_ns
         << ",\"voluntary_context_switches\":" << aggregate.voluntary_context_switches
         << ",\"involuntary_context_switches\":" << aggregate.involuntary_context_switches
         << ",\"minor_page_faults\":" << aggregate.minor_page_faults
         << ",\"major_page_faults\":" << aggregate.major_page_faults << "}},\n";
}

}  // namespace holoscan::benchmarking
