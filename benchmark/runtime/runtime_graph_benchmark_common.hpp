/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <mutex>
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

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/threading/thread_placement.hpp>

#include "../common/transport_benchmark_common.hpp"
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

}  // namespace holoscan::benchmarking
