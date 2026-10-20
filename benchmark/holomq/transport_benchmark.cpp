/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#include <holomq/buffer.hpp>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/errors.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/subscriber_builder.hpp>

namespace {

constexpr std::uint64_t kConnectionProbeSequence = std::numeric_limits<std::uint64_t>::max();
constexpr unsigned char kDeviceBufferPattern = 0xA5U;
constexpr int kCudaUnavailableExitCode = 77;
constexpr std::size_t kMinimumP999Samples = 10'000U;

// A not-yet-connected receiver can accept up to one queue's capacity. One additional complete
// delivery proves that the receiver connected and drained at least one probe.
[[nodiscard]] constexpr std::uint64_t required_connection_probe_deliveries(
    const std::uint32_t queue_depth) noexcept {
  return static_cast<std::uint64_t>(queue_depth) + 1U;
}

enum class Workload { kThroughput, kLatency };
enum class PublisherPacing { kSleep, kSpin };
enum class PayloadKind { kInline, kBufferHandle };
enum class StorageKind { kHostPageable, kHostPinned, kDevice };
enum class DeviceSync { kNone, kPublisherReady, kSubscriberRelease, kReady };
enum class PayloadValidation { kNone, kFinal, kAll };

class BenchmarkClock {
 public:
  [[nodiscard]] static std::uint64_t now_ns() noexcept {
#if defined(__linux__)
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0U;
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
#else
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
#endif
  }

  [[nodiscard]] static constexpr const char* source() noexcept {
#if defined(__linux__)
    return "clock_gettime(CLOCK_MONOTONIC)";
#else
    return "std::chrono::steady_clock";
#endif
  }

  [[nodiscard]] static constexpr const char* scope() noexcept {
#if defined(__linux__)
    return "same_host_cross_process";
#else
    return "process_local";
#endif
  }
};

class ScopedPrimaryCudaContext {
 public:
  ScopedPrimaryCudaContext() = default;
  ScopedPrimaryCudaContext(const ScopedPrimaryCudaContext&) = delete;
  ScopedPrimaryCudaContext& operator=(const ScopedPrimaryCudaContext&) = delete;
  ~ScopedPrimaryCudaContext() {
    if (retained_) {
      static_cast<void>(cuCtxSetCurrent(nullptr));
      static_cast<void>(cuDevicePrimaryCtxRelease(device_));
    }
  }

  [[nodiscard]] bool initialize() noexcept {
    if (cuInit(0U) != CUDA_SUCCESS || cuDeviceGet(&device_, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context_, device_) != CUDA_SUCCESS) {
      return false;
    }
    retained_ = true;
    return cuCtxSetCurrent(context_) == CUDA_SUCCESS;
  }

 private:
  CUdevice device_{};
  CUcontext context_{nullptr};
  bool retained_{false};
};

class ScopedCudaStream {
 public:
  ScopedCudaStream() = default;
  ScopedCudaStream(const ScopedCudaStream&) = delete;
  ScopedCudaStream& operator=(const ScopedCudaStream&) = delete;
  ~ScopedCudaStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cuStreamDestroy(stream_));
    }
  }

  [[nodiscard]] bool create() noexcept {
    return cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS;
  }
  [[nodiscard]] bool synchronize() const noexcept {
    return stream_ != nullptr && cuStreamSynchronize(stream_) == CUDA_SUCCESS;
  }
  [[nodiscard]] CUstream get() const noexcept { return stream_; }

 private:
  CUstream stream_{nullptr};
};

struct Options {
  std::string role;
  std::string service_name;
  Workload workload{Workload::kThroughput};
  PublisherPacing publisher_pacing{PublisherPacing::kSpin};
  PayloadKind payload_kind{PayloadKind::kInline};
  StorageKind storage_kind{StorageKind::kHostPageable};
  DeviceSync device_sync{DeviceSync::kNone};
  PayloadValidation payload_validation{PayloadValidation::kFinal};
  std::uint64_t samples{100'000U};
  std::uint64_t warmup{10'000U};
  std::uint32_t payload_bytes{64U};
  std::uint32_t queue_depth{64U};
  std::uint32_t subscriber_count{1U};
  std::uint32_t subscriber_index{0U};
  std::uint32_t publisher_period_us{0U};
  std::size_t buffer_bytes{1U << 20U};
  std::uint32_t pool_slots{66U};
  std::uint32_t timeout_seconds{30U};
  int cpu{-1};
  int ready_fd{-1};
};

[[nodiscard]] holomq::PubSubServiceOptions make_service_options(const Options& options) {
  holomq::PubSubServiceOptions service_options{};
  service_options.queue_policy = holomq::QueuePolicy::kBlocking;
  service_options.max_subscribers = options.subscriber_count;
  service_options.max_subscriber_queue_depth = options.queue_depth;
  service_options.max_subscriber_borrowed_samples = options.queue_depth;
  service_options.publisher_history_depth = 0U;
  service_options.enable_event_wake = false;
  return service_options;
}

template <std::size_t PayloadBytes>
struct FixedPayload {
  static_assert(PayloadBytes >= 2U * sizeof(std::uint64_t));

  std::uint64_t sequence{};
  std::uint64_t send_time_ns{};
  std::array<std::byte, PayloadBytes - (2U * sizeof(std::uint64_t))> padding{};
};

struct BufferPayload {
  std::uint64_t sequence{};
  std::uint64_t send_time_ns{};
  std::uint64_t buffer_bytes{};
  holomq::BufferHandle buffer_handle{};
};

static_assert(std::is_standard_layout_v<BufferPayload>);
static_assert(std::is_trivially_copyable_v<BufferPayload>);

template <std::size_t PayloadBytes>
inline constexpr bool kValidFixedPayload = sizeof(FixedPayload<PayloadBytes>) == PayloadBytes &&
                                           std::is_standard_layout_v<FixedPayload<PayloadBytes>> &&
                                           std::is_trivially_copyable_v<FixedPayload<PayloadBytes>>;

static_assert(kValidFixedPayload<64U>);
static_assert(kValidFixedPayload<128U>);
static_assert(kValidFixedPayload<440U>);

void print_usage(std::ostream& output) {
  output << "usage: holomq_transport_benchmark --role publisher|subscriber [options]\n\n"
            "options:\n"
            "  --service-name NAME       unique HoloMQ service name (required)\n"
            "  --workload MODE           throughput or latency\n"
            "  --payload KIND            inline or buffer_handle\n"
            "  --storage KIND            pageable_host, pinned_host, or device for buffer_handle\n"
            "  --device-sync MODE        none, publisher_ready, subscriber_release, or ready\n"
            "  --payload-bytes N         fixed payload size: 64, 128, or 440\n"
            "  --buffer-bytes N          bytes per external buffer slot; default 1048576\n"
            "  --pool-slots N            external host buffer pool slots\n"
            "  --samples N               measured samples\n"
            "  --warmup N                unmeasured warmup samples\n"
            "  --queue-depth N           blocking subscriber queue depth\n"
            "  --subscriber-count N      expected fanout recipient count\n"
            "  --subscriber-index N      zero-based subscriber index\n"
            "  --publisher-period-us N   latency-mode publish period\n"
            "  --publisher-pacing MODE   sleep or spin\n"
            "  --payload-validation MODE none, final, or all\n"
            "  --timeout-seconds N       bounded role timeout\n"
            "  --cpu N                   CPU to pin, or omit for no pinning\n"
            "  --ready-fd N              subscriber readiness pipe descriptor\n";
}

template <typename T>
[[nodiscard]] bool parse_unsigned(const std::string_view text, T& value) noexcept {
  static_assert(std::is_unsigned_v<T>);
  if (text.empty() || text.front() == '-') {
    return false;
  }
  T parsed{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [cursor, error] = std::from_chars(begin, end, parsed);
  if (error != std::errc() || cursor != end) {
    return false;
  }
  value = parsed;
  return true;
}

[[nodiscard]] std::optional<Options> parse_options(const std::span<char*> arguments) {
  Options options;
  for (std::size_t index = 1U; index < arguments.size(); ++index) {
    const std::string_view name(arguments[index]);
    if (name == "--help") {
      print_usage(std::cout);
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1U >= arguments.size()) {
      std::cerr << "missing value for " << name << '\n';
      return std::nullopt;
    }
    const std::string_view value(arguments[++index]);
    if (name == "--role") {
      options.role = value;
    } else if (name == "--service-name") {
      options.service_name = value;
    } else if (name == "--workload") {
      if (value == "throughput") {
        options.workload = Workload::kThroughput;
      } else if (value == "latency") {
        options.workload = Workload::kLatency;
      } else {
        return std::nullopt;
      }
    } else if (name == "--payload") {
      if (value == "inline") {
        options.payload_kind = PayloadKind::kInline;
      } else if (value == "buffer_handle") {
        options.payload_kind = PayloadKind::kBufferHandle;
      } else {
        return std::nullopt;
      }
    } else if (name == "--storage") {
      if (value == "pageable_host") {
        options.storage_kind = StorageKind::kHostPageable;
      } else if (value == "pinned_host") {
        options.storage_kind = StorageKind::kHostPinned;
      } else if (value == "device") {
        options.storage_kind = StorageKind::kDevice;
      } else {
        return std::nullopt;
      }
    } else if (name == "--device-sync") {
      if (value == "none") {
        options.device_sync = DeviceSync::kNone;
      } else if (value == "publisher_ready") {
        options.device_sync = DeviceSync::kPublisherReady;
      } else if (value == "subscriber_release") {
        options.device_sync = DeviceSync::kSubscriberRelease;
      } else if (value == "ready") {
        options.device_sync = DeviceSync::kReady;
      } else {
        return std::nullopt;
      }
    } else if (name == "--publisher-pacing") {
      if (value == "sleep") {
        options.publisher_pacing = PublisherPacing::kSleep;
      } else if (value == "spin") {
        options.publisher_pacing = PublisherPacing::kSpin;
      } else {
        return std::nullopt;
      }
    } else if (name == "--payload-validation") {
      if (value == "none") {
        options.payload_validation = PayloadValidation::kNone;
      } else if (value == "final") {
        options.payload_validation = PayloadValidation::kFinal;
      } else if (value == "all") {
        options.payload_validation = PayloadValidation::kAll;
      } else {
        return std::nullopt;
      }
    } else if (name == "--payload-bytes") {
      if (!parse_unsigned(value, options.payload_bytes)) {
        return std::nullopt;
      }
    } else if (name == "--buffer-bytes") {
      if (!parse_unsigned(value, options.buffer_bytes)) {
        return std::nullopt;
      }
    } else if (name == "--pool-slots") {
      if (!parse_unsigned(value, options.pool_slots)) {
        return std::nullopt;
      }
    } else if (name == "--samples") {
      if (!parse_unsigned(value, options.samples)) {
        return std::nullopt;
      }
    } else if (name == "--warmup") {
      if (!parse_unsigned(value, options.warmup)) {
        return std::nullopt;
      }
    } else if (name == "--queue-depth") {
      if (!parse_unsigned(value, options.queue_depth)) {
        return std::nullopt;
      }
    } else if (name == "--subscriber-count") {
      if (!parse_unsigned(value, options.subscriber_count)) {
        return std::nullopt;
      }
    } else if (name == "--subscriber-index") {
      if (!parse_unsigned(value, options.subscriber_index)) {
        return std::nullopt;
      }
    } else if (name == "--publisher-period-us") {
      if (!parse_unsigned(value, options.publisher_period_us)) {
        return std::nullopt;
      }
    } else if (name == "--timeout-seconds") {
      if (!parse_unsigned(value, options.timeout_seconds)) {
        return std::nullopt;
      }
    } else if (name == "--cpu" || name == "--ready-fd") {
      std::uint32_t parsed{};
      if (!parse_unsigned(value, parsed) ||
          parsed > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
      }
      if (name == "--cpu") {
        options.cpu = static_cast<int>(parsed);
      } else {
        options.ready_fd = static_cast<int>(parsed);
      }
    } else {
      std::cerr << "unknown option: " << name << '\n';
      return std::nullopt;
    }
  }

  if ((options.role != "publisher" && options.role != "subscriber") ||
      options.service_name.empty() || options.samples == 0U || options.queue_depth == 0U ||
      options.subscriber_count == 0U || options.subscriber_index >= options.subscriber_count ||
      options.timeout_seconds == 0U ||
      (options.payload_bytes != 64U && options.payload_bytes != 128U &&
       options.payload_bytes != 440U) ||
      (options.role == "subscriber" && options.ready_fd < 0)) {
    std::cerr << "invalid benchmark options\n";
    return std::nullopt;
  }
  if (options.warmup > std::numeric_limits<std::uint64_t>::max() - options.samples) {
    std::cerr << "warmup plus samples overflows uint64\n";
    return std::nullopt;
  }
  if ((options.workload == Workload::kThroughput && options.publisher_period_us != 0U) ||
      (options.workload == Workload::kLatency &&
       (options.publisher_period_us == 0U || options.queue_depth != 1U ||
        options.subscriber_count != 1U))) {
    std::cerr << "latency requires one subscriber, queue depth 1, and a nonzero publisher period; "
                 "throughput requires publisher period 0\n";
    return std::nullopt;
  }
  if (options.payload_kind == PayloadKind::kBufferHandle &&
      (options.subscriber_count != 1U || options.buffer_bytes < 2U * sizeof(std::uint64_t) ||
       static_cast<std::uint64_t>(options.pool_slots) <
           static_cast<std::uint64_t>(options.queue_depth) + 2U)) {
    std::cerr << "buffer_handle currently requires 1:1 operation, at least 16 buffer bytes, and "
                 "at least queue depth plus two pool slots\n";
    return std::nullopt;
  }
  if (options.payload_kind == PayloadKind::kInline &&
      (options.storage_kind != StorageKind::kHostPageable ||
       options.device_sync != DeviceSync::kNone)) {
    std::cerr << "inline payload requires default storage and device synchronization\n";
    return std::nullopt;
  }
  if (options.storage_kind != StorageKind::kDevice && options.device_sync != DeviceSync::kNone) {
    std::cerr << "non-default device synchronization requires device storage\n";
    return std::nullopt;
  }
  if (options.storage_kind == StorageKind::kDevice &&
      options.payload_validation == PayloadValidation::kAll) {
    std::cerr << "device storage supports none or final payload validation\n";
    return std::nullopt;
  }
  return options;
}

[[nodiscard]] constexpr std::string_view workload_name(const Workload workload) noexcept {
  return workload == Workload::kLatency ? "latency" : "throughput";
}

[[nodiscard]] constexpr std::string_view publisher_pacing_name(
    const PublisherPacing pacing) noexcept {
  return pacing == PublisherPacing::kSleep ? "sleep" : "spin";
}

[[nodiscard]] constexpr std::string_view payload_kind_name(
    const PayloadKind payload_kind) noexcept {
  return payload_kind == PayloadKind::kBufferHandle ? "buffer_handle" : "inline";
}

[[nodiscard]] constexpr std::string_view payload_validation_name(
    const PayloadValidation validation) noexcept {
  switch (validation) {
    case PayloadValidation::kNone:
      return "none";
    case PayloadValidation::kFinal:
      return "final";
    case PayloadValidation::kAll:
      return "all";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view validation_timing_name(
    const PayloadValidation validation) noexcept {
  switch (validation) {
    case PayloadValidation::kNone:
      return "none";
    case PayloadValidation::kFinal:
      return "after_timed_operations";
    case PayloadValidation::kAll:
      return "between_timed_operations";
  }
  return "unknown";
}

[[nodiscard]] bool payload_contents_required(const Options& options, const std::uint64_t sequence,
                                             const std::uint64_t total_samples) noexcept {
  return options.payload_validation == PayloadValidation::kAll ||
         (options.payload_validation == PayloadValidation::kFinal &&
          sequence + 1U == total_samples);
}

[[nodiscard]] constexpr std::string_view storage_kind_name(
    const StorageKind storage_kind) noexcept {
  switch (storage_kind) {
    case StorageKind::kHostPageable:
      return "pageable_host";
    case StorageKind::kHostPinned:
      return "pinned_host";
    case StorageKind::kDevice:
      return "device";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view device_sync_name(const DeviceSync device_sync) noexcept {
  switch (device_sync) {
    case DeviceSync::kPublisherReady:
      return "publisher_ready";
    case DeviceSync::kSubscriberRelease:
      return "subscriber_release";
    case DeviceSync::kReady:
      return "ready";
    case DeviceSync::kNone:
      break;
  }
  return "none";
}

[[nodiscard]] constexpr bool uses_publisher_stream(const DeviceSync device_sync) noexcept {
  return device_sync == DeviceSync::kPublisherReady || device_sync == DeviceSync::kReady;
}

[[nodiscard]] constexpr bool uses_subscriber_stream(const DeviceSync device_sync) noexcept {
  return device_sync == DeviceSync::kSubscriberRelease || device_sync == DeviceSync::kReady;
}

[[nodiscard]] std::uint64_t now_ns() noexcept {
  return BenchmarkClock::now_ns();
}

void emit_clock(std::ostream& output) {
  output << ",\"clock\":{\"source\":\"" << BenchmarkClock::source()
         << "\",\"steady\":true,\"scope\":\"" << BenchmarkClock::scope() << "\"}";
}

void pace_publisher(const Options& options,
                    std::chrono::steady_clock::time_point& next_publish_time) {
  if (options.workload != Workload::kLatency) {
    return;
  }
  next_publish_time += std::chrono::microseconds(options.publisher_period_us);
  if (options.publisher_pacing == PublisherPacing::kSleep) {
    std::this_thread::sleep_until(next_publish_time);
    return;
  }
  while (std::chrono::steady_clock::now() < next_publish_time) {
  }
}

[[nodiscard]] bool pin_to_cpu(const int cpu) noexcept {
  if (cpu < 0) {
    return true;
  }
  if (cpu >= CPU_SETSIZE) {
    std::cerr << "CPU " << cpu << " exceeds CPU_SETSIZE\n";
    return false;
  }
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu, &cpu_set);
  if (::sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
    std::cerr << "sched_setaffinity failed for CPU " << cpu << ": errno=" << errno << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool signal_ready(const int ready_fd) noexcept {
  constexpr char kReady = 'R';
  for (;;) {
    const ssize_t written = ::write(ready_fd, &kReady, sizeof(kReady));
    if (written == static_cast<ssize_t>(sizeof(kReady))) {
      return true;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
}

[[nodiscard]] std::string json_escape(const std::string_view text) {
  constexpr std::string_view kHexDigits = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(text.size());
  for (const char character : text) {
    switch (character) {
      case '"':
        escaped.append("\\\"");
        break;
      case '\\':
        escaped.append("\\\\");
        break;
      case '\b':
        escaped.append("\\b");
        break;
      case '\f':
        escaped.append("\\f");
        break;
      case '\n':
        escaped.append("\\n");
        break;
      case '\r':
        escaped.append("\\r");
        break;
      case '\t':
        escaped.append("\\t");
        break;
      default: {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 0x20U) {
          escaped.append("\\u00");
          escaped.push_back(kHexDigits[byte >> 4U]);
          escaped.push_back(kHexDigits[byte & 0x0FU]);
        } else {
          escaped.push_back(character);
        }
        break;
      }
    }
  }
  return escaped;
}

template <std::size_t PayloadBytes>
[[nodiscard]] FixedPayload<PayloadBytes> make_payload(const std::uint64_t sequence) noexcept {
  FixedPayload<PayloadBytes> payload{.sequence = sequence};
  const std::byte pattern = static_cast<std::byte>(sequence & 0xFFU);
  payload.padding.fill(pattern);
  return payload;
}

template <std::size_t PayloadBytes>
[[nodiscard]] std::uint64_t checksum_payload(const FixedPayload<PayloadBytes>& payload) noexcept {
  return payload.sequence ^ payload.send_time_ns ^
         static_cast<std::uint64_t>(payload.padding.front()) ^
         (static_cast<std::uint64_t>(payload.padding.back()) << 8U);
}

template <std::size_t PayloadBytes>
[[nodiscard]] bool validate_payload(const FixedPayload<PayloadBytes>& payload) noexcept {
  const std::byte expected = static_cast<std::byte>(payload.sequence & 0xFFU);
  return std::all_of(payload.padding.begin(),
                     payload.padding.end(),
                     [expected](const std::byte value) { return value == expected; });
}

[[nodiscard]] std::uint64_t buffer_tail_marker(const std::uint64_t sequence) noexcept {
  return sequence ^ 0x9E3779B97F4A7C15ULL;
}

[[nodiscard]] std::uint64_t buffer_checksum(const std::uint64_t sequence) noexcept {
  return (sequence * 0xD6E8FEB86659FD93ULL) ^ buffer_tail_marker(sequence);
}

void write_buffer_markers(holomq::Buffer& buffer, const std::uint64_t sequence) noexcept {
  std::memcpy(buffer.ptr(), &sequence, sizeof(sequence));
  const std::uint64_t tail = buffer_tail_marker(sequence);
  auto* const bytes = static_cast<std::byte*>(buffer.ptr());
  std::memcpy(bytes + buffer.size_bytes() - sizeof(tail), &tail, sizeof(tail));
}

[[nodiscard]] bool validate_buffer_markers(const holomq::ConstBuffer& buffer,
                                           const std::uint64_t sequence) noexcept {
  std::uint64_t head{};
  std::uint64_t tail{};
  const auto* const bytes = static_cast<const std::byte*>(buffer.ptr());
  std::memcpy(&head, bytes, sizeof(head));
  std::memcpy(&tail, bytes + buffer.size_bytes() - sizeof(tail), sizeof(tail));
  return head == sequence && tail == buffer_tail_marker(sequence);
}

[[nodiscard]] CUdeviceptr device_pointer(const void* const pointer) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer));
}

[[nodiscard]] bool initialize_device_pool(holomq::BufferPool& pool, const std::uint32_t pool_slots,
                                          const std::size_t buffer_bytes, const CUstream stream) {
  std::vector<holomq::Buffer> buffers;
  buffers.reserve(pool_slots);
  for (std::uint32_t index = 0U; index < pool_slots; ++index) {
    auto acquired = pool.acquire();
    if (!acquired.has_value()) {
      return false;
    }
    holomq::Buffer buffer = std::move(acquired).value();
    if (cuMemsetD8Async(device_pointer(buffer.ptr()), kDeviceBufferPattern, buffer_bytes, stream) !=
        CUDA_SUCCESS) {
      return false;
    }
    buffers.push_back(std::move(buffer));
  }
  if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
    return false;
  }
  buffers.clear();
  return true;
}

[[nodiscard]] bool validate_device_buffer(const holomq::ConstBuffer& buffer) noexcept {
  unsigned char head{};
  unsigned char tail{};
  const CUdeviceptr base = device_pointer(buffer.ptr());
  return cuMemcpyDtoH(&head, base, sizeof(head)) == CUDA_SUCCESS &&
         cuMemcpyDtoH(&tail, base + buffer.size_bytes() - sizeof(tail), sizeof(tail)) ==
             CUDA_SUCCESS &&
         head == kDeviceBufferPattern && tail == kDeviceBufferPattern;
}

template <typename Publisher>
[[nodiscard]] bool wait_for_any_subscriber(Publisher& publisher,
                                           const std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated.has_value()) {
      std::cerr << "publisher connection update failed\n";
      return false;
    }
    if (publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

template <typename Publisher>
[[nodiscard]] auto loan_with_wait(Publisher& publisher,
                                  const std::chrono::steady_clock::time_point deadline,
                                  std::uint64_t& backpressured_count) {
  using Result = decltype(publisher.loan());
  while (std::chrono::steady_clock::now() < deadline) {
    Result loaned = publisher.loan();
    if (loaned.has_value()) {
      return loaned;
    }
    if (loaned.error().code != holomq::ErrorCode::kBackpressured) {
      return loaned;
    }
    ++backpressured_count;
    std::this_thread::yield();
  }
  return publisher.loan();
}

[[nodiscard]] auto acquire_with_wait(holomq::BufferPool& pool,
                                     const std::chrono::steady_clock::time_point deadline,
                                     std::uint64_t& backpressured_count) {
  using Result = decltype(pool.acquire());
  while (std::chrono::steady_clock::now() < deadline) {
    Result acquired = pool.acquire();
    if (acquired.has_value()) {
      return acquired;
    }
    if (acquired.error().code != holomq::ErrorCode::kBackpressured) {
      return acquired;
    }
    ++backpressured_count;
    std::this_thread::yield();
  }
  return pool.acquire();
}

template <std::size_t PayloadBytes, typename Publisher>
[[nodiscard]] bool wait_for_recipients(Publisher& publisher,
                                       const std::uint32_t expected_recipients,
                                       const std::uint64_t required_probe_deliveries,
                                       const std::chrono::steady_clock::time_point deadline,
                                       std::uint64_t& backpressured_count,
                                       std::uint64_t& connection_probe_sends) {
  if (!wait_for_any_subscriber(publisher, deadline)) {
    return false;
  }
  std::uint64_t completed_probe_deliveries = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto outbound = loan_with_wait(publisher, deadline, backpressured_count);
    if (!outbound.has_value()) {
      return false;
    }
    outbound->payload() = make_payload<PayloadBytes>(kConnectionProbeSequence);
    outbound->payload().send_time_ns = now_ns();
    auto sent = publisher.send(std::move(outbound).value());
    ++connection_probe_sends;
    if (!sent.has_value() || sent.value() < 0 ||
        sent.value() > static_cast<int>(expected_recipients)) {
      return false;
    }
    if (sent.value() == static_cast<int>(expected_recipients)) {
      ++completed_probe_deliveries;
      if (completed_probe_deliveries == required_probe_deliveries) {
        return true;
      }
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::yield();
  }
  return false;
}

template <typename Publisher>
[[nodiscard]] bool wait_for_buffer_recipients(Publisher& publisher,
                                              const std::uint64_t required_probe_deliveries,
                                              const std::chrono::steady_clock::time_point deadline,
                                              std::uint64_t& backpressured_count,
                                              std::uint64_t& connection_probe_sends) {
  if (!wait_for_any_subscriber(publisher, deadline)) {
    return false;
  }
  std::uint64_t completed_probe_deliveries = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto outbound = loan_with_wait(publisher, deadline, backpressured_count);
    if (!outbound.has_value()) {
      return false;
    }
    outbound->payload().sequence = kConnectionProbeSequence;
    outbound->payload().send_time_ns = now_ns();
    auto sent = publisher.send(std::move(outbound).value());
    ++connection_probe_sends;
    if (!sent.has_value() || sent.value() < 0 || sent.value() > 1) {
      return false;
    }
    if (sent.value() == 1) {
      ++completed_probe_deliveries;
      if (completed_probe_deliveries == required_probe_deliveries) {
        return true;
      }
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::yield();
  }
  return false;
}

[[nodiscard]] std::uint64_t percentile(std::vector<std::uint64_t>& sorted_values,
                                       const double fraction) {
  if (sorted_values.empty()) {
    return 0U;
  }
  const double position = fraction * static_cast<double>(sorted_values.size() - 1U);
  const auto index = static_cast<std::size_t>(std::ceil(position));
  return sorted_values[index];
}

void emit_percentiles(std::ostream& output, const std::string_view name,
                      std::vector<std::uint64_t>& values) {
  std::sort(values.begin(), values.end());
  output << ",\"" << name << "_count\":" << values.size() << ",\"" << name
         << "_p50_ns\":" << percentile(values, 0.50) << ",\"" << name
         << "_p99_ns\":" << percentile(values, 0.99) << ",\"" << name << "_p999_ns\":";
  if (values.size() >= kMinimumP999Samples) {
    output << percentile(values, 0.999);
  } else {
    output << "null";
  }
  output << ",\"" << name << "_max_ns\":" << (values.empty() ? 0U : values.back());
}

template <std::size_t PayloadBytes>
int run_publisher(const Options& options) {  // NOLINT(readability-function-cognitive-complexity)
  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    std::cerr << "failed to create HoloMQ context\n";
    return EXIT_FAILURE;
  }

  const auto service_options = make_service_options(options);
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = options.queue_depth;
  publisher_options.enable_sample_release_callbacks = false;
  auto publisher = context->publisher_builder(options.service_name)
                       .service_options(service_options)
                       .enable_event_wake(false)
                       .publisher_options(publisher_options)
                       .payload_type<FixedPayload<PayloadBytes>>()
                       .create();
  if (!publisher.has_value()) {
    std::cerr << "failed to create HoloMQ publisher\n";
    return EXIT_FAILURE;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_seconds);
  std::uint64_t connection_probe_backpressured_loans = 0U;
  std::uint64_t connection_probe_sends = 0U;
  if (!wait_for_recipients<PayloadBytes>(publisher.value(),
                                         options.subscriber_count,
                                         required_connection_probe_deliveries(options.queue_depth),
                                         deadline,
                                         connection_probe_backpressured_loans,
                                         connection_probe_sends)) {
    std::cerr << "timed out waiting for all HoloMQ subscribers\n";
    return EXIT_FAILURE;
  }
  if (!pin_to_cpu(options.cpu)) {
    return EXIT_FAILURE;
  }

  const std::uint64_t total_samples = options.warmup + options.samples;
  std::uint64_t measured_start_ns = 0U;
  std::uint64_t measured_end_ns = 0U;
  std::uint64_t checksum = 0U;
  std::uint64_t backpressured_count = 0U;
  auto next_publish_time = std::chrono::steady_clock::now();
  for (std::uint64_t sequence = 0U; sequence < total_samples; ++sequence) {
    pace_publisher(options, next_publish_time);
    if (sequence == options.warmup) {
      measured_start_ns = now_ns();
    }
    auto outbound = loan_with_wait(publisher.value(), deadline, backpressured_count);
    if (!outbound.has_value()) {
      std::cerr << "publisher loan failed or timed out\n";
      return EXIT_FAILURE;
    }
    outbound->payload() = make_payload<PayloadBytes>(sequence);
    outbound->payload().send_time_ns = now_ns();
    const std::uint64_t payload_checksum =
        payload_contents_required(options, sequence, total_samples)
            ? checksum_payload(outbound->payload())
            : 0U;
    auto sent = publisher->send(std::move(outbound).value());
    if (!sent.has_value() || sent.value() != static_cast<int>(options.subscriber_count)) {
      std::cerr << "publisher send did not reach every expected subscriber\n";
      return EXIT_FAILURE;
    }
    if (sequence >= options.warmup) {
      checksum ^= payload_checksum;
    }
    if (sequence + 1U == total_samples) {
      measured_end_ns = now_ns();
    }
  }

  std::cout << "{\"schema_version\":1,\"benchmark\":\"holomq_inline_transport\""
            << ",\"role\":\"publisher\",\"domain\":\"interprocess\"" << ",\"workload\":\""
            << workload_name(options.workload) << "\",\"payload\":\""
            << payload_kind_name(options.payload_kind) << "\",\"storage\":\"inline\""
            << ",\"service_name\":\"" << json_escape(options.service_name)
            << "\",\"payload_bytes\":" << PayloadBytes << ",\"samples\":" << options.samples
            << ",\"warmup\":" << options.warmup
            << ",\"queue_policy\":\"blocking\",\"queue_depth\":" << options.queue_depth
            << ",\"subscriber_count\":" << options.subscriber_count << ",\"cpu\":" << options.cpu
            << ",\"publisher_period_us\":" << options.publisher_period_us
            << ",\"publisher_pacing\":\"" << publisher_pacing_name(options.publisher_pacing)
            << "\",\"rate_limited\":" << (options.workload == Workload::kLatency ? "true" : "false")
            << ",\"event_wake\":false,\"vmm_fd_export\":false" << ",\"payload_validation\":\""
            << payload_validation_name(options.payload_validation) << "\",\"validation_timing\":\""
            << validation_timing_name(options.payload_validation) << "\""
            << ",\"measured_start_ns\":" << measured_start_ns
            << ",\"measured_end_ns\":" << measured_end_ns << ",\"checksum\":" << checksum
            << ",\"backpressured_loans\":" << backpressured_count
            << ",\"connection_probe_backpressured_loans\":" << connection_probe_backpressured_loans
            << ",\"connection_probe_sends\":" << connection_probe_sends;
  emit_clock(std::cout);
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

template <std::size_t PayloadBytes>
int run_subscriber(const Options& options) {  // NOLINT(readability-function-cognitive-complexity)
  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    std::cerr << "failed to create HoloMQ context\n";
    return EXIT_FAILURE;
  }

  const auto service_options = make_service_options(options);
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = options.queue_depth;
  subscriber_options.max_borrowed_samples = options.queue_depth;
  auto subscriber = context->subscriber_builder(options.service_name)
                        .service_options(service_options)
                        .enable_event_wake(false)
                        .subscriber_options(subscriber_options)
                        .payload_type<FixedPayload<PayloadBytes>>()
                        .create();
  if (!subscriber.has_value()) {
    std::cerr << "failed to create HoloMQ subscriber\n";
    return EXIT_FAILURE;
  }
  if (!pin_to_cpu(options.cpu)) {
    return EXIT_FAILURE;
  }
  if (!signal_ready(options.ready_fd)) {
    std::cerr << "failed to signal subscriber readiness\n";
    return EXIT_FAILURE;
  }
  static_cast<void>(::close(options.ready_fd));

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_seconds);
  const std::uint64_t total_samples = options.warmup + options.samples;
  std::vector<std::uint64_t> latencies_ns;
  latencies_ns.reserve(options.samples);
  std::uint64_t measured_start_ns = 0U;
  std::uint64_t measured_end_ns = 0U;
  std::uint64_t checksum = 0U;
  std::uint64_t empty_polls = 0U;
  std::uint64_t connection_probes = 0U;
  std::uint64_t next_sequence = 0U;
  std::optional<FixedPayload<PayloadBytes>> final_payload;
  while (next_sequence < total_samples) {
    auto inbound = subscriber->try_take();
    if (!inbound.has_value()) {
      if (inbound.error().code != holomq::ErrorCode::kNotReady) {
        std::cerr << "subscriber take failed\n";
        return EXIT_FAILURE;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        std::cerr << "subscriber timed out waiting for samples\n";
        return EXIT_FAILURE;
      }
      ++empty_polls;
      std::this_thread::yield();
      continue;
    }
    const FixedPayload<PayloadBytes>& payload = inbound->payload();
    if (payload.sequence == kConnectionProbeSequence) {
      if (!validate_payload(payload)) {
        std::cerr << "subscriber connection probe validation failed\n";
        return EXIT_FAILURE;
      }
      ++connection_probes;
      continue;
    }
    const bool validate_contents = options.payload_validation == PayloadValidation::kAll;
    if (payload.sequence != next_sequence || (validate_contents && !validate_payload(payload))) {
      std::cerr << "subscriber payload validation failed at sequence " << next_sequence << '\n';
      return EXIT_FAILURE;
    }
    if (next_sequence >= options.warmup) {
      const std::uint64_t received_ns = now_ns();
      if (measured_start_ns == 0U) {
        measured_start_ns = received_ns;
      }
      measured_end_ns = received_ns;
      if (options.payload_validation == PayloadValidation::kAll) {
        checksum ^= checksum_payload(payload);
      }
      latencies_ns.push_back(received_ns - payload.send_time_ns);
    }
    if (options.payload_validation == PayloadValidation::kFinal &&
        next_sequence + 1U == total_samples) {
      final_payload.emplace(payload);
    }
    ++next_sequence;
  }
  if (options.payload_validation == PayloadValidation::kFinal) {
    if (!final_payload.has_value() || !validate_payload(*final_payload)) {
      std::cerr << "subscriber final payload validation failed\n";
      return EXIT_FAILURE;
    }
    checksum ^= checksum_payload(*final_payload);
  }

  std::sort(latencies_ns.begin(), latencies_ns.end());
  const std::uint64_t latency_p50_ns = percentile(latencies_ns, 0.50);
  const std::uint64_t latency_p99_ns = percentile(latencies_ns, 0.99);
  const std::uint64_t latency_max_ns = latencies_ns.back();
  std::cout << "{\"schema_version\":1,\"benchmark\":\"holomq_inline_transport\""
            << ",\"role\":\"subscriber\",\"domain\":\"interprocess\"" << ",\"workload\":\""
            << workload_name(options.workload) << "\",\"payload\":\""
            << payload_kind_name(options.payload_kind) << "\",\"storage\":\"inline\""
            << ",\"service_name\":\"" << json_escape(options.service_name)
            << "\",\"payload_bytes\":" << PayloadBytes << ",\"samples\":" << options.samples
            << ",\"warmup\":" << options.warmup
            << ",\"queue_policy\":\"blocking\",\"queue_depth\":" << options.queue_depth
            << ",\"subscriber_count\":" << options.subscriber_count
            << ",\"subscriber_index\":" << options.subscriber_index << ",\"cpu\":" << options.cpu
            << ",\"publisher_period_us\":" << options.publisher_period_us
            << ",\"publisher_pacing\":\"" << publisher_pacing_name(options.publisher_pacing)
            << "\",\"rate_limited\":" << (options.workload == Workload::kLatency ? "true" : "false")
            << ",\"event_wake\":false,\"vmm_fd_export\":false" << ",\"payload_validation\":\""
            << payload_validation_name(options.payload_validation) << "\",\"validation_timing\":\""
            << validation_timing_name(options.payload_validation) << "\""
            << ",\"measured_start_ns\":" << measured_start_ns
            << ",\"measured_end_ns\":" << measured_end_ns << ",\"checksum\":" << checksum
            << ",\"empty_polls\":" << empty_polls << ",\"connection_probes\":" << connection_probes
            << ",\"latency_p50_ns\":" << latency_p50_ns << ",\"latency_p99_ns\":" << latency_p99_ns
            << ",\"latency_p999_ns\":";
  if (latencies_ns.size() >= kMinimumP999Samples) {
    std::cout << percentile(latencies_ns, 0.999);
  } else {
    std::cout << "null";
  }
  std::cout << ",\"latency_max_ns\":" << latency_max_ns;
  emit_clock(std::cout);
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

int run_buffer_publisher(
    const Options& options) {  // NOLINT(readability-function-cognitive-complexity)
  ScopedPrimaryCudaContext cuda_context;
  if (!cuda_context.initialize()) {
    std::cerr << "failed to initialize the buffer publisher CUDA context\n";
    return kCudaUnavailableExitCode;
  }
  ScopedCudaStream cuda_stream;
  if (options.storage_kind == StorageKind::kDevice && !cuda_stream.create()) {
    std::cerr << "failed to create the buffer publisher CUDA stream\n";
    return EXIT_FAILURE;
  }
  auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  if (!context.has_value()) {
    std::cerr << "failed to create HoloMQ buffer publisher context\n";
    return EXIT_FAILURE;
  }
  auto pool = [&]() {
    if (options.storage_kind == StorageKind::kDevice) {
      return context->create_device_buffer_pool(options.buffer_bytes, options.pool_slots);
    }
    if (options.storage_kind == StorageKind::kHostPinned) {
      return context->create_host_pinned_buffer_pool(
          options.buffer_bytes, options.pool_slots, CU_MEMHOSTREGISTER_PORTABLE);
    }
    return context->create_host_buffer_pool(options.buffer_bytes, options.pool_slots);
  }();
  if (!pool.has_value()) {
    std::cerr << "failed to create the HoloMQ buffer pool\n";
    return EXIT_FAILURE;
  }
  if (options.storage_kind == StorageKind::kDevice &&
      !initialize_device_pool(
          pool.value(), options.pool_slots, options.buffer_bytes, cuda_stream.get())) {
    std::cerr << "failed to initialize the HoloMQ device buffer pool\n";
    return EXIT_FAILURE;
  }

  const auto service_options = make_service_options(options);
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = options.queue_depth;
  auto publisher_or = context->publisher_builder(options.service_name)
                          .service_options(service_options)
                          .enable_event_wake(false)
                          .publisher_options(publisher_options)
                          .payload_type<BufferPayload>()
                          .create();
  if (!publisher_or.has_value()) {
    std::cerr << "failed to create HoloMQ buffer publisher\n";
    return EXIT_FAILURE;
  }
  using BufferPublisher = std::remove_cvref_t<decltype(std::move(publisher_or).value())>;
  std::optional<BufferPublisher> publisher;
  publisher.emplace(std::move(publisher_or).value());

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_seconds);
  std::uint64_t connection_probe_backpressured_loans = 0U;
  std::uint64_t connection_probe_sends = 0U;
  if (!wait_for_buffer_recipients(publisher.value(),
                                  required_connection_probe_deliveries(options.queue_depth),
                                  deadline,
                                  connection_probe_backpressured_loans,
                                  connection_probe_sends)) {
    std::cerr << "timed out waiting for the HoloMQ buffer subscriber\n";
    return EXIT_FAILURE;
  }
  if (!pin_to_cpu(options.cpu)) {
    return EXIT_FAILURE;
  }

  const std::uint64_t total_samples = options.warmup + options.samples;
  std::uint64_t measured_start_ns = 0U;
  std::uint64_t measured_end_ns = 0U;
  std::uint64_t checksum = 0U;
  std::uint64_t backpressured_loans = 0U;
  std::uint64_t backpressured_acquires = 0U;
  std::vector<std::uint64_t> acquire_ns;
  std::vector<std::uint64_t> export_ns;
  acquire_ns.reserve(options.samples);
  export_ns.reserve(options.samples);
  auto next_publish_time = std::chrono::steady_clock::now();
  for (std::uint64_t sequence = 0U; sequence < total_samples; ++sequence) {
    pace_publisher(options, next_publish_time);
    if (sequence == options.warmup) {
      measured_start_ns = now_ns();
    }
    auto outbound = loan_with_wait(publisher.value(), deadline, backpressured_loans);
    if (!outbound.has_value()) {
      std::cerr << "buffer publisher loan failed or timed out\n";
      return EXIT_FAILURE;
    }

    const std::uint64_t acquire_start_ns = now_ns();
    auto acquired = acquire_with_wait(pool.value(), deadline, backpressured_acquires);
    const std::uint64_t acquire_elapsed_ns = now_ns() - acquire_start_ns;
    if (!acquired.has_value()) {
      std::cerr << "host buffer pool acquire failed or timed out\n";
      return EXIT_FAILURE;
    }
    holomq::Buffer buffer = std::move(acquired).value();
    if (options.storage_kind != StorageKind::kDevice) {
      write_buffer_markers(buffer, sequence);
    }

    const std::uint64_t export_start_ns = now_ns();
    auto exported =
        options.storage_kind == StorageKind::kDevice && uses_publisher_stream(options.device_sync)
            ? buffer.export_handle(cuda_stream.get())
            : buffer.export_handle_unsynchronized();
    const std::uint64_t export_elapsed_ns = now_ns() - export_start_ns;
    if (!exported.has_value()) {
      std::cerr << "buffer handle export failed\n";
      return EXIT_FAILURE;
    }

    BufferPayload& payload = outbound->payload();
    payload.sequence = sequence;
    payload.buffer_bytes = options.buffer_bytes;
    payload.buffer_handle = exported.value();
    payload.send_time_ns = now_ns();
    auto sent = publisher->send(std::move(outbound).value(), [buffer = std::move(buffer)]() {});
    if (!sent.has_value() || sent.value() != 1) {
      std::cerr << "buffer publisher send did not reach the subscriber\n";
      return EXIT_FAILURE;
    }
    if (sequence >= options.warmup) {
      if (payload_contents_required(options, sequence, total_samples)) {
        checksum ^= buffer_checksum(sequence);
      }
      acquire_ns.push_back(acquire_elapsed_ns);
      export_ns.push_back(export_elapsed_ns);
    }
    if (sequence + 1U == total_samples) {
      measured_end_ns = now_ns();
    }
  }
  publisher.reset();
  if (options.storage_kind == StorageKind::kDevice && uses_publisher_stream(options.device_sync) &&
      !cuda_stream.synchronize()) {
    std::cerr << "failed to synchronize the buffer publisher CUDA stream\n";
    return EXIT_FAILURE;
  }

  std::cout << "{\"schema_version\":1,\"benchmark\":\"holomq_buffer_handle_transport\""
            << ",\"role\":\"publisher\",\"domain\":\"interprocess\"" << ",\"workload\":\""
            << workload_name(options.workload) << "\",\"payload\":\"buffer_handle\""
            << ",\"storage\":\"" << storage_kind_name(options.storage_kind)
            << "\",\"service_name\":\"" << json_escape(options.service_name)
            << "\",\"payload_bytes\":" << sizeof(BufferPayload)
            << ",\"buffer_bytes\":" << options.buffer_bytes
            << ",\"pool_slots\":" << options.pool_slots << ",\"samples\":" << options.samples
            << ",\"warmup\":" << options.warmup
            << ",\"queue_policy\":\"blocking\",\"queue_depth\":" << options.queue_depth
            << ",\"subscriber_count\":1,\"cpu\":" << options.cpu << ",\"device_data_work\":\""
            << (options.storage_kind == StorageKind::kDevice ? "preinitialized" : "none") << "\""
            << ",\"device_sync\":\"" << device_sync_name(options.device_sync) << "\""
            << ",\"publisher_period_us\":" << options.publisher_period_us
            << ",\"publisher_pacing\":\"" << publisher_pacing_name(options.publisher_pacing)
            << "\",\"rate_limited\":" << (options.workload == Workload::kLatency ? "true" : "false")
            << ",\"event_wake\":false,\"vmm_fd_export\":true" << ",\"payload_validation\":\""
            << payload_validation_name(options.payload_validation) << "\",\"validation_timing\":\""
            << validation_timing_name(options.payload_validation) << "\""
            << ",\"measured_start_ns\":" << measured_start_ns
            << ",\"measured_end_ns\":" << measured_end_ns << ",\"checksum\":" << checksum
            << ",\"backpressured_loans\":" << backpressured_loans
            << ",\"connection_probe_backpressured_loans\":" << connection_probe_backpressured_loans
            << ",\"backpressured_acquires\":" << backpressured_acquires
            << ",\"connection_probe_sends\":" << connection_probe_sends;
  emit_percentiles(std::cout, "buffer_acquire", acquire_ns);
  emit_percentiles(std::cout, "buffer_export", export_ns);
  emit_clock(std::cout);
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

int run_buffer_subscriber(
    const Options& options) {  // NOLINT(readability-function-cognitive-complexity)
  ScopedPrimaryCudaContext cuda_context;
  if (!cuda_context.initialize()) {
    std::cerr << "failed to initialize the buffer subscriber CUDA context\n";
    return kCudaUnavailableExitCode;
  }
  ScopedCudaStream cuda_stream;
  if (options.storage_kind == StorageKind::kDevice && !cuda_stream.create()) {
    std::cerr << "failed to create the buffer subscriber CUDA stream\n";
    return EXIT_FAILURE;
  }
  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    std::cerr << "failed to create HoloMQ buffer subscriber context\n";
    return EXIT_FAILURE;
  }

  const auto service_options = make_service_options(options);
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = options.queue_depth;
  subscriber_options.max_borrowed_samples = options.queue_depth;
  auto subscriber = context->subscriber_builder(options.service_name)
                        .service_options(service_options)
                        .enable_event_wake(false)
                        .subscriber_options(subscriber_options)
                        .payload_type<BufferPayload>()
                        .create();
  if (!subscriber.has_value()) {
    std::cerr << "failed to create HoloMQ buffer subscriber\n";
    return EXIT_FAILURE;
  }
  if (!pin_to_cpu(options.cpu)) {
    return EXIT_FAILURE;
  }
  if (!signal_ready(options.ready_fd)) {
    std::cerr << "failed to signal buffer subscriber readiness\n";
    return EXIT_FAILURE;
  }
  static_cast<void>(::close(options.ready_fd));

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_seconds);
  const std::uint64_t total_samples = options.warmup + options.samples;
  std::vector<std::uint64_t> latencies_ns;
  std::vector<std::uint64_t> import_ns;
  latencies_ns.reserve(options.samples);
  import_ns.reserve(options.samples);
  std::uint64_t measured_start_ns = 0U;
  std::uint64_t measured_end_ns = 0U;
  std::uint64_t checksum = 0U;
  std::uint64_t empty_polls = 0U;
  std::uint64_t connection_probes = 0U;
  std::uint64_t cold_first_import_ns = 0U;
  std::uint64_t next_sequence = 0U;
  std::optional<holomq::ConstBuffer> final_import;
  while (next_sequence < total_samples) {
    auto inbound = subscriber->try_take();
    if (!inbound.has_value()) {
      if (inbound.error().code != holomq::ErrorCode::kNotReady) {
        std::cerr << "buffer subscriber take failed\n";
        return EXIT_FAILURE;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        std::cerr << "buffer subscriber timed out waiting for samples\n";
        return EXIT_FAILURE;
      }
      ++empty_polls;
      std::this_thread::yield();
      continue;
    }
    const BufferPayload& payload = inbound->payload();
    if (payload.sequence == kConnectionProbeSequence) {
      ++connection_probes;
      continue;
    }
    if (payload.sequence != next_sequence || payload.buffer_bytes != options.buffer_bytes ||
        !payload.buffer_handle) {
      std::cerr << "buffer subscriber descriptor validation failed at sequence " << next_sequence
                << '\n';
      return EXIT_FAILURE;
    }

    const std::uint64_t import_start_ns = now_ns();
    auto imported =
        options.storage_kind == StorageKind::kDevice && options.device_sync == DeviceSync::kReady
            ? context->import_buffer(payload.buffer_handle, cuda_stream.get())
            : context->import_buffer_unsynchronized(payload.buffer_handle);
    if (imported.has_value() && options.storage_kind == StorageKind::kDevice &&
        options.device_sync == DeviceSync::kSubscriberRelease &&
        !imported->enqueue_ready_wait(cuda_stream.get())) {
      std::cerr << "failed to arm subscriber deferred release\n";
      return EXIT_FAILURE;
    }
    const std::uint64_t import_elapsed_ns = now_ns() - import_start_ns;
    const holomq::BufferMemoryType expected_memory_type =
        options.storage_kind == StorageKind::kDevice
            ? holomq::BufferMemoryType::Device
            : (options.storage_kind == StorageKind::kHostPinned
                   ? holomq::BufferMemoryType::HostPinned
                   : holomq::BufferMemoryType::HostPageable);
    if (!imported.has_value() || imported->size_bytes() != options.buffer_bytes ||
        imported->memory_type() != expected_memory_type ||
        (options.storage_kind != StorageKind::kDevice &&
         options.payload_validation == PayloadValidation::kAll &&
         !validate_buffer_markers(imported.value(), payload.sequence))) {
      std::cerr << "buffer subscriber import or marker validation failed\n";
      return EXIT_FAILURE;
    }
    if (next_sequence == 0U) {
      cold_first_import_ns = import_elapsed_ns;
    }
    if (next_sequence >= options.warmup) {
      const std::uint64_t received_ns = now_ns();
      if (measured_start_ns == 0U) {
        measured_start_ns = received_ns;
      }
      measured_end_ns = received_ns;
      if (options.payload_validation == PayloadValidation::kAll) {
        checksum ^= buffer_checksum(next_sequence);
      }
      latencies_ns.push_back(received_ns - payload.send_time_ns);
      if (next_sequence != 0U) {
        import_ns.push_back(import_elapsed_ns);
      }
    }
    if (options.payload_validation == PayloadValidation::kFinal &&
        next_sequence + 1U == total_samples) {
      final_import.emplace(std::move(imported).value());
    }
    ++next_sequence;
  }

  std::uint64_t validation_errors = 0U;
  const std::optional<std::uint64_t> final_validation_start_ns =
      options.payload_validation == PayloadValidation::kFinal
          ? std::optional<std::uint64_t>{now_ns()}
          : std::nullopt;
  if (options.storage_kind == StorageKind::kDevice) {
    const bool stream_synchronized =
        !uses_subscriber_stream(options.device_sync) || cuda_stream.synchronize();
    const bool payload_valid =
        options.payload_validation == PayloadValidation::kNone ||
        (final_import.has_value() && validate_device_buffer(final_import.value()));
    if (!stream_synchronized || !payload_valid) {
      ++validation_errors;
    } else if (options.payload_validation == PayloadValidation::kFinal) {
      checksum ^= buffer_checksum(total_samples - 1U);
    }
    final_import.reset();
    if (uses_subscriber_stream(options.device_sync) && !cuda_stream.synchronize()) {
      ++validation_errors;
    }
  } else if (options.payload_validation == PayloadValidation::kFinal) {
    if (!final_import.has_value() || !validate_buffer_markers(*final_import, total_samples - 1U)) {
      ++validation_errors;
    } else {
      checksum ^= buffer_checksum(total_samples - 1U);
    }
  }
  if (validation_errors != 0U) {
    std::cerr << "device buffer final validation or synchronization failed\n";
    return EXIT_FAILURE;
  }
  const std::optional<std::uint64_t> final_validation_ns =
      final_validation_start_ns
          ? std::optional<std::uint64_t>{now_ns() - *final_validation_start_ns}
          : std::nullopt;

  std::cout << "{\"schema_version\":1,\"benchmark\":\"holomq_buffer_handle_transport\""
            << ",\"role\":\"subscriber\",\"domain\":\"interprocess\"" << ",\"workload\":\""
            << workload_name(options.workload) << "\",\"payload\":\"buffer_handle\""
            << ",\"storage\":\"" << storage_kind_name(options.storage_kind)
            << "\",\"service_name\":\"" << json_escape(options.service_name)
            << "\",\"payload_bytes\":" << sizeof(BufferPayload)
            << ",\"buffer_bytes\":" << options.buffer_bytes
            << ",\"pool_slots\":" << options.pool_slots << ",\"samples\":" << options.samples
            << ",\"warmup\":" << options.warmup
            << ",\"queue_policy\":\"blocking\",\"queue_depth\":" << options.queue_depth
            << ",\"subscriber_count\":1,\"subscriber_index\":0,\"cpu\":" << options.cpu
            << ",\"device_data_work\":\""
            << (options.storage_kind == StorageKind::kDevice ? "preinitialized" : "none") << "\""
            << ",\"device_sync\":\"" << device_sync_name(options.device_sync) << "\""
            << ",\"publisher_period_us\":" << options.publisher_period_us
            << ",\"publisher_pacing\":\"" << publisher_pacing_name(options.publisher_pacing)
            << "\",\"rate_limited\":" << (options.workload == Workload::kLatency ? "true" : "false")
            << ",\"event_wake\":false,\"vmm_fd_export\":false" << ",\"payload_validation\":\""
            << payload_validation_name(options.payload_validation) << "\",\"validation_timing\":\""
            << validation_timing_name(options.payload_validation) << "\""
            << ",\"measured_start_ns\":" << measured_start_ns
            << ",\"measured_end_ns\":" << measured_end_ns << ",\"checksum\":" << checksum
            << ",\"empty_polls\":" << empty_polls << ",\"connection_probes\":" << connection_probes
            << ",\"cold_first_import_ns\":" << cold_first_import_ns
            << ",\"validation_errors\":" << validation_errors << ",\"final_validation_ns\":";
  if (final_validation_ns) {
    std::cout << *final_validation_ns;
  } else {
    std::cout << "null";
  }
  emit_percentiles(std::cout, "buffer_import", import_ns);
  emit_percentiles(std::cout, "latency", latencies_ns);
  emit_clock(std::cout);
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

template <std::size_t PayloadBytes>
int run_role(const Options& options) {
  if (options.role == "publisher") {
    return run_publisher<PayloadBytes>(options);
  }
  return run_subscriber<PayloadBytes>(options);
}

}  // namespace

int main(const int argc, char** argv) {
  const auto options = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options.has_value()) {
    print_usage(std::cerr);
    return EXIT_FAILURE;
  }
  if (options->payload_kind == PayloadKind::kBufferHandle) {
    return options->role == "publisher" ? run_buffer_publisher(options.value())
                                        : run_buffer_subscriber(options.value());
  }
  switch (options->payload_bytes) {
    case 64U:
      return run_role<64U>(options.value());
    case 128U:
      return run_role<128U>(options.value());
    case 440U:
      return run_role<440U>(options.value());
    default:
      break;
  }
  return EXIT_FAILURE;
}
