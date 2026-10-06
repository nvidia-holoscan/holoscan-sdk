/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <sched.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <holomq/context_builder.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/channel_tensor_transport_payload.hpp"
#include "../../src/channel/holomq_tensor.hpp"
#include "../common/benchmark_pod.hpp"
#include "../common/channel_benchmark_common.hpp"
#include "../common/transport_benchmark_common.hpp"

namespace {

using holoscan::BenchmarkPOD64;
using holoscan::benchmarking::BenchmarkMonotonicClock;
using holoscan::benchmarking::cleanup_holomq_benchmark_socket_path;
using holoscan::benchmarking::json_escape;
using holoscan::benchmarking::kDescriptorBackedImagePayloadBytes;
using holoscan::benchmarking::kTensorTransportPayloadBytes;
using holoscan::benchmarking::make_unique_holomq_benchmark_socket_path;
using holoscan::benchmarking::payload_validation_name;
using holoscan::benchmarking::PayloadValidation;
using holoscan::benchmarking::summarize_percentiles;
using holoscan::benchmarking::TransportPercentiles;
using holoscan::benchmarking::validation_timing_name;
using holoscan::benchmarking::wait_for_channel_subscribers;
using BenchmarkClock = BenchmarkMonotonicClock;
using Clock = std::chrono::steady_clock;
constexpr std::uint8_t kAsyncCudaPattern = 0xa5U;
constexpr std::uint64_t kBenchmarkEndpointGeneration = 1U;

enum class PublisherPacing { kSleep, kSpin };

using DescriptorBackedImagePayload =
    holoscan::schema::DescriptorFlatBufferPayloadFor<holoscan::schema::Image>;
using ImageInlinePublisher = holoscan::ChannelPublisher<DescriptorBackedImagePayload>;
using ImageSidecarPublisher = holoscan::ChannelTensorSchemaPublisher<DescriptorBackedImagePayload>;
using ImagePublisher = std::variant<ImageInlinePublisher, ImageSidecarPublisher>;

static_assert(sizeof(holoscan::channel_internal::TensorTransportPayload) ==
              kTensorTransportPayloadBytes);
static_assert(sizeof(DescriptorBackedImagePayload) == kDescriptorBackedImagePayloadBytes);

struct Options {
  std::string role{"publisher"};
  std::string domain{"interprocess"};
  std::string payload{"pod"};
  std::string tensor_storage{"pinned_host"};
  std::string workload{"throughput"};
  std::string channel_name{"holoscan_channel_interprocess_benchmark"};
  std::uint64_t samples{100000U};
  std::uint64_t warmup{1000U};
  std::uint64_t tensor_bytes{1U << 20U};
  std::uint32_t subscriber_queue_depth{64U};
  std::uint32_t max_subscriber_queue_depth{0U};
  std::uint32_t expected_subscribers{1U};
  std::uint32_t publisher_hold_ms{500U};
  std::uint32_t publisher_period_us{0U};
  PublisherPacing publisher_pacing{PublisherPacing::kSleep};
  int cpu{-1};
  int publisher_cpu{-1};
  std::vector<int> subscriber_cpus;
  bool async_cuda_work{false};
  PayloadValidation payload_validation{PayloadValidation::kFinal};
};

class PublisherPacer {
 public:
  explicit PublisherPacer(const Options& options)
      : period_(std::chrono::microseconds(options.publisher_period_us)),
        pacing_(options.publisher_pacing) {}

  void wait() {
    if (period_.count() == 0) {
      return;
    }
    if (pacing_ == PublisherPacing::kSleep) {
      std::this_thread::sleep_for(period_);
      return;
    }

    const Clock::time_point now = Clock::now();
    if (!initialized_) {
      next_deadline_ = now + period_;
      initialized_ = true;
    } else {
      next_deadline_ += period_;
      if (next_deadline_ < now) {
        next_deadline_ = now;
      }
    }
    while (Clock::now() < next_deadline_) {
      // Busy-wait to avoid scheduler wake-up latency.
    }
  }

 private:
  Clock::duration period_{};
  PublisherPacing pacing_{PublisherPacing::kSleep};
  Clock::time_point next_deadline_{};
  bool initialized_{false};
};

[[nodiscard]] std::int64_t now_ns() {
  return BenchmarkClock::now_ns();
}

[[nodiscard]] std::uint64_t elapsed_ns(const std::int64_t start_ns) {
  return static_cast<std::uint64_t>(now_ns() - start_ns);
}

void emit_percentiles(std::ostream& os, const char* const prefix,
                      const TransportPercentiles& values) {
  os << ",\"" << prefix << "_pMin_ns\":" << values.p_min << ",\"" << prefix
     << "_p50_ns\":" << values.p50 << ",\"" << prefix << "_p99_ns\":" << values.p99 << ",\""
     << prefix << "_p999_ns\":";
  if (values.count >= 10'000U) {
    os << values.p999;
  } else {
    os << "null";
  }
  os << ",\"" << prefix << "_pMax_ns\":" << values.p_max;
}

void emit_throughput(std::ostream& os, const std::uint64_t samples,
                     const std::int64_t measured_start_ns, const std::int64_t measured_end_ns) {
  const std::int64_t measured_loop_ns = measured_end_ns - measured_start_ns;
  const double messages_per_second =
      measured_loop_ns > 0
          ? static_cast<double>(samples) * 1.0e9 / static_cast<double>(measured_loop_ns)
          : 0.0;
  os << ",\"measured_start_ns\":" << measured_start_ns << ",\"measured_end_ns\":" << measured_end_ns
     << ",\"measured_loop_ns\":" << (measured_loop_ns > 0 ? measured_loop_ns : 0)
     << ",\"messages_per_second\":" << messages_per_second;
}

void print_help(std::ostream& os) {
  os << "channel_interprocess_benchmark\n\n"
     << "Usage:\n"
     << "  channel_interprocess_benchmark --role subscriber --channel-name <name> [options]\n"
     << "  channel_interprocess_benchmark --role publisher --channel-name <name> [options]\n"
     << "  channel_interprocess_benchmark --role coordinator --domain intraprocess "
        "--channel-name <name> [options]\n\n"
     << "Options:\n"
     << "  --role <publisher|subscriber|coordinator>\n"
     << "  --domain <interprocess|intraprocess|intraprocess_sidecar|intraprocess_serial>\n"
     << "                                    Execution domain; default interprocess.\n"
     << "  --payload <pod|tensor|image>     Payload mode; default pod.\n"
     << "  --workload <throughput|latency> Saturated capacity or paced low-queueing latency; "
        "default throughput.\n"
     << "  --tensor-storage <pageable_host|pinned_host|device>\n"
     << "                                    Tensor/Image backing pool; default pinned_host.\n"
     << "  --channel-name <name>\n"
     << "  --samples <n>                    Measured samples after warmup; default 100000.\n"
     << "  --warmup <n>                     Warmup samples before stats; default 1000.\n"
     << "  --tensor-bytes <n>               Tensor/Image backing bytes; default 1048576.\n"
     << "  --subscriber-queue-depth <n>     Per-subscriber queue depth; default 64.\n"
     << "  --max-subscriber-queue-depth <n> Publisher service max queue depth; default queue "
        "depth.\n"
     << "  --expected-subscribers <n>       Publisher waits for this count; default 1.\n"
     << "  --publisher-hold-ms <ms>         Publisher sleep after sending; default 500.\n"
     << "  --publisher-period-us <us>       Publisher pacing interval; default 0.\n"
     << "  --publisher-pacing <sleep|spin> Relative sleep or deadline-based spin pacing; "
        "default sleep.\n"
     << "  --async-cuda-work                Enqueue S0/S1 work for device Tensor mode.\n"
     << "  --payload-validation <none|final|all>\n"
        "                                    Full content checks disabled, after timing, or per "
        "sample; default final.\n"
     << "  --cpu <n>                        Pin this role to CPU n; default -1.\n"
     << "  --publisher-cpu <n>              Coordinator publisher thread CPU; default -1.\n"
     << "  --subscriber-cpu <n>             Coordinator subscriber thread CPU; repeat once per "
        "subscriber.\n";
}

[[nodiscard]] bool parse_u64(const char* const text, std::uint64_t& out) {
  if (text == nullptr || text[0] < '0' || text[0] > '9') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      value > std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  out = static_cast<std::uint64_t>(value);
  return true;
}

[[nodiscard]] bool parse_u32(const char* const text, std::uint32_t& out) {
  std::uint64_t value = 0U;
  if (!parse_u64(text, value) || value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  out = static_cast<std::uint32_t>(value);
  return true;
}

[[nodiscard]] bool parse_int(const char* const text, int& out) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,cppcoreguidelines-avoid-c-arrays)
[[nodiscard]] std::optional<Options> parse_args(const int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto need_value = [&](const char* const name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << '\n';
        std::exit(2);
      }
      return argv[++i];  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    };

    if (arg == "--help" || arg == "-h") {
      print_help(std::cout);
      std::exit(0);
    }
    if (arg == "--role") {
      options.role = need_value("--role");
    } else if (arg == "--domain") {
      options.domain = need_value("--domain");
    } else if (arg == "--payload") {
      options.payload = need_value("--payload");
    } else if (arg == "--tensor-storage") {
      options.tensor_storage = need_value("--tensor-storage");
    } else if (arg == "--workload") {
      options.workload = need_value("--workload");
    } else if (arg == "--channel-name") {
      options.channel_name = need_value("--channel-name");
    } else if (arg == "--samples") {
      if (!parse_u64(need_value("--samples"), options.samples)) {
        std::cerr << "invalid --samples\n";
        return std::nullopt;
      }
    } else if (arg == "--warmup") {
      if (!parse_u64(need_value("--warmup"), options.warmup)) {
        std::cerr << "invalid --warmup\n";
        return std::nullopt;
      }
    } else if (arg == "--tensor-bytes") {
      if (!parse_u64(need_value("--tensor-bytes"), options.tensor_bytes)) {
        std::cerr << "invalid --tensor-bytes\n";
        return std::nullopt;
      }
    } else if (arg == "--subscriber-queue-depth") {
      if (!parse_u32(need_value("--subscriber-queue-depth"), options.subscriber_queue_depth)) {
        std::cerr << "invalid --subscriber-queue-depth\n";
        return std::nullopt;
      }
    } else if (arg == "--max-subscriber-queue-depth") {
      if (!parse_u32(need_value("--max-subscriber-queue-depth"),
                     options.max_subscriber_queue_depth)) {
        std::cerr << "invalid --max-subscriber-queue-depth\n";
        return std::nullopt;
      }
    } else if (arg == "--expected-subscribers") {
      if (!parse_u32(need_value("--expected-subscribers"), options.expected_subscribers)) {
        std::cerr << "invalid --expected-subscribers\n";
        return std::nullopt;
      }
    } else if (arg == "--publisher-hold-ms") {
      if (!parse_u32(need_value("--publisher-hold-ms"), options.publisher_hold_ms)) {
        std::cerr << "invalid --publisher-hold-ms\n";
        return std::nullopt;
      }
    } else if (arg == "--publisher-period-us") {
      if (!parse_u32(need_value("--publisher-period-us"), options.publisher_period_us)) {
        std::cerr << "invalid --publisher-period-us\n";
        return std::nullopt;
      }
    } else if (arg == "--publisher-pacing") {
      const std::string_view pacing = need_value("--publisher-pacing");
      if (pacing == "sleep") {
        options.publisher_pacing = PublisherPacing::kSleep;
      } else if (pacing == "spin") {
        options.publisher_pacing = PublisherPacing::kSpin;
      } else {
        std::cerr << "--publisher-pacing must be sleep or spin\n";
        return std::nullopt;
      }
    } else if (arg == "--async-cuda-work") {
      options.async_cuda_work = true;
    } else if (arg == "--payload-validation") {
      const std::string_view mode = need_value("--payload-validation");
      if (mode == "none") {
        options.payload_validation = PayloadValidation::kNone;
      } else if (mode == "final") {
        options.payload_validation = PayloadValidation::kFinal;
      } else if (mode == "all") {
        options.payload_validation = PayloadValidation::kAll;
      } else {
        std::cerr << "--payload-validation must be none, final, or all\n";
        return std::nullopt;
      }
    } else if (arg == "--cpu") {
      if (!parse_int(need_value("--cpu"), options.cpu)) {
        std::cerr << "invalid --cpu\n";
        return std::nullopt;
      }
    } else if (arg == "--publisher-cpu") {
      if (!parse_int(need_value("--publisher-cpu"), options.publisher_cpu)) {
        std::cerr << "invalid --publisher-cpu\n";
        return std::nullopt;
      }
    } else if (arg == "--subscriber-cpu") {
      int subscriber_cpu = -1;
      if (!parse_int(need_value("--subscriber-cpu"), subscriber_cpu)) {
        std::cerr << "invalid --subscriber-cpu\n";
        return std::nullopt;
      }
      options.subscriber_cpus.push_back(subscriber_cpu);
    } else {
      std::cerr << "unknown option: " << arg << '\n';
      return std::nullopt;
    }
  }

  if (options.role != "publisher" && options.role != "subscriber" &&
      options.role != "coordinator") {
    std::cerr << "--role must be publisher, subscriber, or coordinator\n";
    return std::nullopt;
  }
  const bool intraprocess = options.domain == "intraprocess" ||
                            options.domain == "intraprocess_sidecar" ||
                            options.domain == "intraprocess_serial";
  if (options.domain != "interprocess" && !intraprocess) {
    std::cerr << "--domain must be interprocess, intraprocess, intraprocess_sidecar, or "
                 "intraprocess_serial\n";
    return std::nullopt;
  }
  if ((options.role == "coordinator") != intraprocess) {
    std::cerr << "--role coordinator requires an intraprocess domain; publisher/subscriber roles "
                 "require --domain interprocess\n";
    return std::nullopt;
  }
  if (options.payload != "pod" && options.payload != "tensor" && options.payload != "image") {
    std::cerr << "--payload must be pod, tensor, or image\n";
    return std::nullopt;
  }
  if (options.tensor_storage != "pageable_host" && options.tensor_storage != "pinned_host" &&
      options.tensor_storage != "device") {
    std::cerr << "--tensor-storage must be pageable_host, pinned_host, or device\n";
    return std::nullopt;
  }
  if (options.workload != "throughput" && options.workload != "latency") {
    std::cerr << "--workload must be throughput or latency\n";
    return std::nullopt;
  }
  if (options.channel_name.empty()) {
    std::cerr << "--channel-name must be non-empty\n";
    return std::nullopt;
  }
  if (options.samples == 0U) {
    std::cerr << "--samples must be non-zero\n";
    return std::nullopt;
  }
  if (options.tensor_bytes == 0U || (options.tensor_bytes % sizeof(std::uint32_t)) != 0U) {
    std::cerr << "--tensor-bytes must be non-zero and divisible by 4\n";
    return std::nullopt;
  }
  if (options.subscriber_queue_depth == 0U) {
    std::cerr << "--subscriber-queue-depth must be non-zero\n";
    return std::nullopt;
  }
  if (options.workload == "latency" && options.publisher_period_us == 0U) {
    std::cerr << "--workload latency requires a nonzero --publisher-period-us\n";
    return std::nullopt;
  }
  if (options.workload == "throughput" && options.publisher_period_us != 0U) {
    std::cerr << "--publisher-period-us requires --workload latency\n";
    return std::nullopt;
  }
  if (options.workload != "latency" && options.publisher_pacing == PublisherPacing::kSpin) {
    std::cerr << "--publisher-pacing spin requires --workload latency\n";
    return std::nullopt;
  }
  if (options.workload == "latency" && options.subscriber_queue_depth > 2U) {
    std::cerr << "--workload latency requires --subscriber-queue-depth 1 or 2\n";
    return std::nullopt;
  }
  if (options.expected_subscribers == 0U) {
    std::cerr << "--expected-subscribers must be non-zero\n";
    return std::nullopt;
  }
  if (options.domain == "intraprocess_serial" && options.expected_subscribers != 1U) {
    std::cerr << "--domain intraprocess_serial requires --expected-subscribers 1\n";
    return std::nullopt;
  }
  if (options.domain == "intraprocess_serial" && options.payload != "tensor") {
    std::cerr << "--domain intraprocess_serial currently requires --payload tensor\n";
    return std::nullopt;
  }
  if (options.domain == "intraprocess_serial" && !options.subscriber_cpus.empty()) {
    std::cerr << "--domain intraprocess_serial uses --publisher-cpu and does not accept "
                 "--subscriber-cpu\n";
    return std::nullopt;
  }
  if (options.role == "coordinator") {
    if (options.subscriber_cpus.empty()) {
      options.subscriber_cpus.resize(options.expected_subscribers, -1);
    } else if (options.subscriber_cpus.size() != options.expected_subscribers) {
      std::cerr << "--role coordinator requires one --subscriber-cpu per expected subscriber\n";
      return std::nullopt;
    }
  }
  if (options.async_cuda_work &&
      (options.payload != "tensor" || options.tensor_storage != "device")) {
    std::cerr << "--async-cuda-work requires --payload tensor --tensor-storage device\n";
    return std::nullopt;
  }
  if (options.async_cuda_work && options.payload_validation == PayloadValidation::kAll) {
    std::cerr << "--payload-validation all is incompatible with --async-cuda-work; "
                 "use its deferred CUDA validation\n";
    return std::nullopt;
  }
  if (options.async_cuda_work &&
      (options.domain == "intraprocess_sidecar" || options.domain == "intraprocess_serial")) {
    std::cerr << "--async-cuda-work is not yet supported by sidecar benchmark domains\n";
    return std::nullopt;
  }
  if (options.max_subscriber_queue_depth == 0U) {
    options.max_subscriber_queue_depth = options.subscriber_queue_depth;
  }
  return options;
}

void emit_optional_duration(std::ostream& os, const char* const name,
                            const std::optional<std::uint64_t> value) {
  os << ",\"" << name << "\":";
  if (value) {
    os << *value;
  } else {
    os << "null";
  }
}

[[nodiscard]] bool uses_device_tensor_storage(const Options& options) noexcept {
  return options.tensor_storage == "device";
}

[[nodiscard]] bool uses_pageable_host_tensor_storage(const Options& options) noexcept {
  return options.tensor_storage == "pageable_host";
}

[[nodiscard]] bool uses_sidecar_transport(const Options& options) noexcept {
  return options.domain == "intraprocess_sidecar" || options.domain == "intraprocess_serial";
}

[[nodiscard]] holoscan::MemoryStorageType tensor_storage_type(const Options& options) noexcept {
  if (uses_device_tensor_storage(options)) {
    return holoscan::MemoryStorageType::kDevice;
  }
  if (uses_pageable_host_tensor_storage(options)) {
    return holoscan::MemoryStorageType::kSystem;
  }
  return holoscan::MemoryStorageType::kHost;
}

[[nodiscard]] holoscan::ChannelPayloadKind tensor_payload_kind(const Options& options) noexcept {
  return uses_sidecar_transport(options) ? holoscan::ChannelPayloadKind::kLocalBufferDescriptor
                                         : holoscan::ChannelPayloadKind::kBufferHandle;
}

[[nodiscard]] const char* transport_name(const Options& options) noexcept {
  if (options.payload == "pod") {
    return "inline";
  }
  return uses_sidecar_transport(options) ? "sidecar" : "buffer_handle";
}

[[nodiscard]] const char* publisher_pacing_name(const PublisherPacing pacing) noexcept {
  return pacing == PublisherPacing::kSpin ? "spin" : "sleep";
}

[[nodiscard]] bool pin_to_cpu(const int cpu, const std::string_view role) {
  if (cpu < 0) {
    return true;
  }
  if (cpu >= CPU_SETSIZE) {
    std::cerr << "cannot pin " << role << " to CPU " << cpu << ": CPU ID must be less than "
              << CPU_SETSIZE << '\n';
    return false;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::cerr << "failed to pin " << role << " to CPU " << cpu << ": " << std::strerror(errno)
              << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] holoscan::ChannelOptions make_channel_options(const Options& options) noexcept {
  holoscan::ChannelOptions channel_options{};
  channel_options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  channel_options.subscriber_queue_depth = options.subscriber_queue_depth;
  channel_options.max_subscriber_queue_depth = options.max_subscriber_queue_depth;
  channel_options.expected_recipient_count = options.expected_subscribers;
  if (options.payload == "tensor" && !uses_sidecar_transport(options)) {
    channel_options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;
  }
  return channel_options;
}

[[nodiscard]] BenchmarkPOD64 make_payload(const std::uint64_t sequence,
                                          const std::int64_t full_start_time_ns,
                                          const std::int64_t send_time_ns) noexcept {
  BenchmarkPOD64 payload{
      .sequence = sequence,
      .source_start_time_ns = full_start_time_ns,
      .send_time_ns = send_time_ns,
      .auxiliary0 = sequence + 1U,
      .auxiliary1 = sequence + 2U,
  };
  payload.checksum = holoscan::benchmark_pod_checksum(payload);
  return payload;
}

[[nodiscard]] holoscan::expected<std::vector<std::byte>, holoscan::Error> build_image_metadata(
    const holoscan::Tensor& tensor, const Options& options, const std::uint64_t sequence,
    const std::int64_t full_start_time_ns) {
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 1,
      .width = static_cast<int32_t>(options.tensor_bytes / sizeof(std::uint32_t)),
      .encoding = holoscan::schema::ImageEncoding_CUSTOM,
      .capture_timestamp_ns = full_start_time_ns,
      .frame_id = "channel_interprocess_image_" + std::to_string(sequence)};
  return holoscan::schema::build_image_metadata(tensor, metadata_options);
}

[[nodiscard]] holoscan::expected<DescriptorBackedImagePayload, holoscan::Error> make_image_payload(
    const holoscan::Tensor& tensor, const Options& options, const std::uint64_t sequence,
    const std::int64_t full_start_time_ns) {
  auto metadata = build_image_metadata(tensor, options, sequence, full_start_time_ns);
  if (!metadata) {
    return holoscan::make_unexpected(std::move(metadata).error());
  }

  auto tensor_descriptor = holoscan::channel_internal::export_holomq_tensor_descriptor(tensor);
  if (!tensor_descriptor) {
    return holoscan::make_unexpected(std::move(tensor_descriptor).error());
  }
  holoscan::PayloadDescriptor payload_descriptor =
      holoscan::channel_internal::payload_descriptor_from_holomq_tensor_descriptor(
          tensor_descriptor.value());
  payload_descriptor.schema_id = holoscan::schema_identity<holoscan::schema::Image>().family_id;
  payload_descriptor.sequence = sequence;
  payload_descriptor.byte_size = static_cast<std::uint64_t>(tensor.nbytes());

  return DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata->data(), metadata->size()),
      payload_descriptor,
      kBenchmarkEndpointGeneration);
}

[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_image(
    ImageInlinePublisher& publisher, const holoscan::Tensor& tensor, const Options& options,
    const std::uint64_t sequence, const std::int64_t full_start_time_ns) {
  auto payload = make_image_payload(tensor, options, sequence, full_start_time_ns);
  if (!payload) {
    return holoscan::make_unexpected(std::move(payload).error());
  }
  return publisher.emit(payload.value());
}

[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_image(
    ImageSidecarPublisher& publisher, const holoscan::Tensor& tensor, const Options& options,
    const std::uint64_t sequence, const std::int64_t full_start_time_ns) {
  auto metadata = build_image_metadata(tensor, options, sequence, full_start_time_ns);
  if (!metadata) {
    return holoscan::make_unexpected(std::move(metadata).error());
  }
  return publisher.emit(std::span<const std::byte>(metadata->data(), metadata->size()),
                        holoscan::Tensor(tensor));
}

[[nodiscard]] std::uint64_t application_payload_bytes(const Options& options) noexcept {
  return options.payload == "pod" ? sizeof(BenchmarkPOD64) : options.tensor_bytes;
}

[[nodiscard]] std::uint64_t payload_bytes(const Options& options) noexcept {
  if (options.payload == "tensor") {
    return options.tensor_bytes;
  }
  return options.payload == "image" ? sizeof(DescriptorBackedImagePayload) : sizeof(BenchmarkPOD64);
}

[[nodiscard]] std::size_t inline_transport_bytes(const Options& options) noexcept {
  if (options.payload == "tensor") {
    return kTensorTransportPayloadBytes;
  }
  if (options.payload == "image") {
    return kDescriptorBackedImagePayloadBytes;
  }
  return sizeof(BenchmarkPOD64);
}

[[nodiscard]] std::uint64_t external_buffer_bytes(const Options& options) noexcept {
  return options.payload == "pod" ? 0U : options.tensor_bytes;
}

[[nodiscard]] const char* payload_write_profile(const Options& options) noexcept {
  if (options.payload == "pod" || options.async_cuda_work) {
    return "full_payload_per_sample";
  }
  if (options.payload == "image") {
    return "metadata_per_sample_reused_external_buffer";
  }
  return "reused_preinitialized_contents";
}

[[nodiscard]] const char* bulk_work_profile(const Options& options) noexcept {
  if (options.payload == "pod") {
    return "not_applicable";
  }
  return options.async_cuda_work ? "full_write" : "descriptor_only";
}

[[nodiscard]] std::optional<std::size_t> bulk_pool_slots(const Options& options) noexcept {
  if (options.payload == "pod") {
    return std::nullopt;
  }
  constexpr std::size_t kMinimumPoolSize = 256U;
  return std::max<std::size_t>(kMinimumPoolSize, options.subscriber_queue_depth + 8U);
}

void emit_common_json(std::ostream& os, const Options& options, const char* const role) {
  const std::optional<std::size_t> pool_slots = bulk_pool_slots(options);
  os << "\"role\":\"" << role << "\",\"channel_name\":\"" << json_escape(options.channel_name)
     << "\",\"domain\":\"" << options.domain << "\",\"payload\":\"" << options.payload << '"'
     << ",\"transport\":\"" << transport_name(options) << '"' << ",\"samples\":" << options.samples
     << ",\"warmup\":" << options.warmup << ",\"tensor_bytes\":" << options.tensor_bytes
     << ",\"tensor_storage\":\"" << options.tensor_storage << '"' << ",\"workload\":\""
     << options.workload << '"' << ",\"subscriber_queue_depth\":" << options.subscriber_queue_depth
     << ",\"max_subscriber_queue_depth\":" << options.max_subscriber_queue_depth
     << ",\"expected_subscribers\":" << options.expected_subscribers
     << ",\"publisher_period_us\":" << options.publisher_period_us << ",\"publisher_pacing\":\""
     << publisher_pacing_name(options.publisher_pacing) << "\",\"clock\":{\"source\":\""
     << BenchmarkClock::source() << "\",\"steady\":true,\"scope\":\"" << BenchmarkClock::scope()
     << "\"},\"payload_bytes\":" << payload_bytes(options)
     << ",\"application_payload_bytes\":" << application_payload_bytes(options)
     << ",\"inline_transport_bytes\":" << inline_transport_bytes(options)
     << ",\"external_buffer_bytes\":" << external_buffer_bytes(options) << ",\"pool_slots\":";
  if (pool_slots.has_value()) {
    os << *pool_slots;
  } else {
    os << "null";
  }
  os << ",\"release_capacity\":";
  if (pool_slots.has_value()) {
    os << *pool_slots;
  } else {
    os << "null";
  }
  os << ",\"publication_backpressure_capacity\":" << options.subscriber_queue_depth
     << ",\"max_fanout_release_streams\":";
  if (pool_slots.has_value()) {
    os << options.expected_subscribers;
  } else {
    os << "null";
  }
  os << ",\"payload_write_profile\":\"" << payload_write_profile(options) << '"'
     << ",\"bulk_work_profile\":\"" << bulk_work_profile(options) << '"'
     << ",\"wake\":{\"event_wake\":false,\"busy_poll\":false,\"receive_polling\":true}"
     << ",\"cpu\":" << options.cpu
     << ",\"rate_limited\":" << (options.workload == "latency" ? "true" : "false")
     << ",\"async_cuda_work\":" << (options.async_cuda_work ? "true" : "false")
     << ",\"payload_validation\":\"" << payload_validation_name(options.payload_validation)
     << "\",\"validation_timing\":\"" << validation_timing_name(options.payload_validation) << '"';
}

struct CudaStreamDeleter {
  void operator()(cudaStream_t stream) const noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream));
    }
  }
};

using UniqueCudaStream = std::unique_ptr<CUstream_st, CudaStreamDeleter>;

[[nodiscard]] bool create_non_blocking_stream(UniqueCudaStream& stream) {
  cudaStream_t raw_stream = nullptr;
  if (cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking) != cudaSuccess) {
    return false;
  }
  stream.reset(raw_stream);
  return true;
}

struct CudaDeviceDeleter {
  void operator()(void* pointer) const noexcept {
    if (pointer != nullptr) {
      static_cast<void>(cudaFree(pointer));
    }
  }
};

[[nodiscard]] bool cuda_available() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::cerr << "no CUDA device available\n";
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    std::cerr << "failed to initialize CUDA\n";
    return false;
  }
  return true;
}

struct HoloMQTensorPool {
  struct SocketCleanup {
    explicit SocketCleanup(std::string socket_path_arg) : socket_path(std::move(socket_path_arg)) {}

    ~SocketCleanup() { cleanup_holomq_benchmark_socket_path(socket_path); }

    SocketCleanup(const SocketCleanup&) = delete;
    SocketCleanup& operator=(const SocketCleanup&) = delete;

    std::string socket_path;
  };

  HoloMQTensorPool(std::string socket_path_arg, holomq::Context context_arg,
                   holomq::BufferPool pool_arg, std::vector<holoscan::Tensor> tensors_arg)
      : socket_cleanup(std::move(socket_path_arg)),
        context(std::move(context_arg)),
        pool(std::move(pool_arg)),
        tensors(std::move(tensors_arg)) {}

  HoloMQTensorPool(const HoloMQTensorPool&) = delete;
  HoloMQTensorPool& operator=(const HoloMQTensorPool&) = delete;

  // Declare cleanup first so it is destroyed last, after tensors, pool, and context teardown.
  SocketCleanup socket_cleanup;
  holomq::Context context;
  holomq::BufferPool pool;
  std::vector<holoscan::Tensor> tensors;
};

struct SidecarTensorPool {
  SidecarTensorPool(std::shared_ptr<holoscan::BlockMemoryPool> allocator_arg,
                    std::vector<holoscan::Tensor> tensors_arg)
      : allocator(std::move(allocator_arg)), tensors(std::move(tensors_arg)) {}

  SidecarTensorPool(const SidecarTensorPool&) = delete;
  SidecarTensorPool& operator=(const SidecarTensorPool&) = delete;

  // Destroy Tensor handles before their allocator releases the backing pool.
  std::shared_ptr<holoscan::BlockMemoryPool> allocator;
  std::vector<holoscan::Tensor> tensors;
};

[[nodiscard]] holoscan::expected<std::optional<holoscan::Tensor>, holoscan::Error>
try_acquire_tensor_lease(HoloMQTensorPool& tensor_pool, const Options& options) {
  auto buffer = tensor_pool.pool.acquire();
  if (!buffer) {
    return std::optional<holoscan::Tensor>{};
  }
  const std::array<int64_t, 1U> shape{
      static_cast<int64_t>(options.tensor_bytes / sizeof(std::uint32_t))};
  constexpr auto dtype = DLDataType{.code = kDLUInt, .bits = 32U, .lanes = 1U};
  auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
      shape, dtype, std::nullopt, std::move(buffer).value());
  if (!tensor) {
    return holoscan::make_unexpected(std::move(tensor).error());
  }
  return std::optional<holoscan::Tensor>{std::move(tensor).value()};
}

[[nodiscard]] bool write_tensor_pattern(void* const ptr, const std::size_t size_bytes,
                                        const std::uint32_t seed) {
  auto* words = static_cast<std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t i = 0U; i < word_count; ++i) {
    words[i] = seed + static_cast<std::uint32_t>(i);
  }
  return true;
}

[[nodiscard]] std::vector<std::uint32_t> make_tensor_pattern(const std::size_t size_bytes,
                                                             const std::uint32_t seed) {
  std::vector<std::uint32_t> words(size_bytes / sizeof(std::uint32_t));
  for (std::size_t i = 0U; i < words.size(); ++i) {
    words[i] = seed + static_cast<std::uint32_t>(i);
  }
  return words;
}

[[nodiscard]] bool initialize_tensor_buffer(holomq::Buffer& buffer, const Options& options,
                                            const std::vector<std::uint32_t>& device_pattern) {
  if (!uses_device_tensor_storage(options)) {
    return write_tensor_pattern(buffer.ptr(), buffer.size_bytes(), 0xC0010000U);
  }
  const cudaError_t result =
      cudaMemcpy(buffer.ptr(), device_pattern.data(), buffer.size_bytes(), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy HostToDevice failed during Tensor setup: "
              << cudaGetErrorString(result) << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool initialize_tensor(holoscan::Tensor& tensor, const Options& options,
                                     const std::vector<std::uint32_t>& device_pattern) {
  auto writable_data = tensor.mutable_data();
  if (!writable_data) {
    std::cerr << "Tensor setup requires writable backing: "
              << holoscan::runtime_error_message(writable_data.error()) << '\n';
    return false;
  }
  if (!uses_device_tensor_storage(options)) {
    return write_tensor_pattern(
        writable_data.value(), static_cast<std::size_t>(tensor.nbytes()), 0xC0010000U);
  }
  const cudaError_t result = cudaMemcpy(writable_data.value(),
                                        device_pattern.data(),
                                        static_cast<std::size_t>(tensor.nbytes()),
                                        cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy HostToDevice failed during Tensor setup: "
              << cudaGetErrorString(result) << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool verify_tensor_pattern(const void* const ptr, const std::size_t size_bytes,
                                         const std::uint32_t seed) {
  const auto* words = static_cast<const std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t i = 0U; i < word_count; ++i) {
    if (words[i] != seed + static_cast<std::uint32_t>(i)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool verify_tensor_pattern(const holoscan::Tensor& tensor, const Options& options,
                                         std::vector<std::uint32_t>& scratch_words) {
  const std::size_t size_bytes = static_cast<std::size_t>(tensor.nbytes());
  if (!uses_device_tensor_storage(options)) {
    const DLDeviceType expected_device_type =
        uses_pageable_host_tensor_storage(options) ? kDLCPU : kDLCUDAHost;
    return tensor.device().device_type == expected_device_type &&
           verify_tensor_pattern(tensor.data(), size_bytes, 0xC0010000U);
  }

  if (tensor.device().device_type != kDLCUDA) {
    return false;
  }
  scratch_words.resize(size_bytes / sizeof(std::uint32_t));
  const cudaError_t result =
      cudaMemcpy(scratch_words.data(), tensor.data(), size_bytes, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy DeviceToHost failed during Tensor validation: "
              << cudaGetErrorString(result) << '\n';
    return false;
  }
  return verify_tensor_pattern(scratch_words.data(), size_bytes, 0xC0010000U);
}

[[nodiscard]] std::unique_ptr<HoloMQTensorPool> make_holomq_tensor_pool(
    const Options& options, const std::size_t pool_size) {
  const std::string socket_path =
      make_unique_holomq_benchmark_socket_path("hsdk_channel_interprocess_benchmark_");
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  if (!context) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    std::cerr << "ContextBuilder::create failed\n";
    return nullptr;
  }
  auto pool = [&]() {
    if (uses_device_tensor_storage(options)) {
      return context->create_device_buffer_pool(static_cast<std::size_t>(options.tensor_bytes),
                                                pool_size);
    }
    if (uses_pageable_host_tensor_storage(options)) {
      return context->create_host_buffer_pool(static_cast<std::size_t>(options.tensor_bytes),
                                              pool_size);
    }
    return context->create_host_pinned_buffer_pool(
        static_cast<std::size_t>(options.tensor_bytes), pool_size, CU_MEMHOSTREGISTER_PORTABLE);
  }();
  if (!pool) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    std::cerr << "Context buffer pool creation failed for tensor-storage=" << options.tensor_storage
              << '\n';
    return nullptr;
  }

  const std::array<int64_t, 1U> shape{
      static_cast<int64_t>(options.tensor_bytes / sizeof(std::uint32_t))};
  constexpr auto dtype = DLDataType{.code = kDLUInt, .bits = 32U, .lanes = 1U};

  std::vector<holoscan::Tensor> tensors;
  tensors.reserve(pool_size);
  const std::vector<std::uint32_t> device_pattern =
      uses_device_tensor_storage(options)
          ? make_tensor_pattern(static_cast<std::size_t>(options.tensor_bytes), 0xC0010000U)
          : std::vector<std::uint32_t>{};
  const std::size_t retained_tensor_count = options.async_cuda_work ? 0U : pool_size;
  for (std::size_t i = 0U; i < retained_tensor_count; ++i) {
    auto buffer = pool->acquire();
    if (!buffer) {
      cleanup_holomq_benchmark_socket_path(socket_path);
      std::cerr << "BufferPool::acquire failed during setup\n";
      return nullptr;
    }
    if (!initialize_tensor_buffer(buffer.value(), options, device_pattern)) {
      cleanup_holomq_benchmark_socket_path(socket_path);
      return nullptr;
    }
    auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
        shape, dtype, std::nullopt, std::move(buffer).value());
    if (!tensor) {
      cleanup_holomq_benchmark_socket_path(socket_path);
      std::cerr << "make_tensor_from_holomq_buffer failed: "
                << holoscan::runtime_error_message(tensor.error()) << '\n';
      return nullptr;
    }
    tensors.emplace_back(std::move(tensor).value());
  }
  return std::make_unique<HoloMQTensorPool>(
      socket_path, std::move(*context), std::move(*pool), std::move(tensors));
}

[[nodiscard]] std::unique_ptr<SidecarTensorPool> make_sidecar_tensor_pool(
    const Options& options, const std::size_t pool_size,
    const holoscan::MemoryBuffer::Options& memory_options) {
  auto allocator = std::make_shared<holoscan::BlockMemoryPool>(
      tensor_storage_type(options), options.tensor_bytes, pool_size);
  auto initialized = allocator->initialize_pool();
  if (!initialized) {
    std::cerr << "BlockMemoryPool::initialize_pool failed: "
              << holoscan::runtime_error_message(initialized.error()) << '\n';
    return nullptr;
  }

  const std::array<int64_t, 1U> shape{
      static_cast<int64_t>(options.tensor_bytes / sizeof(std::uint32_t))};
  constexpr auto dtype = DLDataType{.code = kDLUInt, .bits = 32U, .lanes = 1U};
  const std::vector<std::uint32_t> device_pattern =
      uses_device_tensor_storage(options)
          ? make_tensor_pattern(static_cast<std::size_t>(options.tensor_bytes), 0xC0010000U)
          : std::vector<std::uint32_t>{};

  std::vector<holoscan::Tensor> tensors;
  tensors.reserve(pool_size);
  for (std::size_t i = 0U; i < pool_size; ++i) {
    auto tensor = holoscan::make_tensor(
        shape, dtype, std::nullopt, tensor_storage_type(options), allocator, memory_options);
    if (!tensor) {
      std::cerr << "make_tensor failed for sidecar pool: "
                << holoscan::runtime_error_message(tensor.error()) << '\n';
      return nullptr;
    }
    if (!initialize_tensor(tensor.value(), options, device_pattern)) {
      return nullptr;
    }
    tensors.emplace_back(std::move(tensor).value());
  }
  return std::make_unique<SidecarTensorPool>(std::move(allocator), std::move(tensors));
}

[[nodiscard]] bool validate_reached_total(const Options& options, const std::uint64_t total,
                                          const std::uint64_t reached_total) {
  const std::uint64_t expected_reached_total =
      total * static_cast<std::uint64_t>(options.expected_subscribers);
  if (reached_total == expected_reached_total) {
    return true;
  }
  std::cerr << "publisher reached " << reached_total << " total recipients; expected "
            << expected_reached_total << '\n';
  return false;
}

class StartGateParticipant {
 public:
  explicit StartGateParticipant(std::barrier<>* const barrier) noexcept : barrier_(barrier) {}

  ~StartGateParticipant() {
    if (barrier_ != nullptr) {
      barrier_->arrive_and_drop();
    }
  }

  StartGateParticipant(const StartGateParticipant&) = delete;
  StartGateParticipant& operator=(const StartGateParticipant&) = delete;

  void arrive_and_wait() {
    if (barrier_ != nullptr) {
      std::barrier<>* const barrier = std::exchange(barrier_, nullptr);
      barrier->arrive_and_wait();
    }
  }

 private:
  std::barrier<>* barrier_;
};

[[nodiscard]] holoscan::ChannelRuntime* select_runtime(
    holoscan::ChannelRuntime* const shared_runtime,
    std::optional<holoscan::ChannelRuntime>& owned_runtime) {
  if (shared_runtime != nullptr) {
    return shared_runtime;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "ChannelRuntime::create failed: "
              << holoscan::runtime_error_message(runtime.error()) << '\n';
    return nullptr;
  }
  owned_runtime.emplace(std::move(runtime).value());
  return &owned_runtime.value();
}

template <typename CreateEndpoint>
[[nodiscard]] auto create_endpoint(std::mutex* const runtime_mutex, CreateEndpoint create) {
  if (runtime_mutex == nullptr) {
    return create();
  }
  const std::lock_guard<std::mutex> lock(*runtime_mutex);
  return create();
}

[[nodiscard]] int run_publisher(const Options& options,
                                holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                std::mutex* const runtime_mutex = nullptr,
                                std::barrier<>* const start_barrier = nullptr,
                                std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 3;
  }

  auto publisher = create_endpoint(runtime_mutex, [&]() {
    return runtime->create_publisher<BenchmarkPOD64>(options.channel_name,
                                                     make_channel_options(options));
  });
  if (!publisher) {
    std::cerr << "create_publisher failed: " << holoscan::runtime_error_message(publisher.error())
              << '\n';
    return 4;
  }
  if (!pin_to_cpu(options.cpu, "publisher")) {
    return 2;
  }
  start_gate.arrive_and_wait();
  if (!wait_for_channel_subscribers(
          publisher.value(), options.expected_subscribers, [&publisher]() {
            const std::int64_t probe_time_ns = now_ns();
            return publisher->emit(make_payload(0U, probe_time_ns, probe_time_ns));
          })) {
    return 5;
  }

  std::vector<std::uint64_t> channel_emit_ns;
  std::vector<std::uint64_t> publisher_full_ns;
  channel_emit_ns.reserve(static_cast<std::size_t>(options.samples));
  publisher_full_ns.reserve(static_cast<std::size_t>(options.samples));

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t failed_emits = 0U;
  std::uint64_t reached_total = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  PublisherPacer publisher_pacer(options);
  for (std::uint64_t sequence = 1U; sequence <= total; ++sequence) {
    publisher_pacer.wait();
    const bool measured = sequence > options.warmup;
    const std::int64_t full_start_time_ns = now_ns();
    const std::int64_t send_time_ns = now_ns();
    const BenchmarkPOD64 payload = make_payload(sequence, full_start_time_ns, send_time_ns);

    const std::int64_t emit_start_ns = now_ns();
    auto emitted = publisher->emit(payload);
    const std::uint64_t emit_elapsed_ns = elapsed_ns(emit_start_ns);
    if (!emitted) {
      ++failed_emits;
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured ||
          emitted.error().code == holoscan::ErrorCode::kNotReady) {
        --sequence;
        std::this_thread::yield();
        continue;
      }
      std::cerr << "emit failed: " << holoscan::runtime_error_message(emitted.error()) << '\n';
      return 6;
    }
    reached_total += emitted.value();
    if (measured) {
      if (measured_start_ns == 0) {
        measured_start_ns = full_start_time_ns;
      }
      channel_emit_ns.push_back(emit_elapsed_ns);
      publisher_full_ns.push_back(elapsed_ns(full_start_time_ns));
      measured_end_ns = emit_start_ns + static_cast<std::int64_t>(emit_elapsed_ns);
    }
  }

  if (!validate_reached_total(options, total, reached_total)) {
    return 7;
  }
  if (options.publisher_hold_ms > 0U) {
    std::this_thread::sleep_for(std::chrono::milliseconds(options.publisher_hold_ms));
  }

  output << '{';
  emit_common_json(output, options, "publisher");
  output << ",\"failed_emits\":" << failed_emits << ",\"reached_total\":" << reached_total;
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_emit", summarize_percentiles(std::move(channel_emit_ns)));
  emit_percentiles(output, "publisher_full", summarize_percentiles(std::move(publisher_full_ns)));
  output << "}\n";
  return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_tensor_publisher(const Options& options,
                                       holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                       std::mutex* const runtime_mutex = nullptr,
                                       std::barrier<>* const start_barrier = nullptr,
                                       std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  if (!cuda_available()) {
    return 2;
  }
  UniqueCudaStream producer_stream;
  if (options.async_cuda_work && !create_non_blocking_stream(producer_stream)) {
    std::cerr << "failed to create producer CUDA stream\n";
    return 3;
  }

  constexpr std::size_t kMinimumPoolSize = 256U;
  const std::size_t pool_size =
      std::max<std::size_t>(kMinimumPoolSize, options.subscriber_queue_depth + 8U);
  // Declare pool ownership before endpoints so release callbacks run before backing teardown.
  std::unique_ptr<HoloMQTensorPool> holomq_tensor_pool;
  std::unique_ptr<SidecarTensorPool> sidecar_tensor_pool;
  std::vector<holoscan::Tensor>* tensor_handles = nullptr;

  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 4;
  }

  auto publisher = create_endpoint(runtime_mutex, [&]() {
    return holoscan::channel_internal::TensorPublisherTestAccess::create(
        *runtime, options.channel_name, make_channel_options(options));
  });
  if (!publisher) {
    std::cerr << "create_publisher failed: " << holoscan::runtime_error_message(publisher.error())
              << '\n';
    return 5;
  }

  if (uses_sidecar_transport(options)) {
    const holoscan::MemoryBuffer::Options memory_options{.max_fanout_release_streams =
                                                             options.expected_subscribers};
    sidecar_tensor_pool = make_sidecar_tensor_pool(options, pool_size, memory_options);
    if (sidecar_tensor_pool == nullptr) {
      return 3;
    }
    tensor_handles = &sidecar_tensor_pool->tensors;
  } else {
    holomq_tensor_pool = make_holomq_tensor_pool(options, pool_size);
    if (holomq_tensor_pool == nullptr) {
      return 3;
    }
    tensor_handles = &holomq_tensor_pool->tensors;
  }

  if (!pin_to_cpu(options.cpu, "publisher")) {
    return 2;
  }
  start_gate.arrive_and_wait();
  if (!wait_for_channel_subscribers(
          publisher.value(),
          options.expected_subscribers,
          [&publisher, &holomq_tensor_pool, tensor_handles, &options]()
              -> holoscan::expected<std::uint32_t, holoscan::Error> {
            const std::int64_t probe_time_ns = now_ns();
            std::optional<holoscan::Tensor> probe_lease;
            holoscan::Tensor* tensor = nullptr;
            if (options.async_cuda_work) {
              auto acquired = try_acquire_tensor_lease(*holomq_tensor_pool, options);
              if (!acquired) {
                return holoscan::make_unexpected(std::move(acquired).error());
              }
              probe_lease = std::move(acquired).value();
              if (!probe_lease) {
                return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
              }
              tensor = &probe_lease.value();
            } else {
              tensor = &tensor_handles->front();
            }
            const holoscan::Sample<holoscan::Tensor> probe(
                *tensor,
                holoscan::SampleMetadata{.sequence = uses_sidecar_transport(options)
                                                         ? std::numeric_limits<std::uint64_t>::max()
                                                         : 0U,
                                         .capture_timestamp_ns = probe_time_ns,
                                         .source_clock_id = holoscan::ClockId{1U},
                                         .publish_timestamp_ns = probe_time_ns,
                                         .publish_clock_id = holoscan::ClockId{1U},
                                         .byte_size = static_cast<std::uint64_t>(tensor->nbytes()),
                                         .payload_kind = holoscan::PayloadKind::kDescriptorBacked});
            return publisher->emit_sample(probe);
          })) {
    return 6;
  }

  std::vector<std::uint64_t> channel_emit_ns;
  std::vector<std::uint64_t> publisher_full_ns;
  std::vector<std::uint64_t> cuda_work_enqueue_ns;
  channel_emit_ns.reserve(static_cast<std::size_t>(options.samples));
  publisher_full_ns.reserve(static_cast<std::size_t>(options.samples));
  cuda_work_enqueue_ns.reserve(static_cast<std::size_t>(options.samples));

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t failed_emits = 0U;
  std::uint64_t async_lease_acquisitions = 0U;
  std::uint64_t reached_total = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  PublisherPacer publisher_pacer(options);
  for (std::uint64_t sequence = 1U; sequence <= total; ++sequence) {
    publisher_pacer.wait();
    const bool measured = sequence > options.warmup;
    const std::int64_t full_start_time_ns = now_ns();
    std::optional<holoscan::Tensor> async_tensor_lease;
    holoscan::Tensor* tensor = nullptr;
    if (options.async_cuda_work) {
      auto acquired = try_acquire_tensor_lease(*holomq_tensor_pool, options);
      if (!acquired) {
        std::cerr << "failed to wrap acquired HoloMQ Tensor lease: "
                  << holoscan::runtime_error_message(acquired.error()) << '\n';
        return 7;
      }
      async_tensor_lease = std::move(acquired).value();
      if (!async_tensor_lease) {
        --sequence;
        std::this_thread::yield();
        continue;
      }
      ++async_lease_acquisitions;
      tensor = &async_tensor_lease.value();
    } else {
      tensor = &(*tensor_handles)[(sequence - 1U) % tensor_handles->size()];
    }
    std::uint64_t cuda_enqueue_elapsed_ns = 0U;
    if (options.async_cuda_work) {
      const std::int64_t cuda_enqueue_start_ns = now_ns();
      auto writable_data = tensor->mutable_data();
      if (!writable_data) {
        std::cerr << "asynchronous Tensor setup requires writable backing: "
                  << holoscan::runtime_error_message(writable_data.error()) << '\n';
        return 7;
      }
      if (cudaMemsetAsync(writable_data.value(),
                          kAsyncCudaPattern,
                          static_cast<std::size_t>(tensor->nbytes()),
                          producer_stream.get()) != cudaSuccess ||
          !tensor->set_readiness_stream(producer_stream.get())) {
        std::cerr << "failed to enqueue producer CUDA work\n";
        return 7;
      }
      cuda_enqueue_elapsed_ns = elapsed_ns(cuda_enqueue_start_ns);
    }
    const std::int64_t send_time_ns = now_ns();
    const holoscan::Sample<holoscan::Tensor> sample(
        *tensor,
        holoscan::SampleMetadata{.sequence = sequence,
                                 .capture_timestamp_ns = full_start_time_ns,
                                 .source_clock_id = holoscan::ClockId{1U},
                                 .publish_timestamp_ns = send_time_ns,
                                 .publish_clock_id = holoscan::ClockId{1U},
                                 .byte_size = static_cast<std::uint64_t>(tensor->nbytes()),
                                 .payload_kind = holoscan::PayloadKind::kDescriptorBacked});

    const std::int64_t emit_start_ns = now_ns();
    auto emitted = publisher->emit_sample(sample);
    const std::uint64_t emit_elapsed_ns = elapsed_ns(emit_start_ns);
    if (!emitted) {
      ++failed_emits;
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured ||
          emitted.error().code == holoscan::ErrorCode::kNotReady) {
        --sequence;
        std::this_thread::yield();
        continue;
      }
      std::cerr << "emit failed: " << holoscan::runtime_error_message(emitted.error()) << '\n';
      return 7;
    }
    reached_total += emitted.value();
    if (measured) {
      if (measured_start_ns == 0) {
        measured_start_ns = full_start_time_ns;
      }
      channel_emit_ns.push_back(emit_elapsed_ns);
      publisher_full_ns.push_back(elapsed_ns(full_start_time_ns));
      if (options.async_cuda_work) {
        cuda_work_enqueue_ns.push_back(cuda_enqueue_elapsed_ns);
      }
      measured_end_ns = emit_start_ns + static_cast<std::int64_t>(emit_elapsed_ns);
    }
  }

  if (!validate_reached_total(options, total, reached_total)) {
    return 9;
  }
  if (options.publisher_hold_ms > 0U) {
    std::this_thread::sleep_for(std::chrono::milliseconds(options.publisher_hold_ms));
  }
  if (options.async_cuda_work && cudaStreamSynchronize(producer_stream.get()) != cudaSuccess) {
    std::cerr << "producer CUDA stream synchronization failed\n";
    return 8;
  }

  output << '{';
  emit_common_json(output, options, "publisher");
  output << ",\"failed_emits\":" << failed_emits
         << ",\"async_lease_acquisitions\":" << async_lease_acquisitions
         << ",\"reached_total\":" << reached_total;
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_emit", summarize_percentiles(std::move(channel_emit_ns)));
  emit_percentiles(output, "publisher_full", summarize_percentiles(std::move(publisher_full_ns)));
  if (options.async_cuda_work) {
    emit_percentiles(
        output, "cuda_work_enqueue", summarize_percentiles(std::move(cuda_work_enqueue_ns)));
  }
  output << "}\n";
  return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_image_publisher(const Options& options,
                                      holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                      std::mutex* const runtime_mutex = nullptr,
                                      std::barrier<>* const start_barrier = nullptr,
                                      std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  if (!cuda_available()) {
    return 2;
  }
  constexpr std::size_t kMinimumPoolSize = 256U;
  const std::size_t pool_size =
      std::max<std::size_t>(kMinimumPoolSize, options.subscriber_queue_depth + 8U);
  // Declare pool ownership before endpoints so release callbacks run before backing teardown.
  std::unique_ptr<HoloMQTensorPool> holomq_tensor_pool;
  std::unique_ptr<SidecarTensorPool> sidecar_tensor_pool;
  std::vector<holoscan::Tensor>* tensor_handles = nullptr;

  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 4;
  }

  std::optional<ImagePublisher> publisher;
  if (uses_sidecar_transport(options)) {
    auto created = create_endpoint(runtime_mutex, [&]() {
      return runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(
          options.channel_name, make_channel_options(options));
    });
    if (!created) {
      std::cerr << "create_tensor_schema_publisher failed: "
                << holoscan::runtime_error_message(created.error()) << '\n';
      return 5;
    }
    publisher.emplace(std::in_place_type<ImageSidecarPublisher>, std::move(created).value());
  } else {
    auto created = create_endpoint(runtime_mutex, [&]() {
      return runtime->create_publisher<DescriptorBackedImagePayload>(options.channel_name,
                                                                     make_channel_options(options));
    });
    if (!created) {
      std::cerr << "create_publisher failed: " << holoscan::runtime_error_message(created.error())
                << '\n';
      return 5;
    }
    publisher.emplace(std::in_place_type<ImageInlinePublisher>, std::move(created).value());
  }

  if (uses_sidecar_transport(options)) {
    auto memory_options = std::get<ImageSidecarPublisher>(*publisher).tensor_memory_options();
    if (!memory_options) {
      std::cerr << "tensor_memory_options failed: "
                << holoscan::runtime_error_message(memory_options.error()) << '\n';
      return 5;
    }
    sidecar_tensor_pool = make_sidecar_tensor_pool(options, pool_size, memory_options.value());
    if (sidecar_tensor_pool == nullptr) {
      return 3;
    }
    tensor_handles = &sidecar_tensor_pool->tensors;
  } else {
    holomq_tensor_pool = make_holomq_tensor_pool(options, pool_size);
    if (holomq_tensor_pool == nullptr) {
      return 3;
    }
    tensor_handles = &holomq_tensor_pool->tensors;
  }

  if (!pin_to_cpu(options.cpu, "publisher")) {
    return 2;
  }
  start_gate.arrive_and_wait();
  const bool subscribers_ready = std::visit(
      [&](auto& endpoint) {
        return wait_for_channel_subscribers(
            endpoint, options.expected_subscribers, [&endpoint, tensor_handles, &options]() {
              const std::int64_t probe_time_ns = now_ns();
              return emit_image(endpoint, tensor_handles->front(), options, 0U, probe_time_ns);
            });
      },
      *publisher);
  if (!subscribers_ready) {
    return 6;
  }

  std::vector<std::uint64_t> channel_emit_ns;
  std::vector<std::uint64_t> publisher_full_ns;
  channel_emit_ns.reserve(static_cast<std::size_t>(options.samples));
  publisher_full_ns.reserve(static_cast<std::size_t>(options.samples));

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t failed_emits = 0U;
  std::uint64_t payload_build_errors = 0U;
  std::uint64_t reached_total = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  PublisherPacer publisher_pacer(options);
  for (std::uint64_t sequence = 1U; sequence <= total; ++sequence) {
    publisher_pacer.wait();
    const bool measured = sequence > options.warmup;
    const std::int64_t full_start_time_ns = now_ns();
    const holoscan::Tensor& tensor = (*tensor_handles)[(sequence - 1U) % tensor_handles->size()];

    const std::int64_t emit_start_ns = now_ns();
    auto emitted = std::visit(
        [&](auto& endpoint) {
          return emit_image(endpoint, tensor, options, sequence, full_start_time_ns);
        },
        *publisher);
    const std::uint64_t emit_elapsed_ns = elapsed_ns(emit_start_ns);
    if (!emitted) {
      ++failed_emits;
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured ||
          emitted.error().code == holoscan::ErrorCode::kNotReady) {
        --sequence;
        std::this_thread::yield();
        continue;
      }
      ++payload_build_errors;
      std::cerr << "image emit failed: " << holoscan::runtime_error_message(emitted.error())
                << '\n';
      return 8;
    }
    reached_total += emitted.value();
    if (measured) {
      if (measured_start_ns == 0) {
        measured_start_ns = full_start_time_ns;
      }
      channel_emit_ns.push_back(emit_elapsed_ns);
      publisher_full_ns.push_back(elapsed_ns(full_start_time_ns));
      measured_end_ns = emit_start_ns + static_cast<std::int64_t>(emit_elapsed_ns);
    }
  }

  if (!validate_reached_total(options, total, reached_total)) {
    return 9;
  }
  if (options.publisher_hold_ms > 0U) {
    std::this_thread::sleep_for(std::chrono::milliseconds(options.publisher_hold_ms));
  }

  output << '{';
  emit_common_json(output, options, "publisher");
  output << ",\"failed_emits\":" << failed_emits
         << ",\"payload_build_errors\":" << payload_build_errors
         << ",\"reached_total\":" << reached_total;
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_emit", summarize_percentiles(std::move(channel_emit_ns)));
  emit_percentiles(output, "publisher_full", summarize_percentiles(std::move(publisher_full_ns)));
  output << "}\n";
  return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_subscriber(const Options& options,
                                 holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                 std::mutex* const runtime_mutex = nullptr,
                                 std::barrier<>* const start_barrier = nullptr,
                                 std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 3;
  }

  auto subscriber = create_endpoint(runtime_mutex, [&]() {
    return runtime->create_subscriber<BenchmarkPOD64>(options.channel_name,
                                                      make_channel_options(options));
  });
  if (!subscriber) {
    std::cerr << "create_subscriber failed: " << holoscan::runtime_error_message(subscriber.error())
              << '\n';
    return 4;
  }
  if (!pin_to_cpu(options.cpu, "subscriber")) {
    return 2;
  }
  start_gate.arrive_and_wait();

  std::vector<std::uint64_t> channel_receive_ns;
  std::vector<std::uint64_t> end_to_end_ns;
  std::vector<std::uint64_t> full_end_to_end_ns;
  channel_receive_ns.reserve(static_cast<std::size_t>(options.samples));
  end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));
  full_end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t received_count = 0U;
  std::uint64_t sequence_gaps = 0U;
  std::uint64_t not_ready_polls = 0U;
  std::uint64_t checksum_errors = 0U;
  std::uint64_t last_sequence = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  const auto deadline = Clock::now() + std::chrono::seconds(60);

  while (received_count < total && Clock::now() < deadline) {
    const std::int64_t receive_start_ns = now_ns();
    auto received = subscriber->receive();
    const std::int64_t receive_done_ns = now_ns();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        std::cerr << "receive failed: " << holoscan::runtime_error_message(received.error())
                  << '\n';
        return 5;
      }
      ++not_ready_polls;
      std::this_thread::yield();
      continue;
    }

    const BenchmarkPOD64 payload = std::move(received).value();
    if (payload.sequence == 0U) {
      continue;
    }
    ++received_count;
    if (last_sequence != 0U && payload.sequence != last_sequence + 1U) {
      sequence_gaps +=
          payload.sequence > last_sequence ? payload.sequence - last_sequence - 1U : 1U;
    }
    last_sequence = payload.sequence;
    const bool validate_payload =
        options.payload_validation == PayloadValidation::kAll ||
        (options.payload_validation == PayloadValidation::kFinal && received_count == total);
    if (validate_payload && !holoscan::benchmark_pod_valid(payload)) {
      ++checksum_errors;
    }

    if (payload.sequence > options.warmup) {
      if (measured_start_ns == 0) {
        measured_start_ns = receive_start_ns;
      }
      channel_receive_ns.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      end_to_end_ns.push_back(static_cast<std::uint64_t>(receive_done_ns - payload.send_time_ns));
      full_end_to_end_ns.push_back(
          static_cast<std::uint64_t>(receive_done_ns - payload.source_start_time_ns));
      measured_end_ns = receive_done_ns;
    }
  }

  if (received_count != total) {
    std::cerr << "timed out after receiving " << received_count << " of " << total << " samples\n";
    return 6;
  }

  output << '{';
  emit_common_json(output, options, "subscriber");
  output << ",\"received_count\":" << received_count << ",\"sequence_gaps\":" << sequence_gaps
         << ",\"not_ready_polls\":" << not_ready_polls
         << ",\"checksum_errors\":" << checksum_errors;
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_receive", summarize_percentiles(std::move(channel_receive_ns)));
  emit_percentiles(output, "end_to_end", summarize_percentiles(std::move(end_to_end_ns)));
  emit_percentiles(output, "full_end_to_end", summarize_percentiles(std::move(full_end_to_end_ns)));
  output << "}\n";
  return checksum_errors == 0U ? 0 : 7;
}

// Benchmark receive loop keeps timing, validation, and JSON counters adjacent so measured
// intervals are easy to audit.
// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] int run_tensor_subscriber(const Options& options,
                                        holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                        std::mutex* const runtime_mutex = nullptr,
                                        std::barrier<>* const start_barrier = nullptr,
                                        std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  if (!cuda_available()) {
    return 2;
  }
  UniqueCudaStream consumer_stream;
  std::unique_ptr<void, CudaDeviceDeleter> device_scratch;
  if (options.async_cuda_work) {
    if (!create_non_blocking_stream(consumer_stream)) {
      std::cerr << "failed to create consumer CUDA stream\n";
      return 3;
    }
    void* scratch = nullptr;
    if (cudaMalloc(&scratch, static_cast<std::size_t>(options.tensor_bytes)) != cudaSuccess) {
      std::cerr << "failed to allocate consumer CUDA scratch\n";
      return 3;
    }
    device_scratch.reset(scratch);
  }

  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 3;
  }

  auto subscriber = create_endpoint(runtime_mutex, [&]() {
    return runtime->create_subscriber<holoscan::Tensor>(options.channel_name,
                                                        make_channel_options(options));
  });
  if (!subscriber) {
    std::cerr << "create_subscriber failed: " << holoscan::runtime_error_message(subscriber.error())
              << '\n';
    return 4;
  }
  if (!pin_to_cpu(options.cpu, "subscriber")) {
    return 2;
  }
  start_gate.arrive_and_wait();

  std::vector<std::uint64_t> channel_receive_ns;
  std::vector<std::uint64_t> end_to_end_ns;
  std::vector<std::uint64_t> full_end_to_end_ns;
  std::vector<std::uint64_t> cuda_work_enqueue_ns;
  channel_receive_ns.reserve(static_cast<std::size_t>(options.samples));
  end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));
  full_end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));
  cuda_work_enqueue_ns.reserve(static_cast<std::size_t>(options.samples));
  std::vector<std::uint32_t> validation_scratch;
  std::optional<holoscan::Tensor> validation_tensor;

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t received_count = 0U;
  std::uint64_t sequence_gaps = 0U;
  std::uint64_t not_ready_polls = 0U;
  std::uint64_t validation_errors = 0U;
  std::uint64_t last_sequence = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  const auto deadline = Clock::now() + std::chrono::seconds(60);

  while (received_count < total && Clock::now() < deadline) {
    const std::int64_t receive_start_ns = now_ns();
    auto received = subscriber->receive_sample();
    const std::int64_t receive_done_ns = now_ns();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        std::cerr << "receive_sample failed: " << holoscan::runtime_error_message(received.error())
                  << '\n';
        return 5;
      }
      ++not_ready_polls;
      std::this_thread::yield();
      continue;
    }

    holoscan::Sample<holoscan::Tensor> sample = std::move(received).value();
    const std::uint64_t sequence = sample.metadata.sequence;
    if (sequence == 0U || sequence == std::numeric_limits<std::uint64_t>::max()) {
      continue;
    }
    ++received_count;
    if (last_sequence != 0U && sequence != last_sequence + 1U) {
      sequence_gaps += sequence > last_sequence ? sequence - last_sequence - 1U : 1U;
    }
    last_sequence = sequence;
    const bool valid_tensor =
        sample.metadata.payload_kind == holoscan::PayloadKind::kDescriptorBacked &&
        sample.metadata.byte_size == options.tensor_bytes &&
        static_cast<std::uint64_t>(sample.data.nbytes()) == options.tensor_bytes &&
        (uses_sidecar_transport(options) || sample.data.is_read_only()) &&
        (!options.async_cuda_work || sample.data.device().device_type == kDLCUDA);
    std::uint64_t cuda_enqueue_elapsed_ns = 0U;
    bool cuda_work_valid = true;
    if (options.async_cuda_work && valid_tensor) {
      const std::int64_t cuda_enqueue_start_ns = now_ns();
      auto synchronized = sample.data.synchronize_for_consumer_stream(consumer_stream.get());
      cuda_work_valid = synchronized.has_value() &&
                        cudaMemcpyAsync(device_scratch.get(),
                                        sample.data.data(),
                                        static_cast<std::size_t>(options.tensor_bytes),
                                        cudaMemcpyDeviceToDevice,
                                        consumer_stream.get()) == cudaSuccess &&
                        sample.data.set_deallocation_stream(consumer_stream.get());
      cuda_enqueue_elapsed_ns = elapsed_ns(cuda_enqueue_start_ns);
    }
    const bool payload_valid = !valid_tensor || options.async_cuda_work ||
                               options.payload_validation != PayloadValidation::kAll ||
                               verify_tensor_pattern(sample.data, options, validation_scratch);
    if (!valid_tensor || !cuda_work_valid || !payload_valid) {
      ++validation_errors;
    }

    if (sequence > options.warmup) {
      if (measured_start_ns == 0) {
        measured_start_ns = receive_start_ns;
      }
      channel_receive_ns.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      end_to_end_ns.push_back(
          static_cast<std::uint64_t>(receive_done_ns - sample.metadata.publish_timestamp_ns));
      full_end_to_end_ns.push_back(
          static_cast<std::uint64_t>(receive_done_ns - sample.metadata.capture_timestamp_ns));
      if (options.async_cuda_work) {
        cuda_work_enqueue_ns.push_back(cuda_enqueue_elapsed_ns);
      }
      measured_end_ns = options.async_cuda_work ? now_ns() : receive_done_ns;
    }
    if (!options.async_cuda_work && options.payload_validation == PayloadValidation::kFinal &&
        received_count == total) {
      validation_tensor.emplace(std::move(sample.data));
    }
  }

  if (received_count != total) {
    std::cerr << "timed out after receiving " << received_count << " of " << total << " samples\n";
    return 6;
  }
  const std::optional<std::int64_t> final_validation_start_ns =
      options.payload_validation == PayloadValidation::kFinal
          ? std::optional<std::int64_t>{now_ns()}
          : std::nullopt;
  if (options.async_cuda_work) {
    if (cudaStreamSynchronize(consumer_stream.get()) != cudaSuccess) {
      std::cerr << "consumer CUDA stream synchronization failed\n";
      return 7;
    }
    if (options.payload_validation != PayloadValidation::kNone) {
      validation_scratch.resize(static_cast<std::size_t>(options.tensor_bytes) /
                                sizeof(std::uint32_t));
      if (cudaMemcpy(validation_scratch.data(),
                     device_scratch.get(),
                     static_cast<std::size_t>(options.tensor_bytes),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          !std::all_of(reinterpret_cast<const std::uint8_t*>(validation_scratch.data()),
                       reinterpret_cast<const std::uint8_t*>(validation_scratch.data()) +
                           options.tensor_bytes,
                       [](const std::uint8_t value) { return value == kAsyncCudaPattern; })) {
        ++validation_errors;
      }
    }
  } else if (options.payload_validation == PayloadValidation::kFinal &&
             (!validation_tensor.has_value() ||
              !verify_tensor_pattern(validation_tensor.value(), options, validation_scratch))) {
    ++validation_errors;
  }
  const std::optional<std::uint64_t> final_validation_ns =
      final_validation_start_ns
          ? std::optional<std::uint64_t>{elapsed_ns(*final_validation_start_ns)}
          : std::nullopt;

  output << '{';
  emit_common_json(output, options, "subscriber");
  output << ",\"received_count\":" << received_count << ",\"sequence_gaps\":" << sequence_gaps
         << ",\"not_ready_polls\":" << not_ready_polls
         << ",\"validation_errors\":" << validation_errors;
  emit_optional_duration(output, "final_validation_ns", final_validation_ns);
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_receive", summarize_percentiles(std::move(channel_receive_ns)));
  emit_percentiles(output, "end_to_end", summarize_percentiles(std::move(end_to_end_ns)));
  emit_percentiles(output, "full_end_to_end", summarize_percentiles(std::move(full_end_to_end_ns)));
  if (options.async_cuda_work) {
    emit_percentiles(
        output, "cuda_work_enqueue", summarize_percentiles(std::move(cuda_work_enqueue_ns)));
  }
  output << "}\n";
  return validation_errors == 0U ? 0 : 7;
}
// NOLINTEND(readability-function-cognitive-complexity)

// Benchmark receive loop keeps timing, validation, and JSON counters adjacent so measured
// intervals are easy to audit.
// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] int run_image_subscriber(const Options& options,
                                       holoscan::ChannelRuntime* const shared_runtime = nullptr,
                                       std::mutex* const runtime_mutex = nullptr,
                                       std::barrier<>* const start_barrier = nullptr,
                                       std::ostream& output = std::cout) {
  StartGateParticipant start_gate(start_barrier);
  if (!cuda_available()) {
    return 2;
  }
  std::optional<holoscan::ChannelRuntime> owned_runtime;
  holoscan::ChannelRuntime* const runtime = select_runtime(shared_runtime, owned_runtime);
  if (runtime == nullptr) {
    return 3;
  }

  auto subscriber = create_endpoint(runtime_mutex, [&]() {
    return runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(
        options.channel_name, make_channel_options(options));
  });
  if (!subscriber) {
    std::cerr << "create_tensor_schema_subscriber failed: "
              << holoscan::runtime_error_message(subscriber.error()) << '\n';
    return 4;
  }
  if (!pin_to_cpu(options.cpu, "subscriber")) {
    return 2;
  }
  start_gate.arrive_and_wait();

  std::vector<std::uint64_t> channel_receive_ns;
  // No publish-to-receive series here. This subscriber resolves a ChannelResolvedTensorSample,
  // which carries no SampleMetadata, and the transport descriptor carries no timestamp, so the
  // publish instant does not cross the process boundary on this path.
  std::vector<std::uint64_t> full_end_to_end_ns;
  channel_receive_ns.reserve(static_cast<std::size_t>(options.samples));
  full_end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));
  std::vector<std::uint32_t> validation_scratch;
  std::optional<holoscan::Tensor> validation_tensor;

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t received_count = 0U;
  std::uint64_t sequence_gaps = 0U;
  std::uint64_t not_ready_polls = 0U;
  std::uint64_t validation_errors = 0U;
  std::uint64_t last_sequence = 0U;
  std::int64_t measured_start_ns = 0;
  std::int64_t measured_end_ns = 0;
  const auto deadline = Clock::now() + std::chrono::seconds(60);

  while (received_count < total && Clock::now() < deadline) {
    const std::int64_t receive_start_ns = now_ns();
    auto received = subscriber->receive();
    const std::int64_t receive_done_ns = now_ns();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        std::cerr << "receive failed: " << holoscan::runtime_error_message(received.error())
                  << '\n';
        return 5;
      }
      ++not_ready_polls;
      std::this_thread::yield();
      continue;
    }

    auto sample = std::move(received).value();
    const std::uint64_t sequence = sample.payload.sequence;
    const holoscan::schema::Image* image = sample.root();
    const auto* header = image == nullptr ? nullptr : image->header();
    const auto* frame_id = header == nullptr ? nullptr : header->frame_id();
    if (sequence == 0U ||
        (frame_id != nullptr &&
         frame_id->string_view() == std::string_view{"channel_interprocess_image_0"})) {
      continue;
    }
    ++received_count;
    if (last_sequence != 0U && sequence != last_sequence + 1U) {
      sequence_gaps += sequence > last_sequence ? sequence - last_sequence - 1U : 1U;
    }
    last_sequence = sequence;

    const bool valid_image_timestamps = header != nullptr;
    const bool valid_tensor =
        sample.payload.payload_kind == tensor_payload_kind(options) &&
        sample.payload.byte_size == options.tensor_bytes &&
        static_cast<std::uint64_t>(sample.tensor.nbytes()) == options.tensor_bytes &&
        (uses_sidecar_transport(options) || sample.tensor.is_read_only());
    const bool payload_valid = !valid_tensor ||
                               options.payload_validation != PayloadValidation::kAll ||
                               verify_tensor_pattern(sample.tensor, options, validation_scratch);
    if (!valid_tensor || !payload_valid || !valid_image_timestamps) {
      ++validation_errors;
    }

    if (received_count > options.warmup) {
      if (measured_start_ns == 0) {
        measured_start_ns = receive_start_ns;
      }
      channel_receive_ns.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      if (valid_image_timestamps) {
        full_end_to_end_ns.push_back(
            static_cast<std::uint64_t>(receive_done_ns - header->capture_timestamp_ns()));
      }
      measured_end_ns = receive_done_ns;
    }
    if (options.payload_validation == PayloadValidation::kFinal && received_count == total) {
      validation_tensor.emplace(std::move(sample.tensor));
    }
  }

  if (received_count != total) {
    std::cerr << "timed out after receiving " << received_count << " of " << total << " samples\n";
    return 6;
  }
  const std::optional<std::int64_t> final_validation_start_ns =
      options.payload_validation == PayloadValidation::kFinal
          ? std::optional<std::int64_t>{now_ns()}
          : std::nullopt;
  if (options.payload_validation == PayloadValidation::kFinal &&
      (!validation_tensor.has_value() ||
       !verify_tensor_pattern(validation_tensor.value(), options, validation_scratch))) {
    ++validation_errors;
  }
  const std::optional<std::uint64_t> final_validation_ns =
      final_validation_start_ns
          ? std::optional<std::uint64_t>{elapsed_ns(*final_validation_start_ns)}
          : std::nullopt;

  output << '{';
  emit_common_json(output, options, "subscriber");
  output << ",\"received_count\":" << received_count << ",\"sequence_gaps\":" << sequence_gaps
         << ",\"not_ready_polls\":" << not_ready_polls
         << ",\"validation_errors\":" << validation_errors;
  emit_optional_duration(output, "final_validation_ns", final_validation_ns);
  emit_throughput(output, options.samples, measured_start_ns, measured_end_ns);
  emit_percentiles(output, "channel_receive", summarize_percentiles(std::move(channel_receive_ns)));
  emit_percentiles(output, "full_end_to_end", summarize_percentiles(std::move(full_end_to_end_ns)));
  output << "}\n";
  return validation_errors == 0U ? 0 : 7;
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_tensor_intraprocess_serial(const Options& options) {
  if (uses_device_tensor_storage(options) && !cuda_available()) {
    return 2;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "ChannelRuntime::create failed: "
              << holoscan::runtime_error_message(runtime.error()) << '\n';
    return 3;
  }

  auto publisher = holoscan::channel_internal::TensorPublisherTestAccess::create(
      runtime.value(), options.channel_name, make_channel_options(options));
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(options.channel_name,
                                                                 make_channel_options(options));
  if (!publisher || !subscriber) {
    std::cerr << "failed to create serial Tensor Channel endpoints\n";
    return 4;
  }

  constexpr std::size_t kMinimumPoolSize = 256U;
  const std::size_t pool_size =
      std::max<std::size_t>(kMinimumPoolSize, options.subscriber_queue_depth + 8U);
  const holoscan::MemoryBuffer::Options memory_options{.max_fanout_release_streams =
                                                           options.expected_subscribers};
  auto tensor_pool = make_sidecar_tensor_pool(options, pool_size, memory_options);
  if (tensor_pool == nullptr || tensor_pool->tensors.empty()) {
    return 4;
  }

  if (!pin_to_cpu(options.publisher_cpu, "serial publisher/subscriber")) {
    return 2;
  }
  if (!wait_for_channel_subscribers(
          publisher.value(),
          1U,
          [&publisher, &tensor_pool, &options]()
              -> holoscan::expected<std::uint32_t, holoscan::Error> {
            const std::int64_t probe_time_ns = now_ns();
            const holoscan::Tensor& tensor = tensor_pool->tensors.front();
            const holoscan::Sample<holoscan::Tensor> probe{
                tensor,
                holoscan::SampleMetadata{.sequence = std::numeric_limits<std::uint64_t>::max(),
                                         .capture_timestamp_ns = probe_time_ns,
                                         .publish_timestamp_ns = probe_time_ns,
                                         .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
                                         .payload_kind = holoscan::PayloadKind::kDescriptorBacked}};
            return publisher->emit_sample(probe);
          })) {
    return 5;
  }
  while (true) {
    auto probe = subscriber->receive_sample();
    if (probe.has_value()) {
      continue;
    }
    if (probe.error().code != holoscan::ErrorCode::kNotReady) {
      std::cerr << "failed to drain serial startup probe: "
                << holoscan::runtime_error_message(probe.error()) << '\n';
      return 5;
    }
    break;
  }

  std::vector<std::uint64_t> channel_emit_ns;
  std::vector<std::uint64_t> publisher_full_ns;
  std::vector<std::uint64_t> channel_receive_ns;
  std::vector<std::uint64_t> end_to_end_ns;
  std::vector<std::uint64_t> full_end_to_end_ns;
  channel_emit_ns.reserve(static_cast<std::size_t>(options.samples));
  publisher_full_ns.reserve(static_cast<std::size_t>(options.samples));
  channel_receive_ns.reserve(static_cast<std::size_t>(options.samples));
  end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));
  full_end_to_end_ns.reserve(static_cast<std::size_t>(options.samples));

  const std::uint64_t total = options.samples + options.warmup;
  std::uint64_t failed_emits = 0U;
  std::uint64_t reached_total = 0U;
  std::uint64_t received_count = 0U;
  std::uint64_t not_ready_polls = 0U;
  std::uint64_t sequence_gaps = 0U;
  std::uint64_t validation_errors = 0U;
  std::uint64_t last_sequence = 0U;
  std::int64_t publisher_measured_start_ns = 0;
  std::int64_t publisher_measured_end_ns = 0;
  std::int64_t subscriber_measured_start_ns = 0;
  std::int64_t subscriber_measured_end_ns = 0;
  std::vector<std::uint32_t> validation_scratch;
  std::optional<holoscan::Tensor> validation_tensor;
  PublisherPacer publisher_pacer(options);

  for (std::uint64_t sequence = 1U; sequence <= total; ++sequence) {
    publisher_pacer.wait();
    const bool measured = sequence > options.warmup;
    const std::int64_t full_start_time_ns = now_ns();
    const holoscan::Tensor& tensor =
        tensor_pool->tensors[(sequence - 1U) % tensor_pool->tensors.size()];
    const std::int64_t send_time_ns = now_ns();
    const holoscan::Sample<holoscan::Tensor> sample{
        tensor,
        holoscan::SampleMetadata{.sequence = sequence,
                                 .capture_timestamp_ns = full_start_time_ns,
                                 .publish_timestamp_ns = send_time_ns,
                                 .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
                                 .payload_kind = holoscan::PayloadKind::kDescriptorBacked}};

    const std::int64_t emit_start_ns = now_ns();
    auto emitted = publisher->emit_sample(sample);
    const std::int64_t emit_done_ns = now_ns();
    if (!emitted) {
      ++failed_emits;
      if (emitted.error().code == holoscan::ErrorCode::kBackpressured ||
          emitted.error().code == holoscan::ErrorCode::kNotReady) {
        --sequence;
        std::this_thread::yield();
        continue;
      }
      std::cerr << "serial emit_sample failed: " << holoscan::runtime_error_message(emitted.error())
                << '\n';
      return 6;
    }
    reached_total += emitted.value();

    const std::int64_t receive_start_ns = now_ns();
    constexpr auto kReceiveTimeout = std::chrono::seconds(5);
    const auto receive_deadline = Clock::now() + kReceiveTimeout;
    std::optional<holoscan::Sample<holoscan::Tensor>> received;
    while (!received.has_value()) {
      auto next = subscriber->receive_sample();
      if (next.has_value()) {
        received.emplace(std::move(next).value());
      } else if (next.error().code == holoscan::ErrorCode::kNotReady) {
        ++not_ready_polls;
        if (Clock::now() >= receive_deadline) {
          std::cerr << "timed out waiting for serial sample " << sequence << '\n';
          return 7;
        }
        std::this_thread::yield();
      } else {
        std::cerr << "serial receive_sample failed: "
                  << holoscan::runtime_error_message(next.error()) << '\n';
        return 7;
      }
    }
    const std::int64_t receive_done_ns = now_ns();
    if (!received.has_value()) {
      std::cerr << "serial receive loop exited without a sample\n";
      return 7;
    }
    auto received_sample =
        std::move(received).value();  // NOLINT(bugprone-unchecked-optional-access)

    ++received_count;
    const std::uint64_t received_sequence = received_sample.metadata.sequence;
    if (last_sequence != 0U && received_sequence != last_sequence + 1U) {
      sequence_gaps +=
          received_sequence > last_sequence ? received_sequence - last_sequence - 1U : 1U;
    }
    last_sequence = received_sequence;
    const bool valid_tensor =
        received_sequence == sequence &&
        received_sample.metadata.payload_kind == holoscan::PayloadKind::kDescriptorBacked &&
        received_sample.metadata.byte_size == options.tensor_bytes &&
        static_cast<std::uint64_t>(received_sample.data.nbytes()) == options.tensor_bytes;
    const bool payload_valid =
        !valid_tensor || options.payload_validation != PayloadValidation::kAll ||
        verify_tensor_pattern(received_sample.data, options, validation_scratch);
    if (!valid_tensor || !payload_valid) {
      ++validation_errors;
    }

    if (measured) {
      if (publisher_measured_start_ns == 0) {
        publisher_measured_start_ns = full_start_time_ns;
        subscriber_measured_start_ns = receive_start_ns;
      }
      channel_emit_ns.push_back(static_cast<std::uint64_t>(emit_done_ns - emit_start_ns));
      publisher_full_ns.push_back(static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
      channel_receive_ns.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      end_to_end_ns.push_back(static_cast<std::uint64_t>(
          receive_done_ns - received_sample.metadata.publish_timestamp_ns));
      full_end_to_end_ns.push_back(static_cast<std::uint64_t>(
          receive_done_ns - received_sample.metadata.capture_timestamp_ns));
      publisher_measured_end_ns = emit_done_ns;
      subscriber_measured_end_ns = receive_done_ns;
    }
    if (options.payload_validation == PayloadValidation::kFinal && received_count == total) {
      validation_tensor.emplace(std::move(received_sample.data));
    }
  }

  if (reached_total != total || received_count != total || sequence_gaps != 0U) {
    std::cerr << "serial Tensor benchmark did not complete every sample exactly once\n";
    return 8;
  }
  const std::optional<std::int64_t> final_validation_start_ns =
      options.payload_validation == PayloadValidation::kFinal
          ? std::optional<std::int64_t>{now_ns()}
          : std::nullopt;
  if (options.payload_validation == PayloadValidation::kFinal &&
      (!validation_tensor.has_value() ||
       !verify_tensor_pattern(validation_tensor.value(), options, validation_scratch))) {
    ++validation_errors;
  }
  const std::optional<std::uint64_t> final_validation_ns =
      final_validation_start_ns
          ? std::optional<std::uint64_t>{elapsed_ns(*final_validation_start_ns)}
          : std::nullopt;

  Options publisher_options = options;
  publisher_options.cpu = options.publisher_cpu;
  std::cout << '{';
  emit_common_json(std::cout, publisher_options, "publisher");
  std::cout << ",\"failed_emits\":" << failed_emits
            << ",\"async_lease_acquisitions\":0,\"reached_total\":" << reached_total;
  emit_throughput(
      std::cout, options.samples, publisher_measured_start_ns, publisher_measured_end_ns);
  emit_percentiles(std::cout, "channel_emit", summarize_percentiles(std::move(channel_emit_ns)));
  emit_percentiles(
      std::cout, "publisher_full", summarize_percentiles(std::move(publisher_full_ns)));
  std::cout << "}\n";

  Options subscriber_options = options;
  subscriber_options.cpu = options.publisher_cpu;
  std::cout << '{';
  emit_common_json(std::cout, subscriber_options, "subscriber");
  std::cout << ",\"received_count\":" << received_count << ",\"sequence_gaps\":" << sequence_gaps
            << ",\"not_ready_polls\":" << not_ready_polls
            << ",\"validation_errors\":" << validation_errors;
  emit_optional_duration(std::cout, "final_validation_ns", final_validation_ns);
  emit_throughput(
      std::cout, options.samples, subscriber_measured_start_ns, subscriber_measured_end_ns);
  emit_percentiles(
      std::cout, "channel_receive", summarize_percentiles(std::move(channel_receive_ns)));
  emit_percentiles(std::cout, "end_to_end", summarize_percentiles(std::move(end_to_end_ns)));
  emit_percentiles(
      std::cout, "full_end_to_end", summarize_percentiles(std::move(full_end_to_end_ns)));
  std::cout << "}\n";
  return validation_errors == 0U ? 0 : 9;
}

[[nodiscard]] int run_role(const Options& options, holoscan::ChannelRuntime* const shared_runtime,
                           std::mutex* const runtime_mutex, std::barrier<>* const start_barrier,
                           std::ostream& output) {
  if (options.payload == "tensor") {
    return options.role == "publisher"
               ? run_tensor_publisher(options, shared_runtime, runtime_mutex, start_barrier, output)
               : run_tensor_subscriber(
                     options, shared_runtime, runtime_mutex, start_barrier, output);
  }
  if (options.payload == "image") {
    return options.role == "publisher"
               ? run_image_publisher(options, shared_runtime, runtime_mutex, start_barrier, output)
               : run_image_subscriber(
                     options, shared_runtime, runtime_mutex, start_barrier, output);
  }
  return options.role == "publisher"
             ? run_publisher(options, shared_runtime, runtime_mutex, start_barrier, output)
             : run_subscriber(options, shared_runtime, runtime_mutex, start_barrier, output);
}

[[nodiscard]] int run_intraprocess(const Options& options) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "ChannelRuntime::create failed: "
              << holoscan::runtime_error_message(runtime.error()) << '\n';
    return 3;
  }

  const auto subscriber_count = static_cast<std::size_t>(options.expected_subscribers);
  holoscan::ChannelRuntime* const shared_runtime = &runtime.value();
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(subscriber_count + 1U));
  std::mutex runtime_mutex;
  std::vector<int> subscriber_return_codes(subscriber_count, 1);
  std::vector<std::string> subscriber_outputs(subscriber_count);
  std::vector<std::thread> subscriber_threads;
  subscriber_threads.reserve(subscriber_count);

  for (std::size_t index = 0U; index < subscriber_count; ++index) {
    Options subscriber_options = options;
    subscriber_options.role = "subscriber";
    subscriber_options.cpu = options.subscriber_cpus[index];
    subscriber_threads.emplace_back([&, index, subscriber_options = std::move(subscriber_options)] {
      try {
        std::ostringstream output;
        subscriber_return_codes[index] =
            run_role(subscriber_options, shared_runtime, &runtime_mutex, &start_barrier, output);
        subscriber_outputs[index] = std::move(output).str();
      } catch (const std::exception& error) {
        std::cerr << "subscriber " << index << " failed with exception: " << error.what() << '\n';
        subscriber_return_codes[index] = 10;
      } catch (...) {
        std::cerr << "subscriber " << index << " failed with an unknown exception\n";
        subscriber_return_codes[index] = 10;
      }
    });
  }

  Options publisher_options = options;
  publisher_options.role = "publisher";
  publisher_options.cpu = options.publisher_cpu;
  int publisher_return_code = 1;
  std::string publisher_output;
  std::thread publisher_thread([&, publisher_options = std::move(publisher_options)] {
    try {
      std::ostringstream output;
      publisher_return_code =
          run_role(publisher_options, shared_runtime, &runtime_mutex, &start_barrier, output);
      publisher_output = std::move(output).str();
    } catch (const std::exception& error) {
      std::cerr << "publisher failed with exception: " << error.what() << '\n';
      publisher_return_code = 10;
    } catch (...) {
      std::cerr << "publisher failed with an unknown exception\n";
      publisher_return_code = 10;
    }
  });

  publisher_thread.join();
  for (std::thread& subscriber_thread : subscriber_threads) {
    subscriber_thread.join();
  }

  if (publisher_return_code != 0) {
    return publisher_return_code;
  }
  for (const int return_code : subscriber_return_codes) {
    if (return_code != 0) {
      return return_code;
    }
  }

  std::cout << publisher_output;
  for (const std::string& subscriber_output : subscriber_outputs) {
    std::cout << subscriber_output;
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  auto options = parse_args(argc, argv);
  if (!options) {
    print_help(std::cerr);
    return 2;
  }
  if (options->role == "coordinator") {
    if (options->domain == "intraprocess_serial") {
      return run_tensor_intraprocess_serial(*options);
    }
    return run_intraprocess(*options);
  }
  return run_role(*options, nullptr, nullptr, nullptr, std::cout);
}
