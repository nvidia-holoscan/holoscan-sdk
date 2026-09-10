/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <holomq/context_builder.hpp>
#include <holomq/detail/publisher_sample_release_registry.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>

namespace {

struct BenchmarkPOD64 {
  std::uint64_t sequence{};
  std::int64_t source_start_time_ns{};
  std::int64_t send_time_ns{};
  std::uint64_t checksum{};
  std::int64_t relay_receive_time_ns{};
  std::int64_t relay_send_time_ns{};
  std::uint64_t auxiliary0{};
  std::uint64_t auxiliary1{};
};

static_assert(sizeof(BenchmarkPOD64) == 64U);
static_assert(std::is_standard_layout_v<BenchmarkPOD64>);
static_assert(std::is_trivially_copyable_v<BenchmarkPOD64>);

template <std::size_t PayloadBytes>
struct FixedPodPayload {
  static_assert(PayloadBytes > sizeof(std::uint64_t));

  std::uint64_t sequence{};
  std::array<std::byte, PayloadBytes - sizeof(std::uint64_t)> padding{};
};

template <std::size_t PayloadBytes>
inline constexpr bool kValidFixedPodPayload =
    sizeof(FixedPodPayload<PayloadBytes>) == PayloadBytes &&
    std::is_standard_layout_v<FixedPodPayload<PayloadBytes>> &&
    std::is_trivially_copyable_v<FixedPodPayload<PayloadBytes>>;

static_assert(kValidFixedPodPayload<64U>);
static_assert(kValidFixedPodPayload<128U>);
static_assert(kValidFixedPodPayload<440U>);

enum class FixedPodMeasuredStage : std::uint8_t { kRoundTrip, kPublish, kTake };

enum class FixedPodPublisherSubstage : std::uint8_t {
  kLoanReturn,
  kLoanAssignReturn,
  kPreloanedSend,
};

enum class FixedPodSubscriberSubstage : std::uint8_t {
  kEmptyProbe,
  kEmptyTryTake,
  kReadyProbe,
  kTryTakeBorrowed,
  kTryTakeCopyOut,
  kRuntimeGraphEquivalent,
};

enum class FixedPodFanoutBackendStage : std::uint8_t {
  kStableConnectionRefresh,
  kEmptyReturnScan,
  kReturnedSampleReclaim,
  kPreloanedSend,
};

enum class ConcurrentReceiveMode : std::uint8_t { kPolling, kKernelWaitSet };

enum class ConcurrentWorkload : std::uint8_t { kThroughput, kLatency, kPublisherSend };

std::atomic<std::uint64_t> g_service_counter{0};

class ScopedCPUAffinity {
 public:
  [[nodiscard]] bool pin(const int cpu) noexcept {
    if (cpu < 0) {
      return true;
    }
    if (cpu >= CPU_SETSIZE) {
      return false;
    }
    if (pthread_getaffinity_np(pthread_self(), sizeof(original_), &original_) != 0) {
      return false;
    }
    restore_ = true;
    cpu_set_t requested{};
    CPU_ZERO(&requested);
    CPU_SET(cpu, &requested);
    return pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) == 0;
  }

  ~ScopedCPUAffinity() {
    if (restore_) {
      static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_));
    }
  }

  ScopedCPUAffinity() = default;
  ScopedCPUAffinity(const ScopedCPUAffinity&) = delete;
  ScopedCPUAffinity& operator=(const ScopedCPUAffinity&) = delete;

 private:
  cpu_set_t original_{};
  bool restore_{};
};

[[nodiscard]] bool pin_current_thread(const int cpu) noexcept {
  if (cpu < 0) {
    return true;
  }
  if (cpu >= CPU_SETSIZE) {
    return false;
  }
  cpu_set_t requested{};
  CPU_ZERO(&requested);
  CPU_SET(cpu, &requested);
  return pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) == 0;
}

[[nodiscard]] std::optional<int> benchmark_cpu(const char* const name) noexcept {
  const char* const value = std::getenv(name);
  if (value == nullptr) {
    return -1;
  }
  int cpu = -1;
  const std::string_view text{value};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), cpu);
  if (error != std::errc() || end != text.data() + text.size() || cpu < -1) {
    return std::nullopt;
  }
  return cpu;
}

[[nodiscard]] std::int64_t now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::string make_unique_service_name(const char* const suffix) {
  return "holomq_benchmark_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(g_service_counter.fetch_add(1U, std::memory_order_relaxed)) + "_" + suffix;
}

[[nodiscard]] holomq::PubSubServiceOptions make_service_options(
    const holomq::QueuePolicy policy, const std::uint32_t max_subscribers = 1U,
    const std::uint32_t max_subscriber_queue_depth = 64U) {
  holomq::PubSubServiceOptions options{};
  options.queue_policy = policy;
  options.max_subscribers = max_subscribers;
  options.max_subscriber_queue_depth = max_subscriber_queue_depth;
  options.publisher_history_depth = 0U;
  return options;
}

[[nodiscard]] holomq::PublisherPubSubOptions make_publisher_options(
    const bool enable_sample_release_callbacks = false) {
  holomq::PublisherPubSubOptions options{};
  options.enable_sample_release_callbacks = enable_sample_release_callbacks;
  return options;
}

[[nodiscard]] holomq::SubscriberPubSubOptions make_subscriber_options(
    const std::uint32_t queue_depth = 64U) {
  holomq::SubscriberPubSubOptions options{};
  options.queue_depth = queue_depth;
  return options;
}

[[nodiscard]] BenchmarkPOD64 make_payload(const std::uint32_t sequence) noexcept {
  BenchmarkPOD64 payload{
      .sequence = sequence,
      .source_start_time_ns = static_cast<std::int64_t>(sequence) * 1000,
      .send_time_ns = static_cast<std::int64_t>(sequence) * 2000,
      .auxiliary0 = sequence ^ 0xA5A5'5A5AU,
      .auxiliary1 = static_cast<std::uint64_t>(sequence) * 3U,
  };
  payload.checksum =
      payload.sequence ^ (static_cast<std::uint64_t>(payload.source_start_time_ns) << 1U) ^
      (static_cast<std::uint64_t>(payload.send_time_ns) << 2U) ^ (payload.auxiliary0 << 5U) ^
      (payload.auxiliary1 << 6U) ^ 0xC001'D00D'5A5A'A5A5ULL;
  return payload;
}

[[nodiscard]] std::uint64_t checksum_payload(const BenchmarkPOD64& payload) noexcept {
  return payload.sequence ^ (static_cast<std::uint64_t>(payload.source_start_time_ns) << 1U) ^
         (static_cast<std::uint64_t>(payload.send_time_ns) << 2U) ^
         (static_cast<std::uint64_t>(payload.relay_receive_time_ns) << 3U) ^
         (static_cast<std::uint64_t>(payload.relay_send_time_ns) << 4U) ^
         (payload.auxiliary0 << 5U) ^ (payload.auxiliary1 << 6U) ^ (payload.checksum << 1U);
}

[[nodiscard]] holomq::expected<holomq::Context, holomq::Error> make_context() {
  return holomq::ContextBuilder().with_vmm_fd_export(false).create();
}

[[nodiscard]] holomq::expected<holomq::Publisher<BenchmarkPOD64>, holomq::Error> make_publisher(
    holomq::Context& context, const std::string& service_name,
    const holomq::PubSubServiceOptions& service_options,
    const holomq::PublisherPubSubOptions& options, const bool enable_event_wake = false) {
  return context.publisher_builder(service_name)
      .service_options(service_options)
      .enable_event_wake(enable_event_wake)
      .publisher_options(options)
      .payload_type<BenchmarkPOD64>()
      .create();
}

[[nodiscard]] holomq::expected<holomq::Subscriber<BenchmarkPOD64>, holomq::Error> make_subscriber(
    holomq::Context& context, const std::string& service_name,
    const holomq::PubSubServiceOptions& service_options,
    const holomq::SubscriberPubSubOptions& options, const bool enable_event_wake = false) {
  return context.subscriber_builder(service_name)
      .service_options(service_options)
      .enable_event_wake(enable_event_wake)
      .subscriber_options(options)
      .payload_type<BenchmarkPOD64>()
      .create();
}

template <typename SubscriberContainer>
void drain_subscribers(SubscriberContainer& subscribers, std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    while (auto sample = subscriber.try_take()) {
      checksum ^= checksum_payload(sample->payload());
    }
  }
}

[[nodiscard]] std::optional<holomq::InboundMessage<BenchmarkPOD64>> wait_for_sample(
    holomq::Subscriber<BenchmarkPOD64>& subscriber) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sample = subscriber.try_take();
    if (sample.has_value()) {
      return std::move(sample).value();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return std::nullopt;
}

[[nodiscard]] bool wait_for_waitset_handle(holomq::WaitSet& waitset,
                                           const holomq::WaitSetHandle expected_handle) {
  constexpr std::size_t kMaximumSpuriousWakes = 64U;
  const holomq::WaitStrategy strategy{
      .mode = holomq::WaitMode::KernelWait,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = std::chrono::seconds{3},
  };
  for (std::size_t attempt = 0U; attempt < kMaximumSpuriousWakes; ++attempt) {
    auto waited = waitset.wait_for_ready(strategy);
    if (!waited.has_value() || *waited != holomq::WaitResult::ready) {
      return false;
    }
    const auto ready = waitset.ready_handles();
    if (std::find(ready.begin(), ready.end(), expected_handle) == ready.end()) {
      continue;
    }
    return true;
  }
  return false;
}

[[nodiscard]] std::optional<holomq::InboundMessage<BenchmarkPOD64>> wait_for_waitset_sample(
    holomq::WaitSet& waitset, const holomq::WaitSetHandle expected_handle,
    holomq::Subscriber<BenchmarkPOD64>& subscriber) {
  if (!wait_for_waitset_handle(waitset, expected_handle)) {
    return std::nullopt;
  }
  auto sample = subscriber.try_take();
  if (sample.has_value()) {
    return std::move(sample).value();
  }
  return std::nullopt;
}

template <std::size_t PayloadBytes>
[[nodiscard]] FixedPodPayload<PayloadBytes> make_fixed_payload(
    const std::uint64_t sequence) noexcept {
  FixedPodPayload<PayloadBytes> payload{.sequence = sequence};
  payload.padding.front() = static_cast<std::byte>(sequence & 0xFFU);
  payload.padding.back() = static_cast<std::byte>((sequence >> 8U) & 0xFFU);
  return payload;
}

template <std::size_t PayloadBytes>
[[nodiscard]] std::uint64_t checksum_fixed_payload(
    const FixedPodPayload<PayloadBytes>& payload) noexcept {
  return payload.sequence ^ static_cast<std::uint64_t>(payload.padding.front()) ^
         (static_cast<std::uint64_t>(payload.padding.back()) << 8U);
}

template <std::size_t PayloadBytes>
[[nodiscard]] holomq::expected<holomq::Publisher<FixedPodPayload<PayloadBytes>>, holomq::Error>
make_fixed_publisher(holomq::Context& context, const std::string& service_name,
                     const holomq::PubSubServiceOptions& service_options,
                     const holomq::PublisherPubSubOptions& options,
                     const bool enable_event_wake = false) {
  return context.publisher_builder(service_name)
      .service_options(service_options)
      .enable_event_wake(enable_event_wake)
      .publisher_options(options)
      .payload_type<FixedPodPayload<PayloadBytes>>()
      .create();
}

template <std::size_t PayloadBytes>
[[nodiscard]] holomq::expected<holomq::Subscriber<FixedPodPayload<PayloadBytes>>, holomq::Error>
make_fixed_subscriber(holomq::Context& context, const std::string& service_name,
                      const holomq::PubSubServiceOptions& service_options,
                      const holomq::SubscriberPubSubOptions& options,
                      const bool enable_event_wake = false) {
  return context.subscriber_builder(service_name)
      .service_options(service_options)
      .enable_event_wake(enable_event_wake)
      .subscriber_options(options)
      .payload_type<FixedPodPayload<PayloadBytes>>()
      .create();
}

template <std::size_t PayloadBytes>
[[nodiscard]] std::optional<holomq::InboundMessage<FixedPodPayload<PayloadBytes>>>
wait_for_fixed_sample(holomq::Subscriber<FixedPodPayload<PayloadBytes>>& subscriber) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sample = subscriber.try_take();
    if (sample.has_value()) {
      return std::move(sample).value();
    }
    std::this_thread::yield();
  }
  return std::nullopt;
}

template <std::size_t PayloadBytes>
[[nodiscard]] int publish_fixed_one(holomq::Publisher<FixedPodPayload<PayloadBytes>>& publisher,
                                    const std::uint64_t sequence) {
  auto outbound = publisher.loan();
  if (!outbound.has_value()) {
    return -1;
  }
  outbound->payload() = make_fixed_payload<PayloadBytes>(sequence);
  auto sent = publisher.send(std::move(outbound).value());
  return sent.has_value() ? sent.value() : -1;
}

template <std::size_t PayloadBytes, typename SubscriberContainer>
[[nodiscard]] bool wait_for_fixed_subscribers(
    holomq::Publisher<FixedPodPayload<PayloadBytes>>& publisher, SubscriberContainer& subscribers,
    const std::size_t expected_subscribers) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  std::uint64_t sequence = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    if (publisher.update_connections().has_value()) {
      const int sent = publish_fixed_one(publisher, ++sequence);
      bool every_subscriber_received = true;
      for (auto& subscriber : subscribers) {
        bool received = false;
        while (auto sample = subscriber.try_take()) {
          received = true;
        }
        every_subscriber_received = every_subscriber_received && received;
      }
      if (sent == static_cast<int>(expected_subscribers) && every_subscriber_received) {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

template <std::size_t PayloadBytes>
[[nodiscard]] bool wait_for_fixed_subscriber(
    holomq::Publisher<FixedPodPayload<PayloadBytes>>& publisher,
    holomq::Subscriber<FixedPodPayload<PayloadBytes>>& subscriber) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  std::uint64_t sequence = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    static_cast<void>(publisher.update_connections());
    if (publish_fixed_one(publisher, ++sequence) == 1) {
      return wait_for_fixed_sample(subscriber).has_value();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

template <std::size_t PayloadBytes, FixedPodMeasuredStage Stage>
void run_fixed_pod_polling(
    benchmark::State& state) {  // NOLINT(readability-function-cognitive-complexity)
  const std::uint32_t batch_size =
      Stage == FixedPodMeasuredStage::kRoundTrip ? 1U : static_cast<std::uint32_t>(state.range(0));
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ context");
    return;
  }

  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, 1U, batch_size);
  const holomq::PublisherPubSubOptions publisher_options = make_publisher_options();
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options(batch_size);

  const std::string service_name = make_unique_service_name("fixed_pod_polling");
  auto publisher = make_fixed_publisher<PayloadBytes>(
      context.value(), service_name, service_options, publisher_options);
  auto subscriber = make_fixed_subscriber<PayloadBytes>(
      context.value(), service_name, service_options, subscriber_options);
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ endpoints");
    return;
  }
  if (!wait_for_fixed_subscriber(publisher.value(), subscriber.value())) {
    state.SkipWithError("timed out waiting for fixed-POD HoloMQ subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  std::int64_t completed_batches = 0;
  auto publish_batch = [&]() {
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      if (publish_fixed_one(publisher.value(), ++sequence) != 1) {
        return false;
      }
    }
    return true;
  };
  auto take_batch = [&]() {
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      auto inbound = subscriber->try_take();
      if (!inbound.has_value()) {
        return false;
      }
      const FixedPodPayload<PayloadBytes> payload = inbound->payload();
      checksum ^= checksum_fixed_payload(payload);
    }
    return true;
  };

  for (auto _ : state) {
    bool succeeded = false;
    if constexpr (Stage == FixedPodMeasuredStage::kRoundTrip) {
      if (publish_fixed_one(publisher.value(), ++sequence) == 1) {
        auto inbound = wait_for_fixed_sample(subscriber.value());
        if (inbound.has_value()) {
          const FixedPodPayload<PayloadBytes> payload = inbound->payload();
          checksum ^= checksum_fixed_payload(payload);
          succeeded = true;
        }
      }
    } else if constexpr (Stage == FixedPodMeasuredStage::kPublish) {
      const auto measured_start = std::chrono::steady_clock::now();
      succeeded = publish_batch();
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
      for (std::uint32_t index = 0U; succeeded && index < batch_size; ++index) {
        auto inbound = wait_for_fixed_sample(subscriber.value());
        if (!inbound.has_value()) {
          succeeded = false;
          break;
        }
        const FixedPodPayload<PayloadBytes> payload = inbound->payload();
        checksum ^= checksum_fixed_payload(payload);
      }
    } else {
      if (!publish_batch()) {
        state.SkipWithError("failed to prefill fixed-POD subscriber batch");
        break;
      }
      const auto measured_start = std::chrono::steady_clock::now();
      succeeded = take_batch();
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    }
    if (!succeeded) {
      state.SkipWithError("failed to process fixed-POD benchmark batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["busy_poll"] = 0.0;
  state.counters["event_wake"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(FixedPodPayload<PayloadBytes>));
  state.counters["publish_stage"] = static_cast<double>(Stage == FixedPodMeasuredStage::kPublish);
  state.counters["take_stage"] = static_cast<double>(Stage == FixedPodMeasuredStage::kTake);
  state.counters["receive_polling"] = 1.0;
  state.counters["vmm_fd_export"] = 0.0;
}

// Enables the SubscriberEmptyProbe, SubscriberEmptyTryTake, SubscriberReadyProbe,
// SubscriberTryTakeBorrowed, SubscriberTryTakeCopyOut, and SubscriberRuntimeGraphEquivalent rows.
template <std::size_t PayloadBytes, FixedPodSubscriberSubstage Substage>
void run_fixed_pod_subscriber_substage(
    benchmark::State& state) {  // NOLINT(readability-function-cognitive-complexity)
  const auto batch_size = static_cast<std::uint32_t>(state.range(0));
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ context");
    return;
  }

  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, 1U, batch_size);
  const holomq::PublisherPubSubOptions publisher_options = make_publisher_options();
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options(batch_size);

  const std::string service_name = make_unique_service_name("fixed_pod_subscriber_substage");
  auto publisher = make_fixed_publisher<PayloadBytes>(
      context.value(), service_name, service_options, publisher_options);
  auto subscriber = make_fixed_subscriber<PayloadBytes>(
      context.value(), service_name, service_options, subscriber_options);
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ endpoints");
    return;
  }
  if (!wait_for_fixed_subscriber(publisher.value(), subscriber.value())) {
    state.SkipWithError("timed out waiting for fixed-POD HoloMQ subscriber");
    return;
  }

  constexpr bool kPrefill = Substage != FixedPodSubscriberSubstage::kEmptyProbe &&
                            Substage != FixedPodSubscriberSubstage::kEmptyTryTake;
  constexpr bool kConsumes = Substage == FixedPodSubscriberSubstage::kTryTakeBorrowed ||
                             Substage == FixedPodSubscriberSubstage::kTryTakeCopyOut ||
                             Substage == FixedPodSubscriberSubstage::kRuntimeGraphEquivalent;
  constexpr bool kCopyOut = Substage == FixedPodSubscriberSubstage::kTryTakeCopyOut ||
                            Substage == FixedPodSubscriberSubstage::kRuntimeGraphEquivalent;
  constexpr std::uint32_t kReadyProbesPerTake =
      Substage == FixedPodSubscriberSubstage::kRuntimeGraphEquivalent ? 2U : 0U;

  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    if constexpr (kPrefill) {
      state.PauseTiming();
      bool prefill_succeeded = true;
      for (std::uint32_t index = 0U; index < batch_size; ++index) {
        if (publish_fixed_one(publisher.value(), ++sequence) != 1) {
          prefill_succeeded = false;
          break;
        }
      }
      state.ResumeTiming();
      if (!prefill_succeeded) {
        state.SkipWithError("failed to prefill fixed-POD subscriber batch");
        break;
      }
    }

    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      if constexpr (Substage == FixedPodSubscriberSubstage::kEmptyProbe ||
                    Substage == FixedPodSubscriberSubstage::kReadyProbe) {
        const auto ready = subscriber->has_samples();
        const bool expected_ready = Substage == FixedPodSubscriberSubstage::kReadyProbe;
        if (!ready.has_value() || ready.value() != expected_ready) {
          succeeded = false;
          break;
        }
        bool ready_value = ready.value();
        benchmark::DoNotOptimize(ready_value);
      } else if constexpr (Substage == FixedPodSubscriberSubstage::kEmptyTryTake) {
        const auto inbound = subscriber->try_take();
        if (inbound.has_value() || inbound.error().code != holomq::ErrorCode::kNotReady) {
          succeeded = false;
          break;
        }
      } else {
        for (std::uint32_t probe = 0U; probe < kReadyProbesPerTake; ++probe) {
          const auto ready = subscriber->has_samples();
          if (!ready.has_value() || !ready.value()) {
            succeeded = false;
            break;
          }
        }
        if (!succeeded) {
          break;
        }
        auto inbound = subscriber->try_take();
        if (!inbound.has_value()) {
          succeeded = false;
          break;
        }
        if constexpr (kCopyOut) {
          FixedPodPayload<PayloadBytes> payload = inbound->payload();
          benchmark::DoNotOptimize(payload);
          checksum ^= checksum_fixed_payload(payload);
        } else {
          checksum ^= checksum_fixed_payload(inbound->payload());
        }
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());

    if constexpr (kPrefill && !kConsumes) {
      state.PauseTiming();
      for (std::uint32_t index = 0U; succeeded && index < batch_size; ++index) {
        auto inbound = wait_for_fixed_sample(subscriber.value());
        if (!inbound.has_value()) {
          succeeded = false;
          break;
        }
        checksum ^= checksum_fixed_payload(inbound->payload());
      }
      state.ResumeTiming();
    }
    if (!succeeded) {
      state.SkipWithError("failed to process fixed-POD subscriber substage batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["borrowed_payload"] = kConsumes && !kCopyOut ? 1.0 : 0.0;
  state.counters["copy_out"] = kCopyOut ? 1.0 : 0.0;
  state.counters["empty_probe"] = Substage == FixedPodSubscriberSubstage::kEmptyProbe ? 1.0 : 0.0;
  state.counters["empty_take"] = Substage == FixedPodSubscriberSubstage::kEmptyTryTake ? 1.0 : 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(FixedPodPayload<PayloadBytes>));
  state.counters["ready_probes_per_take"] = static_cast<double>(kReadyProbesPerTake);
  state.counters["take"] = kConsumes ? 1.0 : 0.0;
  state.counters["vmm_fd_export"] = 0.0;
}

// Enables the PublisherLoanReturn, PublisherLoanAssignReturn, PublisherPreloanedSend,
// PublisherPreloanedSendExpectedRecipient, PublisherPreloanedSendBoundedRetry,
// PublisherPreloanedSendEventWake, PublisherPreloanedSendRuntimeEventless, and
// PublisherPreloanedSendGraphEquivalent rows.
template <std::size_t PayloadBytes, FixedPodPublisherSubstage Substage,
          bool EnforceExpectedRecipient = false, bool EnableEventWake = false,
          bool EnableBoundedRetry = false>
void run_fixed_pod_publisher_substage(
    benchmark::State& state) {  // NOLINT(readability-function-cognitive-complexity)
  const auto batch_size = static_cast<std::uint32_t>(state.range(0));
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ context");
    return;
  }

  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, 1U, batch_size);
  holomq::PublisherPubSubOptions publisher_options = make_publisher_options();
  if constexpr (Substage == FixedPodPublisherSubstage::kPreloanedSend) {
    publisher_options.max_loaned_samples = batch_size;
    publisher_options.expected_recipient_count = EnforceExpectedRecipient ? 1U : 0U;
    if constexpr (EnableBoundedRetry) {
      publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
      publisher_options.bounded_retry_budget_ns = 5'000'000U;
    }
  }
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options(batch_size);

  const std::string service_name = make_unique_service_name("fixed_pod_publisher_substage");
  auto publisher = make_fixed_publisher<PayloadBytes>(
      context.value(), service_name, service_options, publisher_options, EnableEventWake);
  auto subscriber = make_fixed_subscriber<PayloadBytes>(
      context.value(), service_name, service_options, subscriber_options, EnableEventWake);
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create fixed-POD HoloMQ endpoints");
    return;
  }
  if (!wait_for_fixed_subscriber(publisher.value(), subscriber.value())) {
    state.SkipWithError("timed out waiting for fixed-POD HoloMQ subscriber");
    return;
  }
  std::optional<holomq::WaitSet> waitset;
  std::optional<holomq::WaitSetAttachment> attachment;
  if constexpr (EnableEventWake) {
    auto created_waitset = context->create_waitset();
    if (!created_waitset.has_value()) {
      state.SkipWithError("failed to create fixed-POD publisher substage WaitSet");
      return;
    }
    waitset.emplace(std::move(*created_waitset));
    auto created_attachment = waitset->attach(*subscriber);
    if (!created_attachment.has_value()) {
      state.SkipWithError("failed to attach fixed-POD publisher substage subscriber");
      return;
    }
    attachment.emplace(std::move(*created_attachment));
  }

  using Message = holomq::OutboundMessage<FixedPodPayload<PayloadBytes>>;
  std::vector<Message> preloaned_messages;
  preloaned_messages.reserve(batch_size);
  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  std::int64_t completed_batches = 0;

  for (auto _ : state) {
    if constexpr (Substage == FixedPodPublisherSubstage::kPreloanedSend) {
      state.PauseTiming();
      bool preparation_failed = false;
      for (std::uint32_t index = 0U; index < batch_size; ++index) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          state.SkipWithError("failed to preloan fixed-POD publisher batch");
          preparation_failed = true;
          break;
        }
        outbound->payload() = make_fixed_payload<PayloadBytes>(++sequence);
        preloaned_messages.emplace_back(std::move(outbound).value());
      }
      state.ResumeTiming();
      if (preparation_failed) {
        break;
      }
    }

    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      if constexpr (Substage == FixedPodPublisherSubstage::kLoanReturn) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          succeeded = false;
          break;
        }
        benchmark::DoNotOptimize(&outbound->payload());
      } else if constexpr (Substage == FixedPodPublisherSubstage::kLoanAssignReturn) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          succeeded = false;
          break;
        }
        outbound->payload() = make_fixed_payload<PayloadBytes>(++sequence);
        benchmark::DoNotOptimize(&outbound->payload());
      } else {
        auto sent = publisher->send(std::move(preloaned_messages[index]));
        if (!sent.has_value() || sent.value() != 1) {
          succeeded = false;
          break;
        }
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());

    if constexpr (Substage == FixedPodPublisherSubstage::kPreloanedSend) {
      state.PauseTiming();
      if constexpr (EnableEventWake) {
        if (!wait_for_waitset_handle(*waitset, attachment->handle())) {
          state.SkipWithError("failed to consume fixed-POD publisher substage event");
          succeeded = false;
        }
      }
      preloaned_messages.clear();
      for (std::uint32_t index = 0U; succeeded && index < batch_size; ++index) {
        auto inbound = wait_for_fixed_sample(subscriber.value());
        if (!inbound.has_value()) {
          succeeded = false;
          break;
        }
        checksum ^= checksum_fixed_payload(inbound->payload());
      }
      state.ResumeTiming();
    }

    if (!succeeded) {
      state.SkipWithError("failed to process fixed-POD publisher substage batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["bounded_retry"] = EnableBoundedRetry ? 1.0 : 0.0;
  state.counters["event_wake"] = EnableEventWake ? 1.0 : 0.0;
  state.counters["expected_recipient_gate"] = EnforceExpectedRecipient ? 1.0 : 0.0;
  state.counters["loan_return"] =
      static_cast<double>(Substage == FixedPodPublisherSubstage::kLoanReturn);
  state.counters["payload_assignment"] =
      static_cast<double>(Substage == FixedPodPublisherSubstage::kLoanAssignReturn);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(FixedPodPayload<PayloadBytes>));
  state.counters["preloaned_send"] =
      static_cast<double>(Substage == FixedPodPublisherSubstage::kPreloanedSend);
  state.counters["vmm_fd_export"] = 0.0;
}

// Enables the FanoutStableRefresh, FanoutEmptyReturnScan, FanoutReturnedSampleReclaim, and
// FanoutPreloanedSend rows without changing public HoloMQ endpoint configuration.
template <std::size_t SubscriberCount, FixedPodFanoutBackendStage Stage,
          bool EnableReleaseCallback = false>
void run_fixed_pod_fanout_backend_stage(
    benchmark::State& state) {  // NOLINT(readability-function-cognitive-complexity)
  static_assert(SubscriberCount > 0U);
  static_assert(
      Stage == FixedPodFanoutBackendStage::kReturnedSampleReclaim || !EnableReleaseCallback,
      "release callbacks apply only to returned-sample reclamation");
  constexpr std::size_t kPayloadBytes = 64U;
  const auto batch_size = static_cast<std::uint32_t>(state.range(0));
  // Registered callbacks can run while subscribers, the publisher, or its context tear down.
  std::atomic<std::uint64_t> release_count{0U};
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create fixed-POD fanout backend context");
    return;
  }

  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, SubscriberCount, batch_size);
  holomq::PublisherPubSubOptions publisher_options = make_publisher_options(EnableReleaseCallback);
  publisher_options.max_loaned_samples = batch_size;
  publisher_options.expected_recipient_count = SubscriberCount;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns = 5'000'000U;
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options(batch_size);

  const std::string service_name = make_unique_service_name("fixed_pod_fanout_backend");
  auto publisher = make_fixed_publisher<kPayloadBytes>(
      context.value(), service_name, service_options, publisher_options);
  if (!publisher.has_value()) {
    state.SkipWithError("failed to create fixed-POD fanout backend publisher");
    return;
  }
  using Subscriber = holomq::Subscriber<FixedPodPayload<kPayloadBytes>>;
  std::vector<Subscriber> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t index = 0U; index < SubscriberCount; ++index) {
    auto subscriber = make_fixed_subscriber<kPayloadBytes>(
        context.value(), service_name, service_options, subscriber_options);
    if (!subscriber.has_value()) {
      state.SkipWithError("failed to create fixed-POD fanout backend subscriber");
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }
  if (!wait_for_fixed_subscribers(publisher.value(), subscribers, SubscriberCount)) {
    state.SkipWithError("timed out waiting for fixed-POD fanout backend subscribers");
    return;
  }

  using Message = holomq::OutboundMessage<FixedPodPayload<kPayloadBytes>>;
  std::vector<Message> preloaned_messages;
  preloaned_messages.reserve(batch_size);
  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  std::int64_t completed_batches = 0;

  for (auto _ : state) {
    if constexpr (Stage == FixedPodFanoutBackendStage::kPreloanedSend) {
      state.PauseTiming();
      bool preparation_failed = false;
      for (std::uint32_t index = 0U; index < batch_size; ++index) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          state.SkipWithError("failed to preloan fixed-POD fanout backend batch");
          preparation_failed = true;
          break;
        }
        outbound->payload() = make_fixed_payload<kPayloadBytes>(++sequence);
        preloaned_messages.emplace_back(std::move(outbound).value());
      }
      state.ResumeTiming();
      if (preparation_failed) {
        break;
      }
    } else if constexpr (Stage == FixedPodFanoutBackendStage::kReturnedSampleReclaim) {
      state.PauseTiming();
      bool preparation_failed = false;
      const std::uint64_t expected_release_count =
          release_count.load(std::memory_order_relaxed) + batch_size;
      for (std::uint32_t index = 0U; index < batch_size; ++index) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          preparation_failed = true;
          break;
        }
        outbound->payload() = make_fixed_payload<kPayloadBytes>(++sequence);
        holomq::expected<int, holomq::Error> sent = [&]() {
          if constexpr (EnableReleaseCallback) {
            return publisher->send(std::move(outbound).value(), [&release_count]() noexcept {
              release_count.fetch_add(1U, std::memory_order_relaxed);
            });
          } else {
            return publisher->send(std::move(outbound).value());
          }
        }();
        if (!sent.has_value() || sent.value() != static_cast<int>(SubscriberCount)) {
          preparation_failed = true;
          break;
        }
      }
      for (auto& subscriber : subscribers) {
        for (std::uint32_t index = 0U; !preparation_failed && index < batch_size; ++index) {
          auto inbound = wait_for_fixed_sample(subscriber);
          if (!inbound.has_value()) {
            preparation_failed = true;
            break;
          }
          checksum ^= checksum_fixed_payload(inbound->payload());
        }
      }
      state.ResumeTiming();
      if (preparation_failed) {
        state.SkipWithError("failed to prepare fixed-POD returned samples");
        break;
      }

      const auto measured_start = std::chrono::steady_clock::now();
      {
        auto reclaiming_loan = publisher->loan();
        if (!reclaiming_loan.has_value()) {
          state.SkipWithError("failed to reclaim fixed-POD returned samples");
          break;
        }
        benchmark::DoNotOptimize(&reclaiming_loan->payload());
      }
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
      if constexpr (EnableReleaseCallback) {
        if (release_count.load(std::memory_order_relaxed) != expected_release_count) {
          state.SkipWithError("fixed-POD release callback count mismatch");
          break;
        }
      }
      ++completed_batches;
      continue;
    }

    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      if constexpr (Stage == FixedPodFanoutBackendStage::kStableConnectionRefresh) {
        if (!publisher->update_connections().has_value()) {
          succeeded = false;
          break;
        }
      } else if constexpr (Stage == FixedPodFanoutBackendStage::kEmptyReturnScan) {
        auto outbound = publisher->loan();
        if (!outbound.has_value()) {
          succeeded = false;
          break;
        }
        benchmark::DoNotOptimize(&outbound->payload());
      } else {
        auto sent = publisher->send(std::move(preloaned_messages[index]));
        if (!sent.has_value() || sent.value() != static_cast<int>(SubscriberCount)) {
          succeeded = false;
          break;
        }
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());

    if constexpr (Stage == FixedPodFanoutBackendStage::kPreloanedSend) {
      state.PauseTiming();
      preloaned_messages.clear();
      for (auto& subscriber : subscribers) {
        for (std::uint32_t index = 0U; succeeded && index < batch_size; ++index) {
          auto inbound = wait_for_fixed_sample(subscriber);
          if (!inbound.has_value()) {
            succeeded = false;
            break;
          }
          checksum ^= checksum_fixed_payload(inbound->payload());
        }
      }
      state.ResumeTiming();
    }
    if (!succeeded) {
      state.SkipWithError("failed to process fixed-POD fanout backend batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["bounded_retry"] = 1.0;
  state.counters["empty_return_scan"] =
      static_cast<double>(Stage == FixedPodFanoutBackendStage::kEmptyReturnScan);
  state.counters["event_wake"] = 0.0;
  state.counters["expected_recipient_gate"] = 1.0;
  state.counters["fanout"] = static_cast<double>(SubscriberCount);
  state.counters["payload_bytes"] = static_cast<double>(kPayloadBytes);
  state.counters["preloaned_send"] =
      static_cast<double>(Stage == FixedPodFanoutBackendStage::kPreloanedSend);
  state.counters["release_callback"] = EnableReleaseCallback ? 1.0 : 0.0;
  state.counters["returned_sample_reclaim"] =
      static_cast<double>(Stage == FixedPodFanoutBackendStage::kReturnedSampleReclaim);
  state.counters["stable_connection_refresh"] =
      static_cast<double>(Stage == FixedPodFanoutBackendStage::kStableConnectionRefresh);
  state.counters["vmm_fd_export"] = 0.0;
}

[[nodiscard]] bool wait_for_release_count(const std::atomic<std::uint64_t>& release_count,
                                          const std::uint64_t expected_count) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (release_count.load(std::memory_order_acquire) < expected_count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

[[nodiscard]] bool wait_for_receive_count(const std::atomic<std::uint64_t>& receive_count,
                                          const std::uint64_t expected_count) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (receive_count.load(std::memory_order_acquire) < expected_count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

[[nodiscard]] int publish_one(holomq::Publisher<BenchmarkPOD64>& publisher,
                              const std::uint32_t sequence) {
  auto outbound = publisher.loan();
  if (!outbound.has_value()) {
    return -1;
  }
  outbound->payload() = make_payload(sequence);
  auto sent = publisher.send(std::move(outbound).value());
  return sent.has_value() ? sent.value() : -1;
}

template <typename ReleaseCallback>
[[nodiscard]] int publish_one(holomq::Publisher<BenchmarkPOD64>& publisher,
                              const std::uint32_t sequence, ReleaseCallback& on_released) {
  auto outbound = publisher.loan();
  if (!outbound.has_value()) {
    return -1;
  }
  outbound->payload() = make_payload(sequence);
  auto sent = publisher.send(std::move(outbound).value(), on_released);
  return sent.has_value() ? sent.value() : -1;
}

template <typename SubscriberContainer>
[[nodiscard]] bool wait_for_subscriber_count(holomq::Publisher<BenchmarkPOD64>& publisher,
                                             SubscriberContainer& subscribers,
                                             const int expected_recipients) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  std::uint32_t sequence = 0U;
  std::uint64_t ignored_checksum = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    const int sent = publish_one(publisher, ++sequence);
    drain_subscribers(subscribers, ignored_checksum);
    if (sent >= expected_recipients) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

[[nodiscard]] bool wait_for_subscriber(holomq::Publisher<BenchmarkPOD64>& publisher) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (publisher.update_connections().has_value() && publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

template <std::size_t SubscriberCount>
void run_pubsub_polling(benchmark::State& state, const holomq::QueuePolicy policy,
                        const char* const service_suffix,
                        const bool enable_sample_release_callbacks = false,
                        const std::uint32_t queue_depth = 64U) {
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create HoloMQ context");
    return;
  }

  const std::string service_name = make_unique_service_name(service_suffix);
  const holomq::PubSubServiceOptions service_options =
      make_service_options(policy, SubscriberCount, queue_depth);
  const holomq::PublisherPubSubOptions publisher_options =
      make_publisher_options(enable_sample_release_callbacks);
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options(queue_depth);
  auto publisher =
      make_publisher(context.value(), service_name, service_options, publisher_options);
  if (!publisher.has_value()) {
    state.SkipWithError("failed to create HoloMQ publisher");
    return;
  }

  std::vector<holomq::Subscriber<BenchmarkPOD64>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0; i < SubscriberCount; ++i) {
    auto subscriber =
        make_subscriber(context.value(), service_name, service_options, subscriber_options);
    if (!subscriber.has_value()) {
      state.SkipWithError("failed to create HoloMQ subscriber");
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  if (!wait_for_subscriber_count(
          publisher.value(), subscribers, static_cast<int>(SubscriberCount))) {
    state.SkipWithError("timed out waiting for HoloMQ subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint32_t sequence = 0U;
  for (auto _ : state) {
    const int sent = publish_one(publisher.value(), ++sequence);
    if (sent < static_cast<int>(SubscriberCount)) {
      state.SkipWithError("HoloMQ publish did not reach all subscribers");
      break;
    }

    bool sample_timeout = false;
    for (auto& subscriber : subscribers) {
      auto sample = wait_for_sample(subscriber);
      if (!sample.has_value()) {
        state.SkipWithError("timed out waiting for HoloMQ sample");
        sample_timeout = true;
        break;
      }
      checksum ^= checksum_payload(sample->payload());
    }
    if (sample_timeout) {
      break;
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["event_wake"] = 0.0;
  state.counters["busy_poll"] = 0.0;
  state.counters["release_handler"] = enable_sample_release_callbacks ? 1.0 : 0.0;
  state.counters["receive_polling"] = 1.0;
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
  state.counters["vmm_fd_export"] = 0.0;
}

void BM_PubSubPOD_OneToOne_Blocking(benchmark::State& state) {
  run_pubsub_polling<1>(state, holomq::QueuePolicy::kBlocking, "pod_1to1_blocking");
}

void BM_PubSubPOD_OneToOne_Blocking_ReleaseHandlerEnabled(benchmark::State& state) {
  run_pubsub_polling<1>(
      state, holomq::QueuePolicy::kBlocking, "pod_1to1_blocking_release_handler", true);
}

void BM_PubSubPOD_OneToOne_DropOldestDepthOne(benchmark::State& state) {
  run_pubsub_polling<1>(
      state, holomq::QueuePolicy::kDropOldest, "pod_1to1_drop_oldest_depth_1", false, 1U);
}

void BM_PubSubPOD_OneToOne_DropOldestDepth64(benchmark::State& state) {
  run_pubsub_polling<1>(state, holomq::QueuePolicy::kDropOldest, "pod_1to1_drop_oldest_depth_64");
}

void BM_PubSubPOD_OneToOne_Blocking_LatchedWaitSet(benchmark::State& state) {
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create HoloMQ WaitSet benchmark context");
    return;
  }

  const std::string service_name = make_unique_service_name("pod_1to1_blocking_waitset");
  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking);
  const holomq::PublisherPubSubOptions publisher_options = make_publisher_options();
  const holomq::SubscriberPubSubOptions subscriber_options = make_subscriber_options();
  auto publisher = make_publisher(*context, service_name, service_options, publisher_options, true);
  auto subscriber =
      make_subscriber(*context, service_name, service_options, subscriber_options, true);
  auto waitset = context->create_waitset();
  if (!publisher.has_value() || !subscriber.has_value() || !waitset.has_value()) {
    state.SkipWithError("failed to create HoloMQ WaitSet benchmark endpoints");
    return;
  }
  auto attachment = waitset->attach(*subscriber);
  if (!attachment.has_value()) {
    state.SkipWithError("failed to attach HoloMQ WaitSet benchmark subscriber");
    return;
  }

  std::vector<holomq::Subscriber<BenchmarkPOD64>> subscribers;
  subscribers.emplace_back(std::move(*subscriber));
  if (!wait_for_subscriber(*publisher)) {
    state.SkipWithError("timed out waiting for HoloMQ WaitSet benchmark subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint32_t sequence = 0U;
  for (auto _ : state) {
    if (publish_one(*publisher, ++sequence) != 1) {
      state.SkipWithError("HoloMQ WaitSet publish did not reach its subscriber");
      break;
    }
    auto sample = wait_for_waitset_sample(*waitset, attachment->handle(), subscribers.front());
    if (!sample.has_value()) {
      state.SkipWithError("timed out waiting for HoloMQ WaitSet sample");
      break;
    }
    checksum ^= checksum_payload(sample->payload());
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["event_wake"] = 1.0;
  state.counters["busy_poll"] = 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
  state.counters["subscribers"] = 1.0;
  state.counters["receive_polling"] = 0.0;
  state.counters["vmm_fd_export"] = 0.0;
  state.counters["waitset"] = 1.0;
  state.counters["waitset_latched"] = 1.0;
}

template <ConcurrentReceiveMode ReceiveMode, ConcurrentWorkload Workload>
void run_concurrent_pubsub(
    benchmark::State& state) {  // NOLINT(readability-function-cognitive-complexity)
  constexpr std::uint32_t kThroughputBatchSize = 64U;
  std::uint32_t batch_size = Workload == ConcurrentWorkload::kLatency ? 1U : kThroughputBatchSize;
  if constexpr (Workload == ConcurrentWorkload::kPublisherSend) {
    batch_size = static_cast<std::uint32_t>(state.range(0));
    if (batch_size == 0U) {
      state.SkipWithError("concurrent HoloMQ publisher batch must be nonzero");
      return;
    }
  }
  const std::uint32_t endpoint_depth =
      Workload == ConcurrentWorkload::kPublisherSend ? batch_size : kThroughputBatchSize;
  constexpr bool kEventWake = ReceiveMode == ConcurrentReceiveMode::kKernelWaitSet;

  const std::optional<int> publisher_cpu = benchmark_cpu("HOLOMQ_BENCHMARK_PUBLISHER_CPU");
  const std::optional<int> subscriber_cpu = benchmark_cpu("HOLOMQ_BENCHMARK_SUBSCRIBER_CPU");
  if (!publisher_cpu.has_value() || !subscriber_cpu.has_value()) {
    state.SkipWithError("invalid HoloMQ benchmark CPU environment");
    return;
  }
  ScopedCPUAffinity publisher_affinity;
  if (!publisher_affinity.pin(*publisher_cpu)) {
    state.SkipWithError("failed to pin HoloMQ benchmark publisher");
    return;
  }

  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create concurrent HoloMQ context");
    return;
  }
  const std::string service_name = make_unique_service_name(
      ReceiveMode == ConcurrentReceiveMode::kPolling ? "pod_concurrent_polling"
                                                     : "pod_concurrent_waitset");
  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, 1U, endpoint_depth);
  holomq::PublisherPubSubOptions publisher_options = make_publisher_options();
  // A smoke row must fail with bounded evidence instead of parking forever in a blocking send.
  // The exact-recipient gate and retry budget are inactive on the healthy steady-state path and
  // keep all concurrent workloads comparable to the Runtime Graph service configuration.
  publisher_options.expected_recipient_count = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns = 5'000'000U;
  auto publisher =
      make_publisher(*context, service_name, service_options, publisher_options, kEventWake);
  auto subscriber = make_subscriber(
      *context, service_name, service_options, make_subscriber_options(endpoint_depth), kEventWake);
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create concurrent HoloMQ endpoints");
    return;
  }

  std::optional<holomq::WaitSet> waitset;
  std::optional<holomq::WaitSetAttachment> attachment;
  if constexpr (ReceiveMode == ConcurrentReceiveMode::kKernelWaitSet) {
    auto created_waitset = context->create_waitset();
    if (!created_waitset.has_value()) {
      state.SkipWithError("failed to create concurrent HoloMQ WaitSet");
      return;
    }
    waitset.emplace(std::move(*created_waitset));
    auto created_attachment = waitset->attach(*subscriber);
    if (!created_attachment.has_value()) {
      state.SkipWithError("failed to attach concurrent HoloMQ subscriber");
      return;
    }
    attachment.emplace(std::move(*created_attachment));
  }
  if (!wait_for_subscriber(*publisher)) {
    state.SkipWithError("timed out waiting for concurrent HoloMQ subscriber");
    return;
  }

  std::barrier startup{2};
  std::atomic<bool> stop{false};
  std::atomic<bool> subscriber_affinity_ok{true};
  std::atomic<bool> subscriber_failed{false};
  std::atomic<std::uint64_t> receive_count{0U};
  std::atomic<std::int64_t> last_receive_time_ns{0};
  std::uint64_t checksum = 0U;
  std::thread subscriber_thread([&]() {
    subscriber_affinity_ok.store(pin_current_thread(*subscriber_cpu), std::memory_order_release);
    startup.arrive_and_wait();
    if (!subscriber_affinity_ok.load(std::memory_order_acquire)) {
      return;
    }

    const holomq::WaitStrategy wait_strategy{
        .mode = holomq::WaitMode::KernelWait,
        .spin_duration = std::chrono::nanoseconds::zero(),
        .timeout = std::chrono::milliseconds{10},
    };
    auto drain_ready_samples = [&]() {
      bool received = false;
      while (auto sample = subscriber->try_take()) {
        received = true;
        checksum ^= checksum_payload(sample->payload());
        last_receive_time_ns.store(now_ns(), std::memory_order_relaxed);
        receive_count.fetch_add(1U, std::memory_order_release);
      }
      return received;
    };
    while (!stop.load(std::memory_order_acquire)) {
      if constexpr (ReceiveMode == ConcurrentReceiveMode::kKernelWaitSet) {
        auto waited = waitset->wait_for_ready(wait_strategy);
        if (!waited.has_value() || *waited == holomq::WaitResult::error) {
          subscriber_failed.store(true, std::memory_order_release);
          return;
        }
        if (*waited == holomq::WaitResult::timeout) {
          continue;
        }
        const auto ready = waitset->ready_handles();
        if (std::find(ready.begin(), ready.end(), attachment->handle()) == ready.end()) {
          continue;
        }
      }

      if constexpr (ReceiveMode == ConcurrentReceiveMode::kPolling) {
        if (!drain_ready_samples()) {
          std::this_thread::yield();
        }
      } else {
        static_cast<void>(drain_ready_samples());
      }
    }
  });
  startup.arrive_and_wait();
  if (!subscriber_affinity_ok.load(std::memory_order_acquire)) {
    stop.store(true, std::memory_order_release);
    subscriber_thread.join();
    state.SkipWithError("failed to pin HoloMQ benchmark subscriber");
    return;
  }

  std::uint32_t sequence = 0U;
  std::int64_t completed_iterations = 0;
  std::uint64_t published_count = 0U;
  for (auto _ : state) {
    const std::int64_t measured_start_ns = now_ns();
    bool publish_failed = false;
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      if (publish_one(*publisher, ++sequence) != 1) {
        publish_failed = true;
        break;
      }
    }
    const std::int64_t publish_end_ns = now_ns();
    const std::uint64_t expected_count = published_count + batch_size;
    if (publish_failed || !wait_for_receive_count(receive_count, expected_count)) {
      state.SkipWithError(publish_failed ? "concurrent HoloMQ publish failed"
                                         : "concurrent HoloMQ receive timed out");
      break;
    }
    published_count = expected_count;
    if constexpr (Workload == ConcurrentWorkload::kLatency) {
      const std::int64_t measured_end_ns = last_receive_time_ns.load(std::memory_order_acquire);
      state.SetIterationTime(static_cast<double>(measured_end_ns - measured_start_ns) * 1.0e-9);
    } else if constexpr (Workload == ConcurrentWorkload::kPublisherSend) {
      state.SetIterationTime(static_cast<double>(publish_end_ns - measured_start_ns) * 1.0e-9);
    }
    ++completed_iterations;
  }

  stop.store(true, std::memory_order_release);
  subscriber_thread.join();
  if (subscriber_failed.load(std::memory_order_acquire)) {
    state.SkipWithError("concurrent HoloMQ subscriber wait failed");
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_iterations * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["busy_poll"] = ReceiveMode == ConcurrentReceiveMode::kPolling ? 1.0 : 0.0;
  state.counters["concurrent"] = 1.0;
  state.counters["event_wake"] = kEventWake ? 1.0 : 0.0;
  state.counters["latency_workload"] = Workload == ConcurrentWorkload::kLatency ? 1.0 : 0.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
  state.counters["publisher_send_stage"] =
      Workload == ConcurrentWorkload::kPublisherSend ? 1.0 : 0.0;
  state.counters["publisher_cpu"] = static_cast<double>(*publisher_cpu);
  state.counters["receive_polling"] = ReceiveMode == ConcurrentReceiveMode::kPolling ? 1.0 : 0.0;
  state.counters["subscriber_cpu"] = static_cast<double>(*subscriber_cpu);
  state.counters["subscribers"] = 1.0;
  state.counters["vmm_fd_export"] = 0.0;
}

void BM_PubSubPOD_ConcurrentPolling_Throughput(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kPolling, ConcurrentWorkload::kThroughput>(state);
}

void BM_PubSubPOD_ConcurrentPolling_Latency(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kPolling, ConcurrentWorkload::kLatency>(state);
}

void BM_PubSubPOD_ConcurrentWaitSet_Throughput(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kKernelWaitSet, ConcurrentWorkload::kThroughput>(
      state);
}

void BM_PubSubPOD_ConcurrentWaitSet_Latency(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kKernelWaitSet, ConcurrentWorkload::kLatency>(state);
}

void BM_PubSubPOD_ConcurrentWaitSet_PublisherSend(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kKernelWaitSet, ConcurrentWorkload::kPublisherSend>(
      state);
}

void BM_PubSubPOD_ConcurrentPolling_PublisherSend(benchmark::State& state) {
  run_concurrent_pubsub<ConcurrentReceiveMode::kPolling, ConcurrentWorkload::kPublisherSend>(state);
}

template <bool WithReleaseCallback>
void run_pubsub_release_callback_batch(benchmark::State& state) {
  const auto in_flight_depth = static_cast<std::uint32_t>(state.range(0));
  // This must outlive publisher teardown on every benchmark error path because registered
  // callbacks retain a reference to it until their transport slots are reclaimed.
  std::atomic<std::uint64_t> release_count{0U};
  auto context = make_context();
  if (!context.has_value()) {
    state.SkipWithError("failed to create HoloMQ context");
    return;
  }

  const std::string service_name = make_unique_service_name(
      WithReleaseCallback ? "pod_release_callback" : "pod_without_release_callback");
  const holomq::PubSubServiceOptions service_options =
      make_service_options(holomq::QueuePolicy::kBlocking, 1U, in_flight_depth);
  const holomq::PublisherPubSubOptions publisher_options =
      make_publisher_options(WithReleaseCallback);
  const holomq::SubscriberPubSubOptions subscriber_options =
      make_subscriber_options(in_flight_depth);

  auto publisher =
      make_publisher(context.value(), service_name, service_options, publisher_options);
  auto subscriber =
      make_subscriber(context.value(), service_name, service_options, subscriber_options);
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create HoloMQ publisher or subscriber");
    return;
  }
  std::vector<holomq::Subscriber<BenchmarkPOD64>> subscribers;
  subscribers.emplace_back(std::move(subscriber).value());
  if (!wait_for_subscriber_count(publisher.value(), subscribers, 1)) {
    state.SkipWithError("timed out waiting for HoloMQ subscriber");
    return;
  }

  auto on_released = [&release_count]() noexcept {
    release_count.fetch_add(1U, std::memory_order_release);
  };
  std::uint64_t checksum = 0U;
  std::uint32_t sequence = 0U;
  std::uint64_t registered_callback_count = 0U;

  // Prime one complete batch outside the timed loop. The next publisher loans reclaim these
  // released transport slots and dispatch this batch's callbacks while registering their own.
  for (std::uint32_t i = 0U; i < in_flight_depth; ++i) {
    int sent = 0;
    if constexpr (WithReleaseCallback) {
      sent = publish_one(publisher.value(), ++sequence, on_released);
      if (sent >= 0) {
        ++registered_callback_count;
      }
    } else {
      sent = publish_one(publisher.value(), ++sequence);
    }
    if (sent < 1) {
      state.SkipWithError("failed to prime the HoloMQ batch path");
      return;
    }
  }
  for (std::uint32_t i = 0U; i < in_flight_depth; ++i) {
    auto sample = wait_for_sample(subscribers.front());
    if (!sample.has_value()) {
      state.SkipWithError("failed to drain the priming HoloMQ batch");
      return;
    }
    checksum ^= checksum_payload(sample->payload());
    sample.reset();
  }

  for (auto _ : state) {
    const std::uint64_t expected_release_count = registered_callback_count;
    bool batch_failed = false;
    for (std::uint32_t i = 0U; i < in_flight_depth; ++i) {
      int sent = 0;
      if constexpr (WithReleaseCallback) {
        sent = publish_one(publisher.value(), ++sequence, on_released);
        if (sent >= 0) {
          ++registered_callback_count;
        }
      } else {
        sent = publish_one(publisher.value(), ++sequence);
      }
      if (sent < 1) {
        state.SkipWithError("HoloMQ publish did not reach the subscriber");
        batch_failed = true;
        break;
      }
    }
    if constexpr (WithReleaseCallback) {
      if (!batch_failed && !wait_for_release_count(release_count, expected_release_count)) {
        state.SkipWithError("timed out waiting for HoloMQ release callback");
        batch_failed = true;
      }
    }

    for (std::uint32_t i = 0U; !batch_failed && i < in_flight_depth; ++i) {
      auto sample = wait_for_sample(subscribers.front());
      if (!sample.has_value()) {
        state.SkipWithError("timed out waiting for HoloMQ sample");
        batch_failed = true;
        break;
      }
      checksum ^= checksum_payload(sample->payload());
      sample.reset();
    }
    if (batch_failed) {
      drain_subscribers(subscribers, checksum);
      break;
    }
  }

  if constexpr (WithReleaseCallback) {
    // Reclaim the final measured batch outside the timed loop so no callbacks retain references
    // to benchmark-local state during publisher teardown.
    for (std::uint32_t i = 0U; i < in_flight_depth; ++i) {
      if (publish_one(publisher.value(), ++sequence) < 1) {
        state.SkipWithError("failed to flush final HoloMQ release callbacks");
        return;
      }
    }
    if (!wait_for_release_count(release_count, registered_callback_count)) {
      state.SkipWithError("timed out flushing final HoloMQ release callbacks");
      return;
    }
    for (std::uint32_t i = 0U; i < in_flight_depth; ++i) {
      auto sample = wait_for_sample(subscribers.front());
      if (!sample.has_value()) {
        state.SkipWithError("failed to drain final HoloMQ callback-flush batch");
        return;
      }
      checksum ^= checksum_payload(sample->payload());
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(in_flight_depth));
  state.counters["event_wake"] = 0.0;
  state.counters["in_flight_depth"] = static_cast<double>(in_flight_depth);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
  state.counters["release_callback"] = WithReleaseCallback ? 1.0 : 0.0;
  state.counters["release_handler"] = WithReleaseCallback ? 1.0 : 0.0;
  state.counters["vmm_fd_export"] = 0.0;
}

void BM_PubSubPOD_BatchWithoutReleaseCallback(benchmark::State& state) {
  run_pubsub_release_callback_batch<false>(state);
}

void BM_PubSubPOD_BatchWithReleaseCallback(benchmark::State& state) {
  run_pubsub_release_callback_batch<true>(state);
}

void BM_PublisherSampleReleaseRegistry_RegisterDispatch(benchmark::State& state) {
  constexpr std::size_t kMaxDepth = 64U;
  const auto in_flight_depth = static_cast<std::size_t>(state.range(0));
  std::array<BenchmarkPOD64, kMaxDepth> payloads{};
  holomq::detail::PublisherSampleReleaseRegistry registry(kMaxDepth + 1U);
  std::uint64_t release_count = 0U;
  auto on_released = [&release_count]() noexcept { ++release_count; };

  // Provision the reusable callback entries before measuring steady-state registration.
  if (!registry.register_pending(&payloads.front(), on_released)) {
    state.SkipWithError("failed to warm the HoloMQ release registry");
    return;
  }
  registry.dispatch(&payloads.front());
  release_count = 0U;

  for (auto _ : state) {
    bool registration_failed = false;
    for (std::size_t i = 0U; i < in_flight_depth; ++i) {
      if (!registry.register_pending(&payloads[i], on_released)) {
        state.SkipWithError("failed to register a HoloMQ release callback");
        registration_failed = true;
        break;
      }
    }
    for (std::size_t i = 0U; !registration_failed && i < in_flight_depth; ++i) {
      registry.dispatch(&payloads[i]);
    }
    if (registration_failed) {
      break;
    }
  }

  benchmark::DoNotOptimize(release_count);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(in_flight_depth));
  state.counters["in_flight_depth"] = static_cast<double>(in_flight_depth);
}

template <bool MeasureRegistration>
void run_publisher_sample_release_registry_stage(benchmark::State& state) {
  constexpr std::size_t kMaxDepth = 64U;
  const auto in_flight_depth = static_cast<std::size_t>(state.range(0));
  std::array<BenchmarkPOD64, kMaxDepth> payloads{};
  holomq::detail::PublisherSampleReleaseRegistry registry(kMaxDepth + 1U);
  std::uint64_t release_count = 0U;
  auto on_released = [&release_count]() noexcept { ++release_count; };

  // Provision reusable callback entries before measuring either steady-state stage.
  if (!registry.register_pending(&payloads.front(), on_released)) {
    state.SkipWithError("failed to warm the HoloMQ release registry");
    return;
  }
  registry.dispatch(&payloads.front());
  release_count = 0U;

  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool registration_failed = false;
    if constexpr (MeasureRegistration) {
      const auto measured_start = std::chrono::steady_clock::now();
      for (std::size_t index = 0U; index < in_flight_depth; ++index) {
        if (!registry.register_pending(&payloads[index], on_released)) {
          registration_failed = true;
          break;
        }
      }
      const auto measured_end = std::chrono::steady_clock::now();
      if (!registration_failed) {
        state.SetIterationTime(
            std::chrono::duration<double>(measured_end - measured_start).count());
      }
      for (std::size_t index = 0U; index < in_flight_depth; ++index) {
        registry.dispatch(&payloads[index]);
      }
    } else {
      for (std::size_t index = 0U; index < in_flight_depth; ++index) {
        if (!registry.register_pending(&payloads[index], on_released)) {
          registration_failed = true;
          break;
        }
      }
      const auto measured_start = std::chrono::steady_clock::now();
      if (!registration_failed) {
        for (std::size_t index = 0U; index < in_flight_depth; ++index) {
          registry.dispatch(&payloads[index]);
        }
      }
      const auto measured_end = std::chrono::steady_clock::now();
      if (!registration_failed) {
        state.SetIterationTime(
            std::chrono::duration<double>(measured_end - measured_start).count());
      }
    }
    if (registration_failed) {
      state.SkipWithError("failed to process a HoloMQ release registry stage batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(release_count);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(in_flight_depth));
  state.counters["dispatch_only"] = MeasureRegistration ? 0.0 : 1.0;
  state.counters["in_flight_depth"] = static_cast<double>(in_flight_depth);
  state.counters["register_only"] = MeasureRegistration ? 1.0 : 0.0;
}

void BM_PublisherSampleReleaseRegistry_RegisterOnly(benchmark::State& state) {
  run_publisher_sample_release_registry_stage<true>(state);
}

void BM_PublisherSampleReleaseRegistry_DispatchOnly(benchmark::State& state) {
  run_publisher_sample_release_registry_stage<false>(state);
}

template <std::size_t SubscriberCount>
void BM_PubSubPOD_Fanout_Blocking(benchmark::State& state) {
  run_pubsub_polling<SubscriberCount>(state, holomq::QueuePolicy::kBlocking, "pod_fanout_blocking");
}

template <std::size_t SubscriberCount>
void BM_PubSubPOD_Fanout_DropOldestDepthOne(benchmark::State& state) {
  run_pubsub_polling<SubscriberCount>(
      state, holomq::QueuePolicy::kDropOldest, "pod_fanout_drop_oldest_depth_1", false, 1U);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PollingRoundTrip(benchmark::State& state) {
  run_fixed_pod_polling<PayloadBytes, FixedPodMeasuredStage::kRoundTrip>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PollingPublish(benchmark::State& state) {
  run_fixed_pod_polling<PayloadBytes, FixedPodMeasuredStage::kPublish>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PollingTake(benchmark::State& state) {
  run_fixed_pod_polling<PayloadBytes, FixedPodMeasuredStage::kTake>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberEmptyProbe(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes, FixedPodSubscriberSubstage::kEmptyProbe>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberEmptyTryTake(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes, FixedPodSubscriberSubstage::kEmptyTryTake>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberReadyProbe(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes, FixedPodSubscriberSubstage::kReadyProbe>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberTryTakeBorrowed(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes, FixedPodSubscriberSubstage::kTryTakeBorrowed>(
      state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberTryTakeCopyOut(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes, FixedPodSubscriberSubstage::kTryTakeCopyOut>(
      state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_SubscriberRuntimeGraphEquivalent(benchmark::State& state) {
  run_fixed_pod_subscriber_substage<PayloadBytes,
                                    FixedPodSubscriberSubstage::kRuntimeGraphEquivalent>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherLoanReturn(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes, FixedPodPublisherSubstage::kLoanReturn>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherLoanAssignReturn(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes, FixedPodPublisherSubstage::kLoanAssignReturn>(
      state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSend(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes, FixedPodPublisherSubstage::kPreloanedSend>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSendExpectedRecipient(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes, FixedPodPublisherSubstage::kPreloanedSend, true>(
      state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSendBoundedRetry(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes,
                                   FixedPodPublisherSubstage::kPreloanedSend,
                                   false,
                                   false,
                                   true>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSendEventWake(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes,
                                   FixedPodPublisherSubstage::kPreloanedSend,
                                   false,
                                   true>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSendRuntimeEventless(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes,
                                   FixedPodPublisherSubstage::kPreloanedSend,
                                   true,
                                   false,
                                   true>(state);
}

template <std::size_t PayloadBytes>
void BM_PubSubFixed_PublisherPreloanedSendGraphEquivalent(benchmark::State& state) {
  run_fixed_pod_publisher_substage<PayloadBytes,
                                   FixedPodPublisherSubstage::kPreloanedSend,
                                   true,
                                   true,
                                   true>(state);
}

template <std::size_t SubscriberCount>
void BM_PubSubFixed_FanoutStableRefresh(benchmark::State& state) {
  run_fixed_pod_fanout_backend_stage<SubscriberCount,
                                     FixedPodFanoutBackendStage::kStableConnectionRefresh>(state);
}

template <std::size_t SubscriberCount>
void BM_PubSubFixed_FanoutEmptyReturnScan(benchmark::State& state) {
  run_fixed_pod_fanout_backend_stage<SubscriberCount, FixedPodFanoutBackendStage::kEmptyReturnScan>(
      state);
}

template <std::size_t SubscriberCount>
void BM_PubSubFixed_FanoutReturnedSampleReclaim(benchmark::State& state) {
  run_fixed_pod_fanout_backend_stage<SubscriberCount,
                                     FixedPodFanoutBackendStage::kReturnedSampleReclaim>(state);
}

template <std::size_t SubscriberCount>
void BM_PubSubFixed_FanoutReturnedSampleReclaimReleaseCallback(benchmark::State& state) {
  run_fixed_pod_fanout_backend_stage<SubscriberCount,
                                     FixedPodFanoutBackendStage::kReturnedSampleReclaim,
                                     true>(state);
}

template <std::size_t SubscriberCount>
void BM_PubSubFixed_FanoutPreloanedSend(benchmark::State& state) {
  run_fixed_pod_fanout_backend_stage<SubscriberCount, FixedPodFanoutBackendStage::kPreloanedSend>(
      state);
}

BENCHMARK(BM_PubSubPOD_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_PubSubPOD_OneToOne_Blocking_ReleaseHandlerEnabled)->UseRealTime();
BENCHMARK(BM_PubSubPOD_OneToOne_DropOldestDepthOne)->UseRealTime();
BENCHMARK(BM_PubSubPOD_OneToOne_DropOldestDepth64)->UseRealTime();
BENCHMARK(BM_PubSubPOD_OneToOne_Blocking_LatchedWaitSet)->UseRealTime();
BENCHMARK(BM_PubSubPOD_ConcurrentPolling_Throughput)->UseRealTime();
BENCHMARK(BM_PubSubPOD_ConcurrentPolling_Latency)->UseManualTime();
BENCHMARK(BM_PubSubPOD_ConcurrentWaitSet_Throughput)->UseRealTime();
BENCHMARK(BM_PubSubPOD_ConcurrentWaitSet_Latency)->UseManualTime();
BENCHMARK(BM_PubSubPOD_ConcurrentWaitSet_PublisherSend)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_PubSubPOD_ConcurrentPolling_PublisherSend)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_PubSubPOD_BatchWithoutReleaseCallback)->Arg(1)->Arg(8)->Arg(64)->UseRealTime();
BENCHMARK(BM_PubSubPOD_BatchWithReleaseCallback)->Arg(1)->Arg(8)->Arg(64)->UseRealTime();
BENCHMARK(BM_PublisherSampleReleaseRegistry_RegisterDispatch)->Arg(1)->Arg(8)->Arg(64);
BENCHMARK(BM_PublisherSampleReleaseRegistry_RegisterOnly)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_PublisherSampleReleaseRegistry_DispatchOnly)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_Blocking, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_Blocking, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_Blocking, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_DropOldestDepthOne, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_DropOldestDepthOne, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubPOD_Fanout_DropOldestDepthOne, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingRoundTrip, 64)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingRoundTrip, 128)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingRoundTrip, 440)->UseRealTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingPublish, 64)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingPublish, 128)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingPublish, 440)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingTake, 64)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingTake, 128)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PollingTake, 440)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberEmptyProbe, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberEmptyTryTake, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberReadyProbe, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberTryTakeBorrowed, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberTryTakeCopyOut, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_SubscriberRuntimeGraphEquivalent, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanReturn, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanReturn, 128)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanReturn, 440)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanAssignReturn, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanAssignReturn, 128)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherLoanAssignReturn, 440)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSend, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSend, 128)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSend, 440)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSendExpectedRecipient, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSendBoundedRetry, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSendEventWake, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSendRuntimeEventless, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_PublisherPreloanedSendGraphEquivalent, 64)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutStableRefresh, 2)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutEmptyReturnScan, 2)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutReturnedSampleReclaim, 1)->Arg(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutReturnedSampleReclaim, 2)->Arg(1)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutReturnedSampleReclaim, 4)->Arg(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutReturnedSampleReclaim, 8)->Arg(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutReturnedSampleReclaimReleaseCallback, 2)
    ->Arg(1)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutPreloanedSend, 1)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutPreloanedSend, 2)->Arg(1)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutPreloanedSend, 4)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_PubSubFixed_FanoutPreloanedSend, 8)->Arg(64)->UseManualTime();

}  // namespace
