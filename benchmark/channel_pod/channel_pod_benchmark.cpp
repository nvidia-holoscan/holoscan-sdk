/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/errors.hpp>

#include "../common/benchmark_pod.hpp"
#include "../common/channel_benchmark_common.hpp"

namespace {

using holoscan::BenchmarkPOD64;
using holoscan::benchmarking::make_channel_benchmark_options;
using holoscan::benchmarking::make_unique_channel_benchmark_name;
using holoscan::benchmarking::wait_for_channel_subscribers;

[[nodiscard]] holoscan::BenchmarkPOD64 make_payload(const std::uint32_t sequence) noexcept {
  holoscan::BenchmarkPOD64 payload{
      .sequence = sequence,
      .source_start_time_ns = static_cast<std::int64_t>(sequence) * 1000,
      .send_time_ns = static_cast<std::int64_t>(sequence) * 2000,
      .auxiliary0 = sequence ^ 0xA5A5'5A5AU,
      .auxiliary1 = static_cast<std::uint64_t>(sequence) * 3U,
  };
  payload.checksum = holoscan::benchmark_pod_checksum(payload);
  return payload;
}

[[nodiscard]] std::uint64_t checksum_payload(const holoscan::BenchmarkPOD64& payload) noexcept {
  return holoscan::benchmark_pod_checksum(payload) ^ (payload.checksum << 1U);
}

template <typename SubscriberContainer>
void drain_subscribers(SubscriberContainer& subscribers, std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    while (true) {
      auto sample = subscriber.receive();
      if (!sample.has_value()) {
        break;
      }
      checksum ^= checksum_payload(sample.value());
    }
  }
}

template <typename SubscriberContainer>
[[nodiscard]] bool wait_for_subscriber_count(
    holoscan::ChannelPublisher<holoscan::BenchmarkPOD64>& publisher,
    SubscriberContainer& subscribers, const std::uint32_t expected_recipients) {
  std::uint32_t sequence = 0U;
  std::uint64_t ignored_checksum = 0U;
  return wait_for_channel_subscribers(
      publisher,
      expected_recipients,
      [&publisher, &sequence]() { return publisher.emit(make_payload(++sequence)); },
      [&subscribers, &ignored_checksum]() { drain_subscribers(subscribers, ignored_checksum); });
}

template <std::size_t SubscriberCount>
void run_channel_pubsub_polling(benchmark::State& state, const holoscan::ChannelQueuePolicy policy,
                                const char* const channel_suffix,
                                const std::uint32_t queue_depth = 64U) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name =
      make_unique_channel_benchmark_name("holoscan_channel_benchmark_", channel_suffix);
  const holoscan::ChannelOptions options = make_channel_benchmark_options(
      policy, static_cast<std::uint32_t>(SubscriberCount), queue_depth);
  auto publisher = runtime->create_publisher<BenchmarkPOD64>(channel_name, options);
  if (!publisher.has_value()) {
    const std::string message =
        "failed to create Channel publisher: " + holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::vector<holoscan::ChannelSubscriber<BenchmarkPOD64>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0; i < SubscriberCount; ++i) {
    auto subscriber = runtime->create_subscriber<BenchmarkPOD64>(channel_name, options);
    if (!subscriber.has_value()) {
      const std::string message = "failed to create Channel subscriber: " +
                                  holoscan::runtime_error_message(subscriber.error());
      state.SkipWithError(message.c_str());
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  if (!wait_for_subscriber_count(
          publisher.value(), subscribers, static_cast<std::uint32_t>(SubscriberCount))) {
    state.SkipWithError("timed out waiting for Channel subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint32_t sequence = 0U;
  for (auto _ : state) {
    auto emitted = publisher->emit(make_payload(++sequence));
    if (!emitted.has_value()) {
      const std::string message =
          "Channel publish failed: " + holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < SubscriberCount) {
      state.SkipWithError("Channel publish did not reach all subscribers");
      break;
    }

    for (auto& subscriber : subscribers) {
      std::optional<BenchmarkPOD64> sample;
      while (!sample.has_value()) {
        auto received = subscriber.receive();
        if (received.has_value()) {
          sample = received.value();
        } else if (received.error().code != holoscan::ErrorCode::kNotReady) {
          const std::string message =
              "Channel receive failed: " + holoscan::runtime_error_message(received.error());
          state.SkipWithError(message.c_str());
          break;
        }
      }
      if (!sample.has_value()) {
        break;
      }
      checksum ^= checksum_payload(sample.value());
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
}

template <bool WaitAndReceive>
// Keep endpoint setup, queue priming, and the measured receive loop together in this benchmark.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void run_channel_queued_receive(benchmark::State& state, const char* const channel_suffix) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name =
      make_unique_channel_benchmark_name("holoscan_channel_benchmark_", channel_suffix);
  auto options = make_channel_benchmark_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.enable_event_wake = true;
  auto publisher = runtime->create_publisher<BenchmarkPOD64>(channel_name, options);
  auto subscriber = runtime->create_subscriber<BenchmarkPOD64>(channel_name, options);
  if (!publisher || !subscriber) {
    state.SkipWithError("failed to create queued-receive benchmark endpoints");
    return;
  }
  std::vector<holoscan::ChannelSubscriber<BenchmarkPOD64>> subscribers;
  subscribers.emplace_back(std::move(*subscriber));
  if (!wait_for_subscriber_count(publisher.value(), subscribers, 1U)) {
    state.SkipWithError("timed out waiting for queued-receive benchmark subscriber");
    return;
  }

  std::uint32_t sequence = 0U;
  std::uint64_t checksum = 0U;
  auto& receiver = subscribers.front();
  for (auto _ : state) {
    state.PauseTiming();
    auto emitted = publisher->emit(make_payload(++sequence));
    if (!emitted || *emitted != 1U) {
      state.ResumeTiming();
      state.SkipWithError("failed to queue benchmark sample");
      break;
    }

    bool queued = false;
    const auto queue_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (std::chrono::steady_clock::now() < queue_deadline) {
      auto ready = receiver.has_samples();
      if (!ready) {
        break;
      }
      if (*ready) {
        queued = true;
        break;
      }
      std::this_thread::yield();
    }
    state.ResumeTiming();
    if (!queued) {
      state.SkipWithError("sample did not become queued before benchmark receive");
      break;
    }

    if constexpr (WaitAndReceive) {
      auto received =
          receiver.wait_and_receive_sample(std::chrono::steady_clock::time_point::max());
      if (!received) {
        const std::string message = "queued wait_and_receive_sample failed: " +
                                    holoscan::runtime_error_message(received.error());
        state.SkipWithError(message.c_str());
        break;
      }
      checksum ^= checksum_payload(received->data);
    } else {
      auto received = receiver.receive_sample();
      if (!received) {
        const std::string message =
            "queued receive_sample failed: " + holoscan::runtime_error_message(received.error());
        state.SkipWithError(message.c_str());
        break;
      }
      checksum ^= checksum_payload(received->data);
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["payload_bytes"] = static_cast<double>(sizeof(BenchmarkPOD64));
}

// REQ: SYSTEM-005
void BM_ChannelPOD_OneToOne_Blocking(benchmark::State& state) {
  run_channel_pubsub_polling<1>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "pod_1to1_blocking");
}

void BM_ChannelPOD_OneToOne_DropOldestDepthOne(benchmark::State& state) {
  run_channel_pubsub_polling<1>(
      state, holoscan::ChannelQueuePolicy::kDropOldest, "pod_1to1_drop_oldest_depth_1", 1U);
}

void BM_ChannelPOD_OneToOne_DropOldestDepth64(benchmark::State& state) {
  run_channel_pubsub_polling<1>(
      state, holoscan::ChannelQueuePolicy::kDropOldest, "pod_1to1_drop_oldest_depth_64");
}

void BM_ChannelPOD_QueuedReceiveSample(benchmark::State& state) {
  run_channel_queued_receive<false>(state, "pod_queued_receive_sample");
}

void BM_ChannelPOD_QueuedWaitAndReceiveSample(benchmark::State& state) {
  run_channel_queued_receive<true>(state, "pod_queued_wait_and_receive_sample");
}

template <std::size_t SubscriberCount>
void BM_ChannelPOD_Fanout_Blocking(benchmark::State& state) {
  run_channel_pubsub_polling<SubscriberCount>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "pod_fanout_blocking");
}

template <std::size_t SubscriberCount>
void BM_ChannelPOD_Fanout_DropOldestDepthOne(benchmark::State& state) {
  run_channel_pubsub_polling<SubscriberCount>(
      state, holoscan::ChannelQueuePolicy::kDropOldest, "pod_fanout_drop_oldest_depth_1", 1U);
}

BENCHMARK(BM_ChannelPOD_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelPOD_OneToOne_DropOldestDepthOne)->UseRealTime();
BENCHMARK(BM_ChannelPOD_OneToOne_DropOldestDepth64)->UseRealTime();
BENCHMARK(BM_ChannelPOD_QueuedReceiveSample);
BENCHMARK(BM_ChannelPOD_QueuedWaitAndReceiveSample);
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_Blocking, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_Blocking, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_Blocking, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_DropOldestDepthOne, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_DropOldestDepthOne, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelPOD_Fanout_DropOldestDepthOne, 8)->UseRealTime();

}  // namespace
