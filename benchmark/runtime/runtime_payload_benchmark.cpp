/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <benchmark/benchmark.h>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "runtime/runtime_channel_binding.hpp"

namespace holoscan {

struct RuntimeBenchmarkPod {
  std::uint32_t frame{};
  std::array<std::uint32_t, 15> values{};
};

template <>
struct ChannelTypeIdentity<RuntimeBenchmarkPod> {
  static constexpr std::string_view name = "nvidia.holoscan.benchmark.RuntimeBenchmarkPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<RuntimeBenchmarkPod> {
  static constexpr auto fields = pod_fields(
      pod_field("frame", &RuntimeBenchmarkPod::frame, offsetof(RuntimeBenchmarkPod, frame)),
      pod_field("values", &RuntimeBenchmarkPod::values, offsetof(RuntimeBenchmarkPod, values)));
};

namespace {

using namespace std::chrono_literals;

template <PublishedPayload T>
class PayloadSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<T> output;
};

[[nodiscard]] std::string unique_endpoint_name(std::string_view payload_name) {
  static std::atomic<std::uint64_t> ordinal{};
  return "runtime_payload_benchmark_" + std::string(payload_name) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

template <PublishedPayload T>
class CompiledCatalogFixture {
 public:
  CompiledCatalogFixture(const CompiledCatalogFixture&) = delete;
  CompiledCatalogFixture& operator=(const CompiledCatalogFixture&) = delete;
  CompiledCatalogFixture(CompiledCatalogFixture&&) = delete;
  CompiledCatalogFixture& operator=(CompiledCatalogFixture&&) = delete;

  ~CompiledCatalogFixture() {
    if (session_) {
      session_->cancel();
      (void)session_->join();
    }
  }

  [[nodiscard]] static expected<std::unique_ptr<CompiledCatalogFixture>, Error> create(
      std::string_view payload_name) {
    auto fixture = std::unique_ptr<CompiledCatalogFixture>{new CompiledCatalogFixture};
    const std::string endpoint_name = unique_endpoint_name(payload_name);

    Graph graph{"runtime-payload-benchmark"};
    const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
    graph.set_default_clock(runtime_clock);
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 1U,
        .max_borrowed_samples = 1U,
    };
    const auto exported = graph.export_topic<T>("holomq://" + endpoint_name, std::move(options));
    const auto source = graph.op<PayloadSourceOperator<T>>("source");
    graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    if (!plan.ok()) {
      return make_unexpected(Error{ErrorCode::kFailure, "payload benchmark graph did not compile"});
    }
    const auto retained = detail::ExecutionPlanAccess::retain(plan);
    if (!retained) {
      return make_unexpected(Error{ErrorCode::kFailure, "payload benchmark plan is empty"});
    }
    fixture->source_.setup(fixture->source_spec_);
    const auto replay_port = detail::OperatorSpecCompilerAccess::port(fixture->source_spec_, 0U);
    const auto planned = std::find_if(
        retained->ports.begin(), retained->ports.end(), [&](const PortPlanRecord& candidate) {
          return candidate.owner_kind == PortOwnerKind::kOperator &&
                 candidate.direction == PortDirection::kOutput &&
                 candidate.local_name == replay_port.name;
        });
    if (planned == retained->ports.end() || replay_port.declaration == nullptr ||
        replay_port.runtime_type_ops == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    fixture->source_owner_ = planned->owner;
    const std::array replay{detail::RuntimeChannelPortReplayRecord{
        .port = planned->id,
        .owner = planned->owner,
        .direction = replay_port.direction,
        .declaration = const_cast<void*>(replay_port.declaration),
        .type_ops = replay_port.runtime_type_ops,
    }};

    auto external_runtime = ChannelRuntime::create();
    if (!external_runtime) {
      return make_unexpected(std::move(external_runtime).error());
    }
    fixture->external_runtime_ = std::make_unique<ChannelRuntime>(std::move(*external_runtime));
    auto subscriber = fixture->external_runtime_->template create_subscriber<T>(
        plan.endpoint_catalog(), endpoint_name);
    if (!subscriber) {
      return make_unexpected(std::move(subscriber).error());
    }
    fixture->subscriber_ = std::make_unique<ChannelSubscriber<T>>(std::move(*subscriber));

    auto session = detail::RuntimeChannelSession::create(*retained, replay);
    if (!session) {
      return make_unexpected(std::move(session).error());
    }
    fixture->session_ = std::move(*session);
    auto refreshed = fixture->session_->refresh_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    std::uint32_t consecutive_observations = 0U;
    while (std::chrono::steady_clock::now() < deadline && consecutive_observations < 2U) {
      auto ready = fixture->subscriber_->has_samples();
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
      if (fixture->subscriber_->has_publishers()) {
        ++consecutive_observations;
      } else {
        consecutive_observations = 0U;
      }
      std::this_thread::sleep_for(1ms);
    }
    if (consecutive_observations < 2U) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    auto armed = fixture->session_->arm();
    if (!armed) {
      return make_unexpected(std::move(armed).error());
    }
    return fixture;
  }

  [[nodiscard]] expected<Sample<T>, Error> round_trip(const T& value) noexcept {
    auto prepared = session_->prepare_activation(source_owner_, generation_, {});
    if (!prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    auto emitted = source_.output.emit(value);
    if (!emitted) {
      (void)session_->retire_activation(source_owner_, generation_, false);
      return make_unexpected(std::move(emitted).error());
    }
    auto retired = session_->retire_activation(source_owner_, generation_, true);
    if (!retired) {
      return make_unexpected(std::move(retired).error());
    }
    ++generation_;

    for (std::uint32_t attempt = 0U; attempt < 1000U; ++attempt) {
      auto received = subscriber_->template receive_sample<T>();
      if (received || received.error().code != ErrorCode::kNotReady) {
        return received;
      }
      std::this_thread::yield();
    }
    return make_unexpected(Error{ErrorCode::kTimeout});
  }

 private:
  CompiledCatalogFixture() = default;

  PayloadSourceOperator<T> source_;
  OperatorSpec source_spec_;
  std::unique_ptr<ChannelRuntime> external_runtime_;
  std::unique_ptr<ChannelSubscriber<T>> subscriber_;
  std::unique_ptr<detail::RuntimeChannelSession> session_;
  VertexId source_owner_{};
  std::uint64_t generation_{1U};
};

void skip_with_error(benchmark::State& state, std::string_view operation, const Error& error) {
  const std::string message =
      std::string{operation} + ": " + std::string{error_code_name(error.code)};
  state.SkipWithError(message.c_str());
}

template <PublishedPayload T>
void run_payload_benchmark(benchmark::State& state, std::string_view payload_name, const T& value) {
  auto fixture = CompiledCatalogFixture<T>::create(payload_name);
  if (!fixture) {
    skip_with_error(state, "failed to create compiled catalog fixture", fixture.error());
    return;
  }

  std::uint64_t allocation_count = 0U;
  std::uint64_t wire_bytes = 0U;
  for (auto _ : state) {
    std::size_t allocations = 0U;
    {
      const test::ScopedAllocationCounter counter;
      auto received = (*fixture)->round_trip(value);
      allocations = counter.count();
      if (!received) {
        skip_with_error(state, "compiled catalog round trip failed", received.error());
        return;
      }
      wire_bytes = received->metadata.byte_size;
      benchmark::DoNotOptimize(received->data);
    }
    allocation_count += allocations;
  }
  const double iterations = static_cast<double>(state.iterations());
  state.counters["allocations_per_sample"] = static_cast<double>(allocation_count) / iterations;
  state.counters["native_bytes"] = static_cast<double>(sizeof(T));
  state.counters["wire_bytes"] = static_cast<double>(wire_bytes);
  state.SetItemsProcessed(state.iterations());
}

// The array field forces the production custom-POD fieldwise codec instead of the inline bit-cast
// path used by the older Channel POD benchmark.
void BM_CompiledCatalogCustomPod(benchmark::State& state) {
  RuntimeBenchmarkPod value{.frame = 17U};
  for (std::size_t index = 0U; index < value.values.size(); ++index) {
    value.values[index] = static_cast<std::uint32_t>(index + 1U);
  }
  run_payload_benchmark(state, "custom_pod", value);
}

void BM_CompiledCatalogTime(benchmark::State& state) {
  run_payload_benchmark(
      state, "time", schema::TimeT{.time_ns = 123456789, .acqtime_ns = 987654321});
}

void BM_CompiledCatalogString(benchmark::State& state) {
  schema::StringT value;
  value.data.assign(static_cast<std::size_t>(state.range(0)), 's');
  run_payload_benchmark(state, "string", value);
}

void BM_CompiledCatalogImu(benchmark::State& state) {
  schema::ImuT value;
  value.header = std::make_shared<schema::HeaderT>();
  value.header->stamp = std::make_shared<schema::TimeT>();
  value.header->stamp->time_ns = 123456789;
  value.header->stamp->acqtime_ns = 987654321;
  value.header->frame_id = "imu-benchmark-frame";
  value.orientation = schema::Quaternionf{1.0F, 0.1F, 0.2F, 0.3F};
  value.angular_velocity = schema::Vector3f{1.0F, 2.0F, 3.0F};
  value.linear_acceleration = schema::Vector3f{4.0F, 5.0F, 6.0F};
  run_payload_benchmark(state, "imu", value);
}

BENCHMARK(BM_CompiledCatalogCustomPod);
BENCHMARK(BM_CompiledCatalogTime);
BENCHMARK(BM_CompiledCatalogString)->Arg(16)->Arg(256)->Arg(384);
BENCHMARK(BM_CompiledCatalogImu);

}  // namespace
}  // namespace holoscan
