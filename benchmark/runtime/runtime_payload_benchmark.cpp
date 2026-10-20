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
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
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
  static constexpr std::string_view name = "holoscan.benchmark.RuntimeBenchmarkPod";
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

struct RoundTripAllocations {
  std::size_t activation{};
  std::size_t publication{};
  std::size_t retirement{};
  std::size_t receive{};
};

template <typename Fn>
auto invoke_counted(Fn&& operation, std::size_t* allocations) {
  if (allocations == nullptr)
    return operation();
  const test::ScopedAllocationCounter counter;
  auto result = operation();
  *allocations += counter.count();
  return result;
}

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

  [[nodiscard]] expected<Sample<T>, Error> round_trip(
      const T& value, RoundTripAllocations* allocations = nullptr) noexcept {
    auto prepared =
        invoke_counted([&] { return session_->prepare_activation(source_owner_, generation_, {}); },
                       allocations ? &allocations->activation : nullptr);
    if (!prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    auto emitted = invoke_counted([&] { return source_.output.emit(value); },
                                  allocations ? &allocations->publication : nullptr);
    if (!emitted) {
      (void)session_->retire_activation(source_owner_, generation_, false);
      return make_unexpected(std::move(emitted).error());
    }
    auto retired = invoke_counted(
        [&] { return session_->retire_activation(source_owner_, generation_, true); },
        allocations ? &allocations->retirement : nullptr);
    if (!retired) {
      return make_unexpected(std::move(retired).error());
    }
    ++generation_;

    for (std::uint32_t attempt = 0U; attempt < 1000U; ++attempt) {
      auto received = invoke_counted([&] { return subscriber_->template receive_sample<T>(); },
                                     allocations ? &allocations->receive : nullptr);
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
bool same_payload(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, RuntimeBenchmarkPod>) {
    return a.frame == b.frame && a.values == b.values;
  } else if constexpr (std::is_same_v<T, schema::TimeT>) {
    return a.time_ns == b.time_ns && a.acqtime_ns == b.acqtime_ns;
  } else if constexpr (std::is_same_v<T, schema::StringT>) {
    return a.data == b.data;
  } else {
    const auto matrix_equal = [](const auto& x, const auto& y) {
      return static_cast<bool>(x) == static_cast<bool>(y) &&
             (!x || std::equal(x->m()->begin(), x->m()->end(), y->m()->begin()));
    };
    if (static_cast<bool>(a.header) != static_cast<bool>(b.header) ||
        !matrix_equal(a.orientation_covariance, b.orientation_covariance) ||
        !matrix_equal(a.angular_velocity_covariance, b.angular_velocity_covariance) ||
        !matrix_equal(a.linear_acceleration_covariance, b.linear_acceleration_covariance))
      return false;
    if (a.header && (a.header->frame_id != b.header->frame_id ||
                     a.header->device_sequence != b.header->device_sequence ||
                     a.header->capture_timestamp_ns != b.header->capture_timestamp_ns ||
                     static_cast<bool>(a.header->stamp) != static_cast<bool>(b.header->stamp)))
      return false;
    if (a.header && a.header->stamp && !same_payload(*a.header->stamp, *b.header->stamp))
      return false;
    return a.flags == b.flags && a.orientation.x() == b.orientation.x() &&
           a.orientation.y() == b.orientation.y() && a.orientation.z() == b.orientation.z() &&
           a.orientation.w() == b.orientation.w() &&
           a.angular_velocity.x() == b.angular_velocity.x() &&
           a.angular_velocity.y() == b.angular_velocity.y() &&
           a.angular_velocity.z() == b.angular_velocity.z() &&
           a.linear_acceleration.x() == b.linear_acceleration.x() &&
           a.linear_acceleration.y() == b.linear_acceleration.y() &&
           a.linear_acceleration.z() == b.linear_acceleration.z();
  }
}

template <PublishedPayload T>
void run_payload_sequence(benchmark::State& state, std::string_view payload_name,
                          std::span<const T> values) {
  if (values.empty()) {
    state.SkipWithError("catalog payload sequence must not be empty");
    return;
  }
  constexpr std::size_t kCountedSamples = 64;
  // Cover every declared value even if future sequences exceed the allocation-counting pass.
  const auto counted_samples = std::max(kCountedSamples + 1, values.size());
  std::uint64_t max_wire_bytes = 0U;
  RoundTripAllocations first, repeated;
  {
    auto counted = CompiledCatalogFixture<T>::create(payload_name);
    if (!counted) {
      skip_with_error(state, "failed to create counted catalog fixture", counted.error());
      return;
    }
    for (std::size_t i = 0; i < counted_samples; ++i) {
      const auto& value = values[i % values.size()];
      auto received = (*counted)->round_trip(value, i == 0 ? &first : &repeated);
      if (!received) {
        skip_with_error(state, "counted catalog round trip failed", received.error());
        return;
      }
      if (!same_payload(received->data, value)) {
        state.SkipWithError("catalog payload or optional presence did not round trip");
        return;
      }
      max_wire_bytes = std::max<std::uint64_t>(max_wire_bytes, received->metadata.byte_size);
    }
  }
  // Count on the actual executing publisher/subscriber thread, never during the timed loop.
  auto fixture = CompiledCatalogFixture<T>::create(payload_name);
  if (!fixture) {
    skip_with_error(state, "failed to create timed catalog fixture", fixture.error());
    return;
  }
  std::size_t index{};
  for (auto _ : state) {
    const auto& value = values[index++ % values.size()];
    auto received = (*fixture)->round_trip(value);
    if (!received) {
      skip_with_error(state, "compiled catalog round trip failed", received.error());
      return;
    }
    if (!same_payload(received->data, value)) {
      state.SkipWithError("timed catalog payload validation failed");
      return;
    }
    benchmark::DoNotOptimize(received->data);
  }
  for (const auto& [name, member] :
       std::array{std::pair{"activation", &RoundTripAllocations::activation},
                  std::pair{"publication", &RoundTripAllocations::publication},
                  std::pair{"retirement", &RoundTripAllocations::retirement},
                  std::pair{"receive", &RoundTripAllocations::receive}}) {
    state.counters[std::string("first_") + name + "_allocations"] =
        static_cast<double>(first.*member);
    state.counters[std::string(name) + "_allocations_per_sample"] =
        static_cast<double>(repeated.*member) / static_cast<double>(counted_samples - 1);
  }
  state.counters["allocations_per_sample"] =
      static_cast<double>(repeated.activation + repeated.publication + repeated.retirement +
                          repeated.receive) /
      static_cast<double>(counted_samples - 1);
  state.counters["counted_samples"] = static_cast<double>(counted_samples);
  state.counters["native_bytes"] = static_cast<double>(sizeof(T));
  state.counters["max_wire_bytes"] = static_cast<double>(max_wire_bytes);
  state.SetItemsProcessed(state.iterations());
}

template <PublishedPayload T>
void run_payload_benchmark(benchmark::State& state, std::string_view payload_name, const T& value) {
  run_payload_sequence<T>(state, payload_name, std::span<const T>{&value, 1});
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
  value.orientation = schema::Quaternionf{0.1F, 0.2F, 0.3F, 1.0F};
  value.angular_velocity = schema::Vector3f{1.0F, 2.0F, 3.0F};
  value.linear_acceleration = schema::Vector3f{4.0F, 5.0F, 6.0F};
  run_payload_benchmark(state, "imu", value);
}

void BM_CompiledCatalogImuTransitions(benchmark::State& state) {
  std::vector<schema::ImuT> values;
  const auto append = [&](unsigned mask, std::size_t length) {
    schema::ImuT value;
    if (mask & 1U) {
      value.header = std::make_shared<schema::HeaderT>();
      value.header->frame_id.assign(length, 'i');
      if (mask & 2U) {
        value.header->stamp = std::make_shared<schema::TimeT>();
        value.header->stamp->time_ns = 123;
      }
    }
    if (mask & 4U)
      value.orientation_covariance = std::make_shared<schema::Matrix3f>();
    if (mask & 8U)
      value.angular_velocity_covariance = std::make_shared<schema::Matrix3f>();
    if (mask & 16U)
      value.linear_acceleration_covariance = std::make_shared<schema::Matrix3f>();
    values.push_back(std::move(value));
  };
  switch (state.range(0)) {
    case 0:
      append(0, 0);
      break;
    case 1:
      append(31, 128);
      break;
    case 2:
      append(31, 128);
      append(0, 0);
      break;
    case 3:
      append(31, 6);
      append(31, 128);
      append(0, 0);
      append(31, 128);
      break;
    default:
      for (unsigned mask = 0; mask < 32; ++mask)
        append(mask, 128);
  }
  run_payload_sequence<schema::ImuT>(state, "imu_transitions", values);
}

BENCHMARK(BM_CompiledCatalogCustomPod);
BENCHMARK(BM_CompiledCatalogTime);
BENCHMARK(BM_CompiledCatalogString)->Arg(16)->Arg(256)->Arg(384);
BENCHMARK(BM_CompiledCatalogImu);
BENCHMARK(BM_CompiledCatalogImuTransitions)->ArgName("pattern")->DenseRange(0, 4);

}  // namespace
}  // namespace holoscan
