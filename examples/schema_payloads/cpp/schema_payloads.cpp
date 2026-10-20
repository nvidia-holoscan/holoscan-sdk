/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/schema/prepared_schema_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Every source ticks at the same period; each lane stops emitting after its
// bounded count, and main() stops the session once all lanes are observed.
constexpr std::chrono::nanoseconds kTickPeriod = 20ms;
constexpr std::size_t kStringLimit = 5U;
constexpr std::size_t kTimeLimit = 4U;
constexpr std::size_t kIMULimit = 4U;
constexpr std::uint64_t kTraceBase = 0xC0FFEE00ULL;
constexpr std::int64_t kAcquisitionLatencyNs = 1'000'000;  // simulated 1 ms

// The package-declared serialized bound for schema::StringT: every emit of a
// generated-schema payload must fit this many bytes on the wire (512 for the
// SDK String schema). The oversize branch below exceeds it on purpose.
constexpr std::size_t kStringBound =
    holoscan::generated_schema_max_serialized_size_v<holoscan::schema::StringT>;

// Cross-thread observation state. Operators run on runtime threads, so the
// results are collected in a static struct (owned values only — the graph
// rejects borrowed captures) and main() waits on the condition variable.
struct SchemaObservation {
  enum class Oversize : std::uint8_t { kNotAttempted, kRejected, kAccepted };

  static void reset() {
    const std::lock_guard lock(mutex);
    strings = 0U;
    times = 0U;
    imus = 0U;
    degraded = 0U;
    last_string.clear();
    trace_match = true;
    time_delta_ok = true;
    imu_fields_ok = true;
    oversize = Oversize::kNotAttempted;
    oversize_error = holoscan::Error{};
  }

  static void record_string(const holoscan::Sample<holoscan::schema::StringT>& sample) {
    {
      const std::lock_guard lock(mutex);
      ++strings;
      last_string = sample.data.data;
      trace_match =
          trace_match && (sample.metadata.trace_id == kTraceBase + sample.metadata.frame_id);
      if (sample.metadata.degraded()) {
        ++degraded;
      }
      std::cout << "string #" << strings << ": \"" << sample.data.data
                << "\" (frame=" << sample.metadata.frame_id
                << ", degraded=" << (sample.metadata.degraded() ? "yes" : "no") << ")\n";
    }
    ready.notify_all();
  }

  static void record_time(const holoscan::Sample<holoscan::schema::TimeT>& sample) {
    {
      const std::lock_guard lock(mutex);
      ++times;
      const std::int64_t delta = sample.data.time_ns - sample.data.acqtime_ns;
      time_delta_ok = time_delta_ok && (delta == kAcquisitionLatencyNs);
      std::cout << "time #" << times << ": time_ns-acqtime_ns=" << delta << "\n";
    }
    ready.notify_all();
  }

  static void record_imu(const holoscan::Sample<holoscan::schema::ImuT>& sample) {
    {
      const std::lock_guard lock(mutex);
      ++imus;
      // Read the orientation only after the producer claims it means something; the quaternion
      // is inline, so an unset one is indistinguishable from a real value without this bit.
      const bool orientation_claimed =
          (sample.data.flags & holoscan::schema::ImuFlags_ORIENTATION_VALID) != 0;
      const bool fields_ok =
          sample.data.header != nullptr && sample.data.header->frame_id == "imu_link" &&
          orientation_claimed && sample.data.orientation.w() == 1.0F &&
          sample.data.angular_velocity.z() == 1.0F && sample.data.linear_acceleration.z() == -9.81F;
      imu_fields_ok = imu_fields_ok && fields_ok;
      std::cout << "imu #" << imus << ": frame=\""
                << (sample.data.header ? sample.data.header->frame_id : std::string{"<none>"})
                << "\" w=" << sample.data.orientation.w()
                << " az=" << sample.data.linear_acceleration.z() << "\n";
    }
    ready.notify_all();
  }

  static void record_oversize(holoscan::Error error) {
    {
      const std::lock_guard lock(mutex);
      oversize = Oversize::kRejected;
      oversize_error = std::move(error);
    }
    ready.notify_all();
  }

  static void record_oversize_accepted() {
    {
      const std::lock_guard lock(mutex);
      oversize = Oversize::kAccepted;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_until_complete() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] {
      return strings >= kStringLimit && times >= kTimeLimit && imus >= kIMULimit &&
             oversize != Oversize::kNotAttempted;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t strings{};
  static inline std::size_t times{};
  static inline std::size_t imus{};
  static inline std::size_t degraded{};
  static inline std::string last_string;
  static inline bool trace_match{true};
  static inline bool time_delta_ok{true};
  static inline bool imu_fields_ok{true};
  static inline Oversize oversize{Oversize::kNotAttempted};
  static inline holoscan::Error oversize_error{};
};

// Emits kStringLimit owning schema::StringT values with caller metadata, and —
// once, mid-stream — attempts an oversize value that cannot fit the schema's
// 512-byte serialized bound. The runtime serializes generated-schema payloads
// at emit time, so the rejection surfaces synchronously as the emit result;
// the operator records it and carries on. No wrapper struct, no traits, no
// codec registration: SDK schema types are admitted port types out of the box.
class StringSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const holoscan::TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
    }

    // One mid-stream activation is spent demonstrating the bounded-size
    // rejection instead of emitting a normal value.
    if (emitted_ == 2U && !oversize_attempted_) {
      oversize_attempted_ = true;
      holoscan::schema::StringT oversize;
      oversize.data = std::string(2U * kStringBound, 'x');  // 1024 chars > 512-byte bound
      auto rejected = output.emit(holoscan::Sample<holoscan::schema::StringT>{
          std::move(oversize),
          holoscan::SampleMetadata{.capture_timestamp_ns = activation_time.timestamp_ns,
                                   .source_clock_id = activation_time.clock_id}});
      if (!rejected) {
        SchemaObservation::record_oversize(std::move(rejected).error());
      } else {
        SchemaObservation::record_oversize_accepted();
      }
      return {};
    }

    if (emitted_ >= kStringLimit) {
      return {};  // Bounded source: idle after the last emission.
    }
    ++emitted_;
    holoscan::schema::StringT value;
    value.data = "schema greeting #" + std::to_string(emitted_);
    return output.emit(holoscan::Sample<holoscan::schema::StringT>{
        std::move(value),
        holoscan::SampleMetadata{
            .capture_timestamp_ns = activation_time.timestamp_ns,
            .source_clock_id = activation_time.clock_id,
            .trace_id = kTraceBase + emitted_,
            .frame_id = emitted_,
            .flags = emitted_ == kStringLimit ? holoscan::SampleFlags::kDegraded
                                              : holoscan::SampleFlags::kNone,
        }});
  }

  holoscan::Output<holoscan::schema::StringT> output;

 private:
  std::size_t emitted_{};
  bool oversize_attempted_{};
};

// Emits kTimeLimit schema::TimeT values whose two fields encode a fixed
// simulated acquisition latency, so the consumer can verify the payload
// round-trips field-for-field.
class TimeSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kTimeLimit) {
      return {};
    }
    const holoscan::TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
    }
    ++emitted_;
    holoscan::schema::TimeT value;
    value.time_ns = activation_time.timestamp_ns;
    value.acqtime_ns = activation_time.timestamp_ns - kAcquisitionLatencyNs;
    return output.emit(holoscan::Sample<holoscan::schema::TimeT>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = activation_time.timestamp_ns,
                                 .source_clock_id = activation_time.clock_id}});
  }

  holoscan::Output<holoscan::schema::TimeT> output;

 private:
  std::size_t emitted_{};
};

// Emits kIMULimit schema::ImuT values: a nested Header table (shared_ptr
// members in the generated owning type), a fixed identity quaternion, and a
// gravity-only acceleration. Nested tables and structs round-trip through the
// same bounded encode/decode pipeline as the flat types.
class IMUSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_ >= kIMULimit) {
      return {};
    }
    const holoscan::TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
    }
    ++emitted_;
    holoscan::schema::ImuT value;
    value.header = std::make_shared<holoscan::schema::HeaderT>();
    value.header->frame_id = "imu_link";
    value.header->capture_timestamp_ns = activation_time.timestamp_ns;
    value.header->device_sequence = emitted_;
    // Wire order is (x, y, z, w), so identity is the last component. Writing {1, 0, 0, 0} here
    // would compile and produce a zero-norm quaternion that normalizes to NaN.
    value.orientation = holoscan::schema::Quaternionf{0.0F, 0.0F, 0.0F, 1.0F};
    value.angular_velocity = holoscan::schema::Vector3f{0.0F, 0.0F, 0.5F};
    value.linear_acceleration = holoscan::schema::Vector3f{0.0F, 0.0F, -9.81F};
    // An identity quaternion and an unset one are the same bytes to a reader, so a producer that
    // means the former has to say so. A raw accelerometer/gyroscope source leaves this bit clear.
    value.flags = holoscan::schema::ImuFlags_ORIENTATION_VALID;
    return output.emit(holoscan::Sample<holoscan::schema::ImuT>{
        std::move(value),
        holoscan::SampleMetadata{.capture_timestamp_ns = activation_time.timestamp_ns,
                                 .source_clock_id = activation_time.clock_id}});
  }

  holoscan::Output<holoscan::schema::ImuT> output;

 private:
  std::size_t emitted_{};
};

// Borrows the prepared receive slot, copies ordinary fields into one cold-created scratch value,
// performs a capacity-preserving edit, and publishes by reference. No generated owning assignment
// or per-message native allocation is needed in this transform.
class IMUTransform final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.output(output, "output");
  }

  void start() override {
    if (scratch_) {
      return;  // In-place lifecycle restart reuses the cold-prepared capacity.
    }
    auto scratch = holoscan::schema::SchemaScratch<holoscan::schema::ImuT>::create();
    if (!scratch) {
      throw holoscan::RuntimeError(scratch.error());
    }
    scratch_.emplace(std::move(*scratch));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    HOLOSCAN_TRY(scratch_->copy_from(view.data()));
    scratch_->value().angular_velocity =
        holoscan::schema::Vector3f{0.0F, 0.0F, view.data().angular_velocity.z() + 0.5F};
    auto emitted =
        output.emit_ref(scratch_->value(), holoscan::sensor_io::derive_from(view.metadata()));
    HOLOSCAN_TRY(scratch_->reset());
    return emitted;
  }

  holoscan::Input<holoscan::schema::ImuT> input;
  holoscan::Output<holoscan::schema::ImuT> output;

 private:
  std::optional<holoscan::schema::SchemaScratch<holoscan::schema::ImuT>> scratch_;
};

// One sink per lane: each receives the materialized owning value plus the
// caller's sample metadata and records both. The consumer never sees bytes —
// the runtime decodes the bounded FlatBuffer wire form back into the owning
// generated type before compute() runs.
class StringSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SchemaObservation::record_string(*sample);
    return {};
  }

  holoscan::Input<holoscan::schema::StringT> input;
};

class TimeSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SchemaObservation::record_time(*sample);
    return {};
  }

  holoscan::Input<holoscan::schema::TimeT> input;
};

class IMUSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SchemaObservation::record_imu(*sample);
    return {};
  }

  holoscan::Input<holoscan::schema::ImuT> input;
};

}  // namespace

// REQ: SYSTEM-014
int main() {
  SchemaObservation::reset();

  holoscan::Graph graph{"schema-payloads"};
  const auto string_source = graph.op<StringSource>("string-source");
  const auto time_source = graph.op<TimeSource>("time-source");
  const auto imu_source = graph.op<IMUSource>("imu-source");
  const auto imu_transform = graph.op<IMUTransform>("imu-transform");
  const auto string_sink = graph.op<StringSink>("string-sink");
  const auto time_sink = graph.op<TimeSink>("time-sink");
  const auto imu_sink = graph.op<IMUSink>("imu-sink");
  graph.add_flow(
      string_source->output, string_sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(
      time_source->output, time_sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(
      imu_source->output, imu_transform->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(
      imu_transform->output, imu_sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.partition("string-schema-path").add(string_source).add(string_sink);
  graph.partition("time-schema-path").add(time_source).add(time_sink);
  graph.partition("imu-schema-path").add(imu_source).add(imu_transform).add(imu_sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = SchemaObservation::wait_until_complete();
  session.request_stop();
  session.wait();

  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "schema_payloads: FAILED (complete=" << complete
              << " diagnostics=" << session.diagnostics().size() << ")\n";
    return 2;
  }
  if (SchemaObservation::oversize == SchemaObservation::Oversize::kAccepted) {
    // Admitted-but-untested territory: if this branch ever fires, the runtime
    // accepted a serialized value beyond the package bound — a product
    // finding, not something this example works around.
    std::cerr << "schema_payloads: PRODUCT-FINDING oversize StringT emit was accepted in-graph\n";
    return 3;
  }
  if (SchemaObservation::strings != kStringLimit || SchemaObservation::times != kTimeLimit ||
      SchemaObservation::imus != kIMULimit || SchemaObservation::degraded != 1U ||
      SchemaObservation::last_string != "schema greeting #5" || !SchemaObservation::trace_match ||
      !SchemaObservation::time_delta_ok || !SchemaObservation::imu_fields_ok ||
      SchemaObservation::oversize_error.code != holoscan::ErrorCode::kResourceExhausted) {
    std::cerr << "schema_payloads: FAILED round-trip verification\n";
    return 2;
  }

  constexpr holoscan::SchemaIdentity identity =
      holoscan::schema_identity<holoscan::schema::StringT>();
  std::cout << "schema_payloads: StringT identity=" << identity.canonical_name
            << " epoch=" << identity.compatibility_epoch << " bound=" << kStringBound << " bytes\n";
  std::cout << "schema_payloads: oversize emit rejected: "
            << holoscan::runtime_error_message(SchemaObservation::oversize_error) << "\n";
  std::cout << "schema_payloads: strings=" << SchemaObservation::strings
            << " times=" << SchemaObservation::times << " imus=" << SchemaObservation::imus
            << " degraded=" << SchemaObservation::degraded
            << " oversize=" << holoscan::error_code_name(SchemaObservation::oversize_error.code)
            << " termination=stopped\n";
  return 0;
}
