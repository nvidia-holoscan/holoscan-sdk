/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/experimental/params/web_tuner.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

namespace webtuner = holoscan::experimental::params;

constexpr std::chrono::nanoseconds kTickPeriod = 10ms;
// ~1 second of tuned samples at 10ms per tick; the safety cap below keeps the
// demo bounded even if the runtime stalls.
constexpr std::size_t kSampleTarget = 100U;

constexpr std::string_view kSourceSchemaId = "holoscan.examples.WebTunerSourceParams";
constexpr std::string_view kGainSchemaId = "holoscan.examples.WebTunerGainParams";

// Both aggregates mark every field .runtime_mutable().external_tuning(): the
// first opt-in admits live generation swaps, the second exposes the field to
// the external tuning surface. A field missing either marker never appears in
// the tuner's browser catalog.
struct SourceParams {
  std::int32_t step{1};  // counter increment per clock tick
};

HOLOSCAN_PARAM_SCHEMA(SourceParams,
                      holoscan::param_schema("holoscan.examples.WebTunerSourceParams", 1),
                      holoscan::field<&SourceParams::step>("step")
                          .description("Counter increment applied on every clock tick.")
                          .range(1, 100)
                          .runtime_mutable()
                          .external_tuning());

static_assert(holoscan::param_schema_availability<SourceParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be declared in SourceParams's namespace");

struct GainParams {
  double gain{1.0};        // multiplicative gain applied to each sample
  std::int32_t offset{0};  // additive offset applied after the gain
};

HOLOSCAN_PARAM_SCHEMA(GainParams, holoscan::param_schema("holoscan.examples.WebTunerGainParams", 1),
                      holoscan::field<&GainParams::gain>("gain")
                          .description("Multiplicative gain applied to each incoming sample.")
                          .unit("x")
                          .range(0.0, 16.0)
                          .runtime_mutable()
                          .external_tuning(),
                      holoscan::field<&GainParams::offset>("offset")
                          .description("Additive offset applied after the gain stage.")
                          .range(-1000, 1000)
                          .runtime_mutable()
                          .external_tuning());

static_assert(holoscan::param_schema_availability<GainParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be declared in GainParams's namespace");

// Operators run on runtime threads; graph.op<> rejects borrowed captures, so
// the sink records into this static observation and main() blocks on the
// condition variable until enough tuned samples arrived (or the cap expires).
struct TuningObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    count = 0U;
    last_tuned = 0.0;
    last_gain = 1.0;
    last_offset = 0;
  }

  static void record(double tuned, double gain, std::int32_t offset) {
    {
      const std::lock_guard lock(mutex);
      last_tuned = tuned;
      last_gain = gain;
      last_offset = offset;
      ++count;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 8s, [expected] { return count >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t count{};
  static inline double last_tuned{};
  static inline double last_gain{1.0};
  static inline std::int32_t last_offset{};
};

// Clock-driven source whose increment is itself tunable: every activation
// reads the generation pinned for this callback, so a tuner edit to "step"
// takes effect at the next compute boundary, never mid-callback.
class SignalSource final : public holoscan::Operator<SourceParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    next_ += static_cast<std::uint32_t>(context.params<SourceParams>().step);
    return out.emit(next_);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t next_{};
};

// Data-driven sink applying the tunable gain/offset. ctx.params<>() returns
// the pinned const& for this activation — a coherent snapshot, never a torn
// read, even while the tuner publishes a new generation concurrently.
class TunableGain final : public holoscan::Operator<GainParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(const std::uint32_t sample, in.receive_data());
    const GainParams& params = context.params<GainParams>();
    const double tuned =
        static_cast<double>(sample) * params.gain + static_cast<double>(params.offset);
    TuningObservation::record(tuned, params.gain, params.offset);
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

}  // namespace

int main() {
  TuningObservation::reset();

  // Compose: an ordinary pipeline — nothing about it is tuner-specific.
  holoscan::Graph graph{"web-tuner-demo"};
  const auto source = graph.op<SignalSource>("source");
  const auto gain_stage = graph.op<TunableGain>("gain_stage");
  graph.add_flow(source->out, gain_stage->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Compile: the tuner requires a plan that admits bounded external Parameter
  // control — Standard (or Research) profile plus the runtime-updates opt-in.
  holoscan::CompileOptions compile_options;
  compile_options.profile = holoscan::Profile::kStandard;
  compile_options.parameters.runtime_updates =
      holoscan::ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, compile_options);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // Start the advanced facade. WebTuner::start owns the RunSession: it calls
  // run_async internally, so main() never touches a session directly. The
  // defaults below are spelled out for the lesson: an ephemeral loopback IPv4
  // port, plus the announcement policy — which drives only the blocking
  // run_with_tuner() facade's single INFO record. The advanced facade never
  // logs the origin; it exposes endpoint(), and this example prints it below.
  webtuner::WebTunerOptions tuner_options;
  tuner_options.port = 0U;  // zero requests an ephemeral port
  tuner_options.address_family = webtuner::WebTunerAddressFamily::kIpv4;
  tuner_options.endpoint_announcement = webtuner::WebTunerEndpointAnnouncement::kLogOrigin;
  auto tuner = webtuner::WebTuner::start(plan, tuner_options);
  if (!tuner.has_value()) {
    std::cerr << "web_tuner: start failed token=" << tuner.error().token() << '\n';
    return 1;
  }

  // The endpoint is public routing identity, not authorization material: the
  // per-run credential travels only through the bootstrap cookie exchange.
  const webtuner::WebTunerEndpoint endpoint = tuner->endpoint();
  if (endpoint.empty() || !endpoint.origin.starts_with("http://127.0.0.1:") ||
      !endpoint.bootstrap_path.starts_with("/__holoscan/tuner/v1/")) {
    std::cerr << "web_tuner: unexpected endpoint shape origin=" << endpoint.origin
              << " bootstrap=" << endpoint.bootstrap_path << '\n';
    return 2;
  }
  std::cout << "web_tuner: endpoint origin " << endpoint.origin << '\n';
  std::cout << "web_tuner: bootstrap URL   " << endpoint.origin << endpoint.bootstrap_path << '\n';
  std::cout << "web_tuner: open the bootstrap URL in a browser on this machine (the\n"
               "web_tuner: tuner binds loopback only) to receive the scoped session\n"
               "web_tuner: cookie, then edit 'step', 'gain', and 'offset' live; this\n"
               "web_tuner: bounded demo stops on its own after ~1s of samples.\n";

  // Bounded demo window: ~1s of tuned samples, then a cooperative stop. An
  // interactive user can rerun with a larger target (or their own predicate).
  const bool complete = TuningObservation::wait_for(kSampleTarget);
  tuner->request_stop();
  auto result = tuner->wait();
  if (!result.has_value()) {
    std::cerr << "web_tuner: wait failed token=" << result.error().token() << '\n';
    return 2;
  }

  // The immutable terminal result: how many edits Core accepted, how many are
  // proven effective in published state, the closed ingress frontier, and a
  // coherent stopped snapshot in which both schemas must appear.
  std::size_t schemas_found = 0U;
  for (const auto& record : result->stopped_snapshot().operators()) {
    if (record.schema_id == kSourceSchemaId || record.schema_id == kGainSchemaId) {
      ++schemas_found;
    }
  }
  if (result->published_edits() > result->accepted_edits()) {
    std::cerr << "web_tuner: published_edits exceeds accepted_edits\n";
    return 2;
  }

  double last_tuned{};
  double last_gain{};
  std::int32_t last_offset{};
  std::size_t samples{};
  {
    const std::lock_guard lock(TuningObservation::mutex);
    last_tuned = TuningObservation::last_tuned;
    last_gain = TuningObservation::last_gain;
    last_offset = TuningObservation::last_offset;
    samples = TuningObservation::count;
  }
  std::cout << "web_tuner: samples=" << samples << " last_tuned=" << last_tuned
            << " gain=" << last_gain << " offset=" << last_offset << '\n';
  std::cout << "web_tuner: terminalized_through=" << result->terminalized_through() << '\n';
  std::cout << "web_tuner: stopped accepted_edits=" << result->accepted_edits()
            << " published_edits=" << result->published_edits()
            << " catalog_schemas=" << schemas_found << " samples=" << (complete ? "ok" : "timeout")
            << '\n';
  return (complete && schemas_found == 2U) ? 0 : 2;
}
