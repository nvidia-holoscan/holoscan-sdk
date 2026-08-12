/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kSampleTarget = 8U;
constexpr std::array<std::uint32_t, kSampleTarget> kScorePattern{
    12U, 55U, 83U, 47U, 91U, 30U, 66U, 74U};

// The parameter aggregate is a plain struct with default member initializers.
// Defaults live here, in one place, not in setup() calls or YAML files.
struct DetectorParams {
  float threshold{0.5F};        // normalized score a candidate must reach
  std::uint32_t stride{2U};     // evaluate every stride-th sample
  std::string label{"person"};  // class label this detector reports
};

// The schema is declared at namespace scope, right next to the struct, in the
// struct's own namespace. The macro expands to ADL free functions: placing it
// in a different namespace compiles but silently degrades DetectorParams to an
// opaque aggregate (no validation, no field metadata).
HOLOSCAN_PARAM_SCHEMA(
    DetectorParams, holoscan::param_schema("com.nvidia.holoscan.examples.DetectorParams", 1),
    holoscan::field<&DetectorParams::threshold>("threshold")
        .description("Minimum normalized score a candidate must reach.")
        .range(0.0F, 1.0F),
    holoscan::field<&DetectorParams::stride>("stride")
        .description("Input decimation stride; only every stride-th sample is evaluated.")
        .allowed_values({1U, 2U, 4U}),
    holoscan::field<&DetectorParams::label>("label").description(
        "Class label reported for accepted candidates."));

// Static guard against the wrong-namespace footgun above: if ADL cannot find
// the schema, this fails at compile time instead of silently losing validation.
static_assert(holoscan::param_schema_availability<DetectorParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be declared in DetectorParams's namespace");

// Collects results from operator threads; main() prints after the run stops so
// stdout lines stay whole (operator-side printing interleaves).
struct DetectionObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    processed = 0U;
    evaluated = 0U;
    accepted = 0U;
    observed = DetectorParams{};
  }

  static void record(const DetectorParams& params, bool was_evaluated, bool was_accepted) {
    {
      const std::lock_guard lock(mutex);
      if (processed >= kSampleTarget) {
        return;  // cap recording so per-run counts stay deterministic
      }
      if (processed == 0U) {
        observed = params;
      }
      ++processed;
      if (was_evaluated) {
        ++evaluated;
      }
      if (was_accepted) {
        ++accepted;
      }
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return processed >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t processed{};
  static inline std::size_t evaluated{};
  static inline std::size_t accepted{};
  static inline DetectorParams observed{};
};

// Snapshot of one bounded run, copied out of the observation after wait().
struct RunOutcome {
  DetectorParams params{};
  std::size_t processed{};
  std::size_t evaluated{};
  std::size_t accepted{};
};

struct RejectionOutcome {
  holoscan::ParamValidationResult verdict;
  std::string detail;
  holoscan::ErrorCode code;
};

class ScoreSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(scores, "scores"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const std::uint32_t score = kScorePattern[next_++ % kScorePattern.size()];
    return scores.emit(score);
  }

  holoscan::Output<std::uint32_t> scores;

 private:
  std::size_t next_{};
};

// Binding the aggregate as Operator<DetectorParams> is the whole integration:
// no per-parameter registration in setup(), no string lookups in compute().
class Detector final : public holoscan::Operator<DetectorParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(scores, "scores").queue_depth(8U);
  }

  // contract() is omitted: a one-input operator defaults to OnAnyInput{}.

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Read the pinned aggregate exactly once at the top of compute(); every
    // later use in this activation sees the same coherent generation.
    const DetectorParams& params = context.params<DetectorParams>();
    auto sample = scores.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const std::size_t index = next_++;
    const bool evaluated = (index % params.stride) == 0U;
    const bool accepted =
        evaluated && static_cast<float>(sample->data) >= params.threshold * 100.0F;
    DetectionObservation::record(params, evaluated, accepted);
    return {};
  }

  holoscan::Input<std::uint32_t> scores;

 private:
  std::size_t next_{};
};

[[nodiscard]] std::string_view validation_code_name(holoscan::ParamValidationCode code) {
  switch (code) {
    case holoscan::ParamValidationCode::kOk:
      return "kOk";
    case holoscan::ParamValidationCode::kOutOfRange:
      return "kOutOfRange";
    case holoscan::ParamValidationCode::kNotAllowed:
      return "kNotAllowed";
    case holoscan::ParamValidationCode::kCardinalityExceeded:
      return "kCardinalityExceeded";
    case holoscan::ParamValidationCode::kInvalidEncoding:
      return "kInvalidEncoding";
    case holoscan::ParamValidationCode::kValidatorRejected:
      return "kValidatorRejected";
  }
  return "unknown";
}

// Builds, compiles, and runs one bounded source-to-detector pipeline. When
// overrides are present they replace the generation-0 aggregate graph-side.
[[nodiscard]] bool run_detector(const std::optional<DetectorParams>& overrides,
                                RunOutcome& outcome) {
  DetectionObservation::reset();

  holoscan::Graph graph{"typed-parameters"};
  const auto source = graph.op<ScoreSource>("source");
  auto detector = graph.op<Detector>("detector");
  graph.add_flow(source->scores, detector->scores, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  if (overrides.has_value()) {
    detector.set_params(*overrides);  // whole-aggregate replacement, validated now
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return false;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = DetectionObservation::wait_for(kSampleTarget);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "bounded run did not stop cleanly\n";
    return false;
  }

  const std::lock_guard lock(DetectionObservation::mutex);
  outcome.params = DetectionObservation::observed;
  outcome.processed = DetectionObservation::processed;
  outcome.evaluated = DetectionObservation::evaluated;
  outcome.accepted = DetectionObservation::accepted;
  return true;
}

[[nodiscard]] std::optional<RejectionOutcome> validate_rejection(const DetectorParams& candidate) {
  RejectionOutcome outcome{
      .verdict = holoscan::validate_params(candidate),
      .detail = {},
      .code = holoscan::ErrorCode::kSuccess,
  };
  if (outcome.verdict.code != holoscan::ParamValidationCode::kOutOfRange ||
      outcome.verdict.field_name != "threshold") {
    std::cerr << "typed validation did not reject threshold as out of range\n";
    return std::nullopt;
  }

  holoscan::Graph graph{"typed-parameters-rejection"};
  auto detector = graph.op<Detector>("detector");
  try {
    detector.set_params(candidate);
    std::cerr << "out-of-range parameters were unexpectedly admitted\n";
    return std::nullopt;
  } catch (const holoscan::RuntimeError& error) {
    outcome.detail = error.what();
    outcome.code = error.code();
  }

  const bool bounded_detail =
      outcome.code == holoscan::ErrorCode::kInvalidArgument &&
      outcome.detail.find("PARAM_VALIDATION_FAILED category=out_of_range") != std::string::npos &&
      outcome.detail.find("field_name=threshold") != std::string::npos &&
      outcome.detail.find("value=") == std::string::npos;
  if (!bounded_detail) {
    std::cerr << "validation rejection did not carry the expected bounded detail\n";
    return std::nullopt;
  }
  return outcome;
}

}  // namespace

int main() {
  const holoscan::ParamSchemaDescriptor* schema =
      holoscan::param_schema_descriptor<DetectorParams>();
  if (schema == nullptr) {
    std::cerr << "DetectorParams schema unexpectedly opaque\n";
    return 1;
  }
  std::cout << "schema: " << schema->stable_id << " v" << schema->version
            << " availability=kDeclared fields=" << schema->fields.size() << '\n';

  // Run 1: authoring defaults straight from the struct's member initializers.
  RunOutcome defaults;
  if (!run_detector(std::nullopt, defaults)) {
    return 2;
  }

  // Run 2: the same graph shape with a graph-side generation-0 override.
  RunOutcome configured;
  if (!run_detector(DetectorParams{.threshold = 0.8F, .stride = 4U, .label = "vehicle"},
                    configured)) {
    return 3;
  }

  // Run 3: an out-of-range candidate. set_params() validates the complete
  // aggregate against the schema and throws RuntimeError before any run.
  const DetectorParams out_of_range{.threshold = 1.5F};
  const std::optional<RejectionOutcome> rejection = validate_rejection(out_of_range);
  if (!rejection.has_value()) {
    return 4;
  }

  std::cout << "threshold=" << defaults.params.threshold
            << " (default) stride=" << defaults.params.stride << " label=" << defaults.params.label
            << " evaluated=" << defaults.evaluated << " accepted=" << defaults.accepted << '\n';
  std::cout << "threshold=" << configured.params.threshold
            << " (configured) stride=" << configured.params.stride
            << " label=" << configured.params.label << " evaluated=" << configured.evaluated
            << " accepted=" << configured.accepted << '\n';
  std::cout << "rejected: " << rejection->detail
            << " (typed validation: field=" << rejection->verdict.field_name
            << " code=" << validation_code_name(rejection->verdict.code) << ")\n";
  std::cout << "summary: default_threshold=" << defaults.params.threshold
            << " configured_threshold=" << configured.params.threshold
            << " configured_accepted=" << configured.accepted << "/" << configured.evaluated
            << " runtime_error=" << holoscan::error_code_name(rejection->code)
            << " typed_validation=" << validation_code_name(rejection->verdict.code)
            << " field=" << rejection->verdict.field_name << '\n';
  std::cout << "typed_parameters example complete\n";
  return 0;
}
