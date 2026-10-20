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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

constexpr std::size_t kSampleTarget = 6U;
constexpr std::array<std::uint32_t, kSampleTarget> kReadingPattern{4U, 8U, 12U, 16U, 20U, 24U};

// A nested aggregate with its own schema. Nesting keeps FilterParams a stable,
// versioned unit inside any parent schema instead of dissolving it into loose fields.
struct FilterParams {
  float cutoff_hz{30.0F};  // low-pass cutoff applied before calibration
};

HOLOSCAN_PARAM_SCHEMA(FilterParams, holoscan::param_schema("holoscan.examples.FilterParams", 1),
                      holoscan::field<&FilterParams::cutoff_hz>("cutoff_hz")
                          .range(0.5F, 500.0F)
                          .unit("Hz")
                          .description("Low-pass cutoff applied before calibration."));

// Custom validator: a noexcept bool(const Field&) callable plus a stable string
// identity. The ID travels with the schema even where the callable cannot
// (other processes, tooling), so external consumers can see *that* a named
// validation applies without being able to run it.
[[nodiscard]] bool even_value(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

// The full-taxonomy aggregate: one field per policy family. Defaults are plain
// member initializers -- valid by construction against the schema below.
struct CalibrationParams {
  std::uint32_t window{4U};         // range + custom validator (must be even)
  std::uint32_t stride{2U};         // closed allowed-value set
  std::string label{"wide-angle"};  // UTF-8 encoding + byte bound + alias
  float gain{1.0F};                 // runtime-mutable teaser + range + tuning markers
  // Sensitive values are redacted from snapshots and external projections.
  std::string auth_token{"demo-token"};
  std::uint32_t scratch_mb{64U};  // internal: not part of the external surface
  FilterParams filter{};          // nested schema
};

// Declared at namespace scope immediately beside the struct -- the macro expands
// to ADL free functions, and ADL only finds them in the struct's own namespace.
// A misplaced declaration compiles but silently degrades the type to an opaque
// aggregate (no validation, no metadata, no portable identity).
HOLOSCAN_PARAM_SCHEMA(
    CalibrationParams, holoscan::param_schema("holoscan.examples.CalibrationParams", 1),
    holoscan::field<&CalibrationParams::window>("window")
        .range(2U, 32U)
        .validator<even_value>("holoscan.examples.validator.even.v1")
        .unit("samples")
        .description("Moving-average window; must be even."),
    holoscan::field<&CalibrationParams::stride>("stride")
        .allowed_values({1U, 2U, 4U})
        .description("Input decimation stride."),
    holoscan::field<&CalibrationParams::label>("label")
        .encoding(holoscan::ParamStringEncoding::kUtf8)
        .max_utf8_bytes(16)
        .alias("legacy_label")
        .description("Calibration profile name; 'legacy_label' accepted on input."),
    holoscan::field<&CalibrationParams::gain>("gain")
        .runtime_mutable()
        .range(0.0F, 4.0F)
        .persisted()
        .external_tuning()
        .unit("x")
        .description("Linear gain applied to every reading."),
    holoscan::field<&CalibrationParams::auth_token>("auth_token")
        .sensitive()
        .description("Upload credential; redacted from snapshots and external projections."),
    holoscan::field<&CalibrationParams::scratch_mb>("scratch_mb")
        .internal()
        .range(16U, 512U)
        .unit("MiB"),
    holoscan::field<&CalibrationParams::filter>("filter").nested<FilterParams>());

// Static guard against the wrong-namespace footgun: if ADL cannot find the
// schema, this fails at compile time instead of silently losing validation.
static_assert(holoscan::param_schema_availability<CalibrationParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be declared in CalibrationParams's namespace");
static_assert(holoscan::param_schema_availability<FilterParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be declared in FilterParams's namespace");

// Counter-case: a plain aggregate with no schema declaration reports
// kOpaqueLocal -- it still works as Operator<T> params, but with no validation
// and no portable identity. The degradation is silent at the use site, which
// is exactly why the availability query exists.
struct LocalScratch {
  std::uint32_t attempts{};
};

static_assert(holoscan::param_schema_availability<LocalScratch>() ==
                  holoscan::ParamSchemaAvailability::kOpaqueLocal,
              "a schema-less aggregate must report kOpaqueLocal");

// Collects results from the operator thread; main() prints only after the run
// stops so stdout stays deterministic for CTest.
struct CalibrationObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    processed = 0U;
    last_calibrated = 0U;
    observed = CalibrationParams{};
  }

  static void record(const CalibrationParams& params, std::uint32_t calibrated) {
    {
      const std::lock_guard lock(mutex);
      if (processed >= kSampleTarget) {
        return;  // cap recording so per-run counts stay deterministic
      }
      if (processed == 0U) {
        observed = params;
      }
      ++processed;
      last_calibrated = calibrated;
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
  static inline std::uint32_t last_calibrated{};
  static inline CalibrationParams observed{};
};

// Snapshot of one bounded run, copied out of the observation after wait().
struct RunOutcome {
  CalibrationParams params{};
  std::size_t processed{};
  std::uint32_t last_calibrated{};
};

class ReadingSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(readings, "readings"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const std::uint32_t reading = kReadingPattern[next_++ % kReadingPattern.size()];
    return readings.emit(reading);
  }

  holoscan::Output<std::uint32_t> readings;

 private:
  std::size_t next_{};
};

// Binding the aggregate as Operator<CalibrationParams> is the whole
// integration: no per-parameter registration, no string lookups in compute().
class Calibrator final : public holoscan::Operator<CalibrationParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(readings, "readings").queue_depth(8U);
  }

  // contract() is omitted: a one-input operator defaults to OnAnyInput{}.

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // One pinned read per activation; every use below sees the same generation.
    const CalibrationParams& params = context.params<CalibrationParams>();
    auto sample = readings.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const auto calibrated =
        static_cast<std::uint32_t>(static_cast<float>(sample->data) * params.gain);
    CalibrationObservation::record(params, calibrated);
    return {};
  }

  holoscan::Input<std::uint32_t> readings;
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

// Builds, compiles, and runs one bounded source-to-calibrator pipeline with a
// graph-side generation-0 override; the override is validated by set_params.
[[nodiscard]] bool run_calibrator(const CalibrationParams& overrides, RunOutcome& outcome) {
  CalibrationObservation::reset();

  holoscan::Graph graph{"parameter-schemas"};
  const auto source = graph.op<ReadingSource>("source");
  auto calibrator = graph.op<Calibrator>("calibrator");
  graph.add_flow(
      source->readings, calibrator->readings, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  calibrator.set_params(overrides);  // whole-aggregate replacement, validated now

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return false;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = CalibrationObservation::wait_for(kSampleTarget);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "bounded run did not stop cleanly\n";
    return false;
  }

  const std::lock_guard lock(CalibrationObservation::mutex);
  outcome.params = CalibrationObservation::observed;
  outcome.processed = CalibrationObservation::processed;
  outcome.last_calibrated = CalibrationObservation::last_calibrated;
  return true;
}

// One rejection attempt per bind-time-enforced policy; each candidate mutates
// exactly one field away from the (valid) authoring defaults.
struct RejectionAttempt {
  std::string_view policy;
  std::string_view candidate_text;
  CalibrationParams candidate;
  holoscan::ParamValidationCode expected_code;
  std::string_view expected_field_name;
  std::string_view expected_validator_id;
};

[[nodiscard]] std::vector<RejectionAttempt> make_rejection_attempts() {
  std::vector<RejectionAttempt> attempts;
  CalibrationParams candidate;

  candidate.window = 40U;  // even, but above range(2, 32)
  attempts.push_back(
      {"range", "window=40", candidate, holoscan::ParamValidationCode::kOutOfRange, "window", ""});

  candidate = CalibrationParams{};
  candidate.stride = 3U;  // not in allowed_values({1, 2, 4})
  attempts.push_back({"allowed_values",
                      "stride=3",
                      candidate,
                      holoscan::ParamValidationCode::kNotAllowed,
                      "stride",
                      ""});

  candidate = CalibrationParams{};
  candidate.window = 5U;  // in range, but odd: rejected by the named validator
  attempts.push_back({"validator",
                      "window=5",
                      candidate,
                      holoscan::ParamValidationCode::kValidatorRejected,
                      "window",
                      "holoscan.examples.validator.even.v1"});

  candidate = CalibrationParams{};
  candidate.label = std::string{"\xC0\xAF", 2};  // overlong sequence: invalid UTF-8
  attempts.push_back({"encoding",
                      "label=<invalid-utf8-bytes>",
                      candidate,
                      holoscan::ParamValidationCode::kInvalidEncoding,
                      "label",
                      ""});

  candidate = CalibrationParams{};
  candidate.label = "wide-angle-night-calibration";  // 28 bytes > max_utf8_bytes(16)
  attempts.push_back({"max_utf8_bytes",
                      "label=wide-angle-night-calibration",
                      candidate,
                      holoscan::ParamValidationCode::kCardinalityExceeded,
                      "label",
                      ""});

  candidate = CalibrationParams{};
  candidate.filter.cutoff_hz = 900.0F;  // violates the *nested* schema's range
  attempts.push_back({"nested",
                      "filter.cutoff_hz=900",
                      candidate,
                      holoscan::ParamValidationCode::kOutOfRange,
                      "cutoff_hz",
                      ""});

  return attempts;
}

}  // namespace

int main() {
  const holoscan::ParamSchemaDescriptor* schema =
      holoscan::param_schema_descriptor<CalibrationParams>();
  const holoscan::ParamSchemaDescriptor* nested_schema =
      holoscan::param_schema_descriptor<FilterParams>();
  if (schema == nullptr || nested_schema == nullptr) {
    std::cerr << "schema unexpectedly opaque\n";
    return 1;
  }

  // Accepted attempt: a fully valid override applied graph-side, then observed
  // from inside compute() via ctx.params<CalibrationParams>().
  CalibrationParams accepted;
  accepted.window = 8U;
  accepted.stride = 4U;
  accepted.label = "night-mode";
  accepted.gain = 2.5F;
  accepted.auth_token = "rotate-me";
  accepted.scratch_mb = 128U;
  accepted.filter.cutoff_hz = 120.0F;

  RunOutcome outcome;
  if (!run_calibrator(accepted, outcome)) {
    return 2;
  }

  // Rejected attempts: one per enforced policy, checked with validate_params,
  // which reports the failing field, code, and (for validators) the stable ID.
  const std::vector<RejectionAttempt> attempts = make_rejection_attempts();
  std::vector<holoscan::ParamValidationResult> verdicts;
  verdicts.reserve(attempts.size());
  for (const RejectionAttempt& attempt : attempts) {
    const holoscan::ParamValidationResult verdict = holoscan::validate_params(attempt.candidate);
    if (verdict.code != attempt.expected_code ||
        verdict.field_name != attempt.expected_field_name ||
        verdict.validator_id != attempt.expected_validator_id) {
      std::cerr << "candidate '" << attempt.candidate_text
                << "' returned the wrong validation identity: code="
                << validation_code_name(verdict.code) << " field=" << verdict.field_name
                << " validator=" << verdict.validator_id << '\n';
      return 3;
    }
    verdicts.push_back(verdict);
  }

  // Bind-time branch: set_params() validates the aggregate and throws a
  // RuntimeError with bounded category, field, and validator identity but no
  // value. validate_params() remains the typed machine-readable result.
  CalibrationParams odd_window;
  odd_window.window = 5U;
  std::string bind_what;
  holoscan::ErrorCode bind_code{holoscan::ErrorCode::kSuccess};
  {
    holoscan::Graph graph{"parameter-schemas-rejection"};
    auto calibrator = graph.op<Calibrator>("calibrator");
    try {
      calibrator.set_params(odd_window);
      std::cerr << "invalid parameters were unexpectedly admitted\n";
      return 4;
    } catch (const holoscan::RuntimeError& error) {
      bind_what = error.what();
      bind_code = error.code();
    }
  }
  if (bind_code != holoscan::ErrorCode::kInvalidArgument) {
    std::cerr << "unexpected bind-time error code: " << holoscan::error_code_name(bind_code)
              << '\n';
    return 5;
  }
  const bool bounded_bind_detail =
      bind_what.find("PARAM_VALIDATION_FAILED category=validator_rejected") != std::string::npos &&
      bind_what.find("field_name=window") != std::string::npos &&
      bind_what.find("validator_id=holoscan.examples.validator.even.v1") != std::string::npos &&
      bind_what.find("value=") == std::string::npos;
  if (!bounded_bind_detail) {
    std::cerr << "bind-time validation detail was incomplete or exposed a value\n";
    return 5;
  }
  const holoscan::ParamValidationResult typed_validation = holoscan::validate_params(odd_window);
  if (typed_validation.code != holoscan::ParamValidationCode::kValidatorRejected ||
      typed_validation.field_name != "window" ||
      typed_validation.validator_id != "holoscan.examples.validator.even.v1") {
    std::cerr << "typed validation returned the wrong validation identity\n";
    return 6;
  }

  // All output below is printed after every session has fully stopped.
  std::cout << "schema: " << schema->stable_id << " v" << schema->version
            << " availability=kDeclared fields=" << schema->fields.size() << '\n';
  std::cout << "nested: " << nested_schema->stable_id << " v" << nested_schema->version
            << " folded into field 'filter'\n";
  std::cout << "accepted: window=" << outcome.params.window << " stride=" << outcome.params.stride
            << " label=" << outcome.params.label << " gain=" << outcome.params.gain
            << " filter.cutoff_hz=" << outcome.params.filter.cutoff_hz
            << " processed=" << outcome.processed << " calibrated_last=" << outcome.last_calibrated
            << '\n';
  for (std::size_t index = 0; index < attempts.size(); ++index) {
    const holoscan::ParamValidationResult& verdict = verdicts[index];
    std::cout << "rejected[" << attempts[index].policy << "]: " << attempts[index].candidate_text
              << " -> " << validation_code_name(verdict.code) << " field=" << verdict.field_name;
    if (!verdict.validator_id.empty()) {
      std::cout << " validator=" << verdict.validator_id;
    }
    std::cout << '\n';
  }
  std::cout << "bind-time: set_params(window=5) threw RuntimeError what=" << bind_what
            << " (bounded detail; value omitted)\n";
  std::cout << "typed: validate_params(window=5) -> " << validation_code_name(typed_validation.code)
            << " field=" << typed_validation.field_name
            << " validator=" << typed_validation.validator_id << '\n';
  std::cout << "parameter_schemas: accepted=1 rejected=" << attempts.size()
            << " typed_validation=" << validation_code_name(typed_validation.code)
            << " termination=stopped\n";
  std::cout << "parameter_schemas example complete\n";
  return 0;
}
