/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/persistence.hpp>
#include <holoscan/params/rebind.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../../common/temporary_workspace.hpp"

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kSampleTarget = 6U;

// A document that names BOTH the canonical field and its declared alias in one
// object. Duplicate *keys* are a parse error (CONFIG_DUPLICATE_KEY); duplicate
// *targets* -- two distinct keys resolving to the same schema field -- parse
// fine and are rejected later by typed decoding, all-or-nothing.
constexpr std::string_view kDuplicateTargetConfig =
    R"({"detector":{"threshold":9,"legacy_threshold":11}})";

// The detector's typed parameters. Default member initializers are the
// type-default baseline; try_decode() below uses an explicit application
// baseline for fields absent from the file.
struct DetectorParams {
  std::uint32_t threshold{4U};  // detection cutoff, patched by the config file
  float gain{1.0F};             // multiplier applied to every reading, patched by the file
  std::uint32_t window{16U};    // absent from the file: proves baseline preservation
};

// Declared at namespace scope right next to the struct: the macro expands to
// ADL free functions, and a declaration in any other namespace silently
// degrades the type to an opaque aggregate. The "legacy_threshold" alias is
// what the duplicate-target branch below trips over.
HOLOSCAN_PARAM_SCHEMA(DetectorParams,
                      holoscan::param_schema("holoscan.examples.ConfigFileDetectorParams", 1),
                      holoscan::field<&DetectorParams::threshold>("threshold")
                          .range(1U, 64U)
                          .alias("legacy_threshold")
                          .description("Detection cutoff compared against gain-scaled readings."),
                      holoscan::field<&DetectorParams::gain>("gain")
                          .range(0.0F, 10.0F)
                          .description("Linear gain applied to every reading."),
                      holoscan::field<&DetectorParams::window>("window").range(1U, 64U).description(
                          "Moving-window size; deliberately absent from the config assets."));

// Results cross from runtime threads to main() through this static observation
// object (graph.op<> forbids pointer/borrowed captures, and main() must not
// poll). main() prints only after the run stops, so stdout stays deterministic.
struct DetectionObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    processed = 0U;
    detections = 0U;
    effective = DetectorParams{};
  }

  static void record(const DetectorParams& params, bool detected) {
    {
      const std::lock_guard lock(mutex);
      if (processed >= kSampleTarget) {
        return;  // cap recording so per-run counts stay deterministic
      }
      if (processed == 0U) {
        effective = params;  // one copy of the pinned generation-0 values
      }
      ++processed;
      if (detected) {
        ++detections;
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
  static inline std::size_t detections{};
  static inline DetectorParams effective{};
};

// Emits 1, 2, 3, ... on every clock period, then idles; main() stops the run.
// This infrastructure operator intentionally has no parameter schema. The
// archive restore below selects only the detector's exact Operator scope.
class SampleSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (next_ >= kSampleTarget) {
      return {};  // exhausted: keep ticking idly until main() requests stop
    }
    return out.emit(++next_);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t next_{};
};

// The configured operator: Operator<DetectorParams> binds the aggregate, and
// compute() reads the resolved values -- baseline, file patch, and schema
// validation all happened before the plan existed.
class Detector final : public holoscan::Operator<DetectorParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});  // OnEach pins the driving sample itself
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // One pinned read per activation: every use below sees the same generation.
    const DetectorParams& params = context.params<DetectorParams>();
    HOLOSCAN_TRY(const std::uint32_t sample, in.receive_data());
    const bool detected =
        static_cast<float>(sample) * params.gain > static_cast<float>(params.threshold);
    DetectionObservation::record(params, detected);
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

// Locates detector.json/detector.yaml. Explicit --config-dir wins; otherwise
// try the binary's own directory (installed layout: assets sit beside the
// binary) and its parent (build layout: binary in cpp/build/, assets in cpp/).
[[nodiscard]] std::optional<std::filesystem::path> resolve_config_dir(int argc, char** argv) {
  std::optional<std::filesystem::path> requested;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--config-dir" && index + 1 < argc) {
      requested = std::filesystem::path{argv[++index]};
    } else {
      std::cerr << "config_file: unknown argument '" << argument
                << "' (usage: config_file [--config-dir <dir>])\n";
      return std::nullopt;
    }
  }

  std::vector<std::filesystem::path> candidates;
  if (requested.has_value()) {
    candidates.push_back(*requested);
  } else {
    std::filesystem::path argv0_dir =
        argc > 0 ? std::filesystem::path{argv[0]}.parent_path() : std::filesystem::path{};
    if (argv0_dir.empty()) {
      argv0_dir = std::filesystem::current_path();
    }
    candidates.push_back(argv0_dir);         // installed: examples/essential/config_file/cpp/
    candidates.push_back(argv0_dir / "..");  // built: cpp/build/config_file next to cpp/*.json
  }

  for (const auto& candidate : candidates) {
    std::error_code ignored;
    if (std::filesystem::exists(candidate / "detector.json", ignored)) {
      return candidate;
    }
  }

  std::cerr << "config_file: could not find detector.json; tried:\n";
  for (const auto& candidate : candidates) {
    std::cerr << "  " << (candidate / "detector.json").lexically_normal().string() << '\n';
  }
  std::cerr << "config_file: pass --config-dir <dir> pointing at the folder that holds "
               "detector.json and detector.yaml\n";
  return std::nullopt;
}

struct ArchiveEvidence {
  std::size_t saved{};
  std::size_t contributed{};
  std::size_t retained{};
};

[[nodiscard]] std::optional<ArchiveEvidence> archive_and_restore_detector(
    holoscan::RunSession& session) {
  // capture_params() needs the stopped frontier that wait() established.
  auto snapshot = session.capture_params();
  if (!snapshot) {
    std::cerr << "config_file: capture_params failed: "
              << holoscan::runtime_error_message(snapshot.error()) << '\n';
    return std::nullopt;
  }
  auto detector_slice = snapshot->slice("detector");
  if (!detector_slice) {
    std::cerr << "config_file: snapshot slice failed: "
              << holoscan::runtime_error_message(detector_slice.error()) << '\n';
    return std::nullopt;
  }

  // Name the detector-only restore scope. Archive source scope and destination
  // authority are separate: a schema-less infrastructure operator outside this
  // scope is irrelevant, while an unknown archive record still fails closed.
  const auto detector_operator_path = holoscan::OperatorPath::parse("detector");
  if (!detector_operator_path) {
    std::cerr << "config_file: could not construct the detector restore scope\n";
    return std::nullopt;
  }
  const auto restore_scope =
      holoscan::params::ParamRebindTargetScope::operator_(*detector_operator_path);

  const std::filesystem::path workspace = holoscan_examples::create_temporary_workspace();
  if (workspace.empty()) {
    std::cerr << "config_file: failed to create a private temporary workspace\n";
    return std::nullopt;
  }
  const holoscan_examples::TemporaryWorkspaceCleanup workspace_cleanup{workspace};
  const std::filesystem::path archive_path = workspace / "snapshot.json";
  const auto saved = holoscan::params::save(*detector_slice, archive_path);
  if (!saved) {
    std::cerr << "config_file: params::save failed: " << saved.error().detail().token() << '\n';
    return std::nullopt;
  }
  std::cout << "config_file: snapshot archived (" << saved->included_count
            << " persisted fields, strict JSON v1, atomic replace)\n";

  // Restore into a fresh graph because the original graph was already compiled and run.
  holoscan::Graph restore_graph{"config-file-restore"};
  auto restore_detector = restore_graph.op<Detector>("detector");
  const auto restored =
      holoscan::params::load_and_set_params(restore_graph, archive_path, restore_scope);
  if (!restored) {
    std::cerr << "config_file: params::load_and_set_params failed: "
              << restored.error().detail().token() << '\n';
    return std::nullopt;
  }

  // Finish the graph after attachment to show that compilation uses the final graph.
  const auto restore_source = restore_graph.op<SampleSource>("source");
  restore_graph.add_flow(
      restore_source->out, restore_detector->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  restore_graph.set_default_clock(restore_graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan restore_plan = holoscan::compile(restore_graph);
  if (!restore_plan.ok()) {
    std::cerr << "config_file: restored graph failed to compile: " << restore_plan.json() << '\n';
    return std::nullopt;
  }

  // The detector's three archived fields come back as contributions. The
  // schema-less source is outside the explicit target and therefore has no
  // provenance entry; scope never follows the archive implicitly.
  ArchiveEvidence evidence{.saved = saved->included_count};
  for (const auto& field : restored->fields) {
    if (field.disposition == holoscan::ParamFieldRebindDisposition::kContributed) {
      ++evidence.contributed;
    } else if (field.disposition == holoscan::ParamFieldRebindDisposition::kRetainedBaseline) {
      ++evidence.retained;
    }
  }
  const bool restored_sparse = restored->mode == holoscan::ParamRebindMode::kSparseContributions;
  const bool detector_only =
      restored->target_scope_kind == holoscan::ParamRebindTargetScopeKind::kOperator &&
      restored->target_scope_paths.size() == 1U &&
      restored->target_scope_paths.front() == "detector";
  std::cout << "config_file: detector-scoped overrides restored from archive ("
            << restored->fields.size() << " provenance fields: " << evidence.contributed
            << " archive-contributed, " << evidence.retained
            << " baseline-retained; mode=" << (restored_sparse ? "sparse" : "complete") << ")\n";
  if (!restored_sparse || !detector_only || evidence.contributed != 3U || evidence.retained != 0U) {
    std::cerr << "config_file: detector-scoped restore evidence did not match\n";
    return std::nullopt;
  }
  return evidence;
}

}  // namespace

int main(int argc, char** argv) try {
  DetectionObservation::reset();

  const std::optional<std::filesystem::path> config_dir = resolve_config_dir(argc, argv);
  if (!config_dir.has_value()) {
    return 1;
  }

  // ---- Load: explicit format selection, fail-closed parsing --------------
  // The loader never sniffs content or file extensions; you name the syntax.
  const auto json_doc =
      holoscan::config_io::load(*config_dir / "detector.json", holoscan::config_io::Format::kJson);
  const auto yaml_doc =
      holoscan::config_io::load(*config_dir / "detector.yaml", holoscan::config_io::Format::kYaml);
  // Same values, two syntaxes: the admitted YAML subset and strict JSON parse
  // to the same semantic tree, so their semantic digests are identical.
  const bool digests_match = json_doc.digest().semantic.sha256 == yaml_doc.digest().semantic.sha256;
  std::cout << "config_file: loaded detector.json and detector.yaml (semantic digests "
            << (digests_match ? "match" : "differ") << ")\n";
  if (!digests_match) {
    return 1;
  }

  // ---- Decode: configuration becomes one prepared typed value ------------
  const DetectorParams detector_baseline{.threshold = 6U, .gain = 1.0F, .window = 8U};
  auto detector_params =
      holoscan::params::try_decode<DetectorParams>(json_doc, "/detector", detector_baseline);
  if (!detector_params) {
    std::cerr << "config_file: detector.json rejected: " << detector_params.error().token() << '\n';
    return 1;
  }

  // ---- Compose: graph structure plus prepared parameters -----------------
  holoscan::Graph graph{"config-file"};
  const auto source = graph.op<SampleSource>("source");
  auto detector = graph.op<Detector>("detector");
  detector.set_params(std::move(*detector_params));
  graph.add_flow(source->out, detector->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  std::cout << "config_file: decoded typed parameters from /detector\n";

  // ---- Duplicate-target rejection: all-or-nothing decoding ---------------
  // "threshold" and its alias "legacy_threshold" are distinct document keys
  // but one schema field; typed decoding rejects the whole object.
  const auto duplicate_doc =
      holoscan::config_io::parse(kDuplicateTargetConfig, holoscan::config_io::Format::kJson);
  const auto duplicate =
      holoscan::params::try_decode<DetectorParams>(duplicate_doc, "/detector", detector_baseline);
  if (duplicate.has_value()) {
    std::cerr << "config_file: expected the canonical+alias document to be rejected\n";
    return 1;
  }
  std::cout << "config_file: canonical+alias contribution rejected: " << duplicate.error().token()
            << " (no partial parameter input)\n";

  // ---- Compile: validate the final graph and prepared parameters ----------
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // ---- Run: the plan is immutable; nothing re-reads a file ---------------
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = DetectionObservation::wait_for(kSampleTarget);
  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "config_file: timed out waiting for " << kSampleTarget << " samples\n";
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "config_file: unexpected termination state\n";
    return 2;
  }

  DetectorParams effective{};
  std::size_t detections = 0U;
  {
    const std::lock_guard lock(DetectionObservation::mutex);
    effective = DetectionObservation::effective;
    detections = DetectionObservation::detections;
  }
  // Typed decode, proven from inside compute(): file values landed, and the
  // unnamed window kept its explicit decode baseline.
  if (effective.threshold != 9U || effective.gain != 2.5F || effective.window != 8U) {
    std::cerr << "config_file: unexpected effective params: threshold=" << effective.threshold
              << " gain=" << effective.gain << " window=" << effective.window << '\n';
    return 2;
  }
  std::cout << "config_file: effective params: threshold=" << effective.threshold
            << " gain=" << effective.gain << " window=" << effective.window
            << " (window preserved from the decode baseline)\n";

  // ---- Persist: snapshot the stopped run, archive it, restore it ---------
  const auto archive = archive_and_restore_detector(session);
  if (!archive) {
    return 2;
  }

  std::cout << "config_file: digests=match effective=(threshold=" << effective.threshold
            << " gain=" << effective.gain << " window=" << effective.window
            << ") duplicate=" << duplicate.error().token() << " roundtrip=(saved=" << archive->saved
            << " contributed=" << archive->contributed << " retained=" << archive->retained
            << ") detections=" << detections << "/" << kSampleTarget << " termination=stopped\n";
  return 0;
} catch (const holoscan::config::ConfigException& exception) {
  std::cerr << "config_file: " << exception.what() << '\n';
  return 1;
}
