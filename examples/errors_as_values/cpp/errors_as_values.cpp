/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// Part 1 - fallible free functions: expected<T, Error> + the propagation macros
//
// HSDK 5 hot paths return holoscan::expected instead of throwing: an Error is
// a cheap value (code + optional message; code-only errors never allocate),
// so "can fail" becomes part of the signature the compiler checks.
//
// The expression-form extension (spelled HOLOSCAN_TRYX in expected_macros.hpp)
// is reserved: HOLOSCAN_HAS_TRY_EXPR is hard-wired 0 on every supported
// frontend today, so portable code uses the HOLOSCAN_TRY statement form below.
// ---------------------------------------------------------------------------

/// Parse an unsigned scale factor. Fails with kInvalidArgument (value-style).
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> parse_scale(
    std::string_view text) {
  std::uint32_t value{};
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc() || ptr != text.data() + text.size()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "scale is not an unsigned integer"});
  }
  return value;
}

/// Validate a parsed scale. expected<void, Error>: success carries no value.
[[nodiscard]] holoscan::expected<void, holoscan::Error> validate_scale(std::uint32_t scale) {
  if (scale == 0U || scale > 1000U) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "scale must be in [1, 1000]"});
  }
  return {};
}

/// HOLOSCAN_ASSIGN_OR_RETURN (value) + HOLOSCAN_RETURN_IF_ERROR (void): each
/// evaluates its operand exactly once and, on failure, moves the original
/// Error into this function's unexpected return - no rethrow, no copy.
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> scaled_offset(
    std::string_view text) {
  HOLOSCAN_ASSIGN_OR_RETURN(const std::uint32_t scale, parse_scale(text));
  HOLOSCAN_RETURN_IF_ERROR(validate_scale(scale));
  return scale * 10U;
}

/// HOLOSCAN_TRY is the one-name spelling: one argument behaves like
/// HOLOSCAN_RETURN_IF_ERROR, two arguments like HOLOSCAN_ASSIGN_OR_RETURN.
[[nodiscard]] holoscan::expected<std::string, holoscan::Error> describe_scale(
    std::string_view text) {
  HOLOSCAN_TRY(const std::uint32_t scale, parse_scale(text));  // two-argument arity
  HOLOSCAN_TRY(validate_scale(scale));                         // one-argument arity
  return std::string("scale=") + std::to_string(scale);
}

/// forward_error is what the macros expand to, written by hand: it moves the
/// same Error into an expected with a *different* value type (uint32_t here
/// becomes std::string), so error types never need converting at call sites.
[[nodiscard]] holoscan::expected<std::string, holoscan::Error> label_for(std::string_view text) {
  auto parsed = parse_scale(text);  // expected<std::uint32_t, Error>
  if (!parsed) {
    return holoscan::forward_error(std::move(parsed));  // -> expected<std::string, Error>
  }
  return std::string("scale-") + std::to_string(*parsed);
}

// ---------------------------------------------------------------------------
// Part 2 - the RuntimeError <-> Error bridge at an exception boundary
//
// Legacy/authoring code may still throw holoscan::RuntimeError (it carries the
// same ErrorCode). make_error() converts a caught RuntimeError back into an
// Error value - and strips the generated "<code name>: " prefix so the
// round-trip Error -> RuntimeError -> Error never duplicates it.
// ---------------------------------------------------------------------------

/// A 4.x-style throwing shim, e.g. a vendor driver wrapper.
void legacy_open_device(int device_id) {
  if (device_id != 0) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kNotFound,
                                 "device " + std::to_string(device_id) + " is not attached"};
  }
}

/// The exception boundary: everything below returns values, everything above
/// may throw. catch once, convert once, and the caller branches on codes.
[[nodiscard]] holoscan::expected<void, holoscan::Error> open_device(int device_id) {
  try {
    legacy_open_device(device_id);
    return {};
  } catch (const holoscan::RuntimeError& error) {
    return holoscan::make_unexpected(holoscan::make_error(error));
  }
}

// ---------------------------------------------------------------------------
// Part 3 - a tiny graph that branches on error codes instead of giving up
//
// kNotReady and kBackpressured are *transient* categories: the right response
// is usually "try again later", not "fail the run". The source below treats
// them as retryable and propagates everything else as fatal.
// ---------------------------------------------------------------------------

/// Simulated flaky device: not ready twice, backpressured once, then values.
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> poll_device(
    std::uint32_t attempt) {
  if (attempt < 2U) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotReady});  // code-only: allocation-free
  }
  if (attempt == 2U) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kBackpressured, "device queue full"});
  }
  return (attempt - 2U) * 10U;  // 10, 20, 30, ...
}

// Operators run on runtime threads; main() reads results only through this
// static observation struct (mutex + condition_variable, slow_fast pattern).
struct ErrorObservation {
  static void reset() noexcept {
    not_ready.store(0U, std::memory_order_relaxed);
    backpressured.store(0U, std::memory_order_relaxed);
    delivered.store(0U, std::memory_order_relaxed);
    last_value.store(0U, std::memory_order_relaxed);
  }

  static void record_transient(holoscan::ErrorCode code) {
    if (code == holoscan::ErrorCode::kNotReady) {
      not_ready.fetch_add(1U, std::memory_order_relaxed);
    } else {
      backpressured.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  static void record_delivery(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      last_value.store(value, std::memory_order_relaxed);
      delivered.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return delivered.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> not_ready{};
  static inline std::atomic<std::size_t> backpressured{};
  static inline std::atomic<std::size_t> delivered{};
  static inline std::atomic<std::uint32_t> last_value{};
};

class FlakySource final : public holoscan::Operator<> {
 public:
  explicit FlakySource(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 2ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (emitted_ >= limit_) {
      return {};  // exhausted: tick idly; main() observes the count and stops the run
    }
    auto sample = poll_device(attempt_++);
    if (!sample) {
      const holoscan::Error& error = sample.error();
      // Branch on the code: transient categories are retried on the next
      // clock tick by succeeding *this* activation without emitting.
      if (error.code == holoscan::ErrorCode::kNotReady ||
          error.code == holoscan::ErrorCode::kBackpressured) {
        ErrorObservation::record_transient(error.code);
        return {};
      }
      // Anything else is fatal: returning the error fails the run and the
      // runtime records it in session.diagnostics().
      return holoscan::forward_error(std::move(sample));
    }
    ++emitted_;
    return out.emit(*sample);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t limit_;
  std::uint32_t attempt_{};
  std::uint32_t emitted_{};
};

class DeliverySink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    ErrorObservation::record_delivery(value);
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

constexpr std::uint32_t kDeliveryTarget = 3U;

}  // namespace

int main() {
  ErrorObservation::reset();

  // -- Run the retry graph first; every stdout line below is post-stop. ------
  holoscan::Graph graph{"errors-as-values"};
  const auto source = graph.op<FlakySource>("source", kDeliveryTarget);
  const auto sink = graph.op<DeliverySink>("sink");
  graph.add_flow(source->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = ErrorObservation::wait_for(kDeliveryTarget);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped) {
    std::cerr << "errors_as_values: run did not deliver " << kDeliveryTarget << " samples\n";
    return 2;
  }

  // -- Part 1: macros over fallible free functions ---------------------------
  const auto offset = scaled_offset("4");
  const auto described = describe_scale("4");
  if (!offset || *offset != 40U || !described || *described != "scale=4") {
    std::cerr << "errors_as_values: macro success paths returned wrong values\n";
    return 2;
  }
  std::cout << "[macros] scaled_offset(\"4\") -> " << *offset
            << " (HOLOSCAN_ASSIGN_OR_RETURN + HOLOSCAN_RETURN_IF_ERROR)\n";
  std::cout << "[macros] describe_scale(\"4\") -> " << *described
            << " (HOLOSCAN_TRY, both arities)\n";

  const auto rejected = scaled_offset("banana");
  if (rejected || rejected.error().code != holoscan::ErrorCode::kInvalidArgument) {
    std::cerr << "errors_as_values: expected kInvalidArgument from scaled_offset\n";
    return 2;
  }
  // runtime_error_message renders "<code name>: <message>" for logs.
  std::cout << "[macros] scaled_offset(\"banana\") -> "
            << holoscan::runtime_error_message(rejected.error()) << '\n';

  const auto forwarded = label_for("banana");
  if (forwarded || forwarded.error().code != holoscan::ErrorCode::kInvalidArgument) {
    std::cerr << "errors_as_values: expected forwarded kInvalidArgument from label_for\n";
    return 2;
  }
  std::cout << "[forward_error] label_for(\"banana\") -> "
            << holoscan::error_code_name(forwarded.error().code)
            << " (same Error, new value type)\n";

  // -- Part 2: the exception boundary ----------------------------------------
  const auto device = open_device(7);
  if (device || device.error().code != holoscan::ErrorCode::kNotFound) {
    std::cerr << "errors_as_values: expected kNotFound from open_device\n";
    return 2;
  }
  std::cout << "[bridge] open_device(7) caught RuntimeError -> Error: "
            << holoscan::runtime_error_message(device.error()) << '\n';

  const holoscan::Error original{holoscan::ErrorCode::kBackpressured, "queue full"};
  const holoscan::RuntimeError bridged{original};                   // Error -> exception
  const holoscan::Error recovered = holoscan::make_error(bridged);  // exception -> Error
  const bool round_trip = recovered.code == original.code && recovered.message == original.message;
  std::cout << "[bridge] Error -> RuntimeError -> Error round-trip preserved: "
            << (round_trip ? "yes" : "no") << '\n';
  if (!round_trip) {
    return 2;
  }

  // -- Part 3: what the retry graph observed ---------------------------------
  const std::size_t not_ready = ErrorObservation::not_ready.load(std::memory_order_relaxed);
  const std::size_t backpressured = ErrorObservation::backpressured.load(std::memory_order_relaxed);
  std::cout << "[retry] transient codes retried: kNotReady=" << not_ready
            << " kBackpressured=" << backpressured << '\n';

  // -- Finale: the three failure styles, side by side ------------------------
  // Style 1 - authoring mistakes throw. Composition APIs like add_flow are
  // cold-path and throw RuntimeError; a second edge into an already-connected
  // input breaks the single-writer rule and is rejected on the spot.
  {
    holoscan::Graph scratch{"finale-authoring"};
    const auto src = scratch.op<FlakySource>("src", kDeliveryTarget);
    const auto dst = scratch.op<DeliverySink>("dst");
    (void)scratch.add_flow(src->out, dst->in);
    try {
      (void)scratch.add_flow(src->out, dst->in);  // deliberate second edge into dst->in
      std::cerr << "errors_as_values: duplicate add_flow unexpectedly accepted\n";
      return 2;
    } catch (const holoscan::RuntimeError& error) {
      std::cout << "[failure-styles] authoring throw: "
                << holoscan::runtime_error_message(holoscan::make_error(error)) << '\n';
    }
  }

  // Style 2 - structural mistakes reject the compile with stable tokens.
  // (A third style lives one step earlier still: contract violations like a
  // selector on an OnEach driver port are compile-time C++ rejections.)
  {
    holoscan::Graph clockless{"finale-clockless"};
    const auto src = clockless.op<FlakySource>("src", kDeliveryTarget);
    const auto dst = clockless.op<DeliverySink>("dst");
    (void)clockless.add_flow(src->out, dst->in);
    // Deliberate: OnClock needs a default clock, and none is declared.
    const holoscan::ExecutionPlan broken = holoscan::compile(clockless);
    if (broken.ok()) {
      std::cerr << "errors_as_values: clockless graph unexpectedly compiled\n";
      return 2;
    }
    std::cout << "[failure-styles] compile rejection tokens:";
    for (const holoscan::Diagnostic& diagnostic : broken.diagnostics()) {
      if (diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
        std::cout << ' ' << diagnostic.token;
      }
    }
    std::cout << '\n';
  }

  // Style 3 - runtime failures land in session.diagnostics() (empty here:
  // transient poll errors were retried, so the run stayed clean).
  const std::size_t run_diagnostics = session.diagnostics().size();
  std::cout << "[failure-styles] run diagnostics after clean stop: " << run_diagnostics << '\n';
  if (run_diagnostics != 0U) {
    return 2;
  }

  std::cout << "errors_as_values: transient_retries=" << (not_ready + backpressured)
            << " delivered=" << ErrorObservation::delivered.load(std::memory_order_relaxed)
            << " (last=" << ErrorObservation::last_value.load(std::memory_order_relaxed)
            << ") run_diagnostics=" << run_diagnostics << " termination=stopped\n";
  return 0;
}
