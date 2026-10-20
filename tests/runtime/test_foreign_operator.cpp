/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/holoscan.hpp>

#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

struct ForeignOperatorObservations {
  std::atomic<std::size_t> setup_count{};
  std::atomic<std::size_t> contract_count{};
  std::atomic<std::size_t> compute_count{};
  std::atomic<std::size_t> start_count{};
  std::atomic<std::size_t> stop_count{};
  std::atomic<std::size_t> configure_count{};
  std::atomic<std::size_t> release_count{};
  std::atomic<std::size_t> destroy_count{};
  std::atomic_bool replay_mismatch{};
  std::atomic_bool fail_compute{};
  std::atomic_bool fail_lifecycle{};
  std::atomic_bool duplicate_output{};
  std::string compute_message{"foreign callback detail"};
};

// Implements the language-neutral callback table used by the ForeignOperatorCore lifecycle,
// diagnostics, and replay tests in this file.
struct ForeignOperatorProbe {
  std::shared_ptr<ForeignOperatorObservations> observations;
  const authoring::RegisteredPayloadType* payload{};

  static void destroy(void* context) noexcept {
    auto* probe = static_cast<ForeignOperatorProbe*>(context);
    probe->observations->destroy_count.fetch_add(1U, std::memory_order_relaxed);
    delete probe;
  }

  static ForeignOperatorResult setup(void* context, ForeignOperatorSpec* spec) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    probe.observations->setup_count.fetch_add(1U, std::memory_order_relaxed);
    const std::string_view name =
        probe.observations->replay_mismatch.load(std::memory_order_relaxed) ? "changed" : "output";
    auto declared = spec->output(*probe.payload, name);
    if (!declared) {
      return ForeignOperatorResult{.code = declared.error().code};
    }
    if (probe.observations->duplicate_output.load(std::memory_order_relaxed)) {
      auto duplicate = spec->output(*probe.payload, name);
      if (!duplicate) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        try {
#endif
          probe.observations->compute_message = duplicate.error().message;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        } catch (...) {
          return ForeignOperatorResult{.code = duplicate.error().code};
        }
#endif
        return ForeignOperatorResult::failure(duplicate.error().code,
                                              probe.observations->compute_message.data(),
                                              probe.observations->compute_message.size());
      }
    }
    return ForeignOperatorResult::success();
  }

  static ForeignOperatorResult contract(void* context, Contract* contract) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    probe.observations->contract_count.fetch_add(1U, std::memory_order_relaxed);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      contract->trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return ForeignOperatorResult{.code = ErrorCode::kFailure};
    }
#endif
    return ForeignOperatorResult::success();
  }

  static ForeignOperatorResult compute(void* context, ExecutionContext*) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    probe.observations->compute_count.fetch_add(1U, std::memory_order_relaxed);
    if (probe.observations->fail_compute.load(std::memory_order_relaxed)) {
      return ForeignOperatorResult::failure(ErrorCode::kFailure,
                                            probe.observations->compute_message.data(),
                                            probe.observations->compute_message.size());
    }
    return ForeignOperatorResult::success();
  }

  static ForeignOperatorResult start(void* context) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    probe.observations->start_count.fetch_add(1U, std::memory_order_relaxed);
    return ForeignOperatorResult::success();
  }

  static ForeignOperatorResult stop(void* context) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    probe.observations->stop_count.fetch_add(1U, std::memory_order_relaxed);
    return ForeignOperatorResult::success();
  }

  static bool has_lifecycle([[maybe_unused]] void* context, LifecycleStage stage) noexcept {
    return stage == LifecycleStage::kConfigure || stage == LifecycleStage::kRelease;
  }

  static bool lifecycle_required([[maybe_unused]] void* context, LifecycleStage stage) noexcept {
    return stage == LifecycleStage::kConfigure;
  }

  static ForeignOperatorResult lifecycle(void* context, LifecycleStage stage,
                                         [[maybe_unused]] LifecycleContext* lifecycle_context,
                                         LifecycleStatus* status) noexcept {
    auto& probe = *static_cast<ForeignOperatorProbe*>(context);
    if (status == nullptr) {
      return ForeignOperatorResult{.code = ErrorCode::kInvalidArgument};
    }
    if (stage == LifecycleStage::kConfigure) {
      probe.observations->configure_count.fetch_add(1U, std::memory_order_relaxed);
      if (probe.observations->fail_lifecycle.load(std::memory_order_relaxed)) {
        return ForeignOperatorResult::failure(ErrorCode::kFailure,
                                              probe.observations->compute_message.data(),
                                              probe.observations->compute_message.size());
      }
      *status = LifecycleStatus::kOk;
      return ForeignOperatorResult::success();
    }
    if (stage == LifecycleStage::kRelease) {
      probe.observations->release_count.fetch_add(1U, std::memory_order_relaxed);
      *status = LifecycleStatus::kSkipped;
      return ForeignOperatorResult::success();
    }
    return ForeignOperatorResult{.code = ErrorCode::kInvalidArgument};
  }
};

// Provides the native structural twin used by
// ForeignOperatorTest.PublishesHostFactAndMatchesNativePlanShape.
class NativeOperatorProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

// Registers exact typed trampolines for the foreign lifecycle dispatcher used by
// ForeignOperatorTest.RunsAdvancedLifecycleCallbacksThroughTheForeignTable.
class LifecycleForeignOperatorProbe final : public ForeignOperatorCore {
 public:
  explicit LifecycleForeignOperatorProbe(ForeignOperatorCallbacks callbacks)
      : ForeignOperatorCore(std::move(callbacks)) {}

  void setup(OperatorSpec& spec) override {
    ForeignOperatorCore::setup(spec);
    spec.lifecycle().stage(LifecycleStage::kConfigure, &LifecycleForeignOperatorProbe::configure);
    spec.lifecycle().optional(LifecycleStage::kRelease, &LifecycleForeignOperatorProbe::release);
  }

 private:
  LifecycleStatus configure(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kConfigure, context);
  }

  LifecycleStatus release(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kRelease, context);
  }
};

[[nodiscard]] ForeignOperatorCallbacks make_callbacks(
    const std::shared_ptr<ForeignOperatorObservations>& observations) {
  auto payload = authoring::authoring_registry().payload_type("holoscan.uint32");
  if (!payload) {
    throw RuntimeError{payload.error()};
  }
  // ForeignOperatorCallbacks adopts context and invokes destroy when its last owner is released.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return ForeignOperatorCallbacks{ForeignOperatorCallbackTable{
      .context = new ForeignOperatorProbe{observations, &payload->get()},
      .destroy = &ForeignOperatorProbe::destroy,
      .setup_cb = &ForeignOperatorProbe::setup,
      .contract_cb = &ForeignOperatorProbe::contract,
      .compute_cb = &ForeignOperatorProbe::compute,
      .start_cb = &ForeignOperatorProbe::start,
      .stop_cb = &ForeignOperatorProbe::stop,
  }};
}

[[nodiscard]] ForeignOperatorCallbacks make_lifecycle_callbacks(
    const std::shared_ptr<ForeignOperatorObservations>& observations) {
  auto payload = authoring::authoring_registry().payload_type("holoscan.uint32");
  if (!payload) {
    throw RuntimeError{payload.error()};
  }
  // ForeignOperatorCallbacks adopts context and invokes destroy when its last owner is released.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return ForeignOperatorCallbacks{ForeignOperatorCallbackTable{
      .context = new ForeignOperatorProbe{observations, &payload->get()},
      .destroy = &ForeignOperatorProbe::destroy,
      .setup_cb = &ForeignOperatorProbe::setup,
      .contract_cb = &ForeignOperatorProbe::contract,
      .compute_cb = &ForeignOperatorProbe::compute,
      .start_cb = &ForeignOperatorProbe::start,
      .stop_cb = &ForeignOperatorProbe::stop,
      .has_lifecycle_cb = &ForeignOperatorProbe::has_lifecycle,
      .lifecycle_required_cb = &ForeignOperatorProbe::lifecycle_required,
      .lifecycle_cb = &ForeignOperatorProbe::lifecycle,
  }};
}

[[nodiscard]] ExecutionPlan make_foreign_plan(
    const std::shared_ptr<ForeignOperatorObservations>& observations) {
  Graph graph{"foreign-operator"};
  static_cast<void>(graph.op<ForeignOperatorCore>("foreign", make_callbacks(observations)));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  return compile(graph);
}

// REQ: OPERATOR-052
// REQ: COMPILE-092
TEST(ForeignOperatorTest, PublishesHostFactAndMatchesNativePlanShape) {
  //   foreign     native
  //
  // Key setup: compile isolated foreign and native Operators with equivalent declarations.
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  const ExecutionPlan foreign = make_foreign_plan(observations);

  Graph native_graph{"foreign-operator"};
  static_cast<void>(native_graph.op<NativeOperatorProbe>("foreign"));
  native_graph.set_default_clock(native_graph.add_clock<RealtimeClock>("clock"));
  const ExecutionPlan native = compile(native_graph);

  // Key assertion: the explicit host fact is the only hosting distinction; port and temporal
  // structure remain ordinary plan records.
  ASSERT_TRUE(foreign.ok()) << foreign.json();
  ASSERT_TRUE(native.ok()) << native.json();
  ASSERT_EQ(foreign.operator_hosts().size(), 1U);
  EXPECT_EQ(foreign.operator_hosts().front().kind, OperatorHostKind::kForeign);
  EXPECT_TRUE(foreign.operator_hosts().front().serialization_domain.empty());
  ASSERT_EQ(native.operator_hosts().size(), 1U);
  EXPECT_EQ(native.operator_hosts().front().kind, OperatorHostKind::kNative);
  EXPECT_TRUE(native.operator_hosts().front().serialization_domain.empty());
  ASSERT_EQ(foreign.graph().ports().size(), native.graph().ports().size());
  EXPECT_EQ(foreign.graph().ports().front().name, native.graph().ports().front().name);
  ASSERT_EQ(foreign.temporal_triggers().size(), native.temporal_triggers().size());
  EXPECT_EQ(foreign.temporal_triggers().front().kind, native.temporal_triggers().front().kind);
  EXPECT_NE(foreign.json().find("\"operator_hosts\""), std::string::npos);
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, ReplaysAndRunsTheFullLifecycle) {
  //   foreign
  //
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  const ExecutionPlan compiled = make_foreign_plan(observations);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = test::enable_direct_runtime(compiled);

  // Key setup: run the same frozen factory that supplied compile-time setup and contract records.
  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: setup/contract replay completes before one compute activation and balanced
  // lifecycle callbacks.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_GE(observations->setup_count.load(std::memory_order_relaxed), 2U);
  EXPECT_GE(observations->contract_count.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(observations->compute_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(observations->start_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(observations->stop_count.load(std::memory_order_relaxed), 1U);
}

// REQ: OPERATOR-053
TEST(ForeignOperatorTest, RunsAdvancedLifecycleCallbacksThroughTheForeignTable) {
  //   lifecycle-foreign
  //
  // Key setup: host required configure and optional release callbacks through exact trampolines.
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  Graph graph{"foreign-advanced-lifecycle"};
  static_cast<void>(graph.op<LifecycleForeignOperatorProbe>(
      "lifecycle-foreign", make_lifecycle_callbacks(observations)));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();

  RunSession session = run_async(test::enable_direct_runtime(compiled));
  session.wait();

  // Key assertion: both advanced stages retain their declared status across the C-shaped table.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(observations->configure_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(observations->release_count.load(std::memory_order_relaxed), 1U);
}

// REQ: OPERATOR-053
TEST(ForeignOperatorTest, PreservesAdvancedLifecycleFailureDetailAndOwner) {
  //   lifecycle-foreign
  //
  // Key setup: fail the required configure callback with adapter-owned bounded detail.
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  observations->fail_lifecycle.store(true, std::memory_order_relaxed);
  Graph graph{"foreign-advanced-lifecycle-failure"};
  static_cast<void>(graph.op<LifecycleForeignOperatorProbe>(
      "lifecycle-foreign", make_lifecycle_callbacks(observations)));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_FALSE(compiled.lifecycle_checkpoints().empty());
  const auto lifecycle_owner = compiled.lifecycle_checkpoints().front().operator_instance.id;

  RunSession session = run_async(test::enable_direct_runtime(compiled));
  session.wait();

  // Key assertion: the terminal diagnostic retains both callback detail and the failing owner.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const auto& diagnostic = session.diagnostics().front();
  ASSERT_TRUE(diagnostic.detail.has_value());
  // ASSERT_TRUE above establishes the optional invariant for the test body.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(diagnostic.detail->error.message, observations->compute_message);
  EXPECT_EQ(diagnostic.source.vertex.value, lifecycle_owner);
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, RejectsAChangedSetupDuringRunReplay) {
  //   foreign
  //
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  const ExecutionPlan compiled = make_foreign_plan(observations);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  observations->replay_mismatch.store(true, std::memory_order_relaxed);

  // Key setup: change the table's declaration only after the compile fingerprint is frozen.
  RunSession session = run_async(test::enable_direct_runtime(compiled));
  session.wait();

  // Key assertion: ordinary replay validation rejects the foreign declaration before lifecycle.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_EQ(observations->start_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(observations->compute_count.load(std::memory_order_relaxed), 0U);
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, CopiesBoundedComputeDetailIntoSessionDiagnostics) {
  //   foreign
  //
  // Key setup: keep callback detail in context-owned storage through the invoking Core method.
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  const ExecutionPlan compiled = make_foreign_plan(observations);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  observations->fail_compute.store(true, std::memory_order_relaxed);
  observations->compute_message.append(kForeignOperatorMessageCapacity, 'x');

  RunSession session = run_async(test::enable_direct_runtime(compiled));
  session.wait();

  // Key assertion: callback text is owned by the terminal diagnostic after the callback returns.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const auto& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  ASSERT_TRUE(diagnostic.detail.has_value());
  // ASSERT_TRUE above establishes the optional invariant for the test body.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(diagnostic.detail->error.message.size(), kForeignOperatorMessageCapacity);
  // ASSERT_TRUE above establishes the optional invariant for the test body.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(diagnostic.detail->error.message.substr(0U, 23U), "foreign callback detail");
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, DestroysTheSharedCallbackContextExactlyOnce) {
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  {
    const ForeignOperatorCallbacks callbacks = make_callbacks(observations);
    ForeignOperatorCallbacks first = callbacks;
    ForeignOperatorCallbacks second = first;
    second = callbacks;
    second = second;
    const ForeignOperatorCallbacks moved = std::move(first);
    first = moved;
    EXPECT_EQ(observations->destroy_count.load(std::memory_order_relaxed), 0U);
  }

  // Key assertion: copied captures share ownership but invoke the foreign release hook once.
  EXPECT_EQ(observations->destroy_count.load(std::memory_order_relaxed), 1U);
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, InvalidTableDestroysItsAdoptedContext) {
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  auto* context = new ForeignOperatorProbe{observations, nullptr};

  // Key assertion: rejection still invokes the supplied ownership-release hook exactly once.
  try {
    // ForeignOperatorCallbacks adopts context before validating and destroys it on rejection.
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    static_cast<void>(ForeignOperatorCallbacks(ForeignOperatorCallbackTable{
        .context = context,
        .destroy = &ForeignOperatorProbe::destroy,
    }));
    FAIL() << "expected invalid callback table rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(std::string_view{error.what()}.find("FOREIGN_OPERATOR_CALLBACK_TABLE_INVALID"),
              std::string_view::npos);
  }
  EXPECT_EQ(observations->destroy_count.load(std::memory_order_relaxed), 1U);
}

// REQ: OPERATOR-053
TEST(ForeignOperatorTest, RejectsAnIncompleteAdvancedLifecycleTable) {
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  auto* context = new ForeignOperatorProbe{observations, nullptr};

  // Key assertion: one lifecycle query without its required peers rejects and releases ownership.
  try {
    // ForeignOperatorCallbacks adopts context before validating and destroys it on rejection.
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    static_cast<void>(ForeignOperatorCallbacks(ForeignOperatorCallbackTable{
        .context = context,
        .destroy = &ForeignOperatorProbe::destroy,
        .compute_cb = &ForeignOperatorProbe::compute,
        .has_lifecycle_cb = &ForeignOperatorProbe::has_lifecycle,
    }));
    FAIL() << "expected incomplete lifecycle table rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(std::string_view{error.what()}.find("FOREIGN_OPERATOR_CALLBACK_TABLE_INVALID"),
              std::string_view::npos);
  }
  EXPECT_EQ(observations->destroy_count.load(std::memory_order_relaxed), 1U);
}

// REQ: OPERATOR-052
TEST(ForeignOperatorTest, ReportsSetupDeclarationFailureAtTheCallbackBoundary) {
  const auto observations = std::make_shared<ForeignOperatorObservations>();
  observations->duplicate_output.store(true, std::memory_order_relaxed);
  ForeignOperatorCore foreign{make_callbacks(observations)};
  OperatorSpec spec;

  // Key assertion: a rejected typed declaration reaches the adapter with local port context.
  try {
    foreign.setup(spec);
    FAIL() << "expected duplicate output rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kDuplicateName);
    EXPECT_NE(std::string_view{error.what()}.find("port_name=output"), std::string_view::npos);
  }
}

}  // namespace
}  // namespace holoscan
