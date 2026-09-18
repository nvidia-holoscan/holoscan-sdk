/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <new>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>

#include "../common/allocation_counter.hpp"
#include "../support/host_unique_test_name.hpp"
#include "compile_test_graph.hpp"
#include "compiler/derivation_replay.hpp"
#include "compiler/diagnostic_detail.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "core/parameter_schema_test_access.hpp"

namespace holoscan {

struct DerivationDiagnosticParams {
  MemoryKind memory{MemoryKind::kHost};
  std::uint64_t bytes{64U};
};

HOLOSCAN_PARAM_SCHEMA(
    DerivationDiagnosticParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.DerivationDiagnosticParams", 1)),
    holoscan::field<&DerivationDiagnosticParams::memory>("memory")
        .enum_values({param_enum_value("host", MemoryKind::kHost),
                      param_enum_value("pinned", MemoryKind::kPinnedHost)})
        .recompile_required(),
    holoscan::field<&DerivationDiagnosticParams::bytes>("bytes").recompile_required());

namespace {

enum class ReplayFault : std::uint8_t {
  kConstructTyped,
  kConstructNested,
  kConstructAllocation,
  kConstructNonstandard,
  kSetupTyped,
  kSetupNested,
  kSetupAllocation,
  kDeclaration,
  kLongChain,
  kSetupLongChain,
};

struct ReplayFaultState {
  ReplayFault fault{};
  std::string message_canary;
  bool contract_seen{};
  bool consumed{};
  std::uint32_t setup_after_contract{};
};

// A real temporal-contract witness arms the next feature-specific replay. Graph authoring,
// Parameter replay and temporal setup must all succeed before this fixture can fail.
[[noreturn]] void throw_replay_fault(ReplayFault fault, std::uint32_t depth = 0U,
                                     std::string_view canary = {}) {
  if (fault == ReplayFault::kConstructAllocation || fault == ReplayFault::kSetupAllocation) {
    throw std::bad_alloc{};
  }
  if (fault == ReplayFault::kConstructNonstandard) {
    throw 7;
  }
  if (fault == ReplayFault::kLongChain || fault == ReplayFault::kSetupLongChain) {
    if (depth == 6U) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "innermost cause"}};
    }
    try {
      throw_replay_fault(fault, depth + 1U, canary);
    } catch (...) {
      std::throw_with_nested(
          RuntimeError{Error{ErrorCode::kInvalidArgument, std::string(1200U, 'x') + "\x1b\n"}});
    }
  }
  const auto message = [canary](std::string_view base) {
    std::string result{base};
    if (!canary.empty()) {
      result += ": ";
      result += canary;
    }
    return result;
  };
  if (fault == ReplayFault::kConstructNested || fault == ReplayFault::kSetupNested) {
    try {
      throw RuntimeError{Error{ErrorCode::kNotSupported, message("nested replay cause")}};
    } catch (...) {
      std::throw_with_nested(
          RuntimeError{Error{ErrorCode::kInvalidArgument, message("outer replay cause")}});
    }
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, message("original replay cause")}};
}

class DerivationDiagnosticSource final : public Operator<DerivationDiagnosticParams> {
 public:
  DerivationDiagnosticSource(std::shared_ptr<ReplayFaultState> state, bool memory_kind)
      : state_(std::move(state)), memory_kind_(memory_kind) {
    if (state_->contract_seen && !state_->consumed &&
        (state_->fault <= ReplayFault::kConstructNonstandard ||
         state_->fault == ReplayFault::kLongChain)) {
      state_->consumed = true;
      throw_replay_fault(state_->fault, 0U, state_->message_canary);
    }
  }

  void setup(OperatorSpec& spec) override {
    if (state_->contract_seen) {
      ++state_->setup_after_contract;
      if (!state_->consumed) {
        state_->consumed = true;
        if (state_->fault == ReplayFault::kDeclaration) {
          spec.output(invalid, "");
        } else {
          throw_replay_fault(state_->fault, 0U, state_->message_canary);
        }
      }
    }
    if (memory_kind_) {
      spec.output(output, "output")
          .produces_tensor(
              TensorOutputSpec{
                  .representation = {.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U},
                  .bounds = tensor_bounds(64U),
                  .storage = TensorOutputStorage::kRuntimePool,
              },
              TensorOutputDerivations{.memory_kind = parameter_memory_kind(
                                          param_ref<&DerivationDiagnosticParams::memory>())});
    } else {
      spec.output(output, "output")
          .produces_tensor(
              TensorOutputSpec{
                  .representation = {.memory_kind = MemoryKind::kHost,
                                     .dtype = DLDataType{kDLUInt, 8U, 1U},
                                     .rank = 1U},
                  .storage = TensorOutputStorage::kRuntimePool,
              },
              parameter_byte_span(param_ref<&DerivationDiagnosticParams::bytes>()));
    }
  }

  Contract contract() const override {
    state_->contract_seen = true;
    return {};
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
  Output<std::uint32_t> invalid;

 private:
  std::shared_ptr<ReplayFaultState> state_;
  bool memory_kind_{};
};

// REQ: COMPILE-076
// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(DerivationDiagnostics, CapturesTheSelectedReplayAfterEarlierPassesSucceed) {
  for (const bool memory_kind : {false, true}) {
    for (const ReplayFault fault : {ReplayFault::kConstructTyped,
                                    ReplayFault::kConstructNested,
                                    ReplayFault::kConstructAllocation,
                                    ReplayFault::kConstructNonstandard,
                                    ReplayFault::kSetupTyped,
                                    ReplayFault::kSetupNested,
                                    ReplayFault::kSetupAllocation,
                                    ReplayFault::kDeclaration,
                                    ReplayFault::kLongChain,
                                    ReplayFault::kSetupLongChain}) {
      SCOPED_TRACE(memory_kind);
      SCOPED_TRACE(static_cast<unsigned>(fault));
      auto control = std::make_shared<ReplayFaultState>();
      control->fault = fault;
      ExecutionPlan plan;
      {
        Graph graph{"derivation-diagnostics"};
        test::add_compile_test_default_clock(graph);
        (void)graph.op<DerivationDiagnosticSource>("replay-owner", control, memory_kind);
        ASSERT_FALSE(control->contract_seen);
        plan = compile(graph);
      }
      ASSERT_TRUE(control->contract_seen);
      ASSERT_TRUE(control->consumed);
      ASSERT_FALSE(plan.ok());
      const bool construction =
          fault <= ReplayFault::kConstructNonstandard || fault == ReplayFault::kLongChain;
      EXPECT_EQ(control->setup_after_contract, construction ? 0U : 1U);
      const std::string token = memory_kind ? "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID"
                                            : "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
      const auto diagnostics = plan.diagnostics();
      const auto found = std::ranges::find(diagnostics, token, &Diagnostic::token);
      ASSERT_NE(found, diagnostics.end());
      EXPECT_TRUE(found->source.vertex);
      EXPECT_FALSE(found->source.port);
      EXPECT_TRUE(found->json_pointer.empty());
      EXPECT_EQ(found->required, construction ? "" : "successful Operator setup replay");
      if (construction) {
        EXPECT_TRUE(found->provided.empty());
      } else {
        EXPECT_EQ(found->provided,
                  fault == ReplayFault::kDeclaration ? "operator declaration name is empty"
                                                     : "setup threw");
      }
      EXPECT_TRUE(std::ranges::none_of(diagnostics, [](const auto& row) {
        return row.token.starts_with("TEMPORAL_") || row.token.starts_with("PARAMETER_");
      }));
      // Copy after the Graph and all replay temporaries have retired; numeric IDs alone cannot
      // supply the owner name to this empty-graph formatter.
      // Keep an independent owning snapshot; a reference would remove this test control.
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      const Diagnostic owned = *found;
      EXPECT_TRUE(owned.detail.has_value());
      if (!owned.detail.has_value()) {
        continue;
      }
      const auto& detail = owned.detail.value();
      EXPECT_EQ(detail.operation,
                construction ? DiagnosticOperation::kConstruct : DiagnosticOperation::kSetup);
      EXPECT_EQ(detail.operator_path, "replay-owner");
      EXPECT_TRUE(detail.port_name.empty());
      const ErrorCode code =
          fault == ReplayFault::kConstructAllocation || fault == ReplayFault::kSetupAllocation
              ? ErrorCode::kResourceExhausted
          : fault == ReplayFault::kConstructNonstandard ? ErrorCode::kFailure
                                                        : ErrorCode::kInvalidArgument;
      EXPECT_EQ(detail.error.code, code);
      const auto rendered = format_diagnostic(owned, {});
      EXPECT_NE(rendered.find("replay-owner"), std::string::npos);
      EXPECT_EQ(rendered.find('\x1b'), std::string::npos);
      EXPECT_EQ(rendered.find('\n'), std::string::npos);
      if (fault == ReplayFault::kConstructNested || fault == ReplayFault::kSetupNested) {
        ASSERT_EQ(detail.causes.size(), 1U);
        EXPECT_EQ(detail.causes[0].code, ErrorCode::kNotSupported);
        EXPECT_EQ(detail.causes[0].message, "nested replay cause");
      }
      if (fault == ReplayFault::kLongChain || fault == ReplayFault::kSetupLongChain) {
        EXPECT_TRUE(detail.truncated);
        EXPECT_EQ(detail.error.code, ErrorCode::kInvalidArgument);
        EXPECT_TRUE(detail.error.message.starts_with(std::string(1200U, 'x')));
        EXPECT_EQ(detail.causes.size(), 3U);
        std::size_t bytes = detail.error.message.size();
        for (const auto& cause : detail.causes) {
          bytes += cause.message.size();
        }
        EXPECT_LE(bytes, 4096U);
        EXPECT_NE(rendered.find("truncated"), std::string::npos);
      }
      EXPECT_LE(format_diagnostics(diagnostics, plan.graph()).size(), 16384U);
      EXPECT_EQ(plan.json().find("original replay cause"), std::string_view::npos);
      EXPECT_EQ(plan.json().find("nested replay cause"), std::string_view::npos);
    }
  }
}

// REQ: COMPILE-076
// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(DerivationDiagnostics, ConcurrentFailingCompilesKeepOwnersAndCausesIndependent) {
  for (const bool memory_kind : {false, true}) {
    for (const ReplayFault fault : {ReplayFault::kConstructNested, ReplayFault::kSetupNested}) {
      SCOPED_TRACE(memory_kind);
      SCOPED_TRACE(static_cast<unsigned>(fault));
      const auto compile_failure = [memory_kind, fault](std::string_view owner,
                                                        std::string canary) {
        auto control = std::make_shared<ReplayFaultState>();
        control->fault = fault;
        control->message_canary = std::move(canary);
        Graph graph{"concurrent-derivation-failure"};
        test::add_compile_test_default_clock(graph);
        (void)graph.op<DerivationDiagnosticSource>(owner, control, memory_kind);
        const bool authored_before_failure = !control->contract_seen && !control->consumed;
        ExecutionPlan plan = compile(graph);
        return std::tuple{std::move(plan), std::move(control), authored_before_failure};
      };
      auto first =
          std::async(std::launch::async, compile_failure, "left-replay-owner", "left-cause-canary");
      auto second = std::async(
          std::launch::async, compile_failure, "right-replay-owner", "right-cause-canary");
      // Join both workers before inspecting results or invoking fatal assertions. A callback
      // barrier could deadlock if a regression rejects one Graph before the selected replay.
      first.wait();
      second.wait();
      const auto first_result = first.get();
      const auto second_result = second.get();
      const auto inspect = [memory_kind, fault](const auto& result,
                                                std::string_view owner,
                                                std::string_view canary,
                                                std::string_view other_owner,
                                                std::string_view other_canary) {
        const auto& [plan, control, authored_before_failure] = result;
        ASSERT_TRUE(authored_before_failure);
        ASSERT_TRUE(control->contract_seen);
        ASSERT_TRUE(control->consumed);
        ASSERT_FALSE(plan.ok());
        const bool construction = fault == ReplayFault::kConstructNested;
        EXPECT_EQ(control->setup_after_contract, construction ? 0U : 1U);
        const std::string token = memory_kind ? "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID"
                                              : "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
        const auto diagnostics = plan.diagnostics();
        const auto found = std::ranges::find(diagnostics, token, &Diagnostic::token);
        ASSERT_NE(found, diagnostics.end());
        EXPECT_TRUE(std::ranges::none_of(diagnostics, [](const auto& row) {
          return row.token.starts_with("TEMPORAL_") || row.token.starts_with("PARAMETER_");
        }));
        EXPECT_EQ(found->severity, DiagnosticSeverity::kError);
        EXPECT_TRUE(found->source.vertex);
        EXPECT_FALSE(found->source.port);
        EXPECT_TRUE(found->json_pointer.empty());
        EXPECT_EQ(found->required, construction ? "" : "successful Operator setup replay");
        EXPECT_EQ(found->provided, construction ? "" : "setup threw");
        // Keep an independent owning snapshot; a reference would remove this test control.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const Diagnostic owned = *found;
        ASSERT_TRUE(owned.detail.has_value());
        if (!owned.detail.has_value()) {
          FAIL() << "Concurrent replay discarded its owning cause";
        }
        const auto& evidence = owned.detail.value();
        EXPECT_EQ(evidence.operation,
                  construction ? DiagnosticOperation::kConstruct : DiagnosticOperation::kSetup);
        EXPECT_EQ(evidence.operator_path, owner);
        EXPECT_TRUE(evidence.port_name.empty());
        EXPECT_EQ(evidence.error.code, ErrorCode::kInvalidArgument);
        EXPECT_EQ(evidence.error.message, "outer replay cause: " + std::string{canary});
        ASSERT_EQ(evidence.causes.size(), 1U);
        EXPECT_EQ(evidence.causes.front().code, ErrorCode::kNotSupported);
        EXPECT_EQ(evidence.causes.front().message, "nested replay cause: " + std::string{canary});

        // Inspect every owning text field, not only the selected diagnostic's display.
        std::string captured;
        for (const auto& row : diagnostics) {
          captured += row.token + row.service + row.endpoint + row.json_pointer + row.required +
                      row.provided;
          if (row.detail.has_value()) {
            const auto& detail = row.detail.value();
            captured += detail.operator_path + detail.port_name + detail.related_operator_path +
                        detail.related_port_name + detail.error.message;
            for (const auto& cause : detail.causes) {
              captured += cause.message;
            }
            for (const auto& mismatch : detail.mismatches) {
              captured += mismatch.field + mismatch.required + mismatch.provided;
            }
          }
        }
        const auto single = format_diagnostic(owned, {});
        const auto report = format_diagnostics(diagnostics, plan.graph());
        EXPECT_NE(single.find(owner), std::string::npos);
        EXPECT_NE(single.find(canary), std::string::npos);
        EXPECT_NE(report.find(owner), std::string::npos);
        EXPECT_NE(report.find(canary), std::string::npos);
        const std::string all_text = captured + single + report;
        EXPECT_EQ(all_text.find(other_owner), std::string::npos);
        EXPECT_EQ(all_text.find(other_canary), std::string::npos);
        EXPECT_EQ(plan.json().find(canary), std::string_view::npos);
        EXPECT_EQ(plan.json().find(other_canary), std::string_view::npos);
        EXPECT_EQ(plan.json().find(other_owner), std::string_view::npos);
      };
      inspect(first_result,
              "left-replay-owner",
              "left-cause-canary",
              "right-replay-owner",
              "right-cause-canary");
      inspect(second_result,
              "right-replay-owner",
              "right-cause-canary",
              "left-replay-owner",
              "left-cause-canary");
    }
  }
}

// The private replay boundary can return an Error or an already allocated instance without
// changing the real factory's schema, lifecycle metadata, or public authoring behavior.
class ReplayBoundaryFactory final : public detail::OperatorFactoryBase {
 public:
  explicit ReplayBoundaryFactory(std::shared_ptr<detail::OperatorFactoryBase> original)
      : original_(std::move(original)) {}
  expected<std::unique_ptr<OperatorBase>, Error> create(detail::DiagnosticCapture*) const override {
    if (return_instance) {
      return std::move(instance);
    }
    return make_unexpected(error);
  }
  std::shared_ptr<detail::OperatorFactoryBase> clone() const override { return original_->clone(); }
  expected<std::shared_ptr<detail::OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* token) const override {
    return original_->clone_with_params(params, token);
  }
  const ParamTypeDescriptor& parameter_type() const noexcept override {
    return original_->parameter_type();
  }
  const ParamSchemaDescriptor* parameter_schema() const noexcept override {
    return original_->parameter_schema();
  }
  const void* initial_parameter_value() const noexcept override {
    return original_->initial_parameter_value();
  }
  bool has_explicit_parameter_baseline() const noexcept override {
    return original_->has_explicit_parameter_baseline();
  }
  bool has_no_params() const noexcept override { return original_->has_no_params(); }
  LifecycleHookSet lifecycle_hooks() const noexcept override {
    return original_->lifecycle_hooks();
  }

  Error error{ErrorCode::kInvalidArgument, "factory returned typed error"};
  bool return_instance{};
  mutable std::unique_ptr<OperatorBase> instance;

 private:
  std::shared_ptr<detail::OperatorFactoryBase> original_;
};

class ReplaySetupWitness final : public Operator<> {
 public:
  explicit ReplaySetupWitness(std::shared_ptr<unsigned> entries) : entries_(std::move(entries)) {}
  void setup(OperatorSpec&) override { ++*entries_; }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  std::shared_ptr<unsigned> entries_;
};

// REQ: COMPILE-076
// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(DerivationDiagnostics, PreservesFactoryReturnedErrorAndPreSetupAllocationBoundary) {
  Graph graph{"private-replay-boundary"};
  test::add_compile_test_default_clock(graph);
  auto entries = std::make_shared<unsigned>();
  (void)graph.op<ReplaySetupWitness>("source", entries);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto original = state->executable_factories.operators.front().factory;
  ReplayBoundaryFactory factory{original};
  detail::DerivationReplay replay;
  detail::DiagnosticCapture capture;
  EXPECT_EQ(detail::replay_derivation(factory, replay, capture),
            detail::DerivationReplayOutcome::kConstructionFailed);
  ASSERT_EQ(capture.count, 1U);
  EXPECT_EQ(capture.causes[0].code, ErrorCode::kInvalidArgument);
  const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
  EXPECT_EQ(evidence.error.message, "factory returned typed error");
  EXPECT_FALSE(replay.instance);
  EXPECT_FALSE(replay.spec);

  auto created = original->create(nullptr);
  ASSERT_TRUE(created.has_value());
  factory.instance = std::move(created.value());
  factory.return_instance = true;
  *entries = 0U;
  bool exhausted{};
  {
    const test::ScopedAllocationFailure fail_spec_allocation{0U};
    try {
      (void)detail::replay_derivation(factory, replay, capture);
    } catch (const std::bad_alloc&) {
      exhausted = true;
    }
  }
  EXPECT_TRUE(exhausted);
  EXPECT_EQ(*entries, 0U);
  EXPECT_EQ(capture.count, 0U);
  EXPECT_FALSE(replay.spec);
}

template <unsigned Depth>
class NestedReplayGraph final : public Graph {
 public:
  explicit NestedReplayGraph(std::shared_ptr<ReplayFaultState> state)
      : fault_state_(std::move(state)) {}

 protected:
  void compose() override {
    if constexpr (Depth == 0U) {
      (void)op<DerivationDiagnosticSource>("leaf", fault_state_, true);
    } else {
      (void)embed<NestedReplayGraph<Depth - 1U>>(std::string(230U, 'p'), fault_state_);
    }
  }

 private:
  std::shared_ptr<ReplayFaultState> fault_state_;
};

// REQ: COMPILE-076
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(DerivationDiagnostics, BoundsAdmittedQualifiedOwnerAndContextBeforeFormatting) {
  auto state = std::make_shared<ReplayFaultState>();
  state->fault = ReplayFault::kConstructTyped;
  Graph graph{"long-derivation-owner"};
  test::add_compile_test_default_clock(graph);
  (void)graph.embed<NestedReplayGraph<10U>>("nested", state);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(state->consumed);
  const auto diagnostics = plan.diagnostics();
  const auto found = std::ranges::find(
      diagnostics, "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID", &Diagnostic::token);
  ASSERT_NE(found, diagnostics.end());
  const auto& retained_detail = found->detail;
  ASSERT_TRUE(retained_detail.has_value());
  if (!retained_detail.has_value()) {
    FAIL() << "Expected bounded replay owner detail";
  }
  const auto& evidence = retained_detail.value();
  const auto full_path = detail::diagnostic_operator_path(plan.graph(), found->source.vertex);
  ASSERT_GT(full_path.size(), 2048U);
  EXPECT_EQ(evidence.operator_path.size(), 2048U);
  EXPECT_TRUE(full_path.starts_with(evidence.operator_path));
  EXPECT_TRUE(evidence.truncated);
  EXPECT_FALSE(found->source.port);
  EXPECT_LE(format_diagnostics(diagnostics, plan.graph()).size(), 16384U);

  Diagnostic context{.token = "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID"};
  context.detail.emplace(DiagnosticDetail{.operation = DiagnosticOperation::kCompile,
                                          .error = Error{ErrorCode::kSuccess}});
  detail::append_derivation_context(
      *context.detail, "parameter_path", std::string("prefix\x1b\n") + std::string(5000U, 'x'));
  const auto& field = context.detail->mismatches.front();
  EXPECT_LE(field.field.size() + field.provided.size(), 1024U);
  EXPECT_TRUE(context.detail->truncated);
  const auto text = format_diagnostic(context, {});
  EXPECT_EQ(text.find('\x1b'), std::string::npos);
  EXPECT_EQ(text.find('\n'), std::string::npos);
}

// REQ: COMPILE-007
// REQ: COMPILE-025
// REQ: COMPILE-076
// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(DerivationDiagnostics, DetailOnlyMutationPreservesRejectedPlanIdentityAndArtifactDenial) {
  const auto structural_key = [](const Diagnostic& row) {
    return std::tuple{row.id.value,
                      row.severity,
                      row.token,
                      row.source.graph_generation,
                      row.source.graph.value,
                      row.source.vertex.value,
                      row.source.port.value(),
                      row.service,
                      row.endpoint,
                      row.json_pointer,
                      row.required,
                      row.provided,
                      row.cause_id};
  };
  for (const bool memory_kind : {false, true}) {
    for (const ReplayFault fault : {ReplayFault::kConstructNested, ReplayFault::kSetupNested}) {
      SCOPED_TRACE(memory_kind);
      SCOPED_TRACE(static_cast<unsigned>(fault));
      ExecutionPlan plan;
      {
        auto control = std::make_shared<ReplayFaultState>();
        control->fault = fault;
        Graph graph{"derivation-detail-artifact"};
        test::add_compile_test_default_clock(graph);
        (void)graph.op<DerivationDiagnosticSource>("replay-owner", control, memory_kind);
        ASSERT_FALSE(control->contract_seen);
        plan = compile(graph);
        ASSERT_TRUE(control->contract_seen);
        ASSERT_TRUE(control->consumed);
        EXPECT_EQ(control->setup_after_contract, fault == ReplayFault::kConstructNested ? 0U : 1U);
      }
      // Both feature-specific compile publication loops ran, and the original Graph and
      // local constructor arguments have retired. The plan still owns its independent data.
      ASSERT_FALSE(plan.ok());
      const std::string token = memory_kind ? "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID"
                                            : "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
      const auto rows = plan.diagnostics();
      const auto found = std::ranges::find(rows, token, &Diagnostic::token);
      ASSERT_NE(found, rows.end());
      const auto index = static_cast<std::size_t>(found - rows.begin());
      // Keep an independent owning snapshot; a reference would remove this test control.
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      const Diagnostic owned = *found;
      if (!owned.detail.has_value()) {
        FAIL() << "The real derivation publication must retain its cause";
      }
      EXPECT_EQ(owned.detail->error.code, ErrorCode::kInvalidArgument);
      EXPECT_EQ(owned.detail->error.message, "outer replay cause");
      ASSERT_EQ(owned.detail->causes.size(), 1U);
      EXPECT_EQ(owned.detail->causes.front().message, "nested replay cause");
      EXPECT_EQ(owned.detail->operator_path, "replay-owner");
      EXPECT_TRUE(owned.detail->port_name.empty());
      EXPECT_TRUE(owned.source.vertex);
      EXPECT_FALSE(owned.source.port);

      const auto* state = detail::ExecutionPlanAccess::state(plan);
      ASSERT_NE(state, nullptr);
      ASSERT_FALSE(state->admitted);
      const std::string original_json{plan.json()};
      const PlanDigest original_digest = plan.digest();
      const ExecutionGraphView owning_graph = plan.graph();
      const auto vertices = owning_graph.vertices();
      const auto vertex = std::ranges::find(vertices, owned.source.vertex, &VertexRecord::id);
      ASSERT_NE(vertex, vertices.end());
      EXPECT_EQ(vertex->name, "replay-owner");
      EXPECT_EQ(owned.source.graph, vertex->graph);
      EXPECT_EQ(owned.source.graph_generation, owning_graph.generation());

      for (const bool remove_detail : {false, true}) {
        SCOPED_TRACE(remove_detail);
        auto changed = std::make_shared<detail::ExecutionPlanState>(*state);
        auto& supplemental = changed->diagnostics[index].detail;
        if (remove_detail) {
          supplemental.reset();
        } else {
          if (!supplemental.has_value()) {
            FAIL() << "Copied published detail disappeared";
          }
          supplemental->error.message = "alternate owned replay detail";
          supplemental->causes.front().message = "alternate owned nested detail";
          supplemental->truncated = true;
        }
        ASSERT_EQ(changed->diagnostics.size(), rows.size());
        for (std::size_t row = 0U; row < rows.size(); ++row) {
          EXPECT_EQ(structural_key(changed->diagnostics[row]), structural_key(rows[row]));
        }
        const auto serialized = detail::finalize_plan_artifact(*changed);
        ASSERT_TRUE(serialized) << serialized.error().message;
        EXPECT_FALSE(changed->admitted);
        EXPECT_EQ(changed->json, original_json);
        EXPECT_EQ(changed->digest, original_digest);
        EXPECT_EQ(changed->json.find("outer replay cause"), std::string::npos);
        EXPECT_EQ(changed->json.find("nested replay cause"), std::string::npos);
        EXPECT_EQ(changed->json.find("alternate owned"), std::string::npos);

        // Failed plans never become deployable merely because their owning public detail
        // changes. Do not change admission, severity, structural rows, or other plan fields.
        const ExecutionPlan changed_plan = detail::ExecutionPlanAccess::make(changed);
        const auto root = std::filesystem::temp_directory_path() /
                          test::make_host_unique_test_name("rejected-derivation-artifacts");
        const auto emitted = emit_artifacts(
            changed_plan, ArtifactEmitOptions{.root = root, .deployment_id = "detail-only"});
        EXPECT_FALSE(emitted);
        if (!emitted) {
          EXPECT_EQ(emitted.error().code, ErrorCode::kInvalidArgument);
          EXPECT_EQ(emitted.error().message, "ARTIFACT_PLAN_OR_DEPLOYMENT_INVALID");
        }
        EXPECT_FALSE(std::filesystem::exists(root));
        // Remove only this test's unique root if a broken emitter unexpectedly created it.
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        EXPECT_FALSE(cleanup_error);
      }

      // Drop every plan/factory owner as well. Only the copied Diagnostic and a separately
      // retained compiled Graph view remain, so empty-context formatting cannot borrow names.
      plan = {};
      const auto empty_report = format_diagnostic(owned, {});
      EXPECT_NE(empty_report.find("operator=replay-owner"), std::string::npos);
      EXPECT_NE(empty_report.find("outer replay cause"), std::string::npos);
      EXPECT_NE(empty_report.find("nested replay cause"), std::string::npos);
      EXPECT_EQ(empty_report.find("port_name="), std::string::npos);
      const auto compiled_report = format_diagnostic(owned, owning_graph);
      EXPECT_NE(compiled_report.find("vertex=replay-owner, vertex_id=" +
                                     std::to_string(owned.source.vertex.value)),
                std::string::npos);
      EXPECT_EQ(compiled_report.find("port_name="), std::string::npos);
    }
  }
}

}  // namespace
}  // namespace holoscan
