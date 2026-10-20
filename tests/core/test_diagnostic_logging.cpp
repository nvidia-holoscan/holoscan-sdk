/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/time/manual_clock.hpp>
#include "common/allocation_counter.hpp"
#include "common/diagnostic_test_environment.hpp"
#include "core/diagnostic_reporting.hpp"

namespace holoscan {
namespace {

// REQ: EXPECTED-032
static_assert(noexcept(set_automatic_diagnostic_logging_enabled(false)));
static_assert(noexcept(automatic_diagnostic_logging_enabled()));

class DiagnosticLogging : public testing::Test {
 protected:
  void SetUp() override {
    policy_environment_ = std::make_unique<test::ScopedTestEnvironment>(
        "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", nullptr);
    logger_ = std::make_unique<test::ScopedDiagnosticLogger>(LogLevel::WARN);
  }
  void TearDown() override {
    // This may be the first policy access after an explicit-helper-only test.
    // Reset while the inherited override is still absent, then restore it last.
    set_automatic_diagnostic_logging_enabled(true);
    logger_.reset();
    policy_environment_.reset();
  }
  std::unique_ptr<test::ScopedTestEnvironment> policy_environment_;
  std::unique_ptr<test::ScopedDiagnosticLogger> logger_;
};

// REQ: EXPECTED-032
TEST(DiagnosticLoggingEnvironment, RestoresLoggerAndEnvironmentAfterEarlyReturn) {
  const test::ScopedDiagnosticLogger outer_logger{LogLevel::INFO};
  set_log_pattern("outer: %v");
  Logger::log_level_set_by_user = false;
  Logger::log_pattern_set_by_user = false;
  for (const char* inherited : {static_cast<const char*>(nullptr), "OFF", "ERROR", "WARN"}) {
    const test::ScopedTestEnvironment level_environment{"HOLOSCAN_LOG_LEVEL", inherited};
    const test::ScopedTestEnvironment format_environment{"HOLOSCAN_LOG_FORMAT", "inherited: %v"};
    const auto exercise_early_return = [] {
      const test::ScopedDiagnosticLogger logger{LogLevel::WARN};
      EXPECT_EQ(log_level(), LogLevel::WARN);
      EXPECT_EQ(Logger::pattern(), "%l: %v");
      EXPECT_EQ(std::getenv("HOLOSCAN_LOG_LEVEL"), nullptr);
      EXPECT_EQ(std::getenv("HOLOSCAN_LOG_FORMAT"), nullptr);
      set_log_level(LogLevel::ERROR);
      set_log_pattern("temporary: %v");
      return;
    };
    exercise_early_return();
    EXPECT_EQ(log_level(), LogLevel::INFO);
    EXPECT_EQ(Logger::pattern(), "outer: %v");
    EXPECT_FALSE(Logger::log_level_set_by_user);
    EXPECT_FALSE(Logger::log_pattern_set_by_user);
    EXPECT_STREQ(std::getenv("HOLOSCAN_LOG_LEVEL"), inherited);
    EXPECT_STREQ(std::getenv("HOLOSCAN_LOG_FORMAT"), "inherited: %v");
  }
}

// REQ: EXPECTED-032
TEST(DiagnosticLoggingEnvironment, RestoresAbsentAndPresentValues) {
  constexpr auto* name = "HOLOSCAN_DIAGNOSTIC_TEST_ENVIRONMENT";
  for (const char* original : {static_cast<const char*>(nullptr), "original"}) {
    const test::ScopedTestEnvironment baseline{name, original};
    {
      const test::ScopedTestEnvironment temporary{name, "temporary"};
      EXPECT_STREQ(std::getenv(name), "temporary");
      EXPECT_EQ(test::set_test_environment(name, nullptr), 0);
      EXPECT_EQ(std::getenv(name), nullptr);
    }
    EXPECT_STREQ(std::getenv(name), original);
  }
}

class DiagnosticLoggingClockOperator final : public Operator<> {
 public:
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// REQ: EXPECTED-033
// REQ: COMPILE-095
TEST_F(DiagnosticLogging, CompileRemainsSilentAndPlanHelperPreservesIdentity) {
  Graph graph{"logging-plan"};
  graph.op<DiagnosticLoggingClockOperator>("source");
  testing::internal::CaptureStderr();
  const auto rejected = compile(graph);
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  ASSERT_FALSE(rejected.ok());
  const auto json = rejected.json();
  const auto digest = rejected.digest();
  testing::internal::CaptureStderr();
  log_diagnostics(rejected);
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("TEMPORAL_CLOCK_UNDECLARED"), std::string::npos);
  EXPECT_EQ(rejected.json(), json);
  EXPECT_EQ(rejected.digest(), digest);
  graph.set_default_clock(graph.add_clock<ManualClock>("clock"));
  const auto accepted = compile(graph);
  ASSERT_TRUE(accepted.ok());
  testing::internal::CaptureStderr();
  log_diagnostics(accepted);
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

// REQ: EXPECTED-033
TEST_F(DiagnosticLogging, ExplicitReportsIgnoreAutomaticPolicyAndRetainIndices) {
  const std::array diagnostics{
      Diagnostic{.severity = DiagnosticSeverity::kWarning, .token = "WARNING_RECORD"},
      Diagnostic{.token = "ERROR_RECORD"}};
  for (const bool enabled : {false, true}) {
    set_automatic_diagnostic_logging_enabled(enabled);
    testing::internal::CaptureStderr();
    log_diagnostics(diagnostics, ExecutionGraphView{});
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("error: error ERROR_RECORD [diagnostic_index=1]"), std::string::npos);
    EXPECT_NE(output.find("warning: warning WARNING_RECORD [diagnostic_index=0]"),
              std::string::npos);
    EXPECT_LT(output.find("ERROR_RECORD"), output.find("WARNING_RECORD"));
  }
}

// REQ: EXPECTED-033
TEST_F(DiagnosticLogging, FiltersSeverityAndLeavesEmptyReportsSilent) {
  const std::array diagnostics{
      Diagnostic{.severity = DiagnosticSeverity::kWarning, .token = "FILTERED_WARNING"},
      Diagnostic{.token = "VISIBLE_ERROR"}};
  set_log_level(LogLevel::ERROR);
  testing::internal::CaptureStderr();
  log_diagnostics(diagnostics, ExecutionGraphView{});
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(output.find("FILTERED_WARNING"), std::string::npos);
  EXPECT_NE(output.find("VISIBLE_ERROR"), std::string::npos);
  set_log_level(LogLevel::OFF);
  testing::internal::CaptureStderr();
  log_diagnostics(diagnostics, ExecutionGraphView{});
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  set_log_level(LogLevel::WARN);
  testing::internal::CaptureStderr();
  log_diagnostics({}, ExecutionGraphView{});
  log_diagnostics(ExecutionPlan{});
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

// REQ: EXPECTED-033
TEST_F(DiagnosticLogging, BoundsOneReportAcrossSeveritiesAndEscapesControlText) {
  std::vector<Diagnostic> diagnostics;
  diagnostics.reserve(80U);
  for (int i = 0; i < 80; ++i) {
    diagnostics.push_back(
        Diagnostic{.severity = i < 20 ? DiagnosticSeverity::kError : DiagnosticSeverity::kWarning,
                   .token = std::string(200, 'x') + "\n\033"});
  }
  testing::internal::CaptureStderr();
  log_diagnostics(diagnostics, ExecutionGraphView{});
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_LE(output.size(), 16410U);  // Two logger-owned severity prefixes and newlines.
  EXPECT_NE(output.find("diagnostic report truncated"), std::string::npos);
  EXPECT_EQ(output.find('\033'), std::string::npos);
}

// REQ: EXPECTED-032
// REQ: RUNTIME-060
// REQ: LIFECYCLE-014
// REQ: GRAPH-044
// REQ: CLOCK-RT-008
// REQ: COMPILE-095
TEST_F(DiagnosticLogging, DisabledClockReportPreservesFailureAndExplicitUserLogging) {
  ManualClock clock;
  set_automatic_diagnostic_logging_enabled(false);
  testing::internal::CaptureStderr();
  const auto result = clock.sleep_for(-1);
  HOLOSCAN_LOG_ERROR("USER_OWNED_ERROR");
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(result.has_value());
  EXPECT_NE(output.find("USER_OWNED_ERROR"), std::string::npos);
  EXPECT_EQ(output, "error: USER_OWNED_ERROR\n");
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, ResourceOptOutRetainsRejectionAndPoolReuse) {
  set_automatic_diagnostic_logging_enabled(false);
  BlockMemoryPool pool{MemoryStorageType::kSystem, 64, 1};
  ASSERT_TRUE(pool.initialize_pool());
  auto block = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(block);
  UnboundedAllocator allocator;
  std::byte foreign{};
  testing::internal::CaptureStderr();
  const auto oversized = pool.allocate(65, MemoryStorageType::kSystem);
  const auto exhausted = pool.allocate(1, MemoryStorageType::kSystem);
  // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
  const auto invalid = allocator.free(&foreign);
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  EXPECT_FALSE(oversized);
  EXPECT_FALSE(exhausted);
  EXPECT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(pool.free(*block));
  auto reused = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(reused);
  EXPECT_EQ(*reused, *block);
  EXPECT_TRUE(pool.free(*reused));
}

// REQ: EXPECTED-033
TEST_F(DiagnosticLogging, FormattingFailureIsContainedAndFilteredReportsDoNotAllocate) {
  const std::array diagnostics{Diagnostic{.token = std::string(240, 'x')}};
  {
    const test::ScopedAllocationExhaustion exhaustion{0};
    log_diagnostics(diagnostics, ExecutionGraphView{});
  }
  set_log_level(LogLevel::OFF);
  std::size_t allocations = 0;
  {
    const test::ScopedAllocationCounter counter;
    log_diagnostics(diagnostics, ExecutionGraphView{});
    allocations = counter.count();
  }
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(diagnostics.front().token.size(), 240U);
}

void automatic_location_probe(int& evaluated) {
  HOLOSCAN_LOG_AUTOMATIC_ERROR("LAZY {}", ++evaluated);
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, AutomaticArgumentsAreLazyAndKeepCallerLocation) {
  int evaluated = 0;
  set_automatic_diagnostic_logging_enabled(false);
  automatic_location_probe(evaluated);
  EXPECT_EQ(evaluated, 0);
  set_automatic_diagnostic_logging_enabled(true);
  set_log_level(LogLevel::OFF);
  automatic_location_probe(evaluated);
  EXPECT_EQ(evaluated, 0);
  set_log_level(LogLevel::ERROR);
  set_log_pattern("%!: %v");
  testing::internal::CaptureStderr();
  automatic_location_probe(evaluated);
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(evaluated, 1);
  EXPECT_NE(output.find("automatic_location_probe"), std::string::npos);
  EXPECT_EQ(output.find("operator()"), std::string::npos);
  EXPECT_NO_THROW(detail::report_automatic_diagnostic(
      LogLevel::ERROR, [] { throw std::runtime_error{"presentation failure"}; }));
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, ReenableAffectsFutureAttemptsWithoutReplay) {
  ManualClock clock;
  set_automatic_diagnostic_logging_enabled(false);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(clock.sleep_for(-1).has_value());
  set_automatic_diagnostic_logging_enabled(true);
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  testing::internal::CaptureStderr();
  EXPECT_FALSE(clock.sleep_for(-1).has_value());
  EXPECT_FALSE(testing::internal::GetCapturedStderr().empty());
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, ConcurrentPreferenceAccessAndReportsAreSafe) {
  std::array<std::thread, 4> threads;
  set_log_level(LogLevel::OFF);
  for (auto& thread : threads) {
    thread = std::thread([] {
      const std::array diagnostics{Diagnostic{.token = "CONCURRENT"}};
      for (int iteration = 0; iteration < 1000; ++iteration) {
        set_automatic_diagnostic_logging_enabled(iteration % 2 == 0);
        static_cast<void>(automatic_diagnostic_logging_enabled());
        log_diagnostics(diagnostics, ExecutionGraphView{});
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  set_automatic_diagnostic_logging_enabled(false);
  EXPECT_FALSE(automatic_diagnostic_logging_enabled());
}

// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, InvalidEnvironmentWarnsOnceAndEmptyUsesPreference) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto* value : {"invalid", "", " ON ", "1"}) {
    EXPECT_EXIT(
        {
          test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", value);
          const test::ScopedDiagnosticLogger logger{LogLevel::WARN};
          testing::internal::CaptureStderr();
          set_automatic_diagnostic_logging_enabled(false);
          if (automatic_diagnostic_logging_enabled())
            std::exit(1);
          set_automatic_diagnostic_logging_enabled(true);
          if (!automatic_diagnostic_logging_enabled())
            std::exit(2);
          const auto output = testing::internal::GetCapturedStderr();
          const auto first = output.find("Ignoring invalid");
          const bool invalid = *value != '\0';
          if ((first != std::string::npos) != invalid)
            std::exit(3);
          if (invalid && output.find("Ignoring invalid", first + 1) != std::string::npos)
            std::exit(4);
          std::exit(0);
        },
        testing::ExitedWithCode(0),
        "");
  }
}

// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, DefaultToggleAndEnvironmentSnapshot) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", nullptr);
        if (!automatic_diagnostic_logging_enabled())
          std::exit(1);
        set_automatic_diagnostic_logging_enabled(false);
        if (automatic_diagnostic_logging_enabled())
          std::exit(2);
        test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "OFF");
        set_automatic_diagnostic_logging_enabled(true);
        std::exit(automatic_diagnostic_logging_enabled() ? 0 : 3);
      },
      testing::ExitedWithCode(0),
      "");
}

// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, EnvironmentOverridesApi) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto* value : {"ON", "on", "oN", "OFF", "off", "oFf"}) {
    EXPECT_EXIT(
        {
          test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", value);
          const bool expected = std::string(value).size() == 2;
          set_automatic_diagnostic_logging_enabled(!expected);
          std::exit(automatic_diagnostic_logging_enabled() == expected ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        "");
  }
}

// REQ: EXPECTED-033
// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, ExplicitHelperDoesNotInitializePolicy) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "invalid");
        const test::ScopedDiagnosticLogger logger{LogLevel::ERROR};
        const std::array diagnostics{Diagnostic{.token = "EXPLICIT"}};
        testing::internal::CaptureStderr();
        log_diagnostics(diagnostics, ExecutionGraphView{});
        const auto output = testing::internal::GetCapturedStderr();
        const bool explicit_only = output.find("EXPLICIT") != std::string::npos &&
                                   output.find("Ignoring invalid") == std::string::npos;
        test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "OFF");
        std::exit(explicit_only && !automatic_diagnostic_logging_enabled() ? 0 : 1);
      },
      testing::ExitedWithCode(0),
      "");
}

class RejectingDiagnosticAllocator final : public Allocator {
 public:
  expected<void, Error> is_available(uint64_t, MemoryStorageType) override { return {}; }
  expected<std::byte*, Error> allocate(uint64_t, MemoryStorageType) override {
    ++allocation_attempts;
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "rejected"});
  }
  expected<void, Error> free(std::byte*) override { return {}; }
  int allocation_attempts{};
};

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, MemoryBufferFailureReportsFollowPolicyAndPreserveOwnership) {
  auto allocator = std::make_shared<RejectingDiagnosticAllocator>();
  for (const bool enabled : {false, true}) {
    set_automatic_diagnostic_logging_enabled(enabled);
    MemoryBuffer buffer;
    testing::internal::CaptureStderr();
    const auto allocated = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
    const auto allocation_output = testing::internal::GetCapturedStderr();
    ASSERT_FALSE(allocated);
    EXPECT_EQ(allocated.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(allocated.error().message, "rejected");
    EXPECT_EQ(buffer.pointer(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_EQ(allocation_output.find("Native allocator failed") != std::string::npos, enabled);

    std::byte backing{};
    bool reject_release = true;
    ASSERT_TRUE(buffer.wrap_memory(
        &backing, 1, MemoryStorageType::kSystem, [&](void*) -> expected<void, Error> {
          if (reject_release) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument, "retained"});
          }
          return {};
        }));
    const auto attempts = allocator->allocation_attempts;
    testing::internal::CaptureStderr();
    const auto released = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
    const auto release_output = testing::internal::GetCapturedStderr();
    ASSERT_FALSE(released);
    EXPECT_EQ(released.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(released.error().message, "retained");
    EXPECT_EQ(buffer.pointer(), &backing);
    EXPECT_EQ(buffer.size(), 1U);
    EXPECT_EQ(allocator->allocation_attempts, attempts);
    EXPECT_EQ(release_output.find("Failed to free memory") != std::string::npos, enabled);
    reject_release = false;
    EXPECT_TRUE(buffer.free_buffer());
  }
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, MemoryBufferSuppressedReportsDoNotFormat) {
  auto allocator = std::make_shared<RejectingDiagnosticAllocator>();
  MemoryBuffer buffer;
  std::byte backing{};
  for (const bool release_failure : {false, true}) {
    bool reject_release = true;
    if (release_failure) {
      ASSERT_TRUE(buffer.wrap_memory(
          &backing, 1, MemoryStorageType::kSystem, [&](void*) -> expected<void, Error> {
            return reject_release ? expected<void, Error>{make_unexpected(
                                        Error{ErrorCode::kInvalidArgument, "retained"})}
                                  : expected<void, Error>{};
          }));
    }
    for (const bool policy_enabled : {false, true}) {
      set_automatic_diagnostic_logging_enabled(policy_enabled);
      set_log_level(policy_enabled ? LogLevel::OFF : LogLevel::ERROR);
      std::size_t allocations{};
      expected<void, Error> result;
      {
        const test::ScopedAllocationCounter counter;
        result = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
        allocations = counter.count();
      }
      EXPECT_EQ(allocations, 0U);
      ASSERT_FALSE(result);
      EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
      EXPECT_EQ(result.error().message, release_failure ? "retained" : "rejected");
    }
    reject_release = false;
    ASSERT_TRUE(buffer.free_buffer());
  }
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, MemoryBufferFormattingFailureCannotReplaceReturnedError) {
  auto allocator = std::make_shared<RejectingDiagnosticAllocator>();
  set_automatic_diagnostic_logging_enabled(true);
  set_log_level(LogLevel::ERROR);
  MemoryBuffer buffer;
  std::byte backing{};
  for (const bool release_failure : {false, true}) {
    bool reject_release = true;
    if (release_failure) {
      ASSERT_TRUE(buffer.wrap_memory(
          &backing, 1, MemoryStorageType::kSystem, [&](void*) -> expected<void, Error> {
            if (reject_release) {
              return make_unexpected(Error{ErrorCode::kInvalidArgument, "retained"});
            }
            return {};
          }));
    }
    expected<void, Error> result;
    bool threw = false;
    {
      const test::ScopedAllocationExhaustion exhaustion{0};
      try {
        result = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
      } catch (...) {
        threw = true;
      }
    }
    EXPECT_FALSE(threw);
    if (!threw) {
      ASSERT_FALSE(result);
      EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
      EXPECT_EQ(result.error().message, release_failure ? "retained" : "rejected");
    }
    reject_release = false;
    ASSERT_TRUE(buffer.free_buffer());
  }
}

// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, MemoryBufferEnvironmentOverridesApiAcrossBothFailureSites) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto* value : {"ON", "OFF"}) {
    EXPECT_EXIT(
        {
          test::set_test_environment("HOLOSCAN_LOG_LEVEL", nullptr);
          test::set_test_environment("HOLOSCAN_LOG_FORMAT", nullptr);
          test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", value);
          const bool enabled = std::string_view(value) == "ON";
          set_automatic_diagnostic_logging_enabled(!enabled);
          set_log_level(LogLevel::ERROR);
          auto allocator = std::make_shared<RejectingDiagnosticAllocator>();
          MemoryBuffer buffer;
          testing::internal::CaptureStderr();
          const auto allocated = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
          const auto allocation_output = testing::internal::GetCapturedStderr();
          if (allocated || allocated.error().message != "rejected" ||
              (allocation_output.find("Native allocator failed") != std::string::npos) != enabled) {
            std::exit(1);
          }
          std::byte backing{};
          const auto wrapped = buffer.wrap_memory(
              &backing, 1, MemoryStorageType::kSystem, [](void*) -> expected<void, Error> {
                return make_unexpected(Error{ErrorCode::kInvalidArgument, "retained"});
              });
          if (!wrapped)
            std::exit(2);
          testing::internal::CaptureStderr();
          const auto released = buffer.resize(allocator, 4096, MemoryStorageType::kSystem);
          const auto release_output = testing::internal::GetCapturedStderr();
          if (released || released.error().message != "retained" || buffer.pointer() != &backing ||
              allocator->allocation_attempts != 1 ||
              (release_output.find("Failed to free memory") != std::string::npos) != enabled) {
            std::exit(3);
          }
          std::exit(0);
        },
        testing::ExitedWithCode(0),
        "");
  }
}

// REQ: EXPECTED-033
TEST_F(DiagnosticLogging, ExactErrorBudgetEmitsWarningMarkerWithoutLeadingNewline) {
  std::vector<Diagnostic> diagnostics;
  while (format_diagnostics(diagnostics, ExecutionGraphView{}).size() < 16050U) {
    diagnostics.push_back(Diagnostic{.token = std::string(100U, 'x')});
  }
  diagnostics.push_back(Diagnostic{.token = ""});
  const auto initial = format_diagnostics(diagnostics, ExecutionGraphView{}).size();
  ASSERT_LE(initial, 16288U);
  diagnostics.back().token.assign(16288U - initial, 'x');
  const auto errors = format_diagnostics(diagnostics, ExecutionGraphView{});
  ASSERT_EQ(errors.size(), 16288U);
  diagnostics.push_back(Diagnostic{.severity = DiagnosticSeverity::kWarning, .token = "warning"});
  const std::string marker = "[diagnostic report truncated; 1 records not fully rendered]";
  EXPECT_EQ(format_diagnostics(diagnostics, ExecutionGraphView{}), errors + "\n" + marker);
  testing::internal::CaptureStderr();
  log_diagnostics(diagnostics, ExecutionGraphView{});
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(output, "error: " + errors + "\nwarning: " + marker + "\n");
}

// REQ: EXPECTED-032
TEST_F(DiagnosticLogging, MemoryBufferReportsRetainInlineCallerLocation) {
  set_automatic_diagnostic_logging_enabled(true);
  set_log_pattern("%s %!: %v");
  MemoryBuffer buffer;
  testing::internal::CaptureStderr();
  const auto result = buffer.resize(
      std::make_shared<RejectingDiagnosticAllocator>(), 4096, MemoryStorageType::kSystem);
  const auto output = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(result);
  EXPECT_NE(output.find("memory_buffer.hpp"), std::string::npos);
  EXPECT_NE(output.find("MemoryBuffer::resize"), std::string::npos);
  EXPECT_EQ(output.find("report_allocation_failure"), std::string::npos);
}

// REQ: EXPECTED-032
TEST(DiagnosticPolicyDeathTest, InvalidEnvironmentFormattingFailureIsContained) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        test::set_test_environment("HOLOSCAN_LOG_LEVEL", nullptr);
        test::set_test_environment("HOLOSCAN_LOG_FORMAT", nullptr);
        test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "invalid");
        set_log_level(LogLevel::WARN);
        bool valid = false;
        {
          const test::ScopedAllocationExhaustion exhaustion{0};
          set_automatic_diagnostic_logging_enabled(false);
          valid = !automatic_diagnostic_logging_enabled();
          set_automatic_diagnostic_logging_enabled(true);
          valid = valid && automatic_diagnostic_logging_enabled();
        }
        std::_Exit(valid ? 0 : 1);
      },
      testing::ExitedWithCode(0),
      "");
}

}  // namespace
}  // namespace holoscan
