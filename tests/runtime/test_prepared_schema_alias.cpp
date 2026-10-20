/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"

namespace holoscan::test {
namespace {
using namespace std::chrono_literals;
enum class AliasKind { kEscapedBorrow, kOwningReceive, kCallbackLocal };

struct AliasProbe {
  std::atomic<unsigned> received{};
  std::shared_ptr<schema::HeaderT> escaped;
};

class AliasSource final : public Operator<> {
 public:
  explicit AliasSource(std::shared_ptr<AliasProbe> probe) : probe_(std::move(probe)) {
    value_.header = std::make_shared<schema::HeaderT>();
    value_.header->frame_id.assign(128, 'a');
  }
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(1); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    const auto received = probe_->received.load();
    if (received == 2) {
      context.request_completion();
      return {};
    }
    if (sent_ != received)
      return {};
    value_.header->device_sequence = sent_ + 1;
    HOLOSCAN_TRY(output.emit_ref(value_));
    ++sent_;
    return {};
  }
  Output<schema::ImuT> output{"output"};

 private:
  std::shared_ptr<AliasProbe> probe_;
  schema::ImuT value_;
  unsigned sent_{};
};

class AliasSink final : public Operator<> {
 public:
  AliasSink(std::shared_ptr<AliasProbe> probe, AliasKind kind)
      : probe_(std::move(probe)), kind_(kind) {}
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(1); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    if (kind_ == AliasKind::kOwningReceive) {
      HOLOSCAN_TRY(auto sample, input.receive());
      return observe(sample.data);
    }
    HOLOSCAN_TRY(auto view, input.selected_sample());
    return observe(view.data());
  }
  Input<schema::ImuT> input{"input"};

 private:
  expected<void, Error> observe(const schema::ImuT& value) {
    auto header = value.header;
    if (!header || header->device_sequence != probe_->received.load() + 1)
      return make_unexpected(Error{ErrorCode::kFailure});
    if (kind_ != AliasKind::kCallbackLocal && probe_->received.load() == 0)
      probe_->escaped = header;
    probe_->received.fetch_add(1);
    return {};  // The callback-local header copy dies before Runtime retires the selection.
  }
  std::shared_ptr<AliasProbe> probe_;
  AliasKind kind_;
};

class PreparedSchemaAlias : public ::testing::TestWithParam<std::tuple<AliasKind, bool>> {};

// REQ: SCHEMA-009
TEST_P(PreparedSchemaAlias, EscapedMetadataReportsLocalStorageViolationNotWireFailure) {
  const auto [kind, transport] = GetParam();
  auto probe = std::make_shared<AliasProbe>();
  Graph graph{"prepared-alias-diagnostic"};
  auto source = graph.op<AliasSource>("source", probe);
  auto sink = graph.op<AliasSink>("sink", probe, kind);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1});
  // Here the shared serial partition is intentional: the source may inspect the sink's count
  // only after the sink selection has retired, not while its callback still owns the sample.
  graph.partition("serial").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan =
      transport ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.partitions().size(), 1U);
  auto session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 15s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  const bool finished = !session.is_running();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(finished) << "Prepared alias graph did not reach a terminal result";
  if (kind == AliasKind::kEscapedBorrow) {
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_EQ(probe->received.load(), 1U);
    bool diagnosed{};
    for (const auto& diagnostic : session.diagnostics()) {
      if (!diagnostic.detail)
        continue;
      EXPECT_NE(diagnostic.detail->error.code, ErrorCode::kProtocolError);
      diagnosed |= diagnostic.detail->error.code == ErrorCode::kPreparedStorageInvalid;
    }
    EXPECT_TRUE(diagnosed) << "Expected the local prepared-storage cause in Runtime diagnostics";
  } else {
    EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
    EXPECT_EQ(probe->received.load(), 2U);
    EXPECT_TRUE(session.diagnostics().empty());
  }
  if (kind != AliasKind::kCallbackLocal) {
    ASSERT_TRUE(probe->escaped);
    EXPECT_EQ(probe->escaped->device_sequence, 1U);
    EXPECT_EQ(probe->escaped->frame_id, std::string(128, 'a'));
  }
}

std::string alias_case_name(const ::testing::TestParamInfo<PreparedSchemaAlias::ParamType>& info) {
  const auto [kind, transport] = info.param;
  return std::string(kind == AliasKind::kEscapedBorrow   ? "EscapedBorrow"
                     : kind == AliasKind::kOwningReceive ? "OwningReceive"
                                                         : "CallbackLocal") +
         (transport ? "HoloMQ" : "Direct");
}

INSTANTIATE_TEST_SUITE_P(PreparedPaths, PreparedSchemaAlias,
                         ::testing::Combine(::testing::Values(AliasKind::kEscapedBorrow,
                                                              AliasKind::kOwningReceive,
                                                              AliasKind::kCallbackLocal),
                                            ::testing::Bool()),
                         alias_case_name);
}  // namespace
}  // namespace holoscan::test
