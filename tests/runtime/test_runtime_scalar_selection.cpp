/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "core/source_id_internal.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {

using namespace runtime_channel_binding_test;

struct ScalarWitnessProbe {
  std::array<RuntimeScalarReadBinding, 2U> captured{};
  std::size_t constructed{};
  std::size_t destroyed{};
  std::size_t fail_construction{};
  bool all_expired_before_destruction{true};
};

// Predicate construction sees the same epoch as the scalar binding. Observe that epoch during
// destruction, not merely after retirement returns, without adding a production hot-path hook.
thread_local ScalarWitnessProbe* witness_probe{};

class RuntimeScalarSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fixture.use_one_slot_drop_oldest_queue();
    fixture.add_compute_only_input_for_sink();
    fixture.set_queue_depth(1U, 1U);
    fixture.use_direct_retained_delivery();
    fixture.plan.process_local_channel_subscribers[1].event_wake = false;

    // Direct retained delivery admits 1:1 edges, not fanout. Give the second input its own source
    // while retaining both inputs under one activation owner for all-or-nothing invalidation.
    second_source.setup(second_source_spec);
    auto source_port = fixture.plan.ports.front();
    source_port.id = PortId{4U};
    source_port.owner = VertexId{3U};
    fixture.plan.ports.push_back(source_port);
    fixture.plan.process_local_channel_services.front().max_subscribers = 1U;
    auto service = fixture.plan.process_local_channel_services.front();
    service.source = PortId{4U};
    service.service_id = "process-local-channel." + std::to_string(fixture.plan.plan_generation) +
                         ".source.4.policy." +
                         std::to_string(static_cast<std::uint32_t>(QueuePolicy::kDropOldest));
    auto publisher = fixture.plan.process_local_channel_publishers.front();
    publisher.source = service.source;
    publisher.service_id = service.service_id;
    auto& subscriber = fixture.plan.process_local_channel_subscribers[1];
    subscriber.source = service.source;
    subscriber.service_id = service.service_id;
    auto& edge = fixture.planned_edge(EdgeId{2U});
    edge.source = service.source;
    edge.fanout = 1U;
    edge.queue_policy = QueuePolicy::kDropOldest;
    fixture.plan.process_local_channel_services.push_back(service);
    fixture.plan.process_local_channel_publishers.push_back(publisher);

    witness_probe = &probe;
    ops = *runtime_port_type_ops<std::int32_t>();
    ops.construct_predicate_view =
        [](void* storage,
           const void* sample,
           InputSelection selection,
           const std::uint64_t* epoch,
           std::uint64_t expected_epoch) noexcept -> expected<void, Error> {
      auto& observed = *witness_probe;
      observed.captured[observed.constructed % observed.captured.size()] = {
          .sample = sample,
          .view_generation = epoch,
          .expected_view_generation = expected_epoch,
      };
      ++observed.constructed;
      if (observed.constructed == observed.fail_construction) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      return runtime_port_type_ops<std::int32_t>()->construct_predicate_view(
          storage, sample, selection, epoch, expected_epoch);
    };
    ops.destroy_predicate_view = [](void* storage) noexcept {
      auto& observed = *witness_probe;
      for (const auto& witness : observed.captured) {
        if (witness.view_generation != nullptr &&
            *witness.view_generation == witness.expected_view_generation) {
          observed.all_expired_before_destruction = false;
        }
      }
      ++observed.destroyed;
      runtime_port_type_ops<std::int32_t>()->destroy_predicate_view(storage);
    };
    fixture.replay[1].type_ops = &ops;
    fixture.replay[2].type_ops = &ops;
    const std::array replay{
        fixture.replay[0],
        fixture.replay[1],
        fixture.replay[2],
        RuntimeChannelPortReplayRecord{.port = PortId{4U},
                                       .owner = VertexId{3U},
                                       .direction = PortDirection::kOutput,
                                       .declaration = &second_source.output,
                                       .type_ops = runtime_port_type_ops<std::int32_t>()},
    };
    auto created = RuntimeChannelSession::create(fixture.plan, replay);
    ASSERT_TRUE(created) << runtime_error_message(created.error());
    session = std::move(*created);
    auto ready =
        session->create_readiness_binding(VertexId{2U},
                                          RuntimePortBindingRequest{
                                              .plan_generation = fixture.plan.plan_generation,
                                              .run_generation = 1U,
                                              .kind = RuntimeEventKind::kInputArrival,
                                              .source = make_source_id(1U),
                                              .source_clock = ClockId{1U},
                                              .reference_clock = ClockId{1U},
                                              .capacity = 2U,
                                          });
    ASSERT_TRUE(ready);
    binding = std::move(*ready);
    ASSERT_TRUE(session->refresh_connections());
    ASSERT_TRUE(session->prepare_activation(VertexId{1U}, 1U, {}));
    ASSERT_TRUE(fixture.source.output.emit(std::int32_t{42}));
    ASSERT_TRUE(session->retire_activation(VertexId{1U}, 1U, true));
    ASSERT_TRUE(session->prepare_activation(VertexId{3U}, 1U, {}));
    ASSERT_TRUE(second_source.output.emit(std::int32_t{42}));
    ASSERT_TRUE(session->retire_activation(VertexId{3U}, 1U, true));
    for (std::size_t index = 0U; index < records.size(); ++index) {
      auto event = wait_for_event(*session, *binding);
      ASSERT_TRUE(event);
      auto record = session->temporal_sample(event->port, event->source_sequence);
      ASSERT_TRUE(record);
      records[event->port.value() - 2U] = *record;
      ASSERT_TRUE(binding->retire(event->source_sequence));
    }
    for (std::size_t index = 0U; index < selections.size(); ++index) {
      selections[index] = {
          .port = PortId{index + 2U},
          .selection = {InputState::kSelected, SelectionReason::kNone},
          .samples = std::span<const TemporalSampleRecord>{records}.subspan(index, 1U),
          .destructive = true,
      };
    }
  }

  void TearDown() override {
    binding.reset();
    // The witness contract ends with the session; do not inspect one input after another input's
    // owner has already been destroyed during teardown of a partially retired activation.
    probe.captured = {};
    session.reset();
    witness_probe = nullptr;
  }

  void expect_current(const RuntimeScalarReadBinding& witness) {
    ASSERT_NE(witness.sample, nullptr);
    ASSERT_NE(witness.view_generation, nullptr);
    EXPECT_NE(witness.expected_view_generation, 0U);
    EXPECT_EQ(*witness.view_generation, witness.expected_view_generation);
    EXPECT_EQ(static_cast<const Sample<std::int32_t>*>(witness.sample)->data, 42);
    EXPECT_EQ(witness.claim.state, nullptr);
    EXPECT_EQ(witness.claim.retain, nullptr);
    EXPECT_EQ(witness.claim.release, nullptr);
  }

  SessionFixture fixture;
  SourceOperator second_source;
  OperatorSpec second_source_spec;
  ScalarWitnessProbe probe;
  RuntimePortTypeOps ops;
  std::unique_ptr<RuntimeChannelSession> session;
  std::unique_ptr<PortBinding> binding;
  std::array<TemporalSampleRecord, 2U> records{};
  std::array<RuntimeChannelActivationSelection, 2U> selections{};
};

// REQ: OPERATOR-055
// REQ: OPERATOR-056
TEST_F(RuntimeScalarSelectionTest, ExpiresBeforeCleanupAndStaysExpiredAcrossCutovers) {
  // Deliberately use an admission ID different from the retirement epoch.
  ASSERT_TRUE(session->prepare_activation(VertexId{2U}, 73U, selections));
  auto witness = RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, PortId{2U});
  ASSERT_TRUE(witness);
  expect_current(*witness);
  EXPECT_NE(witness->expected_view_generation, 73U);
  const auto borrowed = fixture.sink.input.selected_sample();
  ASSERT_TRUE(borrowed);
  EXPECT_EQ(&borrowed->sample(), witness->sample);
  EXPECT_EQ(borrowed->data(), 42);
  ASSERT_TRUE(session->retire_activation(VertexId{2U}, 73U, true));
  EXPECT_NE(*witness->view_generation, witness->expected_view_generation);
  EXPECT_EQ(probe.destroyed, 2U);
  EXPECT_TRUE(probe.all_expired_before_destruction);
#if GTEST_HAS_DEATH_TEST
  const auto view = *borrowed;
  EXPECT_DEATH((void)view.sample(), "");
  EXPECT_DEATH((void)view.data(), "");
  EXPECT_DEATH((void)view.metadata(), "");
  EXPECT_DEATH((void)*view, "");
  EXPECT_DEATH((void)view.operator->(), "");
#endif
  for (const auto& selected : selections) {
    ASSERT_TRUE(session->retire_samples(selected.port, selected.samples));
  }
  const std::array affected{VertexId{1U}, VertexId{2U}};
  ASSERT_TRUE(session->reset());
  EXPECT_NE(*witness->view_generation, witness->expected_view_generation);
  ASSERT_TRUE(session->reset(VertexId{1U}));
  EXPECT_NE(*witness->view_generation, witness->expected_view_generation);
  ASSERT_TRUE(session->reset(VertexId{2U}));
  EXPECT_NE(*witness->view_generation, witness->expected_view_generation);
  ASSERT_TRUE(session->commit_restart(VertexId{1U}, affected));
  EXPECT_NE(*witness->view_generation, witness->expected_view_generation);
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH((void)view.sample(), "");
  EXPECT_DEATH((void)view.data(), "");
  EXPECT_DEATH((void)view.metadata(), "");
  EXPECT_DEATH((void)*view, "");
  EXPECT_DEATH((void)view.operator->(), "");
#endif
}

// REQ: OPERATOR-055
TEST_F(RuntimeScalarSelectionTest, PartialRetirementFailureExpiresEveryInput) {
  // The next publication is retained in direct ingress while the selected typed slots stay
  // protected. Exhaust the global arrival counter before activation, using the existing cold seam.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_arrival_counters(
      *session, PortId{2U}, 2U, std::numeric_limits<std::uint64_t>::max()));
  ASSERT_TRUE(session->prepare_activation(VertexId{2U}, 73U, selections));
  auto first = RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, PortId{2U});
  auto second = RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, PortId{3U});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  expect_current(*first);
  expect_current(*second);
  ASSERT_TRUE(session->prepare_activation(VertexId{1U}, 2U, {}));
  ASSERT_TRUE(fixture.source.output.emit(std::int32_t{43}));
  ASSERT_TRUE(session->retire_activation(VertexId{1U}, 2U, true));
  for (const auto& selected : selections) {
    ASSERT_TRUE(session->retire_samples(selected.port, selected.samples));
  }
  const auto retired = session->retire_activation(VertexId{2U}, 73U, true);
  ASSERT_FALSE(retired);
  EXPECT_EQ(retired.error().code, ErrorCode::kResourceExhausted);
  EXPECT_NE(*first->view_generation, first->expected_view_generation);
  EXPECT_NE(*second->view_generation, second->expected_view_generation);
  EXPECT_EQ(probe.destroyed, 1U);
  EXPECT_TRUE(probe.all_expired_before_destruction);
}

// REQ: OPERATOR-055
TEST_F(RuntimeScalarSelectionTest, FailedPreparationInvalidatesBeforePredicateDestruction) {
  probe.fail_construction = 2U;
  auto rejected = session->prepare_activation(VertexId{2U}, 73U, selections);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kFailure);
  EXPECT_EQ(probe.destroyed, 1U);
  EXPECT_TRUE(probe.all_expired_before_destruction);
  for (const auto& witness : probe.captured) {
    ASSERT_NE(witness.view_generation, nullptr);
    EXPECT_NE(*witness.view_generation, witness.expected_view_generation);
  }
  ASSERT_TRUE(session->prepare_activation(VertexId{2U}, 73U, selections));
  auto witness = RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, PortId{2U});
  ASSERT_TRUE(witness);
  expect_current(*witness);
  ASSERT_TRUE(session->retire_activation(VertexId{2U}, 73U, false));
}

// REQ: OPERATOR-055
TEST_F(RuntimeScalarSelectionTest, IncrementalRollbackPreservesPreviouslyCommittedSelection) {
  ASSERT_TRUE(session->prepare_activation(
      VertexId{2U}, 73U, std::span<const RuntimeChannelActivationSelection>{selections}.first(1U)));
  auto first = RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, PortId{2U});
  ASSERT_TRUE(first);
  probe.fail_construction = 2U;
  auto rejected = session->prepare_activation(
      VertexId{2U},
      73U,
      std::span<const RuntimeChannelActivationSelection>{selections}.subspan(1U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kFailure);
  expect_current(*first);
  const auto rolled_back = probe.captured[1U];
  EXPECT_NE(*rolled_back.view_generation, rolled_back.expected_view_generation);
  ASSERT_TRUE(session->retire_activation(VertexId{2U}, 73U, false));
  EXPECT_NE(*first->view_generation, first->expected_view_generation);
}

// REQ: OPERATOR-055
TEST_F(RuntimeScalarSelectionTest, ExhaustedEpochRejectsPreparationWithoutMutation) {
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::set_input_reset_counters(
      *session, PortId{3U}, 1U, std::numeric_limits<std::uint64_t>::max(), 1U));
  auto rejected = session->prepare_activation(VertexId{2U}, 73U, selections);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(probe.constructed, 0U);
  for (const auto& selected : selections) {
    auto state = RuntimeChannelSessionTestAccess::input_reset_state(*session, selected.port);
    ASSERT_TRUE(state);
    EXPECT_EQ(state->active_generation, 1U);
    const auto scalar =
        RuntimeChannelSessionTestAccess::selected_scalar_binding(*session, selected.port);
    ASSERT_FALSE(scalar);
    EXPECT_EQ(scalar.error().code, ErrorCode::kStaleEndpoint);
  }
}

}  // namespace
}  // namespace holoscan::detail
