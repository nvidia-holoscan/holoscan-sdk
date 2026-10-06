/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

#include "channel/protected_terminal_channel.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_test_binding.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"

namespace holoscan::detail::runtime_channel_binding_test {

template <typename Callback>
class ScopeExit final {
 public:
  explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    if (active_) {
      callback_();
    }
  }

  void release() noexcept { active_ = false; }

 private:
  Callback callback_;
  bool active_{true};
};

template <typename PayloadT>
class EmptySource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output).max_emits_per_compute(8U); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<PayloadT> output{"output"};
};

template <typename PayloadT>
class EmptySink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(8U); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<PayloadT> input{"input"};
};

using SourceOperator = EmptySource<std::int32_t>;
using SinkOperator = EmptySink<std::int32_t>;

// Script publication, refresh, and terminal outcomes without a backend race. For example,
// ExhaustedRequiredExportAbandonmentReturnsClosed removes the accepted recipient at refresh.
class ScriptedInlinePublisher final : public channel_internal::InlinePayloadPublisher {
 public:
  explicit ScriptedInlinePublisher(std::vector<channel_internal::InlinePublicationResult> results,
                                   std::vector<ErrorCode> refresh_results = {},
                                   std::vector<ErrorCode> close_results = {},
                                   bool throw_on_close = false);

  [[nodiscard]] channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte> payload, const SampleMetadata& metadata) override;
  [[nodiscard]] expected<void, Error> close_stream() override;
  [[nodiscard]] expected<void, Error> update_connections() noexcept override;
  [[nodiscard]] bool has_subscribers() const noexcept override;

  std::vector<std::uint64_t> sequences;
  std::vector<std::int32_t> values;
  std::vector<std::vector<std::byte>> payloads;
  std::vector<SampleMetadata> metadata_records;
  std::size_t close_count{};
  std::size_t refresh_count{};
  bool subscribers_present{true};
  bool remove_subscribers_on_refresh{};

 private:
  std::vector<channel_internal::InlinePublicationResult> results_;
  std::vector<ErrorCode> refresh_results_;
  std::vector<ErrorCode> close_results_;
  std::size_t next_result_{};
  std::size_t next_refresh_result_{};
  std::size_t next_close_result_{};
  bool throw_on_close_{};
};

enum class InlineReceivePath : std::uint8_t {
  kBorrowed,
  kCopyFallback,
};

/// Script one scalar data record followed by a stable closed transport head.
class ScriptedInlineSubscriber final : public channel_internal::InlinePayloadSubscriber,
                                       public channel_internal::StagedDataWaitSetSubscriber {
 public:
  ScriptedInlineSubscriber(std::int32_t value, std::uint64_t sequence, InlineReceivePath path);

  [[nodiscard]] expected<bool, Error> has_samples() const override;
  // This double is scripted in memory and has no transport, so it has no event lane and
  // no runtime to build a wait context from. Reporting that honestly makes
  // wait_for_readiness fail with kNotSupported rather than pretend to wait.
  [[nodiscard]] bool event_wake_enabled() const noexcept override;
  [[nodiscard]] void* wait_context_runtime() noexcept override;
  [[nodiscard]] bool has_publishers() const noexcept override;
  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveResult, Error> receive(
      std::span<std::byte> payload) override;
  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveGuard, Error> receive_borrowed()
      override;
  [[nodiscard]] expected<channel_internal::InlinePayloadReceiveGuard, Error>
  receive_borrowed_if_data_accepted(bool accept_data) override;
  void release_payload_loan() noexcept override;
  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void*) override;
  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override;
  [[nodiscard]] std::optional<channel_internal::StagedDataState> staged_data_state()
      const noexcept override;
  [[nodiscard]] std::optional<std::uint64_t> staged_terminal_sequence() const noexcept override;

  std::size_t borrowed_receive_count{};
  std::size_t copy_receive_count{};
  std::size_t loan_release_count{};
  mutable std::size_t staged_data_query_count{};

 private:
  std::array<std::byte, sizeof(std::int32_t)> payload_{};
  SampleMetadata metadata_{};
  InlineReceivePath path_{};
  bool data_delivered_{};
  bool loan_active_{};
};

[[nodiscard]] std::string unique_service();

struct SessionFixture {
  SourceOperator source;
  SinkOperator sink;
  SinkOperator second_sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  OperatorSpec second_sink_spec;
  ExecutionPlanState plan;
  std::array<RuntimeChannelPortReplayRecord, 3U> replay;

  SessionFixture();

  [[nodiscard]] PortPlanRecord& planned_port(PortId id);
  [[nodiscard]] ProcessLocalChannelServicePlanRecord& planned_service(std::string_view service_id);
  [[nodiscard]] ProcessLocalChannelPublisherPlanRecord& planned_publisher(
      std::string_view service_id);
  [[nodiscard]] ProcessLocalChannelSubscriberPlanRecord& planned_subscriber(std::size_t index);
  [[nodiscard]] PayloadEdgePlanRecord& planned_edge(EdgeId id);
  void expect_queue_depth(std::size_t subscriber_index, std::uint32_t depth);
  void set_queue_depth(std::size_t subscriber_index, std::uint32_t depth);
  void set_primary_queue_policy(QueuePolicy policy);
  void use_direct_retained_delivery();
  void use_one_slot_direct_retained_delivery();
  void use_one_slot_blocking_transport();
  void use_same_partition_eventless_transport();
  void use_single_partition_runner();
  void use_cross_process_edge();
  void add_mixed_policy_fanout();
  void add_blocking_fanout();
  void use_one_slot_blocking_fanout();
  void use_one_slot_drop_oldest_fanout();
  void add_compute_only_input_for_sink();
  void use_drop_oldest_queue_policy();
  void use_one_slot_drop_oldest_queue();
  void use_two_slot_drop_oldest_queue();
  void use_two_slot_blocking_queue();
  void require_direct_capture_conversion();
  void enable_reuse_last();
  void enable_scalar(InputSelectorKind selector, TemporalRecordMissingKind missing);
  void enable_latest_within();
  void enable_drain_all(std::uint32_t max_selected_samples);
  void enable_bracketed_window(std::uint32_t max_selected_samples);
};

[[nodiscard]] expected<RuntimeEvent, Error> wait_for_event(RuntimeChannelSession& session,
                                                           PortBinding& binding,
                                                           ClockId observed_clock = ClockId{1U});
[[nodiscard]] expected<RuntimeChannelInputResetState, Error> wait_for_occupied_sample(
    RuntimeChannelSession& session, PortId input);

struct PublishedScalarScript {
  std::vector<RuntimeEvent> events;
  std::vector<TemporalSampleRecord> records;
};

[[nodiscard]] expected<PublishedScalarScript, Error> publish_scalar_script(
    RuntimeChannelSession& session, PortBinding& binding, SourceOperator& source,
    std::span<const std::int32_t> values, std::span<const std::int64_t> capture_times);
[[nodiscard]] std::vector<std::int32_t> batch_values(const SelectedBatchView<std::int32_t>& batch);

enum class InputAccessorKind : std::uint8_t {
  kSample,
  kData,
  kBatch,
};

enum class ChannelGuardApplicability : std::uint8_t {
  kAccessor,
  kSyntheticClosedOnly,
  kSyntheticBackendOnly,
  kSessionBackpressureOnly,
};

struct InputGuardExpectation {
  std::string_view name;
  InputAccessorKind accessor;
  SyntheticInputState state;
  ErrorCode expected;
  ChannelGuardApplicability channel;
  bool unbound{};
  bool exhaust_claims{};
};

[[nodiscard]] ErrorCode observe_input_accessor(Input<std::int32_t>& input,
                                               InputAccessorKind accessor);
[[nodiscard]] ErrorCode observe_exhausted_scalar_claims(Input<std::int32_t>& input);
[[nodiscard]] expected<ErrorCode, Error> observe_channel_guard(
    const InputGuardExpectation& expectation);

}  // namespace holoscan::detail::runtime_channel_binding_test
