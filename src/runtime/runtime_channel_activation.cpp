/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_channel_session_impl.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <holoscan/profiler/nvtx3.hpp>

namespace holoscan::detail {

// Input selection, activation preparation, claim publication, and exact retirement.

RuntimeChannelSession::Impl::InputPortState::~InputPortState() {
  if (predicate_view_constructed) {
    type_ops->destroy_predicate_view(predicate_view_storage.data());
  }
  if (claim_pool != nullptr) {
    claim_pool->release_owner();
  }
}

[[nodiscard]] std::uint64_t
RuntimeChannelSession::Impl::InputPortState::publish_readiness_change() noexcept {
  const std::uint64_t published = readiness_revision.fetch_add(1U, std::memory_order_release) + 1U;
  count_phase(phase_counters, RuntimePhaseCounter::kChannelReadinessPublished);
  return published;
}

[[nodiscard]] InputSelection RuntimeChannelSession::Impl::InputPortState::selection(
    const void* opaque, std::uint64_t generation) noexcept {
  const auto& state = *static_cast<const InputPortState*>(opaque);
  if (!state.active || generation != state.active_generation) {
    return {InputState::kAbsent, SelectionReason::kNeverSeen};
  }
  return state.active_selection;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::InputPortState::classify(
    const RuntimeChannelSession::Impl::InputPortState& state, std::uint64_t generation,
    bool batch_accessor) noexcept {
  if (!state.active || generation != state.active_generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (state.active_selection.state == InputState::kNotSelected) {
    return make_unexpected(Error{ErrorCode::kInputNotSelected});
  }
  const bool empty_state = state.active_selection.state == InputState::kAbsent ||
                           state.active_selection.state == InputState::kSafeDefault;
  if (state.batch != batch_accessor) {
    return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
  }
  if (empty_state || (!batch_accessor && state.active_slots.empty())) {
    return make_unexpected(Error{ErrorCode::kInputEmpty});
  }
  return {};
}

[[nodiscard]] expected<RuntimeScalarReadBinding, Error>
RuntimeChannelSession::Impl::InputPortState::scalar(const void* opaque, std::uint64_t generation,
                                                    bool acquire_claim) noexcept {
  auto& state = *static_cast<InputPortState*>(const_cast<void*>(opaque));
  auto classified = classify(state, generation, false);
  if (!classified) {
    return make_unexpected(std::move(classified).error());
  }
  if (state.active_slots.size() != 1U) {
    return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
  }
  PublishedReadClaimBinding claim{};
  if (acquire_claim) {
    auto acquired = state.claim_pool->acquire();
    if (!acquired) {
      return make_unexpected(std::move(acquired).error());
    }
    claim = *acquired;
  }
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelInputReads);
  const SampleSlot* const slot = state.active_slots.front();
  if (slot->subscriber->receive_nvtx_message != nullptr &&
      slot->subscriber->source_identity_nvtx_string != nullptr) {
    holoscan::profiler::EmitReceiveNVTXPayload payload{
        .source_identity = slot->subscriber->source_identity_nvtx_string,
        .sequence = slot->temporal.source_sequence,
    };
    const nvtx3::payload_data payload_data{payload};
    nvtx3::mark_in<holoscan::profiler::holoscan_operator_domain>(
        nvtx3::message{slot->subscriber->receive_nvtx_message}, payload_data);
  }
  return RuntimeScalarReadBinding{
      .sample = slot->sample.data(),
      .claim = claim,
  };
}

[[nodiscard]] expected<RuntimeBatchReadBinding, Error>
RuntimeChannelSession::Impl::InputPortState::batch_read(const void* opaque,
                                                        std::uint64_t generation) noexcept {
  const auto& state = *static_cast<const InputPortState*>(opaque);
  auto classified = classify(state, generation, true);
  if (!classified) {
    return make_unexpected(std::move(classified).error());
  }
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelInputReads);
  if (state.batch_receive_nvtx_enabled) {
    for (const SampleSlot* const slot : state.active_slots) {
      if (slot->subscriber->receive_nvtx_message != nullptr &&
          slot->subscriber->source_identity_nvtx_string != nullptr) {
        holoscan::profiler::EmitReceiveNVTXPayload payload{
            .source_identity = slot->subscriber->source_identity_nvtx_string,
            .sequence = slot->temporal.source_sequence,
        };
        const nvtx3::payload_data payload_data{payload};
        nvtx3::mark_in<holoscan::profiler::holoscan_operator_domain>(
            nvtx3::message{slot->subscriber->receive_nvtx_message}, payload_data);
      }
    }
  }
  return RuntimeBatchReadBinding{
      .state = &state,
      .size = state.active_slots.size(),
      .active_generation = &state.active_generation,
      .expected_generation = generation,
      .sample_at = [](const void* batch_state, std::size_t index) noexcept -> const void* {
        const auto& input = *static_cast<const InputPortState*>(batch_state);
        return index < input.active_slots.size() ? input.active_slots[index]->sample.data()
                                                 : nullptr;
      },
  };
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::bind_activation_owners(
    std::size_t port_count) {
  if (!activation_owners_.empty() || !input_by_port_.empty() || port_count == 0U ||
      std::any_of(
          inputs_.begin(), inputs_.end(), [](const auto& input) { return !input->owner; }) ||
      std::any_of(subscribers_.begin(),
                  subscribers_.end(),
                  [](const auto& subscriber) {
                    return !subscriber->owner || subscriber->input == nullptr;
                  }) ||
      std::any_of(
          outputs_.begin(), outputs_.end(), [](const auto& output) { return !output->owner; })) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::uint64_t maximum_owner = 0U;
  for (const auto& input : inputs_) {
    maximum_owner = std::max(maximum_owner, input->owner.value);
  }
  for (const auto& output : outputs_) {
    maximum_owner = std::max(maximum_owner, output->owner.value);
  }
  if (maximum_owner > std::numeric_limits<std::size_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  activation_owners_.resize(static_cast<std::size_t>(maximum_owner));
  input_by_port_.resize(port_count);
  for (const auto& input : inputs_) {
    if (!input->port || input->port.value() > input_by_port_.size()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    InputPortState*& indexed = input_by_port_[static_cast<std::size_t>(input->port.value() - 1U)];
    if (indexed != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    indexed = input.get();
    activation_owners_[static_cast<std::size_t>(input->owner.value - 1U)].inputs.push_back(
        input.get());
  }
  for (const auto& subscriber : subscribers_) {
    activation_owners_[static_cast<std::size_t>(subscriber->owner.value - 1U)]
        .subscribers.push_back(subscriber.get());
  }
  for (const auto& output : outputs_) {
    activation_owners_[static_cast<std::size_t>(output->owner.value - 1U)].outputs.push_back(
        output.get());
  }
#if !defined(NDEBUG)
  owner_mutation_active_ =
      std::make_unique<std::atomic_flag[]>(static_cast<std::size_t>(maximum_owner));
  for (std::size_t index = 0U; index < static_cast<std::size_t>(maximum_owner); ++index) {
    owner_mutation_active_[index].clear(std::memory_order_relaxed);
  }
#endif
  return {};
}

[[nodiscard]] RuntimeChannelSession::Impl::ActivationOwnerState*
RuntimeChannelSession::Impl::find_activation_owner(VertexId owner) noexcept {
  if (!owner || owner.value > activation_owners_.size()) {
    return nullptr;
  }
  auto& state = activation_owners_[static_cast<std::size_t>(owner.value - 1U)];
  return state.inputs.empty() && state.outputs.empty() ? nullptr : &state;
}

[[nodiscard]] const RuntimeChannelSession::Impl::ActivationOwnerState*
RuntimeChannelSession::Impl::find_activation_owner(VertexId owner) const noexcept {
  if (!owner || owner.value > activation_owners_.size()) {
    return nullptr;
  }
  const auto& state = activation_owners_[static_cast<std::size_t>(owner.value - 1U)];
  return state.inputs.empty() && state.outputs.empty() ? nullptr : &state;
}

#if !defined(NDEBUG)
[[nodiscard]] std::atomic_flag& RuntimeChannelSession::Impl::owner_mutation_flag(
    VertexId owner) noexcept {
  assert(owner && owner.value <= activation_owners_.size());
  assert(owner_mutation_active_ != nullptr);
  return owner_mutation_active_[static_cast<std::size_t>(owner.value - 1U)];
}
#endif

void RuntimeChannelSession::Impl::cache_direct_continuation_candidate(
    const VertexId destination, ReadinessBinding& binding) noexcept {
  ActivationOwnerState* const destination_state = find_activation_owner(destination);
  if (destination_state == nullptr || destination_state->inputs.size() != 1U ||
      destination_state->subscribers.size() != 1U || destination_state->inputs.front() == nullptr ||
      destination_state->subscribers.front() == nullptr ||
      destination_state->subscribers.front()->input != destination_state->inputs.front() ||
      destination_state->inputs.front()->readiness_binding != &binding ||
      binding.polls_external_transport_ || binding.input_revisions_.size() != 1U ||
      binding.input_revisions_.front().input != destination_state->inputs.front()) {
    return;
  }

  SubscriberState& subscriber = *destination_state->subscribers.front();
  ActivationOwnerState* const producer_state = find_activation_owner(subscriber.source_owner);
  if (producer_state == nullptr || !subscriber.direct_process_local ||
      subscriber.cross_process_transport || subscriber.externally_driven_import ||
      subscriber.service.empty() ||
      (subscriber.queue_policy != QueuePolicy::kBlocking &&
       subscriber.queue_policy != QueuePolicy::kDropOldest)) {
    return;
  }

  OutputState* output = nullptr;
  for (OutputState* const candidate : producer_state->outputs) {
    if (candidate->direct_subscriber == nullptr) {
      continue;
    }
    if (output != nullptr) {
      return;
    }
    output = candidate;
  }
  if (output == nullptr || output->direct_subscriber != &subscriber ||
      output->owner != subscriber.source_owner || output->port != subscriber.source) {
    return;
  }
  producer_state->direct_continuation_output = output;
}

[[nodiscard]] std::optional<RuntimeChannelDirectContinuationTarget>
RuntimeChannelSession::Impl::direct_continuation_target(
    const VertexId producer, const std::uint32_t execution_lane, const bool require_clean_revision,
    const bool require_empty_binding) const noexcept {
  const ActivationOwnerState* const producer_state = find_activation_owner(producer);
  const OutputState* const output =
      producer_state == nullptr ? nullptr : producer_state->direct_continuation_output;
  if (output == nullptr || output->direct_subscriber == nullptr) {
    return std::nullopt;
  }
  const SubscriberState& subscriber = *output->direct_subscriber;
  const InputPortState* const input = subscriber.input;
  const ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
  if (subscriber.source_owner != producer || input == nullptr || binding == nullptr ||
      binding->closed() || binding->request_.execution_lane != execution_lane ||
      binding->input_revisions_.size() != 1U || binding->input_revisions_.front().input != input ||
      (require_empty_binding && binding->size_ != 0U) || binding->timeout_count_ != 0U ||
      binding->closure_count_ != 0U) {
    return std::nullopt;
  }
  const std::uint64_t current_revision = input->readiness_revision.load(std::memory_order_acquire);
  if (require_clean_revision && binding->input_revisions_.front().observed != current_revision) {
    return std::nullopt;
  }
  return RuntimeChannelDirectContinuationTarget{
      .plan_generation = plan_generation_,
      .run_generation = binding->request_.run_generation,
      .producer = producer,
      .destination = subscriber.owner,
      .source_output = output->port,
      .destination_input = subscriber.target,
      .service = subscriber.service,
      .readiness_source = binding->request_.source,
      .reference_clock = binding->request_.reference_clock,
      .source_generation = binding->request_.source_generation,
      .readiness_sequence = binding->next_source_sequence_,
      .destination_revision = current_revision,
      .observed_destination_revision = binding->input_revisions_.front().observed,
      .source_activation_generation = output->active_generation,
      .execution_lane = execution_lane,
      .readiness_capacity = binding->storage_.size(),
  };
}

void RuntimeChannelSession::Impl::note_direct_continuation_publication(
    OutputState& output, SubscriberState& subscriber, const std::uint64_t publication_sequence,
    const std::uint64_t prior_revision, const std::uint64_t destination_revision) noexcept {
  ActivationOwnerState* const owner = find_activation_owner(output.owner);
  if (owner == nullptr) {
    return;
  }
  if (!owner->direct_continuation) {
    owner->direct_continuation.emplace(ActivationOwnerState::DirectContinuationPublication{
        .output = &output,
        .subscriber = &subscriber,
        .publication_sequence = publication_sequence,
        .producer_epoch = subscriber.producer_epoch,
        .prior_destination_revision = prior_revision,
        .destination_revision = destination_revision,
        .source_activation_generation = output.active_generation,
        .successor_publications = 1U,
    });
    return;
  }
  if (owner->direct_continuation->successor_publications !=
      std::numeric_limits<std::uint32_t>::max()) {
    ++owner->direct_continuation->successor_publications;
  }
}

[[nodiscard]] expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
RuntimeChannelSession::Impl::poll_direct_continuation(const VertexId producer,
                                                      const std::uint32_t execution_lane,
                                                      const TimePoint observed_at) noexcept {
  ActivationOwnerState* const owner = find_activation_owner(producer);
  const auto target = direct_continuation_target(producer, execution_lane, false, true);
  if (owner == nullptr || !target || !owner->direct_continuation) {
    if (owner != nullptr) {
      owner->direct_continuation.reset();
    }
    return std::nullopt;
  }
  const ActivationOwnerState::DirectContinuationPublication publication =
      *owner->direct_continuation;
  owner->direct_continuation.reset();
  OutputState* const output = publication.output;
  SubscriberState* const subscriber = publication.subscriber;
  InputPortState* const input = subscriber != nullptr ? subscriber->input : nullptr;
  ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
  std::uint64_t total_publications = 0U;
  for (const OutputState* const candidate : owner->outputs) {
    total_publications += candidate->emitted_this_activation;
  }
  if (output == nullptr || subscriber == nullptr || input == nullptr || binding == nullptr ||
      output->direct_subscriber != subscriber || output->owner != producer ||
      output->active_generation != publication.source_activation_generation ||
      publication.successor_publications != 1U || total_publications != 1U ||
      subscriber->producer_epoch != publication.producer_epoch ||
      subscriber->last_producer_sequence != publication.publication_sequence ||
      publication.prior_destination_revision == std::numeric_limits<std::uint64_t>::max() ||
      publication.destination_revision != publication.prior_destination_revision + 1U ||
      input->readiness_revision.load(std::memory_order_acquire) !=
          publication.destination_revision ||
      binding->input_revisions_.front().observed != publication.prior_destination_revision ||
      !observed_at.valid() || observed_at.clock_id != target->reference_clock) {
    return std::nullopt;
  }
  const std::uint64_t prior_source_sequence = binding->next_source_sequence_;
  auto polled = binding->poll(observed_at);
  if (!polled) {
    return make_unexpected(std::move(polled).error());
  }
  if (prior_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
      binding->next_source_sequence_ != prior_source_sequence + 1U || binding->size_ != 1U ||
      binding->input_revisions_.front().observed != publication.destination_revision) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return RuntimeChannelDirectContinuationReceipt{
      .plan_generation = target->plan_generation,
      .run_generation = target->run_generation,
      .producer = target->producer,
      .destination = target->destination,
      .source_output = target->source_output,
      .destination_input = target->destination_input,
      .service = target->service,
      .readiness_source = target->readiness_source,
      .reference_clock = target->reference_clock,
      .source_generation = target->source_generation,
      .readiness_sequence = binding->next_source_sequence_,
      .producer_epoch = publication.producer_epoch,
      .publication_sequence = publication.publication_sequence,
      .prior_destination_revision = publication.prior_destination_revision,
      .destination_revision = publication.destination_revision,
      .source_activation_generation = publication.source_activation_generation,
      .execution_lane = target->execution_lane,
      .successor_publications = publication.successor_publications,
      .publication_status = ErrorCode::kSuccess,
  };
}

[[nodiscard]] bool RuntimeChannelSession::Impl::validate_direct_continuation_receipt(
    const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept {
  const auto target =
      direct_continuation_target(receipt.producer, receipt.execution_lane, false, false);
  if (!target || !runtime_channel_direct_continuation_matches(receipt, *target)) {
    return false;
  }
  const ActivationOwnerState* const owner = find_activation_owner(receipt.producer);
  const OutputState* const output = owner == nullptr ? nullptr : owner->direct_continuation_output;
  if (output == nullptr || output->port != receipt.source_output ||
      output->direct_subscriber == nullptr) {
    return false;
  }
  const SubscriberState& subscriber = *output->direct_subscriber;
  const InputPortState* const input = subscriber.input;
  const ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
  return binding != nullptr && subscriber.producer_epoch == receipt.producer_epoch &&
         subscriber.last_producer_sequence == receipt.publication_sequence &&
         input->readiness_revision.load(std::memory_order_acquire) ==
             receipt.destination_revision &&
         binding->request_.source_generation == receipt.source_generation &&
         binding->next_source_sequence_ == receipt.readiness_sequence && binding->size_ == 1U;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::rollback_tensor_access(
    RuntimeChannelSession::Impl::InputPortState& input, std::size_t access_checkpoint,
    std::size_t lease_checkpoint, Error error) noexcept {
  input.tensor_access_slots.resize(access_checkpoint);
  input.tensor_read_leases.resize(lease_checkpoint);
  return make_unexpected(std::move(error));
}

[[nodiscard]] Tensor* RuntimeChannelSession::Impl::tensor_for_access(
    RuntimeChannelSession::Impl::InputPortState& input, SampleSlot& slot, const bool direct_tensor,
    const std::uint32_t index) noexcept {
  if (direct_tensor) {
    return index == 0U ? &static_cast<Sample<Tensor>*>(slot.sample.data())->data : nullptr;
  }
  const void* payload = input.type_ops->sample_payload(slot.sample.data());
  return payload == nullptr || input.type_ops->descriptor_tensor_at == nullptr
             ? nullptr
             : const_cast<Tensor*>(input.type_ops->descriptor_tensor_at(payload, index));
}

[[nodiscard]] bool RuntimeChannelSession::Impl::tensor_device_requires_consumer_stream(
    const DLDeviceType device_type) noexcept {
  return device_type == kDLCUDA || device_type == kDLCUDAHost || device_type == kDLCUDAManaged;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::prepare_tensor_stream_access(
    Tensor& tensor, cudaStream_t consumer_stream) noexcept {
  const DLDeviceType device_type = tensor.device().device_type;
  if (!tensor_device_requires_consumer_stream(device_type)) {
    return {};
  }
  if (device_type == kDLCUDAHost) {
    if (!tensor.has_readiness_stream_contract()) {
      // A host-written pinned Tensor has no asynchronous producer work or CUDA device binding.
      return {};
    }
    if (consumer_stream == nullptr) {
      // A pinned-host descriptor endpoint may deliberately admit no compute stream. Complete
      // imported BufferHandle readiness before the synchronous CPU callback instead of
      // manufacturing an operator CUDA-stream obligation.
      return tensor.synchronize_for_host();
    }
  }
  if (consumer_stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  auto synchronized = tensor.synchronize_for_consumer_stream(consumer_stream);
  if (!synchronized) {
    return make_unexpected(std::move(synchronized).error());
  }
  if (!tensor.set_deallocation_stream(consumer_stream)) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return {};
}

// Validate every selected Tensor field before publishing any typed callback view.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::prepare_tensor_input_access(
    InputPortState& input, cudaStream_t consumer_stream) noexcept {
  const bool direct_tensor = input.type_ops == runtime_port_type_ops<Tensor>();
  const bool tensor_schema =
      input.type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
  if (!direct_tensor && !tensor_schema) {
    return {};
  }
  const std::size_t access_checkpoint = input.tensor_access_slots.size();
  const std::size_t lease_checkpoint = input.tensor_read_leases.size();
  for (SampleSlot* const slot : input.active_slots) {
    if (std::find(input.tensor_access_slots.begin(), input.tensor_access_slots.end(), slot) !=
        input.tensor_access_slots.end()) {
      continue;
    }
    if (slot == nullptr || !slot->occupied) {
      return rollback_tensor_access(
          input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
    }
    const std::uint32_t tensor_count =
        direct_tensor ? 1U : input.type_ops->descriptor_tensor_static_count;
    if (tensor_count == 0U || tensor_count > channel_internal::kMaximumDescriptorTensorFields) {
      return rollback_tensor_access(
          input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
    }
    if (input.tensor_access_slots.size() == input.tensor_access_slots.capacity() ||
        tensor_count > input.tensor_read_leases.capacity() - input.tensor_read_leases.size()) {
      return rollback_tensor_access(
          input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kResourceExhausted});
    }
    for (std::uint32_t index = 0U; index < tensor_count; ++index) {
      Tensor* const tensor = tensor_for_access(input, *slot, direct_tensor, index);
      if (tensor == nullptr) {
        return rollback_tensor_access(
            input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
      }
      auto lease = TensorAccess::acquire_read_lease(*tensor);
      if (!lease) {
        return rollback_tensor_access(
            input, access_checkpoint, lease_checkpoint, std::move(lease).error());
      }
      auto stream_access = prepare_tensor_stream_access(*tensor, consumer_stream);
      if (!stream_access) {
        return rollback_tensor_access(
            input, access_checkpoint, lease_checkpoint, std::move(stream_access).error());
      }
      input.tensor_read_leases.push_back(std::move(*lease));
    }
    input.tensor_access_slots.push_back(slot);
  }
  return {};
}

[[nodiscard]] RuntimeChannelSession::Impl::InputPortState*
RuntimeChannelSession::Impl::find_activation_input(ActivationOwnerState& owner,
                                                   PortId port) noexcept {
  const auto found = std::find_if(owner.inputs.begin(), owner.inputs.end(), [&](const auto* input) {
    return input->port == port;
  });
  return found == owner.inputs.end() ? nullptr : *found;
}

RuntimeChannelSession::Impl::ReadinessBinding::ReadinessBinding(
    Impl* owner, VertexId vertex, RuntimePortBindingRequest request, std::size_t candidate_capacity,
    std::size_t terminal_source_count, std::span<InputPortState* const> inputs,
    bool polls_external_transport)
    : owner_(owner),
      vertex_(vertex),
      request_(request),
      storage_(request.capacity),
      timeout_sequences_(request.capacity),
      closure_sequences_(request.capacity),
      terminal_source_count_(terminal_source_count),
      polls_external_transport_(polls_external_transport) {
  candidates_.reserve(candidate_capacity);
  input_revisions_.reserve(inputs.size());
  for (InputPortState* const input : inputs) {
    input_revisions_.push_back(InputRevision{.input = input});
  }
}

RuntimeChannelSession::Impl::ReadinessBinding::~ReadinessBinding() {
  if (owner_ != nullptr) {
    owner_->unregister_readiness_binding(*this);
  }
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::ReadinessBinding::poll(
    TimePoint observed_at) noexcept {
  return poll_with_acquisition(observed_at, RuntimeChannelIncompletePollKind::kNonblocking);
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::ReadinessBinding::poll_blocking(
    TimePoint observed_at) noexcept {
  return poll_with_acquisition(observed_at, RuntimeChannelIncompletePollKind::kBlocking);
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::ReadinessBinding::poll_after_contention(
    TimePoint observed_at) noexcept {
  return poll_with_acquisition(observed_at, RuntimeChannelIncompletePollKind::kContentionRetry);
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::ReadinessBinding::poll_with_acquisition(
    TimePoint observed_at, const RuntimeChannelIncompletePollKind flavour) noexcept {
  if (closed_ || owner_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  last_poll_lock_contended_ = false;
  for (InputRevision& revision : input_revisions_) {
    revision.pending = revision.input->readiness_revision.load(std::memory_order_acquire);
  }
  if (owner_->incomplete_poll_test_control_ != nullptr &&
      owner_->incomplete_poll_test_control_->applies_to(flavour) &&
      owner_->incomplete_poll_test_control_->consume()) {
    // A forced nonblocking or contention-retry decline models a lost input cache lock.
    last_poll_lock_contended_ = flavour != RuntimeChannelIncompletePollKind::kBlocking;
    last_poll_incomplete_ = true;
    count_phase(owner_->phase_counters_, RuntimePhaseCounter::kChannelPollIncomplete);
    return {};
  }
  auto polled = owner_->poll_inputs(*this, observed_at, flavour);
  if (!polled) {
    return make_unexpected(std::move(polled).error());
  }
  last_poll_incomplete_ = !*polled;
  if (!*polled) {
    count_phase(owner_->phase_counters_, RuntimePhaseCounter::kChannelPollIncomplete);
  }
  if (*polled) {
    for (InputRevision& revision : input_revisions_) {
      revision.observed = revision.pending;
    }
  }
  return {};
}

[[nodiscard]] SourceId RuntimeChannelSession::Impl::ReadinessBinding::source() const noexcept {
  return request_.source;
}

[[nodiscard]] std::optional<PortBindingPollTarget>
RuntimeChannelSession::Impl::ReadinessBinding::poll_target() const noexcept {
  if (closed()) {
    return std::nullopt;
  }
  const bool local_input_changed =
      std::any_of(input_revisions_.begin(), input_revisions_.end(), [](const auto& revision) {
        return revision.input->readiness_revision.load(std::memory_order_acquire) !=
               revision.observed;
      });
  if (!polls_external_transport_ && !local_input_changed) {
    return std::nullopt;
  }
  // External imports merge transport and clock-health heads inside poll(), before a typed
  // cache revision exists, so they retain their compiler-selected polling lane.
  return PortBindingPollTarget{
      .execution_lane = request_.execution_lane,
      .reference_clock = request_.reference_clock,
      .busy_poll = false,
  };
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::ReadinessBinding::enqueue_window_timeout(
    PortId port, TimePoint terminal_time) noexcept {
  if (!port || !terminal_time.valid() || terminal_time.clock_id != request_.reference_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (closed()) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (storage_.empty() || size_ == storage_.size() || timeout_count_ == timeout_sequences_.size()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const auto timeout_slot = std::find(timeout_sequences_.begin(), timeout_sequences_.end(), 0U);
  if (timeout_slot == timeout_sequences_.end()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  ++next_source_sequence_;
  storage_[tail_] = RuntimeEvent{
      .kind = RuntimeEventKind::kWindowTimeout,
      .source = request_.source,
      .port = port,
      .source_decision_time = terminal_time,
      .reference_time = terminal_time,
      .source_generation = request_.source_generation,
      .source_sequence = next_source_sequence_,
  };
  *timeout_slot = next_source_sequence_;
  ++timeout_count_;
  tail_ = (tail_ + 1U) % storage_.size();
  ++size_;
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::ReadinessBinding::retire(
    std::uint64_t sequence) noexcept {
  if (owner_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto timeout = std::find(timeout_sequences_.begin(), timeout_sequences_.end(), sequence);
  if (timeout != timeout_sequences_.end()) {
    *timeout = 0U;
    --timeout_count_;
    publish_deferred_terminal_retry();
    return {};
  }
  const auto closure = std::find(closure_sequences_.begin(), closure_sequences_.end(), sequence);
  if (closure != closure_sequences_.end()) {
    *closure = 0U;
    --closure_count_;
    if (terminal_sources_retired_ == terminal_source_count_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    ++terminal_sources_retired_;
    // One readiness binding owns every physical input source for this Operator. The final
    // retired EOS proves that every producer has closed and that all earlier source-local
    // readiness witnesses have passed the Runtime retirement boundary. Close only this
    // downstream readiness source; selector-held payload claims were reconciled before the
    // EOS witness became eligible, and later generations reopen through the restart fence.
    if (terminal_sources_retired_ == terminal_source_count_) {
      closed_ = true;
    }
    publish_deferred_terminal_retry();
    return {};
  }
  auto retired = owner_->retire_witness(vertex_, sequence);
  if (retired) {
    publish_deferred_terminal_retry();
  }
  return retired;
}

void RuntimeChannelSession::Impl::ReadinessBinding::cancel() noexcept {
  closed_ = true;
  size_ = 0U;
  head_ = 0U;
  tail_ = 0U;
  std::fill(timeout_sequences_.begin(), timeout_sequences_.end(), 0U);
  std::fill(closure_sequences_.begin(), closure_sequences_.end(), 0U);
  timeout_count_ = 0U;
  closure_count_ = 0U;
  terminal_retry_pending_ = false;
}

[[nodiscard]] expected<std::uint64_t, Error>
RuntimeChannelSession::Impl::ReadinessBinding::advance_source_generation() noexcept {
  if (owner_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (size_ != 0U || timeout_count_ != 0U || closure_count_ != 0U ||
      request_.source_generation == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{size_ != 0U || timeout_count_ != 0U || closure_count_ != 0U
                                     ? ErrorCode::kBackpressured
                                     : ErrorCode::kResourceExhausted});
  }
  ++request_.source_generation;
  next_source_sequence_ = request_.initial_source_sequence;
  terminal_sources_retired_ = 0U;
  closed_ = false;
  return request_.source_generation;
}

[[nodiscard]] expected<std::size_t, Error> RuntimeChannelSession::Impl::ReadinessBinding::drain(
    std::span<RuntimeEvent> destination) noexcept {
  const std::size_t count = std::min(destination.size(), size_);
  for (std::size_t index = 0U; index < count; ++index) {
    destination[index] = storage_[head_];
    head_ = (head_ + 1U) % storage_.size();
  }
  size_ -= count;
  return count;
}

[[nodiscard]] std::size_t RuntimeChannelSession::Impl::ReadinessBinding::capacity() const noexcept {
  return storage_.size();
}

[[nodiscard]] bool RuntimeChannelSession::Impl::ReadinessBinding::closed() const noexcept {
  return closed_ || owner_ == nullptr || owner_->closed_.load(std::memory_order_acquire);
}

[[nodiscard]] bool RuntimeChannelSession::Impl::ReadinessBinding::later_candidate(
    const SlotCandidate& lhs, const SlotCandidate& rhs) noexcept {
  return lhs.arrival_order > rhs.arrival_order;
}

[[nodiscard]] bool RuntimeChannelSession::Impl::ReadinessBinding::owns_input(
    const InputPortState* input) const noexcept {
  return std::any_of(input_revisions_.begin(),
                     input_revisions_.end(),
                     [input](const InputRevision& revision) { return revision.input == input; });
}

void RuntimeChannelSession::Impl::ReadinessBinding::detach_from_owner() noexcept {
  owner_ = nullptr;
  input_revisions_.clear();
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::ReadinessBinding::push_end_of_stream(PortId port,
                                                                  TimePoint observed_at) noexcept {
  if (!port || !observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (closed()) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (storage_.empty() || size_ == storage_.size() || closure_count_ == closure_sequences_.size()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const auto closure_slot = std::find(closure_sequences_.begin(), closure_sequences_.end(), 0U);
  if (closure_slot == closure_sequences_.end()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  ++next_source_sequence_;
  storage_[tail_] = RuntimeEvent{
      .kind = RuntimeEventKind::kInputEndOfStream,
      .source = request_.source,
      .port = port,
      .source_decision_time = observed_at,
      .reference_time = observed_at,
      .source_generation = request_.source_generation,
      .source_sequence = next_source_sequence_,
  };
  *closure_slot = next_source_sequence_;
  ++closure_count_;
  tail_ = (tail_ + 1U) % storage_.size();
  ++size_;
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::ReadinessBinding::push(
    SampleSlot& slot, std::uint64_t& sequence, TimePoint observed_at) noexcept {
  if (closed()) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (storage_.empty() || size_ == storage_.size()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  auto coordinates = owner_->event_coordinates(slot, request_, observed_at);
  if (!coordinates) {
    return make_unexpected(std::move(coordinates).error());
  }
  // Temporal selectors operate only in the compiler-frozen reference clock. Preserve the
  // caller's original capture coordinate in the typed Sample metadata, but normalize a
  // present payload-independent capture coordinate exactly once while the readiness witness
  // still owns both domains. A capture-less process-local sample uses arrival time only for
  // readiness ordering; its TemporalSampleRecord must remain capture-less so selection applies
  // the authored missing-record policy.
  if (slot.temporal.capture_time.valid()) {
    slot.temporal.capture_time = coordinates->second;
  }
  ++next_source_sequence_;
  sequence = next_source_sequence_;
  storage_[tail_] = RuntimeEvent{
      .kind = RuntimeEventKind::kInputArrival,
      .source = request_.source,
      .port = slot.subscriber->target,
      .source_decision_time = coordinates->first,
      .reference_time = coordinates->second,
      .source_generation = request_.source_generation,
      .source_sequence = sequence,
      .clock_health_disposition = slot.clock_health_disposition,
      .clock_health_generation = slot.clock_health_generation,
  };
  tail_ = (tail_ + 1U) % storage_.size();
  ++size_;
  return {};
}

void RuntimeChannelSession::Impl::unregister_readiness_binding(ReadinessBinding& binding) noexcept {
  const auto found = std::find(readiness_bindings_.begin(), readiness_bindings_.end(), &binding);
  if (found != readiness_bindings_.end()) {
    readiness_bindings_.erase(found);
  }
  for (const auto& revision : binding.input_revisions_) {
    if (revision.input != nullptr && revision.input->readiness_binding == &binding) {
      revision.input->readiness_binding = nullptr;
    }
  }
  for (ActivationOwnerState& owner : activation_owners_) {
    OutputState* const output = owner.direct_continuation_output;
    if (output == nullptr || output->direct_subscriber == nullptr ||
        !binding.owns_input(output->direct_subscriber->input)) {
      continue;
    }
    output->direct_continuation_capture = false;
    owner.direct_continuation.reset();
    owner.direct_continuation_output = nullptr;
  }
  binding.detach_from_owner();
}

void RuntimeChannelSession::Impl::detach_readiness_bindings() noexcept {
  while (!readiness_bindings_.empty()) {
    unregister_readiness_binding(*readiness_bindings_.back());
  }
}

// Translate and validate both event coordinates before publishing any readiness witness.
[[nodiscard]] expected<std::pair<TimePoint, TimePoint>, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::event_coordinates(const SampleSlot& slot,
                                               const RuntimePortBindingRequest& request,
                                               TimePoint observed_at) const noexcept {
  const InputPortState& input = *slot.subscriber->input;
  const TimePoint capture = slot.temporal.capture_time;
  if (!capture.valid()) {
    // Only process-local samples can reach retained storage without capture time. Catalog
    // imports that promise capture are rejected at the transport boundary before a slot or
    // arrival identity is committed. Preserve local absence for selector policy while using
    // observed arrival solely as deterministic readiness-order evidence.
    return std::pair{observed_at, observed_at};
  }

  if (input.reference_clock.valid()) {
    if (capture.clock_id != input.capture_clock) {
      return make_unexpected(Error{
          ErrorCode::kProtocolError,
          "RUNTIME_CLOCK_BINDING_UNAVAILABLE: sample capture clock differs from input clock"});
    }
    auto reference = convert_identical_or_direct(
        capture,
        input.reference_clock,
        input.capture_to_reference ? &*input.capture_to_reference : nullptr);
    if (!reference) {
      return make_unexpected(std::move(reference).error());
    }
    // Input selection always normalizes capture into its compiler-selected reference clock.
    // The readiness source, however, uses that reference coordinate as both decision and
    // reference time unless a deadline explicitly retained a distinct source clock. Exposing
    // the original capture coordinate in the ordinary no-deadline case would violate the
    // RuntimeSourceBindingRecord identity consumed by the coordinator.
    if (request.source_clock == request.reference_clock) {
      if (request.source_to_reference_conversion_id != 0U ||
          request.reference_clock != reference->clock_id) {
        return make_unexpected(
            Error{ErrorCode::kProtocolError,
                  "RUNTIME_CLOCK_BINDING_UNAVAILABLE: readiness clock identities disagree"});
      }
      return std::pair{*reference, *reference};
    }
    if (request.source_to_reference_conversion_id == 0U ||
        request.source_clock != capture.clock_id ||
        request.reference_clock != reference->clock_id) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError,
                "RUNTIME_CLOCK_BINDING_UNAVAILABLE: readiness conversion identities disagree"});
    }
    const auto source_conversion =
        std::find_if(clock_conversions_.begin(), clock_conversions_.end(), [&](const auto& edge) {
          return edge.id == request.source_to_reference_conversion_id &&
                 edge.plan_generation == plan_generation_ && edge.from == request.source_clock &&
                 edge.to == request.reference_clock;
        });
    if (source_conversion == clock_conversions_.end()) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError,
                "RUNTIME_CLOCK_BINDING_UNAVAILABLE: readiness conversion is absent"});
    }
    const DirectAffineClockEdge source_edge{
        .source = source_conversion->from,
        .target = source_conversion->to,
        .conversion = AffineClockConversion{.scale = source_conversion->scale,
                                            .offset_ns = source_conversion->offset_ns},
    };
    auto source_reference =
        convert_identical_or_direct(capture, request.reference_clock, &source_edge);
    if (!source_reference || *source_reference != *reference) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError,
                "RUNTIME_CLOCK_BINDING_UNAVAILABLE: readiness conversion results disagree"});
    }
    return std::pair{capture, *reference};
  }

  // Arrival-backed triggers such as OnEach do not create an input-selection record, so their
  // readiness binding cites only the Runtime reference clock. Preserve capture-derived ordering
  // whenever the coordinate is already in that clock or the plan declares the direct affine
  // conversion. Samples in an unrelated clock remain ordered by their observed arrival.
  if (request.source_clock == request.reference_clock &&
      request.source_to_reference_conversion_id == 0U) {
    if (capture.clock_id == request.reference_clock) {
      return std::pair{capture, capture};
    }
    const auto conversion =
        std::find_if(clock_conversions_.begin(), clock_conversions_.end(), [&](const auto& edge) {
          return edge.plan_generation == plan_generation_ && edge.from == capture.clock_id &&
                 edge.to == request.reference_clock;
        });
    if (conversion == clock_conversions_.end()) {
      return std::pair{observed_at, observed_at};
    }
    const DirectAffineClockEdge edge{
        .source = conversion->from,
        .target = conversion->to,
        .conversion =
            AffineClockConversion{.scale = conversion->scale, .offset_ns = conversion->offset_ns},
    };
    auto reference = convert_identical_or_direct(capture, request.reference_clock, &edge);
    if (!reference) {
      return make_unexpected(std::move(reference).error());
    }
    return std::pair{*reference, *reference};
  }

  if (capture.clock_id == request.reference_clock) {
    return std::pair{capture, capture};
  }
  const auto conversion =
      std::find_if(clock_conversions_.begin(), clock_conversions_.end(), [&](const auto& edge) {
        return edge.id == request.source_to_reference_conversion_id &&
               edge.plan_generation == plan_generation_ && edge.from == capture.clock_id &&
               edge.to == request.reference_clock;
      });
  if (conversion == clock_conversions_.end()) {
    return make_unexpected(Error{ErrorCode::kProtocolError,
                                 "RUNTIME_CLOCK_BINDING_UNAVAILABLE: input conversion is absent"});
  }
  const DirectAffineClockEdge edge{
      .source = conversion->from,
      .target = conversion->to,
      .conversion =
          AffineClockConversion{.scale = conversion->scale, .offset_ns = conversion->offset_ns},
  };
  auto reference = convert_identical_or_direct(capture, request.reference_clock, &edge);
  if (!reference) {
    return make_unexpected(std::move(reference).error());
  }
  return std::pair{capture, *reference};
}

namespace {

// A contention retry waits out one holder's critical section on an input cache mutex: one
// kContentionRetryBudget interval per contended acquisition. poll_inputs acquires once per owner
// subscriber in its scan and once per popped candidate in its commit, and the first acquisition
// that fails ends the poll, so a poll accrues budget only across acquisitions that contend and then
// succeed; a wide fan-in owner may therefore spend up to (subscribers + popped candidates) x
// budget, each attempt overshooting by one yield. Holders are far shorter than the budget, so it
// expires only when a holder is preempted mid-section, and that expiry is reported exactly like a
// declined nonblocking poll: the dirty revision stays for the next round and the 1 ms fallback park
// follows as before. The wait is a yield loop, which under SCHED_FIFO or SCHED_RR cannot hand the
// CPU to a lower-priority holder sharing it; a real-time lane then pays the full budget before that
// park. Execution lanes run SCHED_OTHER unless an application places them, and the pump thread is
// never placed.
//
// The budget is the session's (kContentionRetryBudget unless a source-tree test widened it). Each
// failed attempt that the loop follows with another try reports to the caller first; the final
// failure that expires the budget is returned as the unowned lock instead. That report is how a
// test releases a real holder only after the loop has actually re-armed; production callers
// install no observer, so it costs one null check per attempt.
template <typename OnFailedRetryAttempt>
[[nodiscard]] std::unique_lock<std::mutex> acquire_input_cache_lock(
    std::mutex& mutex, const RuntimeChannelIncompletePollKind flavour,
    const std::chrono::nanoseconds retry_budget,
    OnFailedRetryAttempt&& on_failed_retry_attempt) noexcept {
  switch (flavour) {
    case RuntimeChannelIncompletePollKind::kBlocking:
      return std::unique_lock<std::mutex>{mutex};
    case RuntimeChannelIncompletePollKind::kContentionRetry: {
      std::unique_lock<std::mutex> lock{mutex, std::try_to_lock};
      if (lock.owns_lock()) {
        return lock;
      }
      const auto deadline = std::chrono::steady_clock::now() + retry_budget;
      do {
        on_failed_retry_attempt();
        std::this_thread::yield();
        if (lock.try_lock()) {
          return lock;
        }
      } while (std::chrono::steady_clock::now() < deadline);
      return lock;
    }
    case RuntimeChannelIncompletePollKind::kNonblocking:
      break;
  }
  return std::unique_lock<std::mutex>{mutex, std::try_to_lock};
}

}  // namespace

// Polling joins every endpoint family into one ordered readiness publication pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<bool, Error> RuntimeChannelSession::Impl::poll_inputs(
    ReadinessBinding& binding, TimePoint observed_at,
    const RuntimeChannelIncompletePollKind flavour) noexcept {
  const auto completion = [this](const bool complete) -> expected<bool, Error> {
    auto current = status();
    return current ? expected<bool, Error>{complete}
                   : expected<bool, Error>{make_unexpected(std::move(current).error())};
  };
  const auto* owner = find_activation_owner(binding.vertex_);
  if (owner == nullptr || owner->subscribers.empty()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto monitored = drain_monitored_inputs(*owner);
  if (!monitored) {
    if (monitored.error().code == ErrorCode::kNotReady) {
      return completion(false);
    }
    return make_unexpected(std::move(monitored).error());
  }
  binding.candidates_.clear();
  for (SubscriberState* const subscriber : owner->subscribers) {
    const std::unique_lock lock = acquire_input_cache_lock(
        subscriber->mutex, flavour, contention_retry_budget_, [&]() noexcept {
          if (subscriber->input_contention_retry_attempt_observer != nullptr) {
            subscriber->input_contention_retry_attempt_observer(
                subscriber->input_contention_retry_attempt_observer_state,
                subscriber->owner,
                subscriber->target,
                binding.source());
          }
        });
    if (!lock.owns_lock()) {
      binding.last_poll_lock_contended_ = true;
      if (flavour == RuntimeChannelIncompletePollKind::kNonblocking &&
          subscriber->input_poll_contention_observer != nullptr) {
        subscriber->input_poll_contention_observer(subscriber->input_poll_contention_observer_state,
                                                   subscriber->owner,
                                                   subscriber->target,
                                                   binding.source());
      }
      // Returning without progress preserves arrival order: an earlier sample may be hidden
      // behind this contended cache lock, so selecting from later subscribers would overtake
      // it. The dispatcher may retry after the bounded owner of the lock makes progress.
      return completion(false);
    }
    auto promoted = promote_direct_ingress(*subscriber);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    // A process-local capture-absent successor is valid missing-record evidence. At depth one,
    // ReuseLast N can pin the only typed slot, so N+1 cannot be promoted without destroying the
    // payload the missing policy must return. Publish the already-materialized staged identity as
    // an arrival while keeping its payload unread and independently owned until evidence retires.
    for (SampleSlot& slot : subscriber->slots) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelStagedEvidenceSweepVisits);
      auto published = publish_staged_arrival_evidence_if_eligible(*subscriber, slot);
      if (!published) {
        return make_unexpected(std::move(published).error());
      }
    }
    for (SampleSlot& slot : subscriber->slots) {
      // An occupied pre-witness replacement candidate remains the incumbent until its reserved
      // receive commits. Do not acknowledge the captured input revision while that incumbent is
      // temporarily hidden, because a failed receive must leave it independently pollable.
      if (slot.typed_ingress_reserved && slot.occupied && !slot.direct_ingress_unread &&
          !slot.event_emitted) {
        return completion(false);
      }
      if (!slot.occupied || (slot.direct_ingress_unread && !slot.staged_arrival_evidence) ||
          slot.event_emitted || slot.typed_ingress_reserved) {
        continue;
      }
      if (binding.candidates_.size() == binding.candidates_.capacity()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
          .subscriber = subscriber, .slot = &slot, .arrival_order = slot.arrival_order});
    }
    if (subscriber->end_of_stream_received && !subscriber->end_of_stream_event_emitted &&
        subscriber->end_of_stream_arrival_order != 0U &&
        !subscriber->has_pending_direct_ingress()) {
      if (binding.candidates_.size() == binding.candidates_.capacity()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
          .subscriber = subscriber,
          .slot = nullptr,
          .arrival_order = subscriber->end_of_stream_arrival_order,
      });
    }
  }
  std::make_heap(
      binding.candidates_.begin(), binding.candidates_.end(), ReadinessBinding::later_candidate);
  while (!binding.candidates_.empty()) {
    std::pop_heap(
        binding.candidates_.begin(), binding.candidates_.end(), ReadinessBinding::later_candidate);
    const ReadinessBinding::SlotCandidate selected = binding.candidates_.back();
    binding.candidates_.pop_back();
    if (poll_selection_observer_ != nullptr) {
      const auto observer = std::exchange(poll_selection_observer_, nullptr);
      void* const state = std::exchange(poll_selection_observer_state_, nullptr);
      observer(state);
    }
    SubscriberState& selected_subscriber = *selected.subscriber;
    const std::unique_lock lock = acquire_input_cache_lock(
        selected_subscriber.mutex, flavour, contention_retry_budget_, [&]() noexcept {
          if (selected_subscriber.input_contention_retry_attempt_observer != nullptr) {
            selected_subscriber.input_contention_retry_attempt_observer(
                selected_subscriber.input_contention_retry_attempt_observer_state,
                selected_subscriber.owner,
                selected_subscriber.target,
                binding.source());
          }
        });
    if (!lock.owns_lock()) {
      binding.last_poll_lock_contended_ = true;
      return completion(false);
    }
    if (selected.slot == nullptr) {
      if (!selected_subscriber.end_of_stream_received ||
          selected_subscriber.end_of_stream_event_emitted ||
          selected_subscriber.end_of_stream_arrival_order != selected.arrival_order) {
        continue;
      }
      auto pushed = binding.push_end_of_stream(selected_subscriber.target, observed_at);
      if (!pushed) {
        if (pushed.error().code == ErrorCode::kBackpressured) {
          // The retained EOS remains the oldest unpublished terminal candidate. A full
          // readiness binding means previously accepted data still owns dispatch capacity;
          // observe the current revision so that data can dispatch, then publish a fresh
          // revision when one event retires to retry EOS without overtaking that data.
          binding.terminal_retry_pending_ = true;
          return completion(true);
        }
        return make_unexpected(std::move(pushed).error());
      }
      selected_subscriber.end_of_stream_event_emitted = true;
    } else {
      SampleSlot& selected_slot = *selected.slot;
      // On the HoloMQ path, a drop-oldest producer may replace a saved slot after the owner-local
      // snapshot releases its lock. Reinsert that later identity into the heap so it cannot
      // overtake an older candidate. Direct ingress slots enter this heap only after promotion.
      if (selected_slot.typed_ingress_reserved) {
        return completion(false);
      }
      if (!selected_slot.occupied || selected_slot.event_emitted) {
        continue;
      }
      if (selected_slot.arrival_order != selected.arrival_order) {
        binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
            .subscriber = &selected_subscriber,
            .slot = &selected_slot,
            .arrival_order = selected_slot.arrival_order,
        });
        std::push_heap(binding.candidates_.begin(),
                       binding.candidates_.end(),
                       ReadinessBinding::later_candidate);
        continue;
      }
      auto pushed = binding.push(selected_slot, selected_slot.readiness_sequence, observed_at);
      if (!pushed) {
        return make_unexpected(std::move(pushed).error());
      }
      // Successful readiness publication transfers one payload-lifetime claim to the temporal
      // selector. The dispatcher later returns that claim through retire_samples() only after the
      // record disappears from every cache, reuse slot, and live or closed window.
      selected_slot.temporal_retained = true;
      selected_slot.event_emitted = true;
      if (selected_slot.owed_to_released_claim) {
        selected_slot.subscriber->input->last_refill_outcome.store(1U, std::memory_order_relaxed);
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelReleasedClaimRefillPublished);
      }
      selected_slot.owed_to_released_claim = false;
    }
  }
  return completion(true);
}

// Witness retirement preserves endpoint-specific acknowledgement and terminal ordering.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::retire_witness(
    VertexId owner, std::uint64_t sequence) noexcept {
  if (!owner || sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ActivationOwnerState* const activation_owner = find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const bool signal_progress = owner_progress_wake_enabled();
  for (SubscriberState* const subscriber : activation_owner->subscribers) {
    bool matched = false;
    bool direct_capacity_available = false;
    {
      const std::scoped_lock lock{subscriber->mutex};
      for (SampleSlot& slot : subscriber->slots) {
        if (slot.occupied && slot.readiness_sequence == sequence) {
          if (!slot.event_emitted || slot.witness_retired) {
            return make_unexpected(Error{ErrorCode::kStaleEndpoint});
          }
          slot.witness_retired = true;
          reclaim_sample_slot_if_unclaimed(slot);
          auto promoted = promote_direct_ingress(*subscriber);
          if (!promoted) {
            return make_unexpected(std::move(promoted).error());
          }
          matched = true;
          direct_capacity_available =
              signal_progress && subscriber->direct_process_local &&
              (subscriber->direct_ingress_capacity != 0U
                   ? writable_direct_ingress_slot(*subscriber) != nullptr
                   : writable_slot(*subscriber, WritableSlotIntent::kProbe) != nullptr);
          break;
        }
      }
    }
    if (!matched) {
      continue;
    }
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelWitnessRetired);
    if (direct_capacity_available) {
      signal_owner_progress(subscriber->source_owner);
    }
    return {};
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

bool runtime_channel_direct_continuation_matches(
    const RuntimeChannelDirectContinuationReceipt& receipt,
    const RuntimeChannelDirectContinuationTarget& target) noexcept {
  return receipt.plan_generation != 0U && receipt.plan_generation == target.plan_generation &&
         receipt.run_generation != 0U && receipt.run_generation == target.run_generation &&
         receipt.producer && receipt.producer == target.producer && receipt.destination &&
         receipt.destination == target.destination && receipt.source_output &&
         receipt.source_output == target.source_output && receipt.destination_input &&
         receipt.destination_input == target.destination_input && !receipt.service.empty() &&
         receipt.service == target.service && receipt.readiness_source &&
         receipt.readiness_source == target.readiness_source && receipt.reference_clock.valid() &&
         receipt.reference_clock == target.reference_clock && receipt.source_generation != 0U &&
         receipt.source_generation == target.source_generation &&
         receipt.readiness_sequence != 0U &&
         receipt.readiness_sequence == target.readiness_sequence && receipt.producer_epoch != 0U &&
         receipt.publication_sequence != 0U && receipt.source_activation_generation != 0U &&
         receipt.source_activation_generation == target.source_activation_generation &&
         receipt.destination_revision == target.destination_revision &&
         receipt.destination_revision == target.observed_destination_revision &&
         receipt.execution_lane == target.execution_lane && receipt.successor_publications == 1U &&
         receipt.publication_status == ErrorCode::kSuccess &&
         !publication_error_consumes_authority(receipt.publication_status) &&
         receipt.prior_destination_revision != std::numeric_limits<std::uint64_t>::max() &&
         receipt.destination_revision == receipt.prior_destination_revision + 1U;
}

bool RuntimeChannelSession::has_materialized_activation_state(const VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return false;
  }
  return impl_->find_activation_owner(owner) != nullptr;
}

bool RuntimeChannelSession::has_materialized_input_state(const VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return false;
  }
  const auto* activation_owner = impl_->find_activation_owner(owner);
  return activation_owner != nullptr &&
         std::any_of(activation_owner->inputs.begin(),
                     activation_owner->inputs.end(),
                     [](const auto* input) { return !input->subscribers.empty(); });
}

expected<std::unique_ptr<PortBinding>, Error> RuntimeChannelSession::create_readiness_binding(
    VertexId owner, RuntimePortBindingRequest request) noexcept {
  if (!impl_ || !owner || request.kind != RuntimeEventKind::kInputArrival || !request.source ||
      !request.source_clock.valid() || !request.reference_clock.valid() || request.capacity == 0U ||
      request.source_generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr || activation_owner->inputs.empty() ||
      request.plan_generation != impl_->plan_generation_ || request.run_generation == 0U ||
      std::any_of(activation_owner->inputs.begin(),
                  activation_owner->inputs.end(),
                  [](const auto* input) { return input->readiness_binding != nullptr; })) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  std::size_t candidate_capacity = 0U;
  for (const auto* subscriber : activation_owner->subscribers) {
    const std::size_t subscriber_capacity = subscriber->slots.size() + 1U;
    if (subscriber_capacity == 0U ||
        candidate_capacity > std::numeric_limits<std::size_t>::max() - subscriber_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    candidate_capacity += subscriber_capacity;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto binding = std::make_unique<Impl::ReadinessBinding>(
        impl_.get(),
        owner,
        request,
        candidate_capacity,
        activation_owner->subscribers.size(),
        activation_owner->inputs,
        std::any_of(activation_owner->subscribers.begin(),
                    activation_owner->subscribers.end(),
                    [](const auto* subscriber) { return subscriber->externally_driven_import; }));
    for (auto* input : activation_owner->inputs) {
      input->readiness_binding = binding.get();
    }
    impl_->readiness_bindings_.push_back(binding.get());
    impl_->cache_direct_continuation_candidate(owner, *binding);
    return std::unique_ptr<PortBinding>{std::move(binding)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<std::uint32_t, Error> RuntimeChannelSession::readiness_capacity(
    VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::uint32_t capacity = 0U;
  const auto* activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr || activation_owner->subscribers.empty()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  for (const auto* subscriber : activation_owner->subscribers) {
    if (capacity > std::numeric_limits<std::uint32_t>::max() - subscriber->readiness_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    capacity += subscriber->readiness_capacity;
  }
  return capacity != 0U
             ? expected<std::uint32_t, Error>{capacity}
             : expected<std::uint32_t, Error>{make_unexpected(Error{ErrorCode::kNotFound})};
}

std::optional<RuntimeChannelDirectContinuationTarget>
RuntimeChannelSession::arm_direct_continuation(const VertexId producer,
                                               const std::uint32_t execution_lane) noexcept {
  if (!impl_) {
    return std::nullopt;
  }
  auto target = impl_->direct_continuation_target(producer, execution_lane, true, true);
  if (!target) {
    return std::nullopt;
  }
  auto* const owner = impl_->find_activation_owner(producer);
  auto* const output = owner == nullptr ? nullptr : owner->direct_continuation_output;
  if (output == nullptr || output->port != target->source_output) {
    return std::nullopt;
  }
  output->direct_continuation_capture = true;
  return target;
}

expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
RuntimeChannelSession::poll_direct_continuation(const VertexId producer,
                                                const std::uint32_t execution_lane,
                                                const TimePoint observed_at) noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->poll_direct_continuation(producer, execution_lane, observed_at);
}

bool RuntimeChannelSession::validate_direct_continuation_receipt(
    const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept {
  return impl_ != nullptr && impl_->validate_direct_continuation_receipt(receipt);
}

expected<TemporalSampleRecord, Error> RuntimeChannelSession::temporal_sample(
    PortId port, std::uint64_t readiness_sequence) const noexcept {
  if (!impl_ || !port || readiness_sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  for (const Impl::SubscriberState* const subscriber : input->subscribers) {
    const std::scoped_lock lock{subscriber->mutex};
    for (const Impl::SampleSlot& slot : subscriber->slots) {
      if (slot.occupied && (!slot.direct_ingress_unread || slot.staged_arrival_evidence) &&
          slot.readiness_sequence == readiness_sequence) {
        return slot.temporal;
      }
    }
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

// Acquire all activation claims before exposing a rollback-capable transaction.
expected<RuntimeChannelSession::Impl::ActivationPrepareTransaction, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::begin_activation_prepare(
    ActivationOwnerState& owner, const std::uint64_t generation,
    const std::span<const RuntimeChannelActivationSelection> selections) noexcept {
  bool any_active = false;
  bool all_active = true;
  for (const auto* input : owner.inputs) {
    any_active = any_active || input->active;
    all_active = all_active && input->active;
    if (input->active && input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (const auto* output : owner.outputs) {
    any_active = any_active || output->active;
    all_active = all_active && output->active;
    if (output->active && output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  if (any_active != all_active) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const bool initial_prepare = !any_active;
  if (initial_prepare) {
    owner.direct_continuation.reset();
    for (auto* output : owner.outputs) {
      output->direct_continuation_capture = false;
    }
  }

  for (std::size_t index = 0U; index < selections.size(); ++index) {
    const auto* input = find_activation_input(owner, selections[index].port);
    if (input == nullptr || !valid_input_selection(selections[index].selection) ||
        (!initial_prepare && input->selection_prepared) ||
        std::any_of(selections.first(index).begin(),
                    selections.first(index).end(),
                    [&](const auto& prior) { return prior.port == selections[index].port; })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }

  if (initial_prepare) {
    for (auto* input : owner.inputs) {
      if (!input->active_slots.empty() || input->predicate_view_constructed ||
          input->selection_prepared || input->staged_now) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      input->active_generation = generation;
      input->binding.generation = generation;
      input->type_ops->bind_input(input->declaration, &input->binding);
      input->active_selection = {InputState::kNotSelected,
                                 SelectionReason::kNotSelectedForActivation};
      input->batch = false;
      input->destructive = false;
      input->active = true;
    }
    for (auto* output : owner.outputs) {
      output->active_generation = generation;
      output->emitted_this_activation = 0U;
      output->active = true;
    }
  }
  return ActivationPrepareTransaction{.owner = &owner, .initial_prepare = initial_prepare};
}

void RuntimeChannelSession::Impl::rollback_activation_prepare(
    ActivationPrepareTransaction& transaction) noexcept {
  for (auto* input : transaction.owner->inputs) {
    if (!input->staged_now) {
      continue;
    }
    if (input->predicate_view_constructed) {
      input->type_ops->destroy_predicate_view(input->predicate_view_storage.data());
      input->predicate_view_constructed = false;
    }
    for (SampleSlot* slot : input->active_slots) {
      const std::scoped_lock lock{slot->subscriber->mutex};
      slot->active_selected = false;
    }
    input->active_slots.clear();
    input->active_selection = {InputState::kNotSelected,
                               SelectionReason::kNotSelectedForActivation};
    input->batch = false;
    input->destructive = false;
    input->selection_prepared = false;
    input->staged_now = false;
  }
  if (transaction.initial_prepare) {
    for (auto* input : transaction.owner->inputs) {
      input->active = false;
    }
    for (auto* output : transaction.owner->outputs) {
      output->active = false;
    }
  }
}

// Stage every selection provisionally before the activation transaction can become visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::Impl::stage_activation_selections(
    ActivationPrepareTransaction& transaction,
    const std::span<const RuntimeChannelActivationSelection> selections) noexcept {
  for (const RuntimeChannelActivationSelection& selected : selections) {
    InputPortState* input = find_activation_input(*transaction.owner, selected.port);
    if (input == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    input->active_slots.clear();
    input->active_selection = selected.selection;
    input->batch = selected.batch;
    input->destructive = selected.destructive;
    input->staged_now = true;
    for (const TemporalSampleRecord& record : selected.samples) {
      if (input->active_slots.size() == input->active_slots.capacity()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      SampleSlot* match = nullptr;
      for (SubscriberState* subscriber : input->subscribers) {
        bool duplicate = false;
        bool unavailable = false;
        {
          const std::scoped_lock lock{subscriber->mutex};
          const auto found = std::find_if(
              subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
                return slot.occupied && !slot.direct_ingress_unread && slot.temporal == record;
              });
          if (found != subscriber->slots.end()) {
            duplicate = match != nullptr;
            unavailable = found->active_selected || found->typed_ingress_reserved;
            if (!duplicate && !unavailable) {
              found->active_selected = true;
              match = &*found;
            }
          }
        }
        if (duplicate || unavailable) {
          if (match != nullptr) {
            const std::scoped_lock lock{match->subscriber->mutex};
            match->active_selected = false;
          }
          return make_unexpected(
              Error{duplicate ? ErrorCode::kProtocolError : ErrorCode::kStaleEndpoint});
        }
      }
      if (match == nullptr) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      input->active_slots.push_back(match);
    }
    const bool value_state = selected.selection.state == InputState::kSelected ||
                             selected.selection.state == InputState::kReused;
    if ((!selected.batch && value_state && input->active_slots.size() != 1U) ||
        (!value_state && !input->active_slots.empty())) {
      return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
    }
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::Impl::construct_activation_predicate_views(
    ActivationPrepareTransaction& transaction) noexcept {
  for (auto* input : transaction.owner->inputs) {
    if (!input->staged_now) {
      continue;
    }
    const bool value_state = input->active_selection.state == InputState::kSelected ||
                             input->active_selection.state == InputState::kReused;
    if (value_state && !input->batch) {
      if (input->active_slots.size() != 1U) {
        return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
      }
      auto constructed =
          input->type_ops->construct_predicate_view(input->predicate_view_storage.data(),
                                                    input->active_slots.front()->sample.data(),
                                                    input->active_selection,
                                                    &input->view_generation,
                                                    input->view_generation);
      if (!constructed) {
        return make_unexpected(std::move(constructed).error());
      }
      input->predicate_view_constructed = true;
    }
  }
  return {};
}

void RuntimeChannelSession::Impl::commit_activation_prepare(
    ActivationPrepareTransaction& transaction) noexcept {
  for (auto* input : transaction.owner->inputs) {
    if (input->staged_now) {
      input->selection_prepared = true;
      input->staged_now = false;
    }
  }
}

expected<void, Error> RuntimeChannelSession::prepare_activation(
    const VertexId owner, const std::uint64_t generation,
    const std::span<const RuntimeChannelActivationSelection> selections) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  // Open and bind, stage claims, construct views, then publish. Any pre-publish failure rolls back
  // only the inputs staged by this call; an initial call also closes the new activation shell.
  auto transaction = impl_->begin_activation_prepare(*activation_owner, generation, selections);
  if (!transaction) {
    return make_unexpected(std::move(transaction).error());
  }
  auto staged = impl_->stage_activation_selections(*transaction, selections);
  if (!staged) {
    impl_->rollback_activation_prepare(*transaction);
    return staged;
  }
  auto views = impl_->construct_activation_predicate_views(*transaction);
  if (!views) {
    impl_->rollback_activation_prepare(*transaction);
    return views;
  }
  Impl::commit_activation_prepare(*transaction);
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelPrepareCalls);
  return {};
}

expected<void, Error> RuntimeChannelSession::prepare_tensor_access(
    const VertexId owner, const std::uint64_t generation, cudaStream_t consumer_stream) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  for (const auto* input : activation_owner->inputs) {
    if (!input->active || input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (const auto* output : activation_owner->outputs) {
    if (!output->active || output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (auto* input : activation_owner->inputs) {
    auto prepared = Impl::prepare_tensor_input_access(*input, consumer_stream);
    if (!prepared) {
      return prepared;
    }
  }
  return {};
}

// Validate the complete selected set before any activation-retirement mutation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::Impl::validate_activation_retirement(
    const ActivationOwnerState& owner, const std::uint64_t generation,
    const bool activation_accepted) const noexcept {
  for (const auto* input : owner.inputs) {
    if (!input->active || input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (input->view_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  if (activation_accepted) {
    for (const auto* input : owner.inputs) {
      if (!input->destructive) {
        continue;
      }
      for (const SampleSlot* slot : input->active_slots) {
        const std::scoped_lock lock{slot->subscriber->mutex};
        if (slot->event_emitted && !slot->witness_retired) {
          return make_unexpected(Error{ErrorCode::kBackpressured});
        }
      }
      const bool value_state = input->active_selection.state == InputState::kSelected ||
                               input->active_selection.state == InputState::kReused;
      if (input->reuse_last && value_state && input->active_slots.size() == 1U &&
          input->reuse_slot != nullptr && input->reuse_slot != input->active_slots.front()) {
        const std::scoped_lock lock{input->reuse_slot->subscriber->mutex};
        if (input->reuse_slot->active_selected ||
            (input->reuse_slot->event_emitted && !input->reuse_slot->witness_retired)) {
          return make_unexpected(Error{ErrorCode::kBackpressured});
        }
      }
    }
  }
  for (const auto* output : owner.outputs) {
    if (!output->active || output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (std::any_of(output->loans.begin(), output->loans.end(), [](const auto* loan) {
          return loan->live;
        })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }

  return {};
}

// Retire typed, direct-ingress, and reuse claims in one exact activation transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::Impl::retire_activation_inputs(
    ActivationOwnerState& owner, const bool activation_accepted,
    const bool signal_direct_progress) noexcept {
  for (auto* input : owner.inputs) {
    if (input->predicate_view_constructed) {
      input->type_ops->destroy_predicate_view(input->predicate_view_storage.data());
      input->predicate_view_constructed = false;
    }
    const bool value_state = input->active_selection.state == InputState::kSelected ||
                             input->active_selection.state == InputState::kReused;
    SampleSlot* retained = activation_accepted && input->destructive && input->reuse_last &&
                                   value_state && input->active_slots.size() == 1U
                               ? input->active_slots.front()
                               : nullptr;
    if (retained != nullptr && input->reuse_slot != nullptr && input->reuse_slot != retained) {
      SampleSlot* prior = input->reuse_slot;
      const std::scoped_lock lock{prior->subscriber->mutex};
      prior->reuse_retained = false;
      reclaim_sample_slot_if_unclaimed(*prior);
    }
    for (SampleSlot* slot : input->active_slots) {
      const std::scoped_lock lock{slot->subscriber->mutex};
      slot->active_selected = false;
      if (slot == retained) {
        slot->reuse_retained = true;
      } else {
        // Selection commit/rollback owns the temporal claim; activation retirement owns only the
        // activation and finite reuse claims. Reclaiming through the shared gate preserves scalar
        // rollback, overlapping windows, and cache-retained non-destructive selections.
        reclaim_sample_slot_if_unclaimed(*slot);
      }
    }
    if (retained != nullptr) {
      input->reuse_slot = retained;
    }

    for (SubscriberState* subscriber : input->subscribers) {
      bool direct_capacity_may_have_changed = false;
      {
        const std::scoped_lock lock{subscriber->mutex};
        auto promoted = promote_direct_ingress(*subscriber);
        if (!promoted) {
          return make_unexpected(std::move(promoted).error());
        }
        for (Impl::SampleSlot& slot : subscriber->slots) {
          count_phase(phase_counters_, RuntimePhaseCounter::kChannelStagedEvidenceSweepVisits);
          auto published = publish_staged_arrival_evidence_if_eligible(*subscriber, slot);
          if (!published) {
            return make_unexpected(std::move(published).error());
          }
        }
        direct_capacity_may_have_changed =
            signal_direct_progress && subscriber->direct_process_local &&
            (subscriber->direct_ingress_capacity != 0U
                 ? writable_direct_ingress_slot(*subscriber) != nullptr
                 : writable_slot(*subscriber, WritableSlotIntent::kProbe) != nullptr);
      }
      if (direct_capacity_may_have_changed) {
        signal_owner_progress(subscriber->source_owner);
      }
    }
    input->tensor_read_leases.clear();
    input->tensor_access_slots.clear();
    input->active_slots.clear();
    input->selection_prepared = false;
    input->staged_now = false;
    input->active = false;
    ++input->view_generation;
  }
  return {};
}

void RuntimeChannelSession::Impl::finish_activation_retirement(
    ActivationOwnerState& owner) noexcept {
  for (auto* output : owner.outputs) {
    output->active = false;
    output->direct_continuation_capture = false;
  }
  owner.direct_continuation.reset();
}

expected<void, Error> RuntimeChannelSession::retire_activation(
    const VertexId owner, const std::uint64_t generation, const bool activation_accepted) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  // Validate every fallible claim before mutation, retire input/reuse state, then close outputs.
  // Direct-ingress promotion preserves its existing per-input failure point during commit.
  auto valid =
      impl_->validate_activation_retirement(*activation_owner, generation, activation_accepted);
  if (!valid) {
    return valid;
  }
  if (impl_->activation_retirement_observer_ != nullptr) {
    auto observed = impl_->activation_retirement_observer_(
        impl_->activation_retirement_observer_state_, owner, generation, activation_accepted);
    if (!observed) {
      return observed;
    }
  }
  auto inputs = impl_->retire_activation_inputs(
      *activation_owner, activation_accepted, impl_->owner_progress_wake_enabled());
  if (!inputs) {
    return inputs;
  }
  Impl::finish_activation_retirement(*activation_owner);
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelRetireCalls);
  return impl_->retry_blocked_process_local_transport();
}

expected<bool, Error> RuntimeChannelSession::has_freshness_refill_candidate(
    const PortId port, const TemporalSampleRecord& retained) const noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const Impl::SubscriberState* retained_subscriber = nullptr;
  bool candidate = false;
  for (const Impl::SubscriberState* const subscriber : input->subscribers) {
    const std::unique_lock lock{subscriber->mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
      return false;
    }
    const auto retained_slot =
        std::find_if(subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
          return slot.occupied && !slot.direct_ingress_unread && slot.temporal == retained;
        });
    if (retained_slot == subscriber->slots.end()) {
      continue;
    }
    if (retained_subscriber != nullptr) {
      // No subscriber-local sequence space can prove which matching claim owns a successor.
      // Nothing has been mutated, so ambiguity is a safe decline rather than a session failure.
      return false;
    }
    retained_subscriber = subscriber;
    const bool reuse_exact = retained_slot->reuse_retained && input->reuse_slot == &*retained_slot;
    const bool reclaimable =
        retained_slot->temporal_retained && !retained_slot->retirement_selected &&
        !retained_slot->typed_ingress_reserved && !retained_slot->active_selected &&
        (!retained_slot->reuse_retained || reuse_exact) &&
        (!retained_slot->event_emitted || retained_slot->witness_retired);
    candidate =
        reclaimable &&
        std::any_of(subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
          return slot.occupied && slot.direct_ingress_unread && !slot.staged_arrival_evidence &&
                 !slot.typed_ingress_reserved && is_strictly_newer_sample(slot.temporal, retained);
        });
  }
  return retained_subscriber != nullptr && candidate;
}

// Keep staged and retained identity checks under one serialized subscriber observation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::optional<RuntimeChannelStagedReuseCandidate>, Error>
RuntimeChannelSession::staged_reuse_replacement_candidate(
    const PortId port, const TemporalSampleRecord& retained) const noexcept {
  if (!impl_ || !port || retained.source_sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const Impl::SubscriberState* match = nullptr;
  std::optional<RuntimeChannelStagedReuseCandidate> candidate;
  for (std::size_t index = 0U; index < input->subscribers.size(); ++index) {
    const Impl::SubscriberState* const subscriber = input->subscribers[index];
    if (subscriber == nullptr || index > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::scoped_lock transport_lock{subscriber->transport_mutex};
    const auto staged = const_cast<Impl::SubscriberState*>(subscriber)->staged_data_state();
    if (!staged || staged->sequence <= retained.source_sequence) {
      continue;
    }
    const std::scoped_lock cache_lock{subscriber->mutex};
    const auto slot = std::find_if(
        subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& candidate) {
          return candidate.occupied && !candidate.direct_ingress_unread &&
                 candidate.temporal == retained && candidate.temporal_retained &&
                 (!candidate.reuse_retained || input->reuse_slot == &candidate) &&
                 !candidate.active_selected && !candidate.retirement_selected &&
                 !candidate.typed_ingress_reserved &&
                 (!candidate.event_emitted || candidate.witness_retired);
        });
    if (slot == subscriber->slots.end()) {
      continue;
    }
    if (match != nullptr) {
      return std::optional<RuntimeChannelStagedReuseCandidate>{};
    }
    TimePoint reference;
    if (staged->capture_time.valid()) {
      TimePoint normalized = staged->capture_time;
      if (subscriber->externally_driven_import) {
        if (!subscriber->catalog_capture_clock.valid() || !subscriber->plan_capture_clock.valid() ||
            normalized.clock_id != subscriber->catalog_capture_clock) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        normalized.clock_id = subscriber->plan_capture_clock;
      }
      if (!input->capture_clock.valid() || !input->reference_clock.valid() ||
          normalized.clock_id != input->capture_clock) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto converted = convert_identical_or_direct(
          normalized,
          input->reference_clock,
          input->capture_to_reference ? &*input->capture_to_reference : nullptr);
      if (!converted) {
        return make_unexpected(std::move(converted).error());
      }
      reference = *converted;
    }
    match = subscriber;
    candidate = RuntimeChannelStagedReuseCandidate{
        .subscriber_index = static_cast<std::uint32_t>(index),
        .source_sequence = staged->sequence,
        .source_capture = staged->capture_time,
        .successor_reference = reference,
    };
  }
  return match != nullptr ? candidate : std::optional<RuntimeChannelStagedReuseCandidate>{};
}

expected<bool, Error> RuntimeChannelSession::input_terminal_observed(
    const PortId port) const noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (input->subscribers.size() != 1U || input->subscribers.front() == nullptr) {
    return false;
  }
  Impl::SubscriberState& subscriber = *input->subscribers.front();
  const std::scoped_lock transport_lock{subscriber.transport_mutex};
  // A staged trigger record precedes any staged terminal in the endpoint's ordered stream. Until
  // that record is admitted, the coordinator cannot see it and therefore cannot prove that the
  // retained trigger value is unobservable merely from its pending EOS event.
  const auto staged_data = subscriber.staged_data_state();
  const auto staged_terminal = subscriber.staged_terminal_sequence();
  const std::scoped_lock cache_lock{subscriber.mutex};
  if (staged_data || subscriber.has_pending_direct_ingress()) {
    count_phase(impl_->phase_counters_,
                RuntimePhaseCounter::kChannelProtectedTerminalProbesWithPendingData);
    return false;
  }
  return subscriber.end_of_stream_received || staged_terminal.has_value();
}

expected<TemporalSampleRecord, Error> RuntimeChannelSession::replace_staged_reuse(
    const PortId port, const TemporalSampleRecord& retained,
    const RuntimeChannelStagedReuseCandidate& candidate) noexcept {
  if (!impl_ || !port || retained.source_sequence == 0U || candidate.source_sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (candidate.subscriber_index >= input->subscribers.size() ||
      input->subscribers[candidate.subscriber_index] == nullptr) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  Impl::SubscriberState* const match = input->subscribers[candidate.subscriber_index];
  Impl::SampleSlot* replacement_slot = nullptr;
  bool replacing_reuse = false;
  ErrorCode injected_error = ErrorCode::kSuccess;
  {
    const std::scoped_lock cache_lock{match->mutex};
    const auto owned =
        std::find_if(match->slots.begin(), match->slots.end(), [&](const auto& slot) {
          return slot.occupied && !slot.direct_ingress_unread && slot.temporal == retained &&
                 slot.temporal_retained && (!slot.reuse_retained || input->reuse_slot == &slot);
        });
    if (owned == match->slots.end()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    replacement_slot = &*owned;
    replacing_reuse = replacement_slot->reuse_retained;
    injected_error = std::exchange(match->retryable_reuse_replacement_receive_error_for_test,
                                   ErrorCode::kSuccess);
  }
  if (injected_error != ErrorCode::kSuccess) {
    return make_unexpected(Error{injected_error});
  }
  const channel_internal::StagedDataState expected_staged{
      .sequence = candidate.source_sequence,
      .capture_time = candidate.source_capture,
  };
  auto replaced = impl_->receive_one(*match, nullptr, nullptr, retained, &expected_staged);
  if (!replaced) {
    return make_unexpected(std::move(replaced).error());
  }
  // This controller-owned receive bypasses the ordinary drain loops, so it must retire the one
  // process-local publication it just consumed itself. The earlier backpressured probe consumed
  // nothing and therefore contributed no receipt; counting only this successful receive preserves
  // exact published/received equality and clears activation-cut retry authority exactly once.
  if (match->process_local_transport) {
    impl_->record_process_local_receive();
  }
  const std::scoped_lock cache_lock{match->mutex};
  if (replacement_slot == nullptr ||
      (replacing_reuse ? input->reuse_slot != nullptr : input->reuse_slot == replacement_slot) ||
      !replacement_slot->occupied || replacement_slot->direct_ingress_unread ||
      replacement_slot->temporal == retained ||
      !is_strictly_newer_sample(replacement_slot->temporal, retained) ||
      std::any_of(match->slots.begin(), match->slots.end(), [&](const auto& candidate) {
        return candidate.occupied && candidate.temporal == retained;
      })) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return replacement_slot->temporal;
}

expected<bool, Error> RuntimeChannelSession::supports_freshness_refill(
    const PortId port) const noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return std::any_of(input->subscribers.begin(), input->subscribers.end(), [&](auto* subscriber) {
    if (subscriber == nullptr) {
      return false;
    }
    const bool direct_freshness = subscriber->queue_policy == QueuePolicy::kDropOldest &&
                                  subscriber->direct_ingress_capacity != 0U;
    return direct_freshness ||
           (input->reuse_last && subscriber->supports_staged_data_replacement());
  });
}

expected<bool, Error> RuntimeChannelSession::supports_drop_oldest_freshness_refill(
    const PortId port) const noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return std::any_of(input->subscribers.begin(), input->subscribers.end(), [&](auto* subscriber) {
    return subscriber != nullptr && subscriber->queue_policy == QueuePolicy::kDropOldest &&
           (subscriber->direct_ingress_capacity != 0U ||
            (input->reuse_last && subscriber->supports_staged_data_replacement()));
  });
}

expected<bool, Error> RuntimeChannelSession::recover_abandoned_freshness_reservation(
    const PortId port, const bool preserve_uncommitted_target) noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  Impl::SubscriberState* const subscriber = input->freshness_reservation_subscriber;
  if (subscriber == nullptr) {
    return true;
  }

  bool direct_capacity_may_have_changed = false;
  {
    // Recovery runs on a freshness-only dispatcher path. Never wait on a producer: an abandoned
    // reservation remains generation-fenced, so declining this pass is safe and bounded.
    const std::unique_lock lock{subscriber->mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
      return false;
    }
    if (input->freshness_reservation_subscriber != subscriber) {
      return true;
    }
    const Impl::SampleSlot* const retained = input->freshness_reservation_retained;
    const bool retained_claim_live =
        retained != nullptr && retained->subscriber == subscriber && retained->occupied &&
        !retained->direct_ingress_unread && retained->temporal_retained &&
        !retained->retirement_selected && !retained->typed_ingress_reserved &&
        !retained->active_selected;
    // A caller about to choose again may keep its speculative target long enough to re-reserve it.
    // Standalone recovery owns no such continuation and releases an uncommitted target even while
    // the retained claim is live. A committed predecessor always leaves a durable prefix fence.
    const bool keep_recovery_marker = preserve_uncommitted_target && retained_claim_live &&
                                      !input->freshness_refill_prefix_committed;
    const bool preserve_target = input->freshness_refill_prefix_committed || keep_recovery_marker;
    if (!preserve_target && !Impl::normalize_direct_ingress_after_freshness_release(*subscriber)) {
      return false;
    }
    if (input->freshness_reservation_successor != nullptr &&
        input->freshness_reservation_successor != input->freshness_refill_target) {
      input->freshness_reservation_successor->refill_prefix_reserved = false;
    }
    if (!preserve_target && input->freshness_refill_target != nullptr) {
      input->freshness_refill_target->refill_reserved = false;
      input->freshness_refill_target = nullptr;
      input->freshness_refill_target_reference = {};
    }
    // When reserve_freshness_refill() is about to choose again, retain this existing pointer as a
    // recovery marker until the new reservation is published. Any intervening early return then
    // remains self-healing; standalone recovery clears the marker with the speculative target.
    input->freshness_reservation_subscriber = keep_recovery_marker ? subscriber : nullptr;
    input->freshness_reservation_retained = nullptr;
    input->freshness_reservation_successor = nullptr;
    input->freshness_reservation_generation = 0U;

    // The retained claim may have disappeared while release was deferred. Promote now so an idle
    // port publishes a fresh revision instead of waiting for another producer arrival to revisit
    // the staged record.
    auto promoted = impl_->promote_direct_ingress(*subscriber);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    direct_capacity_may_have_changed =
        impl_->owner_progress_wake_enabled() && subscriber->direct_process_local &&
        (subscriber->direct_ingress_capacity != 0U
             ? impl_->writable_direct_ingress_slot(*subscriber) != nullptr
             : impl_->writable_slot(*subscriber, Impl::WritableSlotIntent::kProbe) != nullptr);
  }
  if (direct_capacity_may_have_changed) {
    impl_->signal_owner_progress(subscriber->source_owner);
  }
  auto retried = impl_->retry_blocked_process_local_transport();
  if (!retried) {
    return make_unexpected(std::move(retried).error());
  }
  return true;
}

// Terminal generation retirement is a two-phase owner transaction: prove every activation,
// transport, payload, loan, and EOS obligation quiescent before releasing finite reuse claims.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<bool, Error> RuntimeChannelSession::retire_terminal_input_generation(
    const VertexId owner) noexcept {
  // Preserve terminal history until EOS is committed and no activation still borrows it.
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return true;
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  if (!impl_->process_local_progress_tracking_valid_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (impl_->blocked_transport_retry_required_.load(std::memory_order_acquire) ||
      impl_->process_local_publications_in_flight_.load(std::memory_order_acquire) != 0U ||
      impl_->process_local_published_deliveries_.load(std::memory_order_acquire) !=
          impl_->process_local_received_deliveries_.load(std::memory_order_acquire)) {
    return false;
  }
  for (Impl::InputPortState* const input : activation_owner->inputs) {
    if (input == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (input->active || input->selection_prepared || input->staged_now ||
        input->predicate_view_constructed || !input->active_slots.empty() ||
        !input->retirement_slots.empty() || !input->tensor_access_slots.empty() ||
        !input->tensor_read_leases.empty() ||
        (input->claim_pool != nullptr && input->claim_pool->has_live_claims())) {
      return false;
    }
  }
  for (Impl::OutputState* const output : activation_owner->outputs) {
    if (output == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (output->active || output->live_loan_count.load(std::memory_order_acquire) != 0U ||
        std::any_of(output->loans.begin(), output->loans.end(), [](const auto* loan) {
          return loan != nullptr && loan->live;
        })) {
      return false;
    }
  }
  // Preflight every subscriber before releasing any finite reuse claim. EOS is explicit control
  // state; an empty cache or transport poll alone never qualifies the owner for propagation.
  for (Impl::SubscriberState* const subscriber : activation_owner->subscribers) {
    if (subscriber == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::scoped_lock lock{subscriber->mutex};
    if (!subscriber->end_of_stream_received || !subscriber->end_of_stream_event_emitted ||
        subscriber->has_pending_direct_ingress() ||
        std::any_of(subscriber->slots.begin(), subscriber->slots.end(), [](const auto& slot) {
          return slot.direct_ingress_unread || slot.typed_ingress_reserved ||
                 slot.active_selected || slot.temporal_retained || slot.retirement_selected ||
                 slot.owed_to_released_claim;
        })) {
      return false;
    }
  }
  // All externally meaningful claims are gone. ReuseLast is a finite selector-owned convenience,
  // not a reason to keep an EOS-closed producer generation alive, so release it in one monotonic
  // owner transaction and reclaim the now-unobservable typed slots.
  for (Impl::InputPortState* const input : activation_owner->inputs) {
    input->reuse_slot = nullptr;
    for (Impl::SubscriberState* const subscriber : input->subscribers) {
      const std::scoped_lock lock{subscriber->mutex};
      for (Impl::SampleSlot& slot : subscriber->slots) {
        slot.reuse_retained = false;
        Impl::reclaim_sample_slot_if_unclaimed(slot);
        if (slot.occupied) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
    }
  }
  return true;
}

// Reserve the exact staged-or-retained replacement without publishing partial ownership.
expected<std::optional<RuntimeChannelFreshnessReservation>, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::reserve_freshness_refill(
    PortId port, const TemporalSampleRecord& retained, std::optional<TimePoint> floor,
    const RuntimeChannelFreshnessSelectionMode selection_mode) noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  // A live reservation here is speculative and must be rolled back before a new choice. A target
  // that already committed a prefix step remains protected by recovery itself; an uncommitted
  // target must not strand headroom after its retained claim disappears.
  auto recovered = recover_abandoned_freshness_reservation(port, true);
  if (!recovered) {
    return make_unexpected(std::move(recovered).error());
  }
  if (!*recovered) {
    return std::optional<RuntimeChannelFreshnessReservation>{};
  }

  Impl::SubscriberState* retained_subscriber = nullptr;
  for (Impl::SubscriberState* const subscriber : input->subscribers) {
    // Never block here. This runs on every dispatch round against a producer that holds this lock
    // constantly, and waiting would cost the dispatcher far more than the freshness refresh is
    // worth. Losing the lock declines the eviction, which is always safe.
    const std::unique_lock lock{subscriber->mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
      return std::optional<RuntimeChannelFreshnessReservation>{};
    }
    const auto found =
        std::find_if(subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
          return slot.occupied && !slot.direct_ingress_unread && slot.temporal == retained;
        });
    if (found == subscriber->slots.end()) {
      continue;
    }
    if (retained_subscriber != nullptr) {
      // The reservation has not been published and no claim has been released. Preserve both
      // matching subscribers unchanged and let ordinary admission make progress.
      return std::optional<RuntimeChannelFreshnessReservation>{};
    }
    retained_subscriber = subscriber;
  }

  if (retained_subscriber == nullptr) {
    return std::optional<RuntimeChannelFreshnessReservation>{};
  }

  // Revalidate the retained claim and choose its successor while holding their one shared mutex.
  // A successor staged by a different subscriber is not comparable and cannot refill this slot.
  const std::unique_lock lock{retained_subscriber->mutex, std::try_to_lock};
  if (!lock.owns_lock()) {
    return std::optional<RuntimeChannelFreshnessReservation>{};
  }
  auto retained_slot = std::find_if(
      retained_subscriber->slots.begin(), retained_subscriber->slots.end(), [&](const auto& slot) {
        return slot.occupied && !slot.direct_ingress_unread && slot.temporal == retained;
      });
  if (retained_slot == retained_subscriber->slots.end() || !retained_slot->temporal_retained ||
      retained_slot->retirement_selected || retained_slot->typed_ingress_reserved ||
      retained_slot->active_selected ||
      (retained_slot->reuse_retained && input->reuse_slot != &*retained_slot) ||
      (retained_slot->event_emitted && !retained_slot->witness_retired)) {
    return std::optional<RuntimeChannelFreshnessReservation>{};
  }

  // A non-trigger LatestWithin side input freezes its newest witness-qualified record as the
  // target, then hands out the target's surviving producer-order prefix one record at a time.
  // Exact instead exposes only the oldest staged producer successor, matching a transported
  // queue head rather than scanning past it to a later exact match. Other selector paths retain
  // the established newest-eligible choice. All remain one bounded scan of staging.
  Impl::SampleSlot* const prior_target = input->freshness_refill_target;
  const bool target_is_committed = input->freshness_refill_prefix_committed;
  const bool preserve_prefix =
      selection_mode == RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix;
  if (target_is_committed && !preserve_prefix) {
    // Once a prefix successor is published, its target is an arrival-order fence. Reinterpreting
    // that target under a single-record mode could let newer producer traffic overtake it.
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (prior_target != nullptr &&
      (prior_target->subscriber != retained_subscriber || !prior_target->occupied ||
       !prior_target->direct_ingress_unread || !prior_target->refill_reserved ||
       prior_target->refill_prefix_reserved || prior_target->staged_arrival_evidence ||
       prior_target->typed_ingress_reserved || !input->freshness_refill_target_reference.valid() ||
       !is_strictly_newer_sample(prior_target->temporal, retained))) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  Impl::SampleSlot* target = target_is_committed ? prior_target : nullptr;
  TimePoint target_reference =
      target_is_committed ? input->freshness_refill_target_reference : TimePoint{};
  if (!target_is_committed) {
    for (Impl::SampleSlot& slot : retained_subscriber->slots) {
      if (!slot.occupied || !slot.direct_ingress_unread || slot.staged_arrival_evidence ||
          slot.typed_ingress_reserved || slot.refill_prefix_reserved ||
          (slot.refill_reserved && &slot != prior_target)) {
        continue;
      }
      const TimePoint capture = slot.temporal.capture_time;
      if (!capture.valid() || !input->capture_clock.valid() || !input->reference_clock.valid() ||
          capture.clock_id != input->capture_clock) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto reference = convert_identical_or_direct(
          capture,
          input->reference_clock,
          input->capture_to_reference ? &*input->capture_to_reference : nullptr);
      if (!reference) {
        return make_unexpected(std::move(reference).error());
      }
      const bool oldest_producer_successor =
          selection_mode == RuntimeChannelFreshnessSelectionMode::kOldestProducerSuccessor;
      if (!oldest_producer_successor && floor &&
          (reference->clock_id != floor->clock_id ||
           reference->timestamp_ns > floor->timestamp_ns)) {
        continue;
      }
      if (!is_strictly_newer_sample(slot.temporal, retained)) {
        continue;
      }
      const bool preferred = [&]() noexcept {
        if (target == nullptr) {
          return true;
        }
        if (oldest_producer_successor) {
          return is_strictly_newer_sample(target->temporal, slot.temporal);
        }
        if (selection_mode == RuntimeChannelFreshnessSelectionMode::kNewest) {
          return is_strictly_newer_sample(slot.temporal, target->temporal);
        }
        // LatestWithin ranks causal records by normalized capture coordinate. Producer order is
        // only the stable tie-breaker and the later admission fence because capture coordinates
        // may be non-monotonic within one epoch.
        return reference->timestamp_ns > target_reference.timestamp_ns ||
               (reference->timestamp_ns == target_reference.timestamp_ns &&
                is_strictly_newer_sample(slot.temporal, target->temporal));
      }();
      if (preferred) {
        target = &slot;
        target_reference = *reference;
      }
    }
  }
  if (target == nullptr) {
    if (prior_target != nullptr) {
      prior_target->refill_reserved = false;
      if (!Impl::normalize_direct_ingress_after_freshness_release(*retained_subscriber)) {
        prior_target->refill_reserved = true;
        return std::optional<RuntimeChannelFreshnessReservation>{};
      }
      input->freshness_refill_target = nullptr;
      input->freshness_refill_target_reference = {};
      input->freshness_refill_prefix_committed = false;
    }
    return std::optional<RuntimeChannelFreshnessReservation>{};
  }

  Impl::SampleSlot* successor = preserve_prefix ? nullptr : target;
  TimePoint successor_reference;
  if (!preserve_prefix) {
    successor_reference = target_reference;
  }
  for (Impl::SampleSlot& slot : retained_subscriber->slots) {
    if (!preserve_prefix) {
      break;
    }
    const bool at_or_before_target = slot.temporal == target->temporal ||
                                     is_strictly_newer_sample(target->temporal, slot.temporal);
    if (!slot.occupied || !slot.direct_ingress_unread || slot.staged_arrival_evidence ||
        slot.typed_ingress_reserved || slot.refill_prefix_reserved ||
        (&slot != target && &slot != prior_target && slot.refill_reserved) ||
        !at_or_before_target || slot.temporal.producer_epoch != target->temporal.producer_epoch ||
        !is_strictly_newer_sample(slot.temporal, retained)) {
      continue;
    }
    if (successor == nullptr || is_strictly_newer_sample(successor->temporal, slot.temporal)) {
      const TimePoint capture = slot.temporal.capture_time;
      if (!capture.valid() || !input->capture_clock.valid() || !input->reference_clock.valid() ||
          capture.clock_id != input->capture_clock) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto reference = convert_identical_or_direct(
          capture,
          input->reference_clock,
          input->capture_to_reference ? &*input->capture_to_reference : nullptr);
      if (!reference) {
        return make_unexpected(std::move(reference).error());
      }
      successor = &slot;
      successor_reference = *reference;
    }
  }
  if (successor == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (input->next_freshness_reservation_generation == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const std::uint64_t generation = input->next_freshness_reservation_generation++;
  if (prior_target != target) {
    if (prior_target != nullptr) {
      prior_target->refill_reserved = false;
    }
    input->freshness_refill_prefix_committed = false;
  }
  target->refill_reserved = true;
  input->freshness_refill_target = target;
  input->freshness_refill_target_reference = target_reference;
  if (successor != target) {
    successor->refill_prefix_reserved = true;
  }
  input->freshness_reservation_subscriber = retained_subscriber;
  input->freshness_reservation_retained = &*retained_slot;
  input->freshness_reservation_successor = successor;
  input->freshness_reservation_generation = generation;
  return std::optional<RuntimeChannelFreshnessReservation>{RuntimeChannelFreshnessReservation{
      .port = port,
      .successor = successor->temporal,
      .successor_reference = successor_reference,
      .target = target->temporal,
      .target_reference = target_reference,
      .generation = generation,
  }};
}

void RuntimeChannelSession::release_freshness_reservation(
    const RuntimeChannelFreshnessReservation& reservation) noexcept {
  if (!impl_ || !reservation.port || reservation.generation == 0U) {
    return;
  }
  Impl::InputPortState* const input = impl_->find_input(reservation.port);
  if (input == nullptr || input->freshness_reservation_generation != reservation.generation ||
      input->freshness_reservation_subscriber == nullptr) {
    return;
  }
  // The caller is abandoning speculative work from the dispatcher thread. If a producer owns the
  // cache, keep the fenced reservation intact; the next reconcile recovery self-heals it before
  // any credit, claim, or successor gate can skip the input.
  const std::unique_lock lock{input->freshness_reservation_subscriber->mutex, std::try_to_lock};
  if (!lock.owns_lock()) {
    count_phase(impl_->phase_counters_,
                RuntimePhaseCounter::kChannelFreshnessReservationReleaseDeferred);
    return;
  }
  if (input->freshness_reservation_generation != reservation.generation) {
    return;
  }
  if (!input->freshness_refill_prefix_committed &&
      !Impl::normalize_direct_ingress_after_freshness_release(
          *input->freshness_reservation_subscriber)) {
    count_phase(impl_->phase_counters_,
                RuntimePhaseCounter::kChannelFreshnessReservationReleaseDeferred);
    return;
  }
  if (input->freshness_reservation_successor != nullptr &&
      input->freshness_reservation_successor != input->freshness_refill_target) {
    input->freshness_reservation_successor->refill_prefix_reserved = false;
  }
  if (!input->freshness_refill_prefix_committed && input->freshness_refill_target != nullptr) {
    input->freshness_refill_target->refill_reserved = false;
    input->freshness_refill_target = nullptr;
    input->freshness_refill_target_reference = {};
  }
  input->freshness_reservation_subscriber = nullptr;
  input->freshness_reservation_retained = nullptr;
  input->freshness_reservation_successor = nullptr;
  input->freshness_reservation_generation = 0U;
}

expected<std::uint8_t, Error> RuntimeChannelSession::last_refill_outcome(
    PortId port) const noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return input->last_refill_outcome.load(std::memory_order_relaxed);
}

// Terminal fanout keeps every endpoint family and its retry marker in one ordered transaction so a
// partially accepted close resumes without publishing EOS twice.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::close_owner_outputs(const VertexId owner) noexcept {
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return {};
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  const auto consumes_terminal_authority = [](const ErrorCode code) noexcept {
    return publication_error_consumes_authority(code) || code == ErrorCode::kProtocolError ||
           code == ErrorCode::kInvalidArgument || code == ErrorCode::kClosed;
  };
  using TerminalCloseState = Impl::OutputState::TerminalCloseState;
  const auto resolve_terminal_publication = [](Impl::OutputState& output, bool& terminal_resolved) {
    terminal_resolved = true;
    output.terminal_close_state = TerminalCloseState::kAuthorityConsumed;
  };
  const auto reject_terminal_publication = [&](Impl::OutputState& output,
                                               Impl::ProcessLocalPublicationScope& publication,
                                               Error error,
                                               bool& terminal_resolved) {
    if (consumes_terminal_authority(error.code)) {
      publication.invalidate();
      resolve_terminal_publication(output, terminal_resolved);
    } else {
      publication.reject();
      error = normalize_terminal_pre_submit_error(std::move(error));
    }
    return expected<void, Error>{make_unexpected(std::move(error))};
  };
  const auto required_export_peer_vanished = [](const bool external_export,
                                                const bool peer_required,
                                                const std::uint64_t next_sequence,
                                                const bool has_subscribers) noexcept {
    // A sequence beyond the initial value proves that this output committed data before its
    // terminal drain. Do not apply the disappearance policy to a required export that is still
    // waiting for its first peer, nor to a process-local lifecycle replacement window.
    return external_export && peer_required && next_sequence > 1U && !has_subscribers;
  };
  const auto fail_vanished_required_export = [](bool& terminal_resolved) {
    // No receiver can observe a later retry from the retired endpoint generation. Resolve the
    // arm monotonically while failing the run; consumer abandonment is not producer EOS.
    terminal_resolved = true;
    return expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
  };
  for (Impl::OutputState* const output : activation_owner->outputs) {
    if (output == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (output->live_loan_count.load(std::memory_order_acquire) != 0U) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (output->terminal_close_state == TerminalCloseState::kComplete) {
      continue;
    }
    auto refreshed = output->refresh_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (output->direct_subscriber != nullptr) {
      if (output->next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const std::scoped_lock lock{output->direct_subscriber->mutex};
      auto closed = impl_->store_end_of_stream(*output->direct_subscriber, output->next_sequence);
      if (!closed) {
        return make_unexpected(std::move(closed).error());
      }
      ++output->next_sequence;
      output->terminal_close_state = TerminalCloseState::kComplete;
      continue;
    }
    // A prior fanout attempt may have closed one endpoint before another backpressured.
    for (auto& endpoint : output->endpoints) {
      if (endpoint.terminal_resolved) {
        continue;
      }
      if (!endpoint.peer_required && !endpoint.terminal_obligated &&
          !endpoint.publisher->has_subscribers()) {
        endpoint.terminal_resolved = true;
        continue;
      }
      if (required_export_peer_vanished(endpoint.refresh_before_emit,
                                        endpoint.peer_required,
                                        output->next_sequence,
                                        endpoint.publisher->has_subscribers())) {
        return fail_vanished_required_export(endpoint.terminal_resolved);
      }
      // Refresh and local resolution do not require a new sequence coordinate.
      if (output->next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      Impl::ProcessLocalPublicationScope publication{*impl_, endpoint.process_local_transport};
      expected<void, Error> closed;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        closed = endpoint.publisher->close_stream();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        return reject_terminal_publication(*output,
                                           publication,
                                           Error{ErrorCode::kPublicationIndeterminate},
                                           endpoint.terminal_resolved);
      }
#endif
      if (!closed) {
        if (endpoint.refresh_before_emit && endpoint.peer_required && output->next_sequence > 1U &&
            closed.error().code == ErrorCode::kNotReady) {
          publication.reject();
          return fail_vanished_required_export(endpoint.terminal_resolved);
        }
        return reject_terminal_publication(
            *output, publication, std::move(closed).error(), endpoint.terminal_resolved);
      }
      publication.accept(endpoint.expected_recipient_count, endpoint.activation_cut_subscribers);
      resolve_terminal_publication(*output, endpoint.terminal_resolved);
    }
    for (auto& endpoint : output->tensor_endpoints) {
      if (endpoint.terminal_resolved) {
        continue;
      }
      if (output->next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const SampleMetadata metadata{.sequence = output->next_sequence,
                                    .byte_size = 0U,
                                    .payload_kind = PayloadKind::kDescriptorBacked};
      Impl::ProcessLocalPublicationScope publication{*impl_, endpoint.process_local_transport};
      auto closed = channel_internal::TensorPublishLoanAccess::emit_end_of_stream(
          endpoint.publisher, metadata);
      if (!closed) {
        return reject_terminal_publication(
            *output, publication, std::move(closed).error(), endpoint.terminal_resolved);
      }
      if (*closed != endpoint.expected_recipient_count) {
        publication.invalidate();
        resolve_terminal_publication(*output, endpoint.terminal_resolved);
        return make_unexpected(
            *closed < endpoint.expected_recipient_count
                ? make_publication_partial_error(*closed, endpoint.expected_recipient_count)
                : Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
      }
      publication.accept(*closed, endpoint.activation_cut_subscribers);
      resolve_terminal_publication(*output, endpoint.terminal_resolved);
    }
    for (auto& endpoint : output->catalog_tensor_endpoints) {
      if (endpoint.terminal_resolved) {
        continue;
      }
      if (!endpoint.peer_required && !endpoint.terminal_obligated &&
          !endpoint.publisher.has_subscribers()) {
        endpoint.terminal_resolved = true;
        continue;
      }
      if (required_export_peer_vanished(true,
                                        endpoint.peer_required,
                                        output->next_sequence,
                                        endpoint.publisher.has_subscribers())) {
        return fail_vanished_required_export(endpoint.terminal_resolved);
      }
      if (output->next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      expected<void, Error> closed;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        closed = endpoint.publisher.close();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        resolve_terminal_publication(*output, endpoint.terminal_resolved);
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }
#endif
      if (!closed) {
        if (endpoint.peer_required && output->next_sequence > 1U &&
            closed.error().code == ErrorCode::kNotReady) {
          return fail_vanished_required_export(endpoint.terminal_resolved);
        }
        if (consumes_terminal_authority(closed.error().code)) {
          resolve_terminal_publication(*output, endpoint.terminal_resolved);
        }
        return make_unexpected(std::move(closed).error());
      }
      resolve_terminal_publication(*output, endpoint.terminal_resolved);
    }
    for (auto& endpoint : output->tensor_schema_endpoints) {
      if (endpoint.terminal_resolved) {
        continue;
      }
      if (!endpoint.peer_required && !endpoint.terminal_obligated &&
          !endpoint.publisher->has_subscribers()) {
        endpoint.terminal_resolved = true;
        continue;
      }
      if (required_export_peer_vanished(endpoint.refresh_before_emit,
                                        endpoint.peer_required,
                                        output->next_sequence,
                                        endpoint.publisher->has_subscribers())) {
        return fail_vanished_required_export(endpoint.terminal_resolved);
      }
      if (output->next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const auto endpoint_sequence = endpoint.refresh_before_emit
                                         ? endpoint.publisher->next_operation_sequence()
                                         : expected<std::uint64_t, Error>{output->next_sequence};
      if (!endpoint_sequence) {
        return make_unexpected(std::move(endpoint_sequence).error());
      }
      const SampleMetadata metadata{.sequence = *endpoint_sequence,
                                    .byte_size = 0U,
                                    .payload_kind = PayloadKind::kDescriptorBacked};
      Impl::ProcessLocalPublicationScope publication{*impl_, endpoint.process_local_transport};
      channel_internal::InlinePublicationResult closed;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        closed = endpoint.publisher->close_stream(metadata);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        publication.invalidate();
        resolve_terminal_publication(*output, endpoint.terminal_resolved);
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }
#endif
      if (closed.phase == channel_internal::InlinePublicationPhase::kPreSubmitRejected) {
        publication.reject();
        if (endpoint.refresh_before_emit && endpoint.peer_required && output->next_sequence > 1U &&
            closed.error.code == ErrorCode::kNotReady) {
          return fail_vanished_required_export(endpoint.terminal_resolved);
        }
        return make_unexpected(normalize_terminal_pre_submit_error(std::move(closed.error)));
      }
      if (closed.phase == channel_internal::InlinePublicationPhase::kPostSubmitUncertain) {
        publication.invalidate();
        resolve_terminal_publication(*output, endpoint.terminal_resolved);
        return make_unexpected(std::move(closed.error));
      }
      if (closed.recipient_count != endpoint.expected_recipient_count) {
        publication.invalidate();
        resolve_terminal_publication(*output, endpoint.terminal_resolved);
        return make_unexpected(
            closed.recipient_count < endpoint.expected_recipient_count
                ? make_publication_partial_error(closed.recipient_count,
                                                 endpoint.expected_recipient_count)
                : Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
      }
      publication.accept(closed.recipient_count, endpoint.activation_cut_subscribers);
      resolve_terminal_publication(*output, endpoint.terminal_resolved);
    }
    // A retry can finish without submitting anything itself. Account once for any earlier
    // authority-consuming submission, but not for an entirely local terminal disposition.
    if (output->terminal_close_state == TerminalCloseState::kAuthorityConsumed) {
      ++output->next_sequence;
    }
    output->terminal_close_state = TerminalCloseState::kComplete;
  }
  return {};
}

expected<const void*, Error> RuntimeChannelSession::predicate_view(
    PortId port, std::uint64_t generation) const noexcept {
  if (!impl_ || !port || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!input->active || input->active_generation != generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (!input->predicate_view_constructed) {
    return make_unexpected(Error{input->active_selection.state == InputState::kNotSelected
                                     ? ErrorCode::kInputNotSelected
                                     : ErrorCode::kInputEmpty});
  }
  return input->predicate_view_storage.data();
}

void RuntimeChannelSession::Impl::rollback_sample_retirement(InputPortState& input) noexcept {
  for (SampleSlot* slot : input.retirement_slots) {
    const std::scoped_lock lock{slot->subscriber->mutex};
    slot->retirement_selected = false;
  }
  input.retirement_slots.clear();
}

// Select and mark the complete retirement proof set before committing any slot mutation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::Impl::select_sample_retirement(
    InputPortState& input, const std::span<const TemporalSampleRecord> samples,
    const bool terminal_history_release) noexcept {
  input.retirement_slots.clear();
  for (const auto& record : samples) {
    if (input.retirement_slots.size() == input.retirement_slots.capacity()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    SampleSlot* match = nullptr;
    bool record_seen = false;
    for (SubscriberState* subscriber : input.subscribers) {
      bool duplicate = false;
      bool unavailable = false;
      bool protected_claim = false;
      {
        const std::scoped_lock lock{subscriber->mutex};
        const auto found =
            std::find_if(subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
              return slot.occupied &&
                     (!slot.direct_ingress_unread || slot.staged_arrival_evidence) &&
                     slot.temporal == record;
            });
        if (found != subscriber->slots.end()) {
          duplicate = record_seen;
          record_seen = true;
          // Ordinary selector retirement may run while the current activation still borrows the
          // selected payload; physical reclamation is then deferred until retire_activation(). A
          // retained sample outside such an activation remains fail-closed unless this is the
          // separately validated EOS terminal-history boundary.
          protected_claim = found->reuse_retained && !input.active && !terminal_history_release;
          unavailable = !found->temporal_retained || found->retirement_selected ||
                        found->typed_ingress_reserved || (found->active_selected && !input.active);
          if (!duplicate && !unavailable && !protected_claim) {
            found->retirement_selected = true;
            match = &*found;
          }
        }
      }
      if (duplicate || unavailable || protected_claim) {
        if (match != nullptr) {
          const std::scoped_lock lock{match->subscriber->mutex};
          match->retirement_selected = false;
        }
        return make_unexpected(Error{duplicate         ? ErrorCode::kProtocolError
                                     : protected_claim ? ErrorCode::kBackpressured
                                                       : ErrorCode::kStaleEndpoint});
      }
    }
    if (match == nullptr) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    input.retirement_slots.push_back(match);
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::Impl::validate_sample_retirement(
    const InputPortState& input, const bool terminal_history_release) noexcept {
  for (SampleSlot* slot : input.retirement_slots) {
    bool unavailable = false;
    {
      const std::scoped_lock lock{slot->subscriber->mutex};
      unavailable = !slot->occupied || !slot->temporal_retained || !slot->retirement_selected;
    }
    if (unavailable) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  if (terminal_history_release) {
    std::size_t retained_claims = 0U;
    for (SubscriberState* const subscriber : input.subscribers) {
      const std::scoped_lock lock{subscriber->mutex};
      retained_claims += static_cast<std::size_t>(
          std::count_if(subscriber->slots.begin(), subscriber->slots.end(), [](const auto& slot) {
            return slot.temporal_retained;
          }));
    }
    const bool reuse_slot_selected =
        input.reuse_slot == nullptr ||
        std::find(input.retirement_slots.begin(), input.retirement_slots.end(), input.reuse_slot) !=
            input.retirement_slots.end();
    const bool reuse_claim_exact = std::all_of(
        input.retirement_slots.begin(), input.retirement_slots.end(), [&](const auto* slot) {
          return !slot->reuse_retained || slot == input.reuse_slot;
        });
    if (retained_claims != input.retirement_slots.size() || !reuse_slot_selected ||
        !reuse_claim_exact || (input.reuse_slot != nullptr && !input.reuse_slot->reuse_retained)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::Impl::commit_sample_retirement(
    InputPortState& input, const bool terminal_history_release, const bool signal_direct_progress,
    std::uint32_t* const promoted_out) noexcept {
  for (SampleSlot* slot : input.retirement_slots) {
    const std::scoped_lock lock{slot->subscriber->mutex};
    slot->temporal_retained = false;
    if (terminal_history_release) {
      slot->reuse_retained = false;
    }
    slot->retirement_selected = false;
    reclaim_sample_slot_if_unclaimed(*slot);
  }
  if (terminal_history_release) {
    input.reuse_slot = nullptr;
  }
  for (SubscriberState* subscriber : input.subscribers) {
    bool direct_capacity_may_have_changed = false;
    {
      const std::scoped_lock lock{subscriber->mutex};
      auto promoted = promote_direct_ingress(*subscriber);
      if (!promoted) {
        input.retirement_slots.clear();
        return make_unexpected(std::move(promoted).error());
      }
      // A retirement that frees a claim and still promotes nothing while a successor is staged
      // is diagnostic only on this path. The ordinary publish path legitimately sees a full
      // depth-one typed slot on nearly every sample.
      if (*promoted == 0U && subscriber->has_pending_direct_ingress()) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelRetirementPromotedNothing);
      }
      if (promoted_out != nullptr) {
        *promoted_out += *promoted;
      }
      direct_capacity_may_have_changed =
          signal_direct_progress && subscriber->direct_process_local &&
          (subscriber->direct_ingress_capacity != 0U
               ? writable_direct_ingress_slot(*subscriber) != nullptr
               : writable_slot(*subscriber, WritableSlotIntent::kProbe) != nullptr);
    }
    if (direct_capacity_may_have_changed) {
      signal_owner_progress(subscriber->source_owner);
    }
  }
  input.retirement_slots.clear();
  return retry_blocked_process_local_transport();
}

expected<void, Error> RuntimeChannelSession::retire_samples(
    PortId port, std::span<const TemporalSampleRecord> samples, std::uint32_t* promoted_out,
    const RuntimeChannelFreshnessReservation* const freshness_reservation,
    std::int64_t* promoted_capture_out) noexcept {
  return retire_samples_impl(port,
                             samples,
                             SampleRetirementKind::kSelectorHistory,
                             promoted_out,
                             freshness_reservation,
                             promoted_capture_out);
}

expected<void, Error> RuntimeChannelSession::retire_terminal_selector_history_samples(
    const VertexId owner, const PortId port,
    const std::span<const TemporalSampleRecord> samples) noexcept {
  if (!owner || !impl_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr || input->owner != owner) {
    return make_unexpected(
        Error{input == nullptr ? ErrorCode::kNotFound : ErrorCode::kInvalidArgument});
  }
  return retire_samples_impl(
      port, samples, SampleRetirementKind::kTerminalHistory, nullptr, nullptr, nullptr);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::retire_samples_impl(
    PortId port, std::span<const TemporalSampleRecord> samples,
    const SampleRetirementKind retirement_kind, std::uint32_t* promoted_out,
    const RuntimeChannelFreshnessReservation* const freshness_reservation,
    std::int64_t* promoted_capture_out) noexcept {
  if (promoted_out != nullptr) {
    *promoted_out = 0U;
  }
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::InputPortState* input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (samples.empty()) {
    return {};
  }
  const bool terminal_history_release = retirement_kind == SampleRetirementKind::kTerminalHistory;
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(input->owner)};
#endif
  if (freshness_reservation != nullptr) {
    if (freshness_reservation->port != port || freshness_reservation->generation == 0U ||
        samples.size() != 1U ||
        input->freshness_reservation_generation != freshness_reservation->generation ||
        input->freshness_reservation_subscriber == nullptr ||
        input->freshness_reservation_retained == nullptr ||
        input->freshness_reservation_successor == nullptr ||
        input->freshness_refill_target == nullptr) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    Impl::SubscriberState* const subscriber = input->freshness_reservation_subscriber;
    bool direct_capacity_may_have_changed = false;
    {
      // This is the freshness handoff's commit boundary. The producer cannot take the retained
      // typed slot between claim release and reserved promotion because both happen under this one
      // subscriber mutex.
      const std::scoped_lock lock{subscriber->mutex};
      Impl::SampleSlot* const retained = input->freshness_reservation_retained;
      Impl::SampleSlot* const successor = input->freshness_reservation_successor;
      Impl::SampleSlot* const target = input->freshness_refill_target;
      const bool successor_is_target = successor == target;
      const bool reservation_live =
          input->freshness_reservation_generation == freshness_reservation->generation &&
          retained->occupied && !retained->direct_ingress_unread &&
          retained->temporal == samples[0] && retained->temporal_retained &&
          !retained->retirement_selected && !retained->typed_ingress_reserved &&
          !retained->active_selected &&
          (!retained->reuse_retained || input->reuse_slot == retained) &&
          (!retained->event_emitted || retained->witness_retired) && successor->occupied &&
          successor->direct_ingress_unread &&
          (successor_is_target ? successor->refill_reserved : successor->refill_prefix_reserved) &&
          !successor->typed_ingress_reserved &&
          successor->temporal == freshness_reservation->successor &&
          successor->temporal.arrival_sequence != 0U && target->occupied &&
          target->direct_ingress_unread && target->refill_reserved &&
          !target->refill_prefix_reserved && !target->typed_ingress_reserved &&
          target->temporal == freshness_reservation->target &&
          input->freshness_refill_target_reference == freshness_reservation->target_reference;
      if (!reservation_live) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      auto arrival_order = impl_->reserve_arrival_order();
      if (!arrival_order) {
        return make_unexpected(std::move(arrival_order).error());
      }

      retained->temporal_retained = false;
      if (retained->reuse_retained) {
        retained->reuse_retained = false;
        input->reuse_slot = nullptr;
      }
      Impl::reset_sample_slot(*retained);
      auto promoted = impl_->promote_direct_ingress(
          *subscriber, *arrival_order, successor, promoted_capture_out, retained);
      if (!promoted || *promoted != 1U) {
        // Every fallible condition was preflighted before the claim release. Reaching this branch
        // means an internal invariant, not producer contention, was violated.
        return make_unexpected(promoted ? Error{ErrorCode::kProtocolError}
                                        : std::move(promoted).error());
      }
      input->freshness_reservation_subscriber = nullptr;
      input->freshness_reservation_retained = nullptr;
      input->freshness_reservation_successor = nullptr;
      input->freshness_reservation_generation = 0U;
      if (successor_is_target) {
        input->freshness_refill_target = nullptr;
        input->freshness_refill_target_reference = {};
        input->freshness_refill_prefix_committed = false;
      } else {
        input->freshness_refill_prefix_committed = true;
      }
      if (promoted_out != nullptr) {
        *promoted_out = 1U;
      }
      direct_capacity_may_have_changed =
          impl_->owner_progress_wake_enabled() && subscriber->direct_process_local &&
          (subscriber->direct_ingress_capacity != 0U
               ? impl_->writable_direct_ingress_slot(*subscriber) != nullptr
               : impl_->writable_slot(*subscriber, Impl::WritableSlotIntent::kProbe) != nullptr);
    }
    if (direct_capacity_may_have_changed) {
      impl_->signal_owner_progress(subscriber->source_owner);
    }
    // The payload replacement is committed at this point. Its matching selector transaction must
    // commit before any fallible auxiliary process-local retry can be reported to RunSession.
    return {};
  }

  if (terminal_history_release) {
    if (input->active || input->selection_prepared || input->staged_now ||
        input->predicate_view_constructed || !input->active_slots.empty() ||
        !input->tensor_access_slots.empty() || !input->tensor_read_leases.empty() ||
        (input->claim_pool != nullptr && input->claim_pool->has_live_claims())) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    for (Impl::SubscriberState* const subscriber : input->subscribers) {
      const std::scoped_lock lock{subscriber->mutex};
      if (!subscriber->end_of_stream_received || !subscriber->end_of_stream_event_emitted ||
          subscriber->has_pending_direct_ingress() ||
          std::any_of(subscriber->slots.begin(), subscriber->slots.end(), [](const auto& slot) {
            return slot.typed_ingress_reserved || slot.active_selected ||
                   slot.retirement_selected || (slot.event_emitted && !slot.witness_retired);
          })) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
    }
  }

  // Select and mark the entire proof set, revalidate every mark, then commit. Any pre-commit
  // failure clears all marks; commit retains its existing promotion and transport-retry order.
  auto selected = Impl::select_sample_retirement(*input, samples, terminal_history_release);
  if (!selected) {
    Impl::rollback_sample_retirement(*input);
    return selected;
  }
  auto valid = Impl::validate_sample_retirement(*input, terminal_history_release);
  if (!valid) {
    Impl::rollback_sample_retirement(*input);
    return valid;
  }

  return impl_->commit_sample_retirement(
      *input, terminal_history_release, impl_->owner_progress_wake_enabled(), promoted_out);
}

}  // namespace holoscan::detail
