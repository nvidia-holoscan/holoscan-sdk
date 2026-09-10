/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <limits>
#include <utility>

#include <holoscan/profiler/nvtx3.hpp>

#include "../channel/pod_codec.hpp"
#include "runtime_channel_session_impl.hpp"

namespace holoscan::detail {

// Output loans and process-local or transported publication transactions.

void RuntimeChannelSession::Impl::LoanControl::release_reference() noexcept {
  if (references.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    type_ops->destroy_value(value.data());
    delete this;
  }
}

void RuntimeChannelSession::Impl::LoanControl::reset_value() noexcept {
  type_ops->destroy_value(value.data());
  type_ops->construct_value(value.data());
}

void RuntimeChannelSession::Impl::LoanControl::retire_live_authority() noexcept {
  if (!live) {
    return;
  }
  live = false;
  if (live_credit != nullptr) {
    live_credit->fetch_sub(1U, std::memory_order_acq_rel);
  }
  reset_value();
  release_reference();
}

void RuntimeChannelSession::Impl::LoanControl::cancel(void* opaque) noexcept {
  auto* control = static_cast<LoanControl*>(opaque);
  control->retire_live_authority();
}

RuntimeChannelSession::Impl::ProcessLocalPublicationScope::ProcessLocalPublicationScope(
    Impl& session, const bool process_local) noexcept
    : session_(process_local ? &session : nullptr),
      in_flight_(session_ != nullptr && session.track_concurrent_process_local_publications_) {
  if (in_flight_) {
    session_->process_local_publications_in_flight_.fetch_add(1U, std::memory_order_acq_rel);
  }
}

RuntimeChannelSession::Impl::ProcessLocalPublicationScope::
    ~ProcessLocalPublicationScope() noexcept {
  if (session_ != nullptr) {
    session_->invalidate_process_local_progress();
    finish();
  }
}

void RuntimeChannelSession::Impl::ProcessLocalPublicationScope::accept(
    const std::uint32_t deliveries,
    const std::span<SubscriberState* const> activation_cut_subscribers) noexcept {
  if (session_ != nullptr) {
    session_->record_process_local_publication(deliveries, activation_cut_subscribers);
    finish();
  }
}

void RuntimeChannelSession::Impl::ProcessLocalPublicationScope::reject() noexcept {
  finish();
}

void RuntimeChannelSession::Impl::ProcessLocalPublicationScope::invalidate() noexcept {
  if (session_ != nullptr) {
    session_->invalidate_process_local_progress();
    finish();
  }
}

void RuntimeChannelSession::Impl::ProcessLocalPublicationScope::finish() noexcept {
  if (session_ == nullptr) {
    return;
  }
  if (in_flight_) {
    const std::uint64_t previous =
        session_->process_local_publications_in_flight_.fetch_sub(1U, std::memory_order_acq_rel);
    static_cast<void>(previous);
    assert(previous != 0U);
  }
  session_ = nullptr;
}

[[nodiscard]] bool RuntimeChannelSession::Impl::OutputState::tensor_payload() const noexcept {
  return type_ops == runtime_port_type_ops<Tensor>();
}

[[nodiscard]] bool RuntimeChannelSession::Impl::OutputState::tensor_schema_payload()
    const noexcept {
  return type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
}

void RuntimeChannelSession::Impl::OutputState::mark_emit_nvtx(
    const std::uint64_t sequence) const noexcept {
  if (emit_nvtx_message == nullptr || source_identity_nvtx_string == nullptr) {
    return;
  }
  holoscan::profiler::EmitReceiveNVTXPayload message_payload{
      .source_identity = source_identity_nvtx_string,
      .sequence = sequence,
  };
  const nvtx3::payload_data payload_data{message_payload};
  nvtx3::mark_in<holoscan::profiler::holoscan_operator_domain>(nvtx3::message{emit_nvtx_message},
                                                               payload_data);
}

RuntimeChannelSession::Impl::OutputState::~OutputState() {
  for (LoanControl* loan : loans) {
    loan->owner = nullptr;
    loan->live_credit = nullptr;
    loan->release_reference();
  }
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::classify_connection_refresh(
    expected<void, Error> refreshed, const bool submitted) noexcept {
  if (refreshed) {
    return {};
  }
  if (!submitted) {
    return make_unexpected(std::move(refreshed).error());
  }
  ++next_sequence;
  ++emitted_this_activation;
  return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
}

// Preserve direct and transported publication authority, accounting, and terminal ordering.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::OutputState::emit_payload(
    std::span<const std::byte> payload, SampleMetadata metadata,
    const channel_internal::InlinePayloadEncoder* const encoder) noexcept {
  if (!active) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (emitted_this_activation == max_emits_per_activation) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  if (next_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const bool direct_encode = encoder != nullptr;
  if ((direct_encode &&
       (type_ops->codec_family != RuntimePayloadCodecFamily::kGeneratedSchema ||
        encoder->capacity != type_ops->wire_capacity || encoder->state == nullptr ||
        encoder->encode == nullptr || direct_subscriber != nullptr ||
        process_local_schema_identity || endpoints.size() != 1U)) ||
      (!direct_encode && payload.empty()) || payload.size() > type_ops->wire_capacity ||
      (type_ops->codec_family == RuntimePayloadCodecFamily::kInlinePod &&
       payload.size() != type_ops->wire_capacity)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
    if (payload.size() != type_ops->wire_capacity) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }
  std::span<const std::byte> process_local_schema_payload;
  if (process_local_schema_identity) {
    if (type_ops->codec_family != RuntimePayloadCodecFamily::kGeneratedSchema ||
        process_local_schema_scratch.data() == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::size_t frame_size = kSchemaEnvelopeHeaderSize + payload.size();
    const std::span<std::byte> frame{static_cast<std::byte*>(process_local_schema_scratch.data()),
                                     frame_size};
    auto envelope = channel_internal::encode_schema_envelope(
        frame.first(kSchemaEnvelopeHeaderSize), *process_local_schema_identity, payload.size());
    if (!envelope) {
      return make_unexpected(std::move(envelope).error());
    }
    std::copy(payload.begin(), payload.end(), frame.begin() + kSchemaEnvelopeHeaderSize);
    if (next_process_local_schema_corruption) {
      corrupt_process_local_schema_frame(frame, *next_process_local_schema_corruption);
      next_process_local_schema_corruption.reset();
    }
    process_local_schema_payload = frame;
  }
  metadata.version = SampleMetadataVersion{};
  metadata.sequence = next_sequence;
  metadata.byte_size = direct_encode ? 0U : static_cast<std::uint32_t>(payload.size());
  metadata.payload_kind = type_ops->payload_kind;
  if (auto stamped = session->stamp_publication_coordinate(metadata); !stamped) {
    return make_unexpected(std::move(stamped).error());
  }
  mark_emit_nvtx(metadata.sequence);
  if (direct_subscriber != nullptr) {
    const bool capture_direct_continuation = direct_continuation_capture;
    const std::uint64_t prior_revision =
        capture_direct_continuation
            ? direct_subscriber->input->readiness_revision.load(std::memory_order_acquire)
            : 0U;
    const std::span<const std::byte> retained_payload =
        process_local_schema_identity ? process_local_schema_payload : payload;
    auto stored =
        session->store_direct_inline_payload(*direct_subscriber, retained_payload, metadata);
    if (!stored) {
      return make_unexpected(std::move(stored).error());
    }
    if (capture_direct_continuation) {
      const std::uint64_t destination_revision =
          direct_subscriber->input->readiness_revision.load(std::memory_order_acquire);
      session->note_direct_continuation_publication(
          *this, *direct_subscriber, metadata.sequence, prior_revision, destination_revision);
    }
    ++next_sequence;
    ++emitted_this_activation;
    return {};
  }
  if (endpoints.empty()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  // Validate every catalog-bound metadata projection before any endpoint accepts the sample.
  // This keeps a deterministic plan/catalog identity error on the pre-submit side even when
  // the same Output also fans out to internal recipients earlier in endpoint order.
  for (auto& endpoint : endpoints) {
    if (!endpoint.refresh_before_emit) {
      continue;
    }
    if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (endpoint.plan_capture_clock.valid()) {
      if (metadata.source_clock_id != endpoint.plan_capture_clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    } else if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }

  trace_output_emit_phase(phase_counters,
                          RuntimePhaseTracePoint::kInlinePayloadTransportPublicationStarted,
                          port,
                          next_sequence);
  bool submitted = false;
  for (auto& endpoint : endpoints) {
    SampleMetadata endpoint_metadata = metadata;
    if (endpoint.plan_capture_clock.valid()) {
      // Internal fanout continues to observe the plan-local metadata above. Only the copy sent
      // through this catalog endpoint enters the independently dense catalog clock namespace.
      endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
    }
    if (endpoint.refresh_before_emit) {
      // The publication coordinate crosses the same namespace boundary as capture time above,
      // and for the same reason: this copy leaves for a peer that resolves clock identities
      // against the catalog rather than the plan.
      session->project_publication_coordinate(endpoint_metadata);
    }
    // External peers may attach after Graph startup, which is deliberately never a peer-wait
    // barrier. Refresh immediately before the submission boundary so a late peer can
    // participate without converting an earlier successful refresh into a first-sample
    // guarantee. A refresh failure is still structurally pre-submit for this endpoint; after
    // an earlier fanout endpoint accepted the sample, the same failure is necessarily an
    // terminal partial or indeterminate publication outcome.
    if (endpoint.refresh_before_emit) {
      auto refreshed =
          classify_connection_refresh(endpoint.publisher->update_connections(), submitted);
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }
    channel_internal::InlinePublicationResult published;
    const std::span<const std::byte> endpoint_payload =
        endpoint.process_local_schema_envelope ? process_local_schema_payload : payload;
    if (endpoint_payload.empty() && !direct_encode) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    ProcessLocalPublicationScope publication{*session, endpoint.process_local_transport};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      published =
          direct_encode ? endpoint.publisher->emit_encoded_phase(*encoder, endpoint_metadata)
          : endpoint.canonical_pod_publisher
              ? endpoint.canonical_pod_publisher.emit_phase(
                    endpoint.canonical_pod_publisher.state, endpoint_payload, endpoint_metadata)
              : endpoint.publisher->emit_phase(endpoint_payload, endpoint_metadata);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      ++next_sequence;
      ++emitted_this_activation;
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
#endif
    switch (published.phase) {
      case channel_internal::InlinePublicationPhase::kAccepted:
        publication.accept(published.recipient_count, endpoint.activation_cut_subscribers);
        endpoint.terminal_obligated |= published.recipient_count != 0U;
        if (published.recipient_count != endpoint.expected_recipient_count &&
            (endpoint.peer_required || published.recipient_count != 0U)) {
          ++next_sequence;
          ++emitted_this_activation;
          if (published.recipient_count < endpoint.expected_recipient_count) {
            return make_unexpected(make_publication_partial_error(
                published.recipient_count, endpoint.expected_recipient_count));
          }
          return make_unexpected(
              Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
        }
        // An optional endpoint with no attached peer is an intentional local drop. No remote
        // observer exists, so it must neither fail the activation nor make a later pre-submit
        // failure on another endpoint indeterminate.
        submitted = submitted || published.recipient_count != 0U;
        break;
      case channel_internal::InlinePublicationPhase::kPostSubmitUncertain:
        publication.invalidate();
        ++next_sequence;
        ++emitted_this_activation;
        if (publication_error_consumes_authority(published.error.code) ||
            published.error.code == ErrorCode::kProtocolError ||
            published.error.code == ErrorCode::kInvalidArgument) {
          return make_unexpected(std::move(published.error));
        }
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      case channel_internal::InlinePublicationPhase::kPreSubmitRejected:
        publication.reject();
        if (!endpoint.peer_required && published.error.code == ErrorCode::kNotReady) {
          // Some backends report the same unobserved optional-peer drop as a pre-submit
          // readiness miss rather than an accepted zero-recipient publication.
          break;
        }
        if (!submitted) {
          return make_unexpected(published.error);
        }
        ++next_sequence;
        ++emitted_this_activation;
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      default:
        // A malformed private phase is never equivalent to the proven no-peer rejection
        // above. Preserve any earlier fanout observation in the returned classification.
        publication.invalidate();
        if (!submitted) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        ++next_sequence;
        ++emitted_this_activation;
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
  }
  trace_output_emit_phase(phase_counters,
                          RuntimePhaseTracePoint::kInlinePayloadTransportPublicationReturned,
                          port,
                          next_sequence);
  ++next_sequence;
  ++emitted_this_activation;
  return {};
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::validate_emit_authority(
    const OutputState& state) noexcept {
  if (!state.active) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (state.emitted_this_activation == state.max_emits_per_activation ||
      state.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return {};
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::classify_tensor_publication(
    OutputState& state, expected<std::uint32_t, Error> published, bool& submitted) noexcept {
  if (published) {
    submitted = true;
    return {};
  }
  return classify_tensor_failure(state, std::move(published).error(), submitted);
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::classify_tensor_failure(OutputState& state, Error error,
                                                                  const bool submitted) noexcept {
  if (publication_error_consumes_authority(error.code)) {
    ++state.next_sequence;
    ++state.emitted_this_activation;
    return make_unexpected(std::move(error));
  }
  if (submitted) {
    ++state.next_sequence;
    ++state.emitted_this_activation;
    return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
  }
  // Manual Tensor publishers report bounded HoloMQ loan/callback capacity as
  // kResourceExhausted. No endpoint has accepted the sample at this boundary, so the
  // graph-bound Output contract classifies the same proven pre-submit pressure as
  // kBackpressured without consuming sequence or publication authority.
  if (error.code == ErrorCode::kResourceExhausted) {
    error = Error{ErrorCode::kBackpressured};
  }
  return make_unexpected(std::move(error));
}

void RuntimeChannelSession::Impl::OutputState::inject_tensor_fanout_failure(
    OutputState& state, std::size_t accepted_count) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (state.throw_tensor_fanout_after_accepts_for_test == accepted_count) {
    state.throw_tensor_fanout_after_accepts_for_test.reset();
    throw std::runtime_error{accepted_count == 0U ? "injected pre-submit Tensor fanout failure"
                                                  : "injected post-submit Tensor fanout failure"};
  }
#else
  (void)state;
  (void)accepted_count;
#endif
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::validate_catalog_tensor_metadata(
    const OutputState& state, const SampleMetadata& metadata) noexcept {
  // Plan and endpoint catalogs use independently dense clock IDs. Validate every projection
  // before any fanout arm can accept the sample so a provenance mismatch remains pre-submit.
  for (const auto& endpoint : state.catalog_tensor_endpoints) {
    if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (endpoint.plan_capture_clock.valid()) {
      if (metadata.source_clock_id != endpoint.plan_capture_clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    } else if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_internal_tensor_endpoints(OutputState& state,
                                                                         Sample<Tensor>& normalized,
                                                                         bool& submitted) {
  if (state.tensor_endpoints.empty()) {
    return {};
  }
  for (std::size_t index = 0U; index + 1U < state.tensor_endpoints.size(); ++index) {
    auto& endpoint = state.tensor_endpoints[index];
    ProcessLocalPublicationScope publication{*state.session, endpoint.process_local_transport};
    auto published = endpoint.publisher.emit_sample(normalized);
    if (published) {
      publication.accept(*published, endpoint.activation_cut_subscribers);
    } else if (publication_error_consumes_authority(published.error().code)) {
      publication.invalidate();
    } else {
      publication.reject();
    }
    auto classified = classify_tensor_publication(state, std::move(published), submitted);
    if (!classified) {
      return classified;
    }
    inject_tensor_fanout_failure(state, index + 1U);
  }
  auto& endpoint = state.tensor_endpoints.back();
  ProcessLocalPublicationScope publication{*state.session, endpoint.process_local_transport};
  auto published = endpoint.publisher.emit_sample(std::move(normalized));
  if (published) {
    publication.accept(*published, endpoint.activation_cut_subscribers);
  } else if (publication_error_consumes_authority(published.error().code)) {
    publication.invalidate();
  } else {
    publication.reject();
  }
  auto classified = classify_tensor_publication(state, std::move(published), submitted);
  if (!classified) {
    return classified;
  }
  inject_tensor_fanout_failure(state, state.tensor_endpoints.size());
  return {};
}

[[nodiscard]] expected<RuntimeChannelSession::Impl::OutputState::CatalogTensorFanoutPreparation,
                       Error>
RuntimeChannelSession::Impl::OutputState::prepare_catalog_tensor_fanout(OutputState& state,
                                                                        const Tensor& supplied) {
  CatalogTensorFanoutPreparation preparation;
  for (std::size_t index = 0U; index < state.catalog_tensor_endpoints.size(); ++index) {
    auto& endpoint = state.catalog_tensor_endpoints[index];
    endpoint.active_for_current_fanout = endpoint.peer_required;
    // Optional endpoints retain their pre-submit skip semantics. Required endpoints defer
    // topology admission to HoloMQ's exact recipient-count gate, avoiding a redundant shared
    // dynamic-config read on the ordinary publication path.
    if (!endpoint.peer_required && !endpoint.publisher.has_subscribers()) {
      auto refreshed =
          state.classify_connection_refresh(endpoint.publisher.update_connections(), false);
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
      if (!endpoint.publisher.has_subscribers()) {
        continue;
      }
    }
    endpoint.active_for_current_fanout = true;

    auto disposition = channel_internal::TensorPublishLoanAccess::classify_foreign_tensor(
        endpoint.publisher, supplied);
    if (!disposition) {
      return make_unexpected(std::move(disposition).error());
    }
    if (preparation.staging_index == std::numeric_limits<std::size_t>::max()) {
      preparation.staging_index = index;
    }
    preparation.copy_required |=
        *disposition == channel_internal::ForeignTensorDisposition::kCopyRequired;
  }
  if (preparation.copy_required) {
    if (preparation.staging_index >= state.catalog_tensor_endpoints.size()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto staged = channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(
        state.catalog_tensor_endpoints[preparation.staging_index].publisher,
        supplied,
        TensorAccess::readiness_stream(supplied),
        state.catalog_tensor_endpoints[preparation.staging_index].staging_stream);
    if (!staged) {
      Error error = std::move(staged).error();
      if (error.code == ErrorCode::kResourceExhausted) {
        error = Error{ErrorCode::kBackpressured};
      }
      return make_unexpected(std::move(error));
    }
    preparation.staged_copy.emplace(std::move(*staged));
    if (state.tensor_test_control != nullptr &&
        state.catalog_tensor_endpoints[preparation.staging_index].staging_stream != nullptr) {
      state.tensor_test_control->record_cuda_service_copy(
          TensorAccess::readiness_stream(supplied),
          state.catalog_tensor_endpoints[preparation.staging_index].staging_stream);
    }
  }
  return preparation;
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_catalog_tensor_endpoint(
    OutputState& state, CatalogTensorEndpointState& endpoint, const Tensor& supplied,
    const SampleMetadata& normalized_metadata, CatalogTensorFanoutPreparation& preparation,
    bool& submitted) {
  if (!endpoint.active_for_current_fanout) {
    return {};
  }
  SampleMetadata endpoint_metadata = normalized_metadata;
  state.session->project_publication_coordinate(endpoint_metadata);
  if (endpoint.plan_capture_clock.valid()) {
    endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
  }
  expected<std::uint32_t, Error> published = make_unexpected(Error{ErrorCode::kFailure});
  if (!preparation.copy_required) {
    published = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
        endpoint.publisher, supplied, endpoint_metadata);
  } else if (preparation.staged_copy) {
    published = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
        endpoint.publisher, preparation.staged_copy->tensor, endpoint_metadata);
  }
  if (!endpoint.peer_required && !published && published.error().code == ErrorCode::kNotReady) {
    // The optional peer disappeared after preparation but before submit. HoloMQ proved the
    // rejection occurred before submission, so the arm remains an optional no-op.
    return {};
  }
  if (published) {
    endpoint.terminal_obligated |= *published != 0U;
  }
  return classify_tensor_publication(state, std::move(published), submitted);
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::OutputState::emit_tensor_fanout(
    OutputState& state, const Sample<Tensor>& supplied, Sample<Tensor> normalized) noexcept {
  auto metadata_valid = validate_catalog_tensor_metadata(state, normalized.metadata);
  if (!metadata_valid) {
    return metadata_valid;
  }
  const SampleMetadata normalized_metadata = normalized.metadata;
  bool submitted = false;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    inject_tensor_fanout_failure(state, 0U);
    auto catalog_preparation = prepare_catalog_tensor_fanout(state, supplied.data);
    if (!catalog_preparation) {
      return make_unexpected(std::move(catalog_preparation).error());
    }
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kTensorTransportPublicationStarted,
                            state.port,
                            normalized_metadata.sequence);
    auto internal = emit_internal_tensor_endpoints(state, normalized, submitted);
    if (!internal) {
      return internal;
    }
    for (std::size_t index = 0U; index < state.catalog_tensor_endpoints.size(); ++index) {
      auto published = emit_catalog_tensor_endpoint(state,
                                                    state.catalog_tensor_endpoints[index],
                                                    supplied.data,
                                                    normalized_metadata,
                                                    *catalog_preparation,
                                                    submitted);
      if (!published) {
        return published;
      }
    }
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kTensorTransportPublicationReturned,
                            state.port,
                            normalized_metadata.sequence);
    ++state.next_sequence;
    ++state.emitted_this_activation;
    return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    if (std::ranges::any_of(state.tensor_endpoints, [](const auto& endpoint) {
          return endpoint.process_local_transport;
        })) {
      state.session->invalidate_process_local_progress();
    }
    if (submitted) {
      ++state.next_sequence;
      ++state.emitted_this_activation;
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::OutputState::emit_tensor_sample(
    OutputState& state, const Sample<Tensor>& supplied) noexcept {
  auto authority = validate_emit_authority(state);
  if (!authority) {
    return authority;
  }
  if (state.direct_subscriber == nullptr && state.tensor_endpoints.empty() &&
      state.catalog_tensor_endpoints.empty()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  const std::uint64_t emitted_sequence = state.next_sequence;
  state.mark_emit_nvtx(emitted_sequence);
  trace_output_emit_phase(state.phase_counters,
                          RuntimePhaseTracePoint::kTensorEmitStarted,
                          state.port,
                          emitted_sequence);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const TensorAccessSnapshot snapshot = TensorAccess::snapshot(supplied.data);
    const bool has_internal_tensor_recipient =
        state.direct_subscriber != nullptr || !state.tensor_endpoints.empty();
    const bool native_contract_matches =
        !has_internal_tensor_recipient ||
        tensor_snapshot_matches_plan(
            snapshot, state.tensor_max_byte_span, state.tensor_memory_kind, state.tensor_device);
    const bool port_contract_matches =
        !state.tensor_contract ||
        tensor_matches_port_contract(supplied.data, snapshot, *state.tensor_contract);
    const bool catalog_contract_matches =
        !state.catalog_tensor_requirements ||
        (snapshot.initialized && snapshot.valid &&
         state.catalog_tensor_requirements->max_byte_span &&
         snapshot.byte_span <= *state.catalog_tensor_requirements->max_byte_span &&
         tensor_matches_logical_requirements(supplied.data, *state.catalog_tensor_requirements));
    if (!native_contract_matches || !port_contract_matches || !catalog_contract_matches) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    Sample<Tensor> normalized{supplied};
    normalized.metadata.version = SampleMetadataVersion{};
    normalized.metadata.sequence = state.next_sequence;
    normalized.metadata.byte_size = snapshot.byte_size;
    normalized.metadata.payload_kind = PayloadKind::kDescriptorBacked;
    if (auto stamped = state.session->stamp_publication_coordinate(normalized.metadata); !stamped) {
      return make_unexpected(std::move(stamped).error());
    }
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kTensorEmitValidated,
                            state.port,
                            emitted_sequence);
    if (state.direct_subscriber != nullptr) {
      // REQ: CHAN-098
      if (!state.tensor_endpoints.empty() || !state.catalog_tensor_endpoints.empty()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto stored = state.session->store_direct_tensor_sample(*state.direct_subscriber,
                                                              std::move(normalized));
      if (!stored) {
        return stored;
      }
      ++state.next_sequence;
      ++state.emitted_this_activation;
      trace_output_emit_phase(state.phase_counters,
                              RuntimePhaseTracePoint::kTensorEmitCompleted,
                              state.port,
                              emitted_sequence);
      return {};
    }
    auto emitted = emit_tensor_fanout(state, supplied, std::move(normalized));
    if (emitted) {
      trace_output_emit_phase(state.phase_counters,
                              RuntimePhaseTracePoint::kTensorEmitCompleted,
                              state.port,
                              emitted_sequence);
    }
    return emitted;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

[[nodiscard]] expected<void, Error>
// Descriptor fanout is one encode, reservation, and per-endpoint publication transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::OutputState::emit_tensor_schema_fanout(
    OutputState& state, const void* payload, const std::span<const Tensor> tensors,
    const std::span<std::byte> destination, const SampleMetadata& metadata,
    const bool require_direct_export) noexcept {
  const auto encode_payload =
      [&](const SampleMetadata& endpoint_metadata,
          const std::uint64_t endpoint_generation) -> expected<std::size_t, Error> {
    expected<std::size_t, Error> encoded = make_unexpected(Error{ErrorCode::kProtocolError});
    if (require_direct_export) {
      if (tensors.size() == 1U && state.type_ops->encode_descriptor_tensor_loan != nullptr) {
        encoded = state.type_ops->encode_descriptor_tensor_loan(
            payload, tensors.front(), endpoint_metadata, endpoint_generation, destination);
      } else if (state.type_ops->encode_descriptor_tensor_loans != nullptr) {
        encoded = state.type_ops->encode_descriptor_tensor_loans(
            payload, tensors, endpoint_metadata, endpoint_generation, destination);
      }
    } else if (state.type_ops->encode_descriptor_tensor_sample != nullptr) {
      encoded = state.type_ops->encode_descriptor_tensor_sample(
          payload, endpoint_metadata, endpoint_generation, destination);
    }
    if (!encoded || *encoded == 0U || *encoded > destination.size() ||
        state.type_ops->descriptor_offset > *encoded ||
        tensors.size() * sizeof(PayloadDescriptor) > *encoded - state.type_ops->descriptor_offset) {
      return encoded ? make_unexpected(Error{ErrorCode::kProtocolError})
                     : make_unexpected(std::move(encoded).error());
    }
    return *encoded;
  };

  bool submitted = false;
  bool destination_has_shared_encoding = false;
  std::size_t shared_encoded_size = 0U;
  for (auto& endpoint : state.tensor_schema_endpoints) {
    SampleMetadata endpoint_metadata = metadata;
    if (endpoint.refresh_before_emit) {
      if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid() ||
          (endpoint.plan_capture_clock.valid() &&
           endpoint_metadata.source_clock_id != endpoint.plan_capture_clock) ||
          (!endpoint.plan_capture_clock.valid() && (endpoint_metadata.source_clock_id.valid() ||
                                                    endpoint_metadata.capture_timestamp_ns != 0))) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto refreshed =
          state.classify_connection_refresh(endpoint.publisher->update_connections(), submitted);
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
      if (!endpoint.peer_required && !endpoint.publisher->has_subscribers()) {
        continue;
      }
      // Output activation serialization is the sole writer for this endpoint, so this peek
      // remains valid through the immediately following publication.
      auto endpoint_sequence = endpoint.publisher->next_operation_sequence();
      if (!endpoint_sequence) {
        return classify_tensor_failure(state, std::move(endpoint_sequence).error(), submitted);
      }
      endpoint_metadata.sequence = *endpoint_sequence;
      if (endpoint.plan_capture_clock.valid()) {
        endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
      }
      state.session->project_publication_coordinate(endpoint_metadata);
    }
    const std::uint64_t endpoint_generation =
        endpoint.refresh_before_emit ? channel_internal::kCompatibilitySchemaEndpointGeneration
                                     : state.session->plan_generation_;
    std::size_t encoded_size = shared_encoded_size;
    if (endpoint.refresh_before_emit || !destination_has_shared_encoding) {
      auto encoded = encode_payload(endpoint_metadata, endpoint_generation);
      if (!encoded) {
        return make_unexpected(std::move(encoded).error());
      }
      encoded_size = *encoded;
      destination_has_shared_encoding = !endpoint.refresh_before_emit;
      if (destination_has_shared_encoding) {
        shared_encoded_size = encoded_size;
      }
    }
    auto reserved = endpoint.publisher->reserve_tensors(tensors, require_direct_export);
    if (!reserved) {
      return classify_tensor_failure(state, std::move(reserved).error(), submitted);
    }
    if (reserved->field_count != tensors.size()) {
      endpoint.publisher->release_tensor(std::move(*reserved));
      return classify_tensor_failure(state, Error{ErrorCode::kProtocolError}, submitted);
    }
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto* field = reserved->field_at(index);
      if (field == nullptr) {
        endpoint.publisher->release_tensor(std::move(*reserved));
        return classify_tensor_failure(state, Error{ErrorCode::kProtocolError}, submitted);
      }
      field->descriptor.schema_id = state.tensor_schema_id;
      auto descriptor_encoded = channel_internal::encode_payload_descriptor_wire(
          field->descriptor,
          destination.subspan(state.type_ops->descriptor_offset + index * sizeof(PayloadDescriptor),
                              sizeof(PayloadDescriptor)));
      if (!descriptor_encoded) {
        endpoint.publisher->release_tensor(std::move(*reserved));
        return classify_tensor_failure(state, std::move(descriptor_encoded).error(), submitted);
      }
    }
    ProcessLocalPublicationScope publication{*state.session, endpoint.process_local_transport};
    auto published = endpoint.publisher->publish_payload(
        destination.first(encoded_size), std::move(*reserved), endpoint_metadata);
    if (!published) {
      if (publication_error_consumes_authority(published.error().code)) {
        publication.invalidate();
      } else {
        publication.reject();
      }
      return classify_tensor_failure(state, std::move(published).error(), submitted);
    }
    publication.accept(*published, endpoint.activation_cut_subscribers);
    endpoint.terminal_obligated |= *published != 0U;
    if (*published != endpoint.expected_recipient_count) {
      ++state.next_sequence;
      ++state.emitted_this_activation;
      if (*published < endpoint.expected_recipient_count) {
        return make_unexpected(
            make_publication_partial_error(*published, endpoint.expected_recipient_count));
      }
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
    }
    submitted = true;
  }
  ++state.next_sequence;
  ++state.emitted_this_activation;
  return {};
}

[[nodiscard]] expected<std::size_t, Error>
RuntimeChannelSession::Impl::OutputState::encode_tensor_schema_transport(
    void* opaque, const std::span<std::byte> destination,
    const channel_internal::TensorSchemaPayloadReservation& reservation) noexcept {
  auto& context = *static_cast<TensorSchemaEncodeContext*>(opaque);
  const OutputState& state = *context.output;
  expected<std::size_t, Error> encoded = make_unexpected(Error{ErrorCode::kProtocolError});
  if (context.separate_tensor) {
    if (context.tensors.size() == 1U && state.type_ops->encode_descriptor_tensor_loan != nullptr) {
      encoded = state.type_ops->encode_descriptor_tensor_loan(context.payload,
                                                              context.tensors.front(),
                                                              context.metadata,
                                                              context.endpoint_generation,
                                                              destination);
    } else if (state.type_ops->encode_descriptor_tensor_loans != nullptr) {
      encoded = state.type_ops->encode_descriptor_tensor_loans(context.payload,
                                                               context.tensors,
                                                               context.metadata,
                                                               context.endpoint_generation,
                                                               destination);
    }
  } else if (state.type_ops->encode_descriptor_tensor_sample != nullptr) {
    encoded = state.type_ops->encode_descriptor_tensor_sample(
        context.payload, context.metadata, context.endpoint_generation, destination);
  }
  if (!encoded || *encoded == 0U || *encoded > destination.size() ||
      reservation.field_count != context.tensors.size() ||
      state.type_ops->descriptor_offset > *encoded ||
      context.tensors.size() * sizeof(PayloadDescriptor) >
          *encoded - state.type_ops->descriptor_offset) {
    return encoded ? make_unexpected(Error{ErrorCode::kProtocolError})
                   : make_unexpected(std::move(encoded).error());
  }
  for (std::size_t index = 0U; index < context.tensors.size(); ++index) {
    const auto* field = reservation.field_at(index);
    if (field == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    PayloadDescriptor descriptor = field->descriptor;
    descriptor.schema_id = state.tensor_schema_id;
    auto descriptor_encoded = channel_internal::encode_payload_descriptor_wire(
        descriptor,
        destination.subspan(state.type_ops->descriptor_offset + index * sizeof(PayloadDescriptor),
                            sizeof(PayloadDescriptor)));
    if (!descriptor_encoded) {
      return make_unexpected(std::move(descriptor_encoded).error());
    }
  }
  return *encoded;
}

[[nodiscard]] expected<void, Error>
// Direct descriptor loans keep refresh, reservation, and recipient accounting indivisible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::OutputState::emit_tensor_schema_direct_loan(
    OutputState& state, const void* payload, const std::span<const Tensor> tensors,
    const SampleMetadata metadata, const bool require_direct_export) noexcept {
  if (state.tensor_schema_endpoints.size() != 1U) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto& endpoint = state.tensor_schema_endpoints.front();
  SampleMetadata endpoint_metadata = metadata;
  if (endpoint.refresh_before_emit) {
    if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid() ||
        (endpoint.plan_capture_clock.valid() &&
         endpoint_metadata.source_clock_id != endpoint.plan_capture_clock) ||
        (!endpoint.plan_capture_clock.valid() && (endpoint_metadata.source_clock_id.valid() ||
                                                  endpoint_metadata.capture_timestamp_ns != 0))) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto refreshed =
        state.classify_connection_refresh(endpoint.publisher->update_connections(), false);
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (!endpoint.peer_required && !endpoint.publisher->has_subscribers()) {
      ++state.next_sequence;
      ++state.emitted_this_activation;
      return {};
    }
    // Output activation serialization is the sole writer for this endpoint, so this peek
    // remains valid through the immediately following publication.
    auto endpoint_sequence = endpoint.publisher->next_operation_sequence();
    if (!endpoint_sequence) {
      return classify_tensor_failure(state, std::move(endpoint_sequence).error(), false);
    }
    endpoint_metadata.sequence = *endpoint_sequence;
    if (endpoint.plan_capture_clock.valid()) {
      endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
    }
    state.session->project_publication_coordinate(endpoint_metadata);
  }
  auto reservation = endpoint.publisher->reserve_tensors(tensors, require_direct_export);
  if (!reservation) {
    return classify_tensor_failure(state, std::move(reservation).error(), false);
  }
  TensorSchemaEncodeContext context{
      .output = &state,
      .payload = payload,
      .tensors = tensors,
      .metadata = endpoint_metadata,
      .endpoint_generation = endpoint.refresh_before_emit
                                 ? channel_internal::kCompatibilitySchemaEndpointGeneration
                                 : state.session->plan_generation_,
      .separate_tensor = require_direct_export,
  };
  const channel_internal::TensorSchemaPayloadEncoder encoder{
      .state = &context,
      .encode = &encode_tensor_schema_transport,
      .capacity = state.type_ops->wire_capacity,
  };
  ProcessLocalPublicationScope publication{*state.session, endpoint.process_local_transport};
  auto published = endpoint.publisher->publish_encoded_payload(
      encoder, std::move(*reservation), endpoint_metadata);
  if (!published) {
    if (publication_error_consumes_authority(published.error().code)) {
      publication.invalidate();
    } else {
      publication.reject();
    }
    return classify_tensor_failure(state, std::move(published).error(), false);
  }
  publication.accept(*published, endpoint.activation_cut_subscribers);
  endpoint.terminal_obligated |= *published != 0U;
  if (*published != endpoint.expected_recipient_count) {
    ++state.next_sequence;
    ++state.emitted_this_activation;
    return make_unexpected(
        *published < endpoint.expected_recipient_count
            ? make_publication_partial_error(*published, endpoint.expected_recipient_count)
            : Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
  }
  ++state.next_sequence;
  ++state.emitted_this_activation;
  return {};
}

[[nodiscard]] expected<void, Error>
// Validate all schema fields before selecting one exact local or transported emission path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::OutputState::emit_tensor_schema_value(
    OutputState& state, const void* payload, const std::span<const Tensor> tensors,
    const SampleMetadata& metadata, const bool separate_tensor) noexcept {
  auto authority = validate_emit_authority(state);
  if (!authority) {
    return authority;
  }
  const std::uint64_t emitted_sequence = state.next_sequence;
  state.mark_emit_nvtx(emitted_sequence);
  const std::size_t tensor_count = state.type_ops->descriptor_tensor_static_count;
  if ((state.direct_subscriber == nullptr && state.tensor_schema_endpoints.empty()) ||
      state.encode_scratch.data() == nullptr || tensor_count == 0U ||
      tensor_count != tensors.size() ||
      tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
      state.type_ops->descriptor_offset > state.type_ops->wire_capacity ||
      tensor_count * sizeof(PayloadDescriptor) >
          state.type_ops->wire_capacity - state.type_ops->descriptor_offset ||
      (separate_tensor && state.type_ops->encode_descriptor_tensor_loans == nullptr &&
       (tensor_count != 1U || state.type_ops->encode_descriptor_tensor_loan == nullptr))) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if ((!state.tensor_fields.empty() && state.tensor_fields.size() != tensor_count) ||
      (!state.tensor_logical_fields.empty() &&
       state.tensor_logical_fields.size() != tensor_count)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::uint64_t aggregate_byte_size = 0U;
  for (std::size_t index = 0U; index < tensor_count; ++index) {
    const TensorAccessSnapshot snapshot = TensorAccess::snapshot(tensors[index]);
    const bool matches =
        state.tensor_fields.empty()
            ? tensor_snapshot_matches_plan(snapshot,
                                           state.tensor_max_byte_span,
                                           state.tensor_memory_kind,
                                           state.tensor_device) &&
                  (!state.tensor_layout ||
                   (tensor_count == 1U &&
                    tensor_matches_port_layout(tensors[index], snapshot, *state.tensor_layout)))
            : state.tensor_fields[index].tensor_index == index &&
                  tensor_matches_field_contract(
                      tensors[index], snapshot, state.tensor_fields[index]);
    const bool logical_matches = state.tensor_logical_fields.empty() ||
                                 (state.tensor_logical_fields[index].tensor_index == index &&
                                  tensor_matches_logical_requirements(
                                      tensors[index], state.tensor_logical_fields[index].tensor));
    if (!matches || !logical_matches ||
        snapshot.byte_size > std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    aggregate_byte_size += snapshot.byte_size;
  }
  SampleMetadata normalized = metadata;
  normalized.version = SampleMetadataVersion{};
  normalized.sequence = state.next_sequence;
  normalized.byte_size = aggregate_byte_size;
  normalized.payload_kind = PayloadKind::kDescriptorBacked;
  if (auto stamped = state.session->stamp_publication_coordinate(normalized); !stamped) {
    return make_unexpected(std::move(stamped).error());
  }
  if (state.direct_subscriber == nullptr && state.tensor_schema_endpoints.size() == 1U) {
    return emit_tensor_schema_direct_loan(state, payload, tensors, normalized, separate_tensor);
  }
  const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                         state.type_ops->wire_capacity};
  if (state.direct_subscriber == nullptr) {
    return emit_tensor_schema_fanout(
        state, payload, tensors, destination, normalized, separate_tensor);
  }
  expected<std::size_t, Error> encoded = make_unexpected(Error{ErrorCode::kProtocolError});
  if (separate_tensor) {
    if (tensor_count == 1U && state.type_ops->encode_descriptor_tensor_loan != nullptr) {
      encoded = state.type_ops->encode_descriptor_tensor_loan(
          payload, tensors.front(), normalized, state.session->plan_generation_, destination);
    } else {
      encoded = state.type_ops->encode_descriptor_tensor_loans(
          payload, tensors, normalized, state.session->plan_generation_, destination);
    }
  } else {
    if (state.type_ops->encode_descriptor_tensor_sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    encoded = state.type_ops->encode_descriptor_tensor_sample(
        payload, normalized, state.session->plan_generation_, destination);
  }
  if (!encoded) {
    return make_unexpected(std::move(encoded).error());
  }
  if (*encoded == 0U || *encoded > state.type_ops->wire_capacity ||
      state.type_ops->descriptor_offset > *encoded ||
      tensor_count * sizeof(PayloadDescriptor) > *encoded - state.type_ops->descriptor_offset) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // REQ: CHAN-099
  if (!state.tensor_schema_endpoints.empty()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto retained = state.session->store_direct_tensor_schema_sample(
      *state.direct_subscriber, destination.first(*encoded), tensors);
  if (!retained) {
    return retained;
  }
  ++state.next_sequence;
  ++state.emitted_this_activation;
  return {};
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_tensor_schema_sample(
    OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept {
  const std::uint32_t tensor_count = state.type_ops->descriptor_tensor_static_count;
  if (state.type_ops->descriptor_tensor_count == nullptr ||
      state.type_ops->descriptor_tensor_at == nullptr || tensor_count == 0U ||
      tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
      state.type_ops->descriptor_tensor_count(payload) != tensor_count) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto& tensors = state.tensor_schema_scratch;
  if (tensors.size() != tensor_count) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const TensorScratchReset reset{tensors};
  for (std::uint32_t index = 0U; index < tensor_count; ++index) {
    const Tensor* tensor = state.type_ops->descriptor_tensor_at(payload, index);
    if (tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    tensors[index] = *tensor;
  }
  return emit_tensor_schema_value(
      state, payload, std::span<const Tensor>{tensors}, metadata, false);
}

[[nodiscard]] expected<std::size_t, Error>
RuntimeChannelSession::Impl::OutputState::encode_generated_schema(
    void* opaque, const std::span<std::byte> destination) noexcept {
  auto& context = *static_cast<GeneratedSchemaEncodeContext*>(opaque);
  if (context.output == nullptr || context.payload == nullptr ||
      context.output->type_ops->encode_payload == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto encoded = context.output->type_ops->encode_payload(context.payload, destination);
  if (encoded) {
    trace_output_emit_phase(context.output->phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEncoded,
                            context.output->port,
                            context.sequence);
  }
  return encoded;
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_generated_schema_payload(
    OutputState& state, const void* payload, const SampleMetadata& metadata,
    const std::uint64_t emitted_sequence) noexcept {
  if (state.type_ops->encode_payload == nullptr || state.encode_scratch.data() == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  GeneratedSchemaEncodeContext context{
      .output = &state,
      .payload = payload,
      .sequence = emitted_sequence,
  };
  const channel_internal::InlinePayloadEncoder encoder{
      .state = &context,
      .encode = &encode_generated_schema,
      .capacity = state.type_ops->wire_capacity,
  };
  if (state.direct_subscriber == nullptr && !state.process_local_schema_identity &&
      state.endpoints.size() == 1U) {
    auto published = state.emit_payload({}, metadata, &encoder);
    if (published || published.error().code != ErrorCode::kNotSupported) {
      return published;
    }
  }
  const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                         state.type_ops->wire_capacity};
  auto encoded = encode_generated_schema(&context, destination);
  if (!encoded) {
    return make_unexpected(std::move(encoded).error());
  }
  return state.emit_payload(destination.first(*encoded), metadata);
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_non_tensor_sample(
    OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept {
  const std::uint64_t emitted_sequence = state.next_sequence;
  trace_output_emit_phase(state.phase_counters,
                          RuntimePhaseTracePoint::kInlinePayloadEmitStarted,
                          state.port,
                          emitted_sequence);
  expected<void, Error> emitted = make_unexpected(Error{ErrorCode::kFailure});
  if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
    emitted = emit_generated_schema_payload(state, payload, metadata, emitted_sequence);
  } else if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
    if (!state.pod_identity || state.encode_scratch.data() == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::span<const std::byte> native{static_cast<const std::byte*>(payload),
                                            state.type_ops->payload_size};
    const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                           state.type_ops->wire_capacity};
    auto encoded = channel_internal::PodFieldwiseCodec::encode_prevalidated(
        *state.pod_identity, native, destination);
    if (!encoded) {
      return make_unexpected(std::move(encoded).error());
    }
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEncoded,
                            state.port,
                            emitted_sequence);
    emitted = state.emit_payload(destination, metadata);
  } else {
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEncoded,
                            state.port,
                            emitted_sequence);
    emitted = state.emit_payload(
        {static_cast<const std::byte*>(payload), state.type_ops->wire_capacity}, metadata);
  }
  if (emitted) {
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEmitCompleted,
                            state.port,
                            emitted_sequence);
  }
  return emitted;
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::OutputState::emit_sample(
    void* opaque, const void* sample) noexcept {
  auto& state = *static_cast<OutputState*>(opaque);
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
  auto published = [&]() -> expected<void, Error> {
    if (sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (state.tensor_payload()) {
      return emit_tensor_sample(state, *static_cast<const Sample<Tensor>*>(sample));
    }
    const void* payload = state.type_ops->sample_payload(sample);
    const SampleMetadata* metadata = state.type_ops->sample_metadata(sample);
    if (payload == nullptr || metadata == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return state.tensor_schema_payload() ? emit_tensor_schema_sample(state, payload, *metadata)
                                         : emit_non_tensor_sample(state, payload, *metadata);
  }();
  if (published) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
  } else if (published.error().code == ErrorCode::kBackpressured) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
  }
  return published;
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_descriptor_tensor(void* opaque,
                                                                 const void* descriptor,
                                                                 const Tensor* tensor,
                                                                 EmitOptions options) noexcept {
  auto& state = *static_cast<OutputState*>(opaque);
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
  auto published = [&]() -> expected<void, Error> {
    if (descriptor == nullptr || tensor == nullptr || !state.tensor_schema_payload()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const TensorAccessSnapshot snapshot = TensorAccess::snapshot(*tensor);
    const std::array tensors{*tensor};
    return emit_tensor_schema_value(
        state, descriptor, tensors, metadata_from_options(options, snapshot.byte_size), true);
  }();
  if (published) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
  } else if (published.error().code == ErrorCode::kBackpressured) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
  }
  return published;
}

[[nodiscard]] expected<void, Error>
RuntimeChannelSession::Impl::OutputState::emit_descriptor_tensors(
    void* opaque, const void* descriptor, const std::span<const Tensor> tensors,
    EmitOptions options) noexcept {
  auto& state = *static_cast<OutputState*>(opaque);
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
  auto published = [&]() -> expected<void, Error> {
    if (descriptor == nullptr || tensors.empty() || !state.tensor_schema_payload()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::uint64_t byte_size = 0U;
    for (const Tensor& tensor : tensors) {
      const TensorAccessSnapshot snapshot = TensorAccess::snapshot(tensor);
      if (!snapshot.initialized || !snapshot.valid ||
          snapshot.byte_size > std::numeric_limits<std::uint64_t>::max() - byte_size) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      byte_size += snapshot.byte_size;
    }
    return emit_tensor_schema_value(
        state, descriptor, tensors, metadata_from_options(options, byte_size), true);
  }();
  if (published) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
  } else if (published.error().code == ErrorCode::kBackpressured) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
  }
  return published;
}

[[nodiscard]] expected<RuntimeLoanAllocation, Error>
RuntimeChannelSession::Impl::OutputState::allocate(void* opaque) noexcept {
  auto& state = *static_cast<OutputState*>(opaque);
  if (!state.active) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  const std::uint32_t prior = state.live_loan_count.fetch_add(1U, std::memory_order_acq_rel);
  if (prior >= state.max_live_loans) {
    state.live_loan_count.fetch_sub(1U, std::memory_order_acq_rel);
    return make_unexpected(
        Error{state.loans.empty() ? ErrorCode::kNotSupported : ErrorCode::kResourceExhausted});
  }
  for (LoanControl* loan : state.loans) {
    if (!loan->live) {
      loan->live = true;
      loan->activation_generation = state.active_generation;
      loan->references.fetch_add(1U, std::memory_order_relaxed);
      return RuntimeLoanAllocation{.value = loan->value.data(), .loan = &loan->binding};
    }
  }
  state.live_loan_count.fetch_sub(1U, std::memory_order_acq_rel);
  return make_unexpected(
      Error{state.loans.empty() ? ErrorCode::kNotSupported : ErrorCode::kResourceExhausted});
}

[[nodiscard]] expected<void, Error> RuntimeChannelSession::Impl::OutputState::emit_loan(
    void* opaque, RuntimeLoanBinding* binding, EmitOptions options) noexcept {
  auto& state = *static_cast<OutputState*>(opaque);
  count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
  if (binding == nullptr || binding->state == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* loan = static_cast<LoanControl*>(binding->state);
  if (loan->owner != &state || !loan->live) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!state.active || loan->activation_generation != state.active_generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  const std::uint64_t emitted_sequence = state.next_sequence;
  trace_output_emit_phase(state.phase_counters,
                          RuntimePhaseTracePoint::kInlinePayloadEmitStarted,
                          state.port,
                          emitted_sequence);
  expected<void, Error> published;
  if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
    published = emit_generated_schema_payload(
        state, loan->value.data(), metadata_from_options(options, 0U), emitted_sequence);
  } else if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
    if (!state.pod_identity || state.encode_scratch.data() == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::span<const std::byte> native{static_cast<const std::byte*>(loan->value.data()),
                                            state.type_ops->payload_size};
    const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                           state.type_ops->wire_capacity};
    auto encoded = channel_internal::PodFieldwiseCodec::encode_prevalidated(
        *state.pod_identity, native, destination);
    if (!encoded) {
      return make_unexpected(std::move(encoded).error());
    }
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEncoded,
                            state.port,
                            emitted_sequence);
    published = state.emit_payload(destination, metadata_from_options(options, destination.size()));
  } else {
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEncoded,
                            state.port,
                            emitted_sequence);
    published = state.emit_payload(
        {static_cast<const std::byte*>(loan->value.data()), state.type_ops->wire_capacity},
        metadata_from_options(options, state.type_ops->wire_capacity));
  }
  if (published || publication_error_consumes_authority(published.error().code)) {
    loan->retire_live_authority();
  }
  if (published) {
    trace_output_emit_phase(state.phase_counters,
                            RuntimePhaseTracePoint::kInlinePayloadEmitCompleted,
                            state.port,
                            emitted_sequence);
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
  } else if (published.error().code == ErrorCode::kBackpressured) {
    count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
  }
  return published;
}

const RuntimeOutputBindingOps RuntimeChannelSession::Impl::OutputState::kBindingOps{
    .emit_sample = &emit_sample,
    .emit_descriptor_tensor = &emit_descriptor_tensor,
    .emit_descriptor_tensors = &emit_descriptor_tensors,
    .allocate = &allocate,
    .emit_loan = &emit_loan,
};

}  // namespace holoscan::detail
