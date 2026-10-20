/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file
/// @brief Carry provenance across an operator that transforms what it received.
///
/// Two independent records travel with a descriptor-backed sample and they forward by different
/// means. Treating them as one is the usual reason a pipeline loses provenance with nothing to
/// indicate which operator dropped it.
///
/// The payload descriptor is the schema value itself: an \c Image's geometry, encoding, exposure,
/// and \c Header. It forwards by handing the received descriptor straight back to
/// \c Output<T>::emit_tensor alongside a fresh Tensor loan. That overload is specified to ignore
/// the descriptor's embedded Tensor member and treat the loan's Tensor as authoritative, so one
/// call publishes new payload bytes under the metadata that arrived. The operator neither reads nor
/// names a single field, and needs no change when the schema grows one. To change one field and
/// forward the rest, update the received owning descriptor, or copy it if the original is still
/// needed. Copy behavior depends on the descriptor type. For \c ImageT, the generated copy
/// constructor copies the \c Header, its optional timestamp and frame ID, the Tensor wrapper, and
/// the plane-layout vector. These copies may allocate, but the Tensor's pixel storage stays shared;
/// copying the descriptor does not copy pixels. The copied \c Header is independent, so editing it
/// does not require another deep copy to protect the original descriptor's header.
///
/// For callback-local read-only use, `Input<T>::selected_sample()` avoids that owning copy.
/// Publish unchanged metadata with `emit_tensor(loan, view.data(), derive_from(view.metadata()))`.
/// For bounded Image metadata edits, cold-prepare `ImageDescriptorScratch`, copy metadata into it,
/// and pass its Image alongside the output loan. Keep its Tensor member null and never retain the
/// borrowed descriptor past the callback. Runtime-admitted stream leases, not the host view,
/// protect asynchronous Tensor reads.
///
/// The runtime envelope — \ref holoscan::SampleMetadata, holding capture time, clock domain, frame
/// identity, trace, and flags — is not part of the payload and does not ride along with it. Runtime
/// rebuilds it from \ref holoscan::EmitOptions on every publication, which is why a forwarded
/// descriptor has to be paired with \ref derive_from. Emit with default options instead and the
/// sample carries no capture time, no clock domain, and no frame identity, while its descriptor
/// still looks complete; that asymmetry is what makes the loss easy to miss.
///
/// `public/tests/sensor_io/test_descriptor_passthrough.cpp` pins all three cases against a running
/// graph, the default-options mistake included.

#include <cstddef>
#include <cstdint>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::sensor_io {

/// @brief Carry a received sample's provenance onto a value derived from it.
///
/// `Output<T>::emit(value, options)` builds a *fresh* metadata record from \p options and fills in
/// only the fields Runtime owns: publication sequence, payload kind, byte size, and the publication
/// coordinate it stamps at the submission boundary.
/// Everything the producer knew about where the data came from — when it was captured, which
/// clock that coordinate belongs to, which frame it was, whether it was already degraded — comes
/// from \p options and from nowhere else. An operator that receives a sample, transforms the
/// payload, and calls `emit(new_value)` with default options therefore publishes a sample whose
/// capture time is *absent*, and Core is explicit that absent means absent: it never substitutes
/// arrival, activation, or publication time for it.
///
/// The loss is invisible at the point where it happens and expensive several operators later. Every
/// temporal input selector except `DrainAll` is defined in terms of capture time, so an unstamped
/// sample is silently skipped by the scalar selectors and rejected outright by the aligned and
/// windowed ones. A pipeline whose first transform drops provenance cannot be aligned, cannot be
/// windowed, and cannot report end-to-end latency, and none of those failures name the operator
/// that caused them.
///
/// @tparam In Payload type of the received sample.
/// @param source Metadata of the sample this publication is derived from.
/// @return Publication options that preserve capture time, frame identity, trace correlation, and
/// validity flags.
/// @note Absent fields stay absent. A source that carries no capture time yields options with no
/// capture time rather than a zero coordinate, because a zero coordinate in a valid clock domain is
/// a real instant and would be a lie.
/// @note `SampleMetadata` retains only the trace identity, so the forwarded trace context is a root
/// context rather than a continuation of the source's span.
/// @note An operator that changes nothing about the payload should forward the whole
/// `Sample<T>` through the `emit(Sample<T>)` overload instead; this helper is for the case where a
/// new value is constructed and only the provenance carries over.
/// @note For a descriptor-backed payload whose Tensor is being replaced, pass the received
/// descriptor and these options together to `emit_tensor`. The descriptor carries the payload
/// metadata and these options carry the envelope, and neither substitutes for the other.
[[nodiscard]] inline EmitOptions derive_from(const SampleMetadata& source) noexcept {
  EmitOptions options{};
  if (source.source_clock_id.valid()) {
    options.capture_time = source.capture_time();
  }
  if (source.frame_id != 0U) {
    options.frame_id = source.frame_id;
  }
  if (source.trace_id != 0U) {
    options.trace = TraceContext::root(source.trace_id);
  }
  options.flags = source.flags;
  return options;
}

/// @brief Carry a received sample's provenance onto a value derived from it.
/// @tparam In Payload type of the received sample.
/// @param source Sample this publication is derived from.
/// @return Publication options that preserve the source's provenance.
/// @see derive_from(const SampleMetadata&)
template <PublishedPayload In>
[[nodiscard]] EmitOptions derive_from(const Sample<In>& source) noexcept {
  return derive_from(source.metadata);
}

/// @brief Select which contributing capture time a many-to-one result inherits.
enum class CaptureTimePolicy : std::uint8_t {
  /// Inherit the oldest contributing capture time. Choose this to measure end-to-end latency, since
  /// the result is no fresher than its stalest input.
  kEarliest = 0,
  /// Inherit the newest contributing capture time. Choose this when the result describes the most
  /// recent observation rather than the whole set.
  kLatest = 1,
};

/// @brief Combine the provenance of several inputs into one derived publication.
///
/// Fusion, inference, and any other many-to-one operator has to answer a question a one-to-one
/// transform does not: which of its inputs did the result come from? There is no universally
/// correct answer, so the usual outcome is that nobody answers it and the result is published
/// unstamped, which removes it from every downstream temporal selector.
///
/// This accumulator makes the choice explicit and refuses the cases where no choice is defensible:
///
/// - **Mixed clock domains are an error, not a silent pick.** Time points from different domains
///   have no ordering without an admitted conversion, so "earliest" is undefined across them.
/// - **A missing contributor demotes the result rather than being ignored.** If some inputs were
///   stamped and some were not, the combined result carries no capture time and is marked degraded,
///   because the extremum of a set with unknown members is unknown. If none were stamped there is
///   nothing to demote: the result is unstamped, exactly as every one of its inputs was, and
///   flagging that would report a loss that did not happen.
/// - **Validity and degradation are unions.** One invalid input makes the result invalid.
/// - **Frame identity survives only when every contributor agrees**, which is what makes it usable
///   for correlating the bands of one subdivided frame and useless for anything else.
/// - **Trace lineage survives unless two contributors claim different roots.** Unlike frame
///   identity, a contributor carrying no trace does not retire it, because tracing is sampled and
///   absence is the common case rather than a contradiction.
///
/// The object is not thread-safe and allocates nothing.
class ProvenanceAccumulator {
 public:
  /// @brief Record one contributing sample's metadata.
  /// @param source Metadata of an input that contributed to the pending result.
  void observe(const SampleMetadata& source) noexcept {
    ++observed_;
    flags_ |= source.flags;

    // Frame identity is only meaningful when the whole set shares it, so one contributor that
    // disagrees or carries none retires it for the rest of this activation.
    if (source.frame_id == 0U || (frame_id_ != 0U && frame_id_ != source.frame_id)) {
      frame_conflict_ = true;
      frame_id_ = 0U;
    } else if (!frame_conflict_) {
      frame_id_ = source.frame_id;
    }

    // Two contributors under different trace roots leave the merged result with no single lineage.
    // Keeping the one that happened to arrive first would attribute the whole result's latency to
    // one arbitrary input, which is worse for a correlation tool than reporting no root at all.
    if (source.trace_id != 0U && !trace_conflict_) {
      if (trace_id_ == 0U) {
        trace_id_ = source.trace_id;
      } else if (trace_id_ != source.trace_id) {
        trace_conflict_ = true;
        trace_id_ = 0U;
      }
    }

    if (!source.source_clock_id.valid()) {
      unstamped_ = true;
      return;
    }

    const TimePoint capture = source.capture_time();
    if (!stamped_) {
      stamped_ = true;
      domain_ = capture.clock_id;
      earliest_ns_ = capture.timestamp_ns;
      latest_ns_ = capture.timestamp_ns;
      return;
    }
    if (domain_ != capture.clock_id) {
      mixed_domains_ = true;
      return;
    }
    earliest_ns_ = capture.timestamp_ns < earliest_ns_ ? capture.timestamp_ns : earliest_ns_;
    latest_ns_ = capture.timestamp_ns > latest_ns_ ? capture.timestamp_ns : latest_ns_;
  }

  /// @brief Record one contributing sample's metadata.
  /// @tparam In Payload type of the contributing sample.
  /// @param source Input that contributed to the pending result.
  template <PublishedPayload In>
  void observe(const Sample<In>& source) noexcept {
    observe(source.metadata);
  }

  /// @brief Build the publication options for the combined result.
  /// @param policy Which contributing capture time the result should inherit.
  /// @return Options carrying the selected capture time and the combined flags, frame identity, and
  /// trace correlation.
  /// @retval ErrorCode::kInvalidArgument No contributor was observed, or contributors carried
  /// capture times from different clock domains, which have no defined ordering.
  [[nodiscard]] expected<EmitOptions, Error> emit_options(CaptureTimePolicy policy) const noexcept {
    if (observed_ == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (mixed_domains_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    EmitOptions options{};
    options.flags = flags_;
    if (frame_id_ != 0U && !frame_conflict_) {
      options.frame_id = frame_id_;
    }
    if (trace_id_ != 0U && !trace_conflict_) {
      options.trace = TraceContext::root(trace_id_);
    }

    if (!stamped_ || unstamped_) {
      // The extremum of a set with unknown members is unknown. Say so, and mark the result so a
      // consumer can tell this apart from a source that never had capture time to begin with.
      if (stamped_) {
        options.flags |= SampleFlags::kDegraded;
      }
      return options;
    }

    options.capture_time = TimePoint{
        .timestamp_ns = policy == CaptureTimePolicy::kEarliest ? earliest_ns_ : latest_ns_,
        .clock_id = domain_};
    return options;
  }

  /// @brief Return how many contributors were observed.
  [[nodiscard]] std::size_t observed() const noexcept { return observed_; }

  /// @brief Return whether contributors carried capture times from different clock domains.
  [[nodiscard]] bool mixed_domains() const noexcept { return mixed_domains_; }

  /// @brief Return whether any contributor arrived without a capture time.
  [[nodiscard]] bool has_unstamped_contributor() const noexcept { return unstamped_; }

  /// @brief Discard the accumulated provenance before the next activation.
  void reset() noexcept { *this = ProvenanceAccumulator{}; }

 private:
  std::size_t observed_{};
  std::int64_t earliest_ns_{};
  std::int64_t latest_ns_{};
  ClockId domain_{};
  std::uint64_t frame_id_{};
  std::uint64_t trace_id_{};
  SampleFlags flags_{SampleFlags::kNone};
  bool stamped_{};
  bool unstamped_{};
  bool mixed_domains_{};
  bool frame_conflict_{};
  bool trace_conflict_{};
};

}  // namespace holoscan::sensor_io
