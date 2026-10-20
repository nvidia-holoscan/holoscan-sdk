/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan::sensor_io {

/// @brief Report what one device sequence number implies about the samples before it.
struct SequenceObservation {
  std::uint64_t sequence{};  ///< Device-reported sequence number for the current sample.
  std::uint64_t gap{};       ///< Samples the device produced but this operator never saw.
  bool first{};              ///< Whether this is the first observation of the current epoch.
  bool regressed{};          ///< Whether the sequence moved backwards or repeated.
};

/// @brief Turn a device's own sample counter into an explicit dropped-sample count.
///
/// Capture devices count the samples they produce, and that counter is the only evidence a pipeline
/// has that samples went missing between the device and the operator. An overrun shows up as a jump
/// in the counter and in nothing else: the sample that arrives after a drop looks entirely normal,
/// carries a plausible timestamp, and is indistinguishable from an on-time sample once the counter
/// is discarded. A pipeline that ignores the counter therefore reports full frame rate while
/// silently dropping frames, and the loss surfaces much later as an unexplained motion artifact or
/// a fusion result that does not line up.
///
/// This object keeps the previous counter value and reports the difference, so the operator can
/// mark the sample degraded and publish the count instead of losing it. It deliberately does not
/// decide what to do about a gap: dropping frames is normal for some sensors and fatal for others,
/// and that judgment belongs to the operator.
///
/// A regression, where the counter repeats or moves backwards, is reported separately rather than
/// folded into the gap. It usually means the device was restarted or the counter wrapped at a width
/// narrower than the operator assumed, which is a different problem from a dropped sample.
///
/// The object is not thread-safe and allocates nothing.
class SequenceTracker {
 public:
  /// @brief Observe the device sequence number carried by the current sample.
  /// @param sequence Device-reported counter value for this sample.
  /// @return What this value implies about samples that were never delivered.
  /// @note The first observation of an epoch reports no gap regardless of where the device's
  /// counter happens to start, because there is no earlier value to compare against.
  [[nodiscard]] SequenceObservation observe(std::uint64_t sequence) noexcept {
    SequenceObservation observation{.sequence = sequence};
    if (!seen_) {
      seen_ = true;
      observation.first = true;
    } else if (sequence <= previous_) {
      observation.regressed = true;
    } else {
      observation.gap = sequence - previous_ - 1U;
      total_gap_ += observation.gap;
    }
    previous_ = sequence;
    return observation;
  }

  /// @brief Return the total number of samples missed since the last reset.
  [[nodiscard]] std::uint64_t total_gap() const noexcept { return total_gap_; }

  /// @brief Return the most recently observed sequence number.
  [[nodiscard]] std::uint64_t previous() const noexcept { return previous_; }

  /// @brief Return whether any sequence number has been observed since the last reset.
  [[nodiscard]] bool seen() const noexcept { return seen_; }

  /// @brief Forget the observed history so the next observation starts a new epoch.
  /// @note Call this when the device counter restarts, such as after a stream is stopped and
  /// started again, so the restart is not reported as an enormous gap.
  void reset() noexcept { *this = SequenceTracker{}; }

 private:
  std::uint64_t previous_{};
  std::uint64_t total_gap_{};
  bool seen_{};
};

}  // namespace holoscan::sensor_io
