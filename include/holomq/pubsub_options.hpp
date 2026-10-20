/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/limits.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace holomq {

/// @brief Queue behavior requested for a typed pub/sub service.
enum class QueuePolicy : std::uint8_t {
  /// Preserve lossless delivery by applying backpressure when subscriber queues are full.
  kBlocking = 0U,
  /// Keep the newest configured number of unread samples by replacing the oldest when full.
  kDropOldest = 1U,
};

/// @brief Select one publisher's response when a blocking service cannot accept a sample.
///
/// This endpoint-local policy does not change the service queue contract. It exists so a
/// scheduler-owned publisher can preserve service-level blocking semantics without waiting while
/// it owns a non-reentrant execution resource.
enum class PublisherPressurePolicy : std::uint8_t {
  /// Follow the queue service policy. A blocking service waits until every recipient accepts.
  kFollowService = 0U,
  /// Reject delivery to each full recipient and return \ref ErrorCode::kBackpressured.
  /// Other fanout recipients may already have accepted the same publication.
  kRejectWhenFull = 1U,
  /// Retry delivery until \ref PublisherPubSubOptions::bounded_retry_budget_ns expires, then
  /// reject the full recipient and return \ref ErrorCode::kBackpressured.
  /// Other fanout recipients may already have accepted the same publication.
  kBoundedRetry = 2U,
};

namespace detail {

// Conservative allowance for the sample header, alignment, pool indices, and segment management.
// The pinned backend uses a small aligned header, 32-bit indices, and a fixed segment
// header. Reserving 4 KiB per sample and per segment keeps all intermediate layouts representable.
inline constexpr std::uint64_t kByteSliceGeometryReserve = 4096U;
inline constexpr std::uint64_t kByteSliceGeometryLimit =
    static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
static_assert(kByteSliceAlignment <= kByteSliceGeometryReserve);

// Options may cross aggregate, deserialization, or test-injection boundaries where an explicit
// cast can produce an out-of-range enum value. Validate before the derived helpers below collapse
// such a value into an apparently supported lossy policy.
[[nodiscard]] constexpr bool is_valid_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] constexpr bool is_valid_publisher_pressure_policy(
    const PublisherPressurePolicy policy) noexcept {
  return policy == PublisherPressurePolicy::kFollowService ||
         policy == PublisherPressurePolicy::kRejectWhenFull ||
         policy == PublisherPressurePolicy::kBoundedRetry;
}

/// Return whether the byte-slice fields match the payload kind: a bounded capacity for a byte-slice
/// service, and neither field for a fixed-payload service.
[[nodiscard]] constexpr bool is_valid_byte_slice_contract(const bool byte_slice,
                                                          const std::uint64_t capacity,
                                                          const std::uint32_t profile) noexcept {
  return byte_slice ? capacity != 0U && capacity <= kByteSliceGeometryLimit -
                                                        2U * kByteSliceGeometryReserve -
                                                        (kByteSliceAlignment - 1U)
                    : capacity == 0U && profile == 0U;
}

}  // namespace detail

/// @brief Static configuration shared by every endpoint of a typed pub/sub service.
///
/// Pass the same option set to publisher and subscriber builders regardless of which endpoint
/// creates the service first. HoloMQ verifies queue policy, queue and borrow maxima, history, and
/// each explicit non-zero subscriber-capacity request when opening an existing service.
/// Endpoint-local requests remain in \ref PublisherPubSubOptions and
/// \ref SubscriberPubSubOptions.
struct PubSubServiceOptions {
  /// Queue behavior for full subscriber queues.
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  /// Maximum subscribers and event-wake listeners supported by the service. Zero uses the backend
  /// defaults.
  ///
  /// Set this to the graph's planned fanout to avoid provisioning and scanning unused backend
  /// connection slots. Subscriber-first creation must use the same planned capacity.
  std::uint32_t max_subscribers{0U};
  /// Maximum per-subscriber queue depth supported by the service. Must be greater than zero.
  std::uint32_t max_subscriber_queue_depth{k_subscriber_queue_depth};
  /// Maximum samples each subscriber may borrow concurrently. Zero derives from the configured
  /// service maximum queue depth.
  std::uint32_t max_subscriber_borrowed_samples{k_subscriber_max_borrowed_samples};
  /// Publisher history retained for late-joining subscribers. Zero disables history.
  std::uint32_t publisher_history_depth{k_publisher_history_depth};
  /// Enable the parallel event service used by kernel-wait publishers and subscribers.
  ///
  /// Every endpoint in a planned service must use the same value. HoloMQ records this field as an
  /// iceoryx2 sample-service attribute. Publisher-first and subscriber-first creation therefore
  /// both reject a peer whose event-wake request differs from the established service contract.
  bool enable_event_wake{true};
  /// Capacity in bytes of every frame of a \ref ByteSlice service.
  ///
  /// Required (non-zero) for a byte-slice service and zero for a fixed-payload service. Each
  /// publisher preallocates its sample pool for this capacity, and a loan above it fails. HoloMQ
  /// records the value as an iceoryx2 service attribute, so every endpoint must use the same one.
  /// Setup returns \c kInvalidArgument before backend creation when the aligned capacity plus
  /// a conservative 4 KiB allowance per sample and 4 KiB per pool exceeds \c PTRDIFF_MAX,
  /// or the configured sample count exceeds \c UINT32_MAX minus one. The count includes all
  /// subscriber queues and borrowed samples, retained history, and publisher loans.
  std::uint64_t byte_slice_capacity{0U};
  /// Application-defined frame layout identifier of a \ref ByteSlice service.
  ///
  /// HoloMQ records the value as an iceoryx2 service attribute, so byte-slice services that carry
  /// different frame layouts never open each other. Zero for a fixed-payload service.
  std::uint32_t byte_slice_profile{0U};

  /// @brief Return whether the backend service uses safe overflow.
  /// @return `false` for \ref QueuePolicy::kBlocking; otherwise `true`.
  [[nodiscard]] bool uses_safe_overflow() const noexcept {
    return queue_policy != QueuePolicy::kBlocking;
  }

  /// @brief Return the configured service queue-depth limit.
  /// @return \ref max_subscriber_queue_depth.
  [[nodiscard]] std::uint32_t effective_max_subscriber_queue_depth() const noexcept {
    return max_subscriber_queue_depth;
  }

  /// @brief Return the service borrowed-sample limit implied by this option set.
  ///
  /// @return \ref max_subscriber_borrowed_samples when non-zero; otherwise
  /// \ref max_subscriber_queue_depth.
  [[nodiscard]] std::uint32_t effective_max_subscriber_borrowed_samples() const noexcept {
    if (max_subscriber_borrowed_samples != 0U) {
      return max_subscriber_borrowed_samples;
    }
    return max_subscriber_queue_depth;
  }
};

/// @brief Publisher-local configuration for a typed pub/sub endpoint.
///
/// These fields configure only the publisher port and its HoloMQ callbacks. Static service
/// capacity and queue behavior belong to \ref PubSubServiceOptions.
struct PublisherPubSubOptions {
  /// Maximum unsent samples this publisher may loan concurrently. Zero derives the current default.
  std::uint32_t max_loaned_samples{0U};
  /// Required connected subscriber count before a send may submit to the backend.
  ///
  /// Zero disables the gate. A non-zero value is an endpoint-local topology contract and must not
  /// exceed a non-zero \ref PubSubServiceOptions::max_subscribers service capacity.
  std::uint32_t expected_recipient_count{0U};
  /// Enable per-send sample-release callbacks.
  ///
  /// Disable this for self-contained payloads that do not retain external resources. Calling
  /// \c Publisher::send(message,on_released) then returns \ref ErrorCode::kNotImplemented.
  bool enable_sample_release_callbacks{true};
  /// Request per-recipient replacement attribution for descriptor-backed lossy channels.
  ///
  /// This is only meaningful for services using \ref QueuePolicy::kDropOldest. Publisher creation
  /// fails when this is true without a matching `PublisherBuilder::on_sample_replaced` callback.
  /// Calling `on_sample_replaced` also sets this option on the builder.
  bool replacement_attribution_required{false};
  /// Publisher-local response to a full recipient queue.
  ///
  /// `kRejectWhenFull` is valid only for a blocking service. It preserves each full recipient's
  /// queued sample and returns `kBackpressured` instead of waiting. Another fanout recipient may
  /// already have accepted the publication, so only an exact single-recipient contract can treat
  /// this result as aggregate zero-delivery, pre-submit evidence.
  PublisherPressurePolicy pressure_policy{PublisherPressurePolicy::kFollowService};
  /// Maximum nanoseconds spent retrying one full recipient under `kBoundedRetry`.
  ///
  /// Must be non-zero only when \ref pressure_policy is `kBoundedRetry`. The budget is checked by
  /// the backend unable-to-deliver callback and does not affect an accepted publication.
  std::uint64_t bounded_retry_budget_ns{0U};
  /// Optional flag that ends `kBoundedRetry` retries early.
  ///
  /// Once it reads true, a full recipient whose budget has not expired is rejected at once instead
  /// of being retried. A send whose only rejections came from this flag fails with `kInterrupted`,
  /// so the caller can tell a cancelled wait from pressure. A send that an expired budget also
  /// rejected keeps `kBackpressured`. Both keep the rejecting recipient's attribution, and
  /// recipients with capacity still accept. Must be null unless \ref pressure_policy is
  /// `kBoundedRetry`, and must outlive the publisher.
  const std::atomic<bool>* bounded_retry_cancellation{nullptr};

  /// @brief Return the publisher loan limit implied by this endpoint and service option set.
  /// @param service_options Static service configuration used by this publisher.
  /// @return \ref max_loaned_samples when non-zero; otherwise the current compatibility default:
  /// max(service queue depth, publisher history depth).
  [[nodiscard]] std::uint32_t effective_max_loaned_samples(
      const PubSubServiceOptions& service_options) const noexcept {
    if (max_loaned_samples != 0U) {
      return max_loaned_samples;
    }
    return service_options.max_subscriber_queue_depth > service_options.publisher_history_depth
               ? service_options.max_subscriber_queue_depth
               : service_options.publisher_history_depth;
  }

  /// @brief Return the maximum transport samples that can retain release callbacks.
  /// @param service_options Static service configuration used by this publisher.
  /// @return The iceoryx2 publisher data-segment sample capacity, or no value when the
  /// capacity cannot be represented by `std::size_t`.
  ///
  /// A sent sample can remain alive in a subscriber queue, as a borrowed sample, or in publisher
  /// history after its publisher loan has returned. Release-callback storage must therefore cover
  /// the complete transport sample window, not only the simultaneous publisher-loan limit.
  [[nodiscard]] std::optional<std::size_t> effective_sample_release_callback_capacity(
      const PubSubServiceOptions& service_options) const noexcept {
    const std::uint64_t configured_subscribers = service_options.max_subscribers == 0U
                                                     ? k_default_max_subscribers
                                                     : service_options.max_subscribers;
    const std::uint64_t subscriber_capacity = configured_subscribers > expected_recipient_count
                                                  ? configured_subscribers
                                                  : expected_recipient_count;
    const std::uint64_t queue_and_borrow_capacity =
        static_cast<std::uint64_t>(service_options.max_subscriber_queue_depth) +
        service_options.effective_max_subscriber_borrowed_samples();
    const std::uint64_t fixed_capacity =
        static_cast<std::uint64_t>(service_options.publisher_history_depth) +
        effective_max_loaned_samples(service_options);
    constexpr std::uint64_t kMaximum = std::numeric_limits<std::size_t>::max();
    if (fixed_capacity > kMaximum ||
        (queue_and_borrow_capacity != 0U &&
         subscriber_capacity > (kMaximum - fixed_capacity) / queue_and_borrow_capacity)) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(subscriber_capacity * queue_and_borrow_capacity +
                                    fixed_capacity);
  }
};

/// @brief Subscriber-local configuration for a typed pub/sub endpoint.
///
/// The requested queue and borrowed-sample limits must not exceed the maxima in the associated
/// \ref PubSubServiceOptions.
struct SubscriberPubSubOptions {
  /// Requested queue depth for this subscriber. Must be greater than zero.
  std::uint32_t queue_depth{k_subscriber_queue_depth};
  /// Required borrowed-sample allowance for this subscriber. Zero derives from the configured queue
  /// depth.
  std::uint32_t max_borrowed_samples{k_subscriber_max_borrowed_samples};

  /// @brief Return the configured backend queue depth for this endpoint.
  /// @param service_options Static service configuration used by this subscriber.
  /// @return \ref queue_depth.
  [[nodiscard]] std::uint32_t effective_queue_depth(
      const PubSubServiceOptions& service_options) const noexcept {
    static_cast<void>(service_options);
    return queue_depth;
  }

  /// @brief Return the borrowed-sample allowance implied by this option set.
  /// @return \ref max_borrowed_samples when non-zero; otherwise \ref queue_depth.
  [[nodiscard]] std::uint32_t effective_max_borrowed_samples() const noexcept {
    if (max_borrowed_samples != 0U) {
      return max_borrowed_samples;
    }
    return queue_depth;
  }
};

namespace detail {

/// Reject unrepresentable byte-slice sample and management pools without performing allocation.
/// \p subscribers is the requested, or already negotiated, backend subscriber count.
[[nodiscard]] inline bool is_valid_byte_slice_pool_geometry(
    const PubSubServiceOptions& service, const std::uint64_t loans,
    const std::uint64_t subscribers) noexcept {
  if (!is_valid_byte_slice_contract(
          true, service.byte_slice_capacity, service.byte_slice_profile)) {
    return false;
  }
  constexpr std::uint64_t kMaximumSamples = std::numeric_limits<std::uint32_t>::max() - 1U;
  const std::uint64_t queue_and_borrow =
      static_cast<std::uint64_t>(service.max_subscriber_queue_depth) +
      service.effective_max_subscriber_borrowed_samples();
  if (loans > kMaximumSamples || service.publisher_history_depth > kMaximumSamples - loans) {
    return false;
  }
  const std::uint64_t fixed_samples = service.publisher_history_depth + loans;
  if (queue_and_borrow != 0U &&
      subscribers > (kMaximumSamples - fixed_samples) / queue_and_borrow) {
    return false;
  }
  const std::uint64_t samples = subscribers * queue_and_borrow + fixed_samples;
  if (samples == 0U) {
    return false;
  }
  const std::uint64_t aligned_capacity = (service.byte_slice_capacity + kByteSliceAlignment - 1U) /
                                         kByteSliceAlignment * kByteSliceAlignment;
  const std::uint64_t sample_budget = aligned_capacity + kByteSliceGeometryReserve;
  return samples <= (kByteSliceGeometryLimit - kByteSliceGeometryReserve) / sample_budget;
}

}  // namespace detail

}  // namespace holomq
