/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_common.hpp"

#include <fmt/format.h>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/memory/memory_buffer.hpp>

#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::channel_internal {
namespace {

constexpr std::string_view kOrderedTerminalServiceSuffix{".hsdk-terminal-v1"};
constexpr std::size_t kMaximumTransportServiceNameSize = 255U;

}  // namespace

bool is_ordered_terminal_service_id(const std::string_view service_id) noexcept {
  return service_id.ends_with(kOrderedTerminalServiceSuffix);
}

std::string ordered_terminal_service_id(const std::string_view data_service_id) {
  if (data_service_id.empty() || is_ordered_terminal_service_id(data_service_id) ||
      data_service_id.size() >
          kMaximumTransportServiceNameSize - kOrderedTerminalServiceSuffix.size()) {
    return {};
  }
  std::string result;
  result.reserve(data_service_id.size() + kOrderedTerminalServiceSuffix.size());
  result.append(data_service_id);
  result.append(kOrderedTerminalServiceSuffix);
  return result;
}

holomq::PubSubServiceOptions ordered_terminal_service_options(
    const ChannelOptions& options) noexcept {
  return holomq::PubSubServiceOptions{
      .queue_policy = holomq::QueuePolicy::kBlocking,
      .max_subscribers = options.expected_recipient_count,
      .max_subscriber_queue_depth = 1U,
      .max_subscriber_borrowed_samples = 1U,
      .publisher_history_depth = 0U,
      .enable_event_wake = options.enable_event_wake,
  };
}

holomq::PublisherPubSubOptions ordered_terminal_publisher_options(
    const ChannelOptions& options) noexcept {
  return holomq::PublisherPubSubOptions{
      .max_loaned_samples = 1U,
      .expected_recipient_count = options.expected_recipient_count,
      .enable_sample_release_callbacks = false,
      .replacement_attribution_required = false,
      .pressure_policy = holomq::PublisherPressurePolicy::kFollowService,
      .bounded_retry_budget_ns = 0U,
  };
}

holomq::SubscriberPubSubOptions ordered_terminal_subscriber_options() noexcept {
  return holomq::SubscriberPubSubOptions{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
}
namespace {

struct HoloMQErrorTranslation {
  ErrorCode code{ErrorCode::kFailure};
  const char* invariant_token{};
};

// Keep the independent domains joined in exactly one named switch. Deliberately omit `default` so
// compiler enum-exhaustiveness diagnostics remain useful when HoloMQ adds a named value; the
// explicit return after the switch is the runtime fail-closed path for injected/future values.
[[nodiscard]] constexpr HoloMQErrorTranslation translate_holomq_error_code(
    const holomq::ErrorCode code, const bool failure_carrier) noexcept {
  switch (code) {
    case holomq::ErrorCode::kSuccess:
      return failure_carrier
                 ? HoloMQErrorTranslation{ErrorCode::kFailure, "HOLOMQ_ERROR_SUCCESS_IN_FAILURE"}
                 : HoloMQErrorTranslation{ErrorCode::kSuccess, nullptr};
    case holomq::ErrorCode::kFailure:
      return {ErrorCode::kFailure, nullptr};
    case holomq::ErrorCode::kReceiveError:
      return {ErrorCode::kReceiveError, nullptr};
    case holomq::ErrorCode::kInvalidArgument:
      return {ErrorCode::kInvalidArgument, nullptr};
    case holomq::ErrorCode::kNotFound:
      return {ErrorCode::kNotFound, nullptr};
    case holomq::ErrorCode::kNotImplemented:
      return {ErrorCode::kNotImplemented, nullptr};
    case holomq::ErrorCode::kBackpressured:
      return {ErrorCode::kBackpressured, nullptr};
    case holomq::ErrorCode::kNotReady:
      return {ErrorCode::kNotReady, nullptr};
    case holomq::ErrorCode::kStaleEndpoint:
      return {ErrorCode::kStaleEndpoint, nullptr};
    case holomq::ErrorCode::kClosed:
      return {ErrorCode::kClosed, nullptr};
    case holomq::ErrorCode::kTimeout:
      return {ErrorCode::kTimeout, nullptr};
    case holomq::ErrorCode::kResourceExhausted:
      return {ErrorCode::kResourceExhausted, nullptr};
    case holomq::ErrorCode::kPermissionDenied:
      return {ErrorCode::kPermissionDenied, nullptr};
    case holomq::ErrorCode::kInterrupted:
      return {ErrorCode::kInterrupted, nullptr};
    case holomq::ErrorCode::kProtocolError:
      return {ErrorCode::kProtocolError, nullptr};
    case holomq::ErrorCode::kIncompatible:
      return {ErrorCode::kIncompatible, nullptr};
    case holomq::ErrorCode::kIncompleteFanout:
      return {ErrorCode::kNotReady, nullptr};
    case holomq::ErrorCode::kNotSupported:
      return {ErrorCode::kNotSupported, nullptr};
  }
  return {ErrorCode::kFailure, "HOLOMQ_ERROR_UNMAPPED"};
}

[[nodiscard]] Error translated_error(holomq::Error error) noexcept {
  const HoloMQErrorTranslation translation = translate_holomq_error_code(error.code, true);
  if (translation.invariant_token == nullptr) {
    return Error{translation.code, std::move(error.message)};
  }
  try {
    return Error{translation.code,
                 error.message.empty()
                     ? std::string{translation.invariant_token}
                     : fmt::format("{}: {}", translation.invariant_token, error.message)};
  } catch (...) {
    return Error{translation.code};
  }
}

}  // namespace

ErrorCode to_holoscan_error_code(const holomq::ErrorCode code) noexcept {
  return translate_holomq_error_code(code, false).code;
}

Error from_holomq_error(holomq::Error error) noexcept {
  return translated_error(std::move(error));
}

Error from_holomq_error(holomq::Error error, const std::string_view context) noexcept {
  const HoloMQErrorTranslation translation = translate_holomq_error_code(error.code, true);
  try {
    if (error.message.empty()) {
      error.message = std::string(context);
    } else if (!context.empty()) {
      error.message = fmt::format("{}: {}", context, error.message);
    }
    return from_holomq_error(std::move(error));
  } catch (...) {
    return Error{translation.code};
  }
}

expected<void, Error> from_holomq_result(holomq::expected<void, holomq::Error> result) noexcept {
  if (!result) {
    return make_unexpected(from_holomq_error(std::move(result).error()));
  }
  return {};
}

expected<bool, Error> from_holomq_result(holomq::expected<bool, holomq::Error> result) noexcept {
  if (!result) {
    return make_unexpected(from_holomq_error(std::move(result).error()));
  }
  return result.value();
}

expected<std::uint32_t, Error> from_holomq_publication_result(
    holomq::expected<int, holomq::Error> result,
    const std::uint32_t expected_recipient_count) noexcept {
  return from_holomq_publication_result(std::move(result), expected_recipient_count, false);
}

namespace {

[[nodiscard]] bool& publication_cancelled_storage() noexcept {
  static thread_local bool cancelled{};
  return cancelled;
}

}  // namespace

void note_publication_cancelled() noexcept {
  publication_cancelled_storage() = true;
}

bool take_publication_cancelled() noexcept {
  return std::exchange(publication_cancelled_storage(), false);
}

expected<std::uint32_t, Error> from_holomq_publication_result(
    holomq::expected<int, holomq::Error> result, const std::uint32_t expected_recipient_count,
    const bool graph_retry_boundary) noexcept {
  try {
    if (!result) {
      holomq::Error transport_error = std::move(result).error();
      if (transport_error.code == holomq::ErrorCode::kIncompleteFanout) {
        return make_unexpected(from_holomq_error(std::move(transport_error)));
      }
      if (transport_error.code == holomq::ErrorCode::kInterrupted) {
        // Only a bounded-retry cancellation reports a send interrupted. It rejected full
        // recipients exactly as an expired budget does, so it keeps the pressure classification
        // and leaves its cause in the per-thread record.
        note_publication_cancelled();
        transport_error.code = holomq::ErrorCode::kBackpressured;
      }
      const Error translated = from_holomq_error(std::move(transport_error));
      if (graph_retry_boundary && expected_recipient_count == 1U &&
          translated.code == ErrorCode::kBackpressured) {
        return make_unexpected(translated);
      }
      return make_unexpected(
          Error{ErrorCode::kPublicationIndeterminate,
                translated.message.empty()
                    ? std::string{"HOLOMQ_PUBLICATION_POST_SUBMIT"}
                    : fmt::format("HOLOMQ_PUBLICATION_POST_SUBMIT: {}", translated.message)});
    }
    if (expected_recipient_count == 0U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "HOLOMQ_PUBLICATION_EXPECTED_RECIPIENTS_ZERO"});
    }
    if (result.value() < 0 ||
        static_cast<std::uint64_t>(result.value()) > expected_recipient_count) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "HOLOMQ_PUBLICATION_RECIPIENT_COUNT_INVALID"});
    }
    if (graph_retry_boundary && result.value() == 0) {
      return make_unexpected(
          Error{ErrorCode::kBackpressured, "HOLOMQ_PUBLICATION_ZERO_RECIPIENTS"});
    }
    if (static_cast<std::uint64_t>(result.value()) < expected_recipient_count) {
      return make_unexpected(make_publication_partial_error(
          static_cast<std::uint32_t>(result.value()), expected_recipient_count));
    }
    return expected_recipient_count;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
  }
}

expected<void, Error> validate_queue_options(const ChannelOptions& options) {
  if (options.subscriber_queue_depth == 0U) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument, "channel subscriber queue depth must be greater than zero"));
  }
  if (options.max_subscriber_queue_depth != 0U &&
      options.max_subscriber_queue_depth < options.subscriber_queue_depth) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "channel publisher maximum subscriber queue depth is smaller than "
                           "the requested subscriber queue depth"));
  }
  return {};
}

expected<void, Error> validate_channel_options(const ChannelOptions& options) {
  if (auto valid = validate_queue_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (options.expected_recipient_count == 0U) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument, "channel expected recipient count must be greater than zero"));
  }
  if (options.expected_recipient_count > 64U) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "channel expected recipient count must be less than or equal to 64"));
  }
  return {};
}

expected<void, Error> validate_tensor_channel_options(const ChannelOptions& options) {
  if (auto valid = validate_channel_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (options.expected_recipient_count > MemoryBuffer::kMaximumFanoutReleaseStreams) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor channel expected recipient count exceeds MemoryBuffer release-stream capacity"));
  }
  if (!options.tensor_pool_plan.has_value()) {
    return {};
  }
  if (options.tensor_transfer_mode != ChannelTensorTransferMode::kBufferHandle) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument, "Tensor pool plan requires the BufferHandle transfer mode"));
  }

  const InterfaceTensorPoolPlan& plan = options.tensor_pool_plan.value();
  if (plan.max_buffer_bytes == 0U) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument, "Tensor pool maximum buffer bytes must be greater than zero"));
  }
  if (plan.slot_count == 0U) {
    return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                              "Tensor pool slot count must be greater than zero"));
  }
  if (plan.fanout_capacity != options.expected_recipient_count) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor pool fanout capacity does not match the channel recipient count"));
  }
  const std::uint32_t effective_max_borrowed_samples =
      options.max_subscriber_borrowed_samples == 0U ? effective_max_subscriber_queue_depth(options)
                                                    : options.max_subscriber_borrowed_samples;
  if (plan.max_borrowed_samples_per_recipient != effective_max_borrowed_samples) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor pool borrowed-sample capacity does not match the channel service limit"));
  }

  switch (plan.storage_type) {
    case MemoryStorageType::kDevice:
      if (plan.cuda_device_ordinal < 0) {
        return make_unexpected(make_channel_error(
            ErrorCode::kInvalidArgument,
            "Device Tensor pool plan requires a non-negative CUDA device ordinal"));
      }
      break;
    case MemoryStorageType::kHost:
    case MemoryStorageType::kSystem:
      if (plan.cuda_device_ordinal != -1) {
        return make_unexpected(
            make_channel_error(ErrorCode::kInvalidArgument,
                               "Host Tensor pool plan must not specify a CUDA device ordinal"));
      }
      break;
    case MemoryStorageType::kCudaManaged:
    default:
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor pool plan storage type is not supported for interprocess export"));
  }
  return {};
}

expected<void, Error> validate_inline_payload_options(const ChannelOptions& options) {
  return validate_queue_options(options);
}

holomq::QueuePolicy to_holomq_queue_policy(const ChannelQueuePolicy policy) noexcept {
  switch (policy) {
    case ChannelQueuePolicy::kBlocking:
      return holomq::QueuePolicy::kBlocking;
    case ChannelQueuePolicy::kDropOldest:
      return holomq::QueuePolicy::kDropOldest;
    default:
      return holomq::QueuePolicy::kBlocking;
  }
}

bool replaces_oldest_unread_sample(const ChannelQueuePolicy policy) noexcept {
  return policy == ChannelQueuePolicy::kDropOldest;
}

std::uint32_t effective_max_subscriber_queue_depth(const ChannelOptions& options) noexcept {
  if (options.max_subscriber_queue_depth != 0U) {
    return options.max_subscriber_queue_depth;
  }
  return options.subscriber_queue_depth;
}

expected<std::size_t, Error> tensor_sidecar_record_capacity(
    const ChannelOptions& options, const std::uint32_t tensor_fields_per_sample) {
  const std::uint64_t queued_sample_capacity =
      static_cast<std::uint64_t>(effective_max_subscriber_queue_depth(options)) *
      static_cast<std::uint64_t>(options.expected_recipient_count);
  // Taking a descriptor frees its transport queue slot before receive_sample() finishes
  // materializing the Tensor and releasing the sidecar claim. Bound one such sample per recipient.
  const std::uint64_t materializing_sample_capacity = options.expected_recipient_count;
  // Every send reserves its descriptor-backed payload before HoloMQ applies queue pressure or
  // identifies a replaced claim. Keep one endpoint-local staging sample so a full queue reaches
  // its configured blocking/replacement policy instead of failing early in the sidecar store.
  constexpr std::uint64_t kPublicationStagingCapacity = 1U;
  const std::uint64_t sample_capacity =
      queued_sample_capacity + materializing_sample_capacity + kPublicationStagingCapacity;
  // A descriptor-backed schema sample holds one record per Tensor field until every recipient
  // resolves or releases it. An out-of-range sample bound is rejected below unscaled, so the
  // product of two 32-bit factors cannot overflow.
  const std::uint64_t capacity = sample_capacity <= std::numeric_limits<std::uint32_t>::max()
                                     ? sample_capacity * tensor_fields_per_sample
                                     : sample_capacity;
  if (capacity == 0U || capacity > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "channel Tensor sidecar record capacity exceeds the endpoint contract range"));
  }
  return static_cast<std::size_t>(capacity);
}

holomq::PubSubServiceOptions to_holomq_service_options(const ChannelOptions& options) noexcept {
  return holomq::PubSubServiceOptions{
      .queue_policy = to_holomq_queue_policy(options.queue_policy),
      .max_subscribers = options.expected_recipient_count,
      .max_subscriber_queue_depth = effective_max_subscriber_queue_depth(options),
      .max_subscriber_borrowed_samples = options.max_subscriber_borrowed_samples,
      .publisher_history_depth = options.publisher_history_depth,
      .enable_event_wake = options.enable_event_wake,
  };
}

holomq::PublisherPubSubOptions to_holomq_publisher_options(
    const ChannelOptions& options, const PublisherPressureOptions pressure) noexcept {
  holomq::PublisherPressurePolicy pressure_policy = holomq::PublisherPressurePolicy::kFollowService;
  switch (pressure.mode) {
    case PublisherPressureMode::kRejectWhenFull:
      pressure_policy = holomq::PublisherPressurePolicy::kRejectWhenFull;
      break;
    case PublisherPressureMode::kBoundedRetry:
      pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
      break;
    case PublisherPressureMode::kFollowService:
    default:
      break;
  }
  return holomq::PublisherPubSubOptions{
      .max_loaned_samples = options.max_loaned_samples,
      .expected_recipient_count = options.expected_recipient_count,
      .replacement_attribution_required = options.replacement_attribution_required,
      .pressure_policy = pressure_policy,
      .bounded_retry_budget_ns = pressure.bounded_retry_budget_ns,
      .bounded_retry_cancellation = pressure.bounded_retry_cancellation,
  };
}

holomq::SubscriberPubSubOptions to_holomq_subscriber_options(
    const ChannelOptions& options) noexcept {
  return holomq::SubscriberPubSubOptions{
      .queue_depth = options.subscriber_queue_depth,
      .max_borrowed_samples = options.max_borrowed_samples,
  };
}

bool supports_queue_policy(const TransportCapabilities& capabilities,
                           const ChannelQueuePolicy policy) noexcept {
  switch (policy) {
    case ChannelQueuePolicy::kBlocking:
      return capabilities.blocking_queue_policy;
    case ChannelQueuePolicy::kDropOldest:
      return capabilities.drop_oldest_queue_policy;
    default:
      return false;
  }
}

expected<void, Error> validate_capabilities(const TransportCapabilities& capabilities,
                                            const ChannelOptions& options) {
  if (!capabilities.typed_payloads || !capabilities.local_descriptor_payload_transfer) {
    return make_unexpected(
        make_channel_error(ErrorCode::kNotImplemented,
                           "channel transport does not support local descriptor payloads"));
  }
  if (!supports_queue_policy(capabilities, options.queue_policy)) {
    return make_unexpected(make_channel_error(
        ErrorCode::kNotImplemented, "channel transport does not support requested queue policy"));
  }
  return {};
}

expected<void, Error> validate_inline_payload_capabilities(
    const TransportCapabilities& capabilities, const ChannelOptions& options) {
  if (!capabilities.typed_payloads) {
    return make_unexpected(make_channel_error(ErrorCode::kNotImplemented,
                                              "channel transport does not support typed payloads"));
  }
  if (!supports_queue_policy(capabilities, options.queue_policy)) {
    return make_unexpected(make_channel_error(
        ErrorCode::kNotImplemented, "channel transport does not support requested queue policy"));
  }
  return {};
}

}  // namespace holoscan::channel_internal
