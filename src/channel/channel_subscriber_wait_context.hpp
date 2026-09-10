/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/wait_set.hpp>
#include <holomq/wait_trigger.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace holomq {
class Context;
}  // namespace holomq

namespace holoscan::channel_internal {

/// @brief Name the fallible step that failed while building a subscriber's wait context.
///
/// The three steps are not separable by \ref ErrorCode: wait-set creation, trigger attachment and
/// lane attachment can each report kFailure, kInvalidArgument or kResourceExhausted, and lane
/// attachment additionally forwards any transport code. These tokens are the discriminator, in the
/// same style as the endpoint tokens in pod_codec.cpp. from_holomq_error prepends the token to
/// the underlying message, so a consumer matches on the prefix rather than the whole string; the
/// only exception is the fail-closed HOLOMQ_ERROR_* path, which deliberately does not match.
inline constexpr std::string_view kWaitSetCreateFailed{"CHANNEL_WAIT_SET_CREATE_FAILED"};
inline constexpr std::string_view kWaitTriggerAttachFailed{"CHANNEL_WAIT_TRIGGER_ATTACH_FAILED"};
inline constexpr std::string_view kWaitTriggerRearmFailed{"CHANNEL_WAIT_TRIGGER_REARM_FAILED"};

/// @brief Own one subscriber's private wait set, lane attachments, and cancellation trigger.
///
/// Constructed on the first wait and cached for the subscriber's lifetime. The wait set, the lane
/// attachments, and the cancellation trigger are all built here, so a warmed wait rebuilds none of
/// them. It is not unconditionally allocation-free: the backend's ready-handle vector is not
/// reserved, and it is normally still empty when this context is built, because a context is only
/// built once readiness has already been reported absent. The first warmed wait that a lane
/// actually satisfies therefore grows it. That growth is bounded by the number of lanes -- at most
/// two for this context -- and never recurs, because the vector is cleared rather than shrunk. A
/// warmed wait allocates nothing beyond that.
///
/// A subscriber under \ref ChannelQueuePolicy::kDropOldest owns two transport lanes, and
/// end-of-stream arrives only on the second. This context therefore never dispatches on wait
/// handles; it asks the subscriber's own \ref WaitSetSubscriber::has_samples, which already reports
/// the union of every lane it owns. The lane handles are deliberately not retained: nothing here
/// reads them, and storing them would have to encode "no terminal lane" as handle zero, which is
/// also a valid handle. The deferred multi-subscriber wait set can reintroduce them when it has a
/// use for them and can represent absence honestly.
class SubscriberWaitContext {
 public:
  SubscriberWaitContext(const SubscriberWaitContext&) = delete;
  SubscriberWaitContext& operator=(const SubscriberWaitContext&) = delete;
  SubscriberWaitContext(SubscriberWaitContext&&) = delete;
  SubscriberWaitContext& operator=(SubscriberWaitContext&&) = delete;
  ~SubscriberWaitContext() = default;

  /// @brief Build one subscriber's wait context and establish its inbound transport connection.
  ///
  /// Setup is transactional with respect to \p subscriber: attaching its lanes is the last fallible
  /// step, and that step cleans up after itself, so a failed call leaves the subscriber unattached
  /// and able to build a context on a later wait. It has to be, because nothing in this layer can
  /// retire a lane attachment once the binding holds it.
  ///
  /// @param context Borrowed transport runtime used to create the private wait set.
  /// @param subscriber Subscriber whose lanes are attached; must outlive this context.
  /// @return The context, or a checked setup error.
  /// @retval ErrorCode::kInvalidArgument This subscriber is already attached to another wait set.
  [[nodiscard]] static expected<std::unique_ptr<SubscriberWaitContext>, Error> create(
      holomq::Context& context, WaitSetSubscriber& subscriber);

  /// @brief Perform one bounded kernel wait.
  /// @param timeout Maximum duration to block; must be positive.
  /// @return Empty success when the backend reported a possible readiness event or the wait
  /// expired. Callers recheck readiness and the deadline either way.
  /// @note This function does not throw and performs no allocation.
  [[nodiscard]] expected<void, Error> wait_for(std::chrono::nanoseconds timeout) noexcept;

  /// @brief Replace a latched cancellation trigger so this context can wait again.
  ///
  /// A \ref holomq::WaitTrigger latch is monotonic and this context is cached for the
  /// subscriber's lifetime, so a trigger latched by an earlier cancellation would end every later
  /// wait immediately, even one given a fresh token or no token at all. Re-arming attaches a
  /// replacement trigger and detaches the latched one.
  ///
  /// @return Empty success, including when the trigger is not latched and nothing is replaced.
  /// @retval ErrorCode::kFailure The wait set refused to attach a replacement trigger.
  /// @note Allocates only after a cancellation; an un-latched trigger returns without work.
  [[nodiscard]] expected<void, Error> rearm_trigger();

  /// @brief Return the cancellation trigger fired to end a blocked wait.
  [[nodiscard]] holomq::WaitTrigger& trigger() const noexcept { return *trigger_; }

 private:
  SubscriberWaitContext(holomq::WaitSet wait_set,
                        holomq::WaitTriggerAttachment trigger_attachment) noexcept;

  holomq::WaitSet wait_set_;
  holomq::WaitTriggerAttachment trigger_attachment_;
  std::shared_ptr<holomq::WaitTrigger> trigger_;
};

}  // namespace holoscan::channel_internal
