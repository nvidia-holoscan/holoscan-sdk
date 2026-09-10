/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_subscriber_wait_context.hpp"

#include <holomq/context.hpp>
#include <holomq/wait_strategy.hpp>

#include "channel_common.hpp"

#include <memory>
#include <utility>

namespace holoscan::channel_internal {

SubscriberWaitContext::SubscriberWaitContext(
    holomq::WaitSet wait_set, holomq::WaitTriggerAttachment trigger_attachment) noexcept
    : wait_set_(std::move(wait_set)),
      trigger_attachment_(std::move(trigger_attachment)),
      trigger_(trigger_attachment_.trigger()) {}

expected<std::unique_ptr<SubscriberWaitContext>, Error> SubscriberWaitContext::create(
    holomq::Context& context, WaitSetSubscriber& subscriber) {
  auto wait_set = context.create_waitset();
  if (!wait_set) {
    return make_unexpected(from_holomq_error(std::move(wait_set).error(), kWaitSetCreateFailed));
  }

  // Ordering is this function's rollback, so attaching the subscriber's lanes is deliberately the
  // last fallible step and the single commit point of the whole setup.
  //
  // Lane attachments do not live here: attach_wait_set() stores them inside the binding, and this
  // layer has no operation that can retire them. A failure after that call would therefore leave
  // them engaged with no context able to release them -- every later attach_wait_set() rejects the
  // repeated attachment with kInvalidArgument, the port stays flagged as owned by a wait set so it
  // can never be handed to the Runtime pump either, and wait_for_readiness caches this context only
  // on success, so the next wait comes straight back here and fails the same way for the rest of
  // that subscriber's life. Everything that can fail without touching the subscriber runs first
  // instead, which leaves nothing to roll back.
  auto trigger_attachment = wait_set->attach_trigger();
  if (!trigger_attachment) {
    return make_unexpected(
        from_holomq_error(std::move(trigger_attachment).error(), kWaitTriggerAttachFailed));
  }

  // Allocated before the commit so a throwing allocation cannot strand an attachment either. A
  // context abandoned below is simply discarded; it owns nothing the subscriber can see.
  auto created = std::unique_ptr<SubscriberWaitContext>(new SubscriberWaitContext(
      std::move(wait_set).value(), std::move(trigger_attachment).value()));

  // Commit. Attaches every lane this subscriber owns; under kDropOldest that is the data lane plus
  // the protected terminal lane that carries end-of-stream. attach_wait_set() is itself
  // self-cleaning -- a terminal-lane failure destroys the local data-lane attachment, which
  // detaches it -- so a failure here leaves the subscriber exactly as it was found and the caller
  // can build a wait context again later.
  //
  // Attaching the trigger first also makes it the wait set's first attachment, so the erase that
  // re-arming performs move-assigns the lane entries down rather than only trigger entries. That
  // path is the reason AttachmentEntry spells out its move-assignment.
  auto data_handle = subscriber.attach_wait_set(&created->wait_set_);
  if (!data_handle) {
    return make_unexpected(std::move(data_handle).error());
  }

  // One throwaway userspace pass before any caller blocks. It drives the readiness query that
  // establishes the inbound transport connection for a subscriber created before its publisher.
  //
  // It does not, however, warm the backend's ready-handle storage. create() is only reached after
  // a receive or a has_samples() probe has already reported nothing, so barring a sample that
  // lands in the gap, no lane is ready here: the pass appends no handle and leaves that vector at
  // capacity zero. It has no reserve either, so the first later wait a lane actually satisfies
  // grows it once. Do not restate this pass as making later waits allocation-free.
  const holomq::WaitStrategy warm{
      .mode = holomq::WaitMode::Return,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = std::nullopt,
  };
  static_cast<void>(created->wait_set_.wait(warm));

  return created;
}

expected<void, Error> SubscriberWaitContext::rearm_trigger() {
  if (!trigger_->is_set()) {
    return {};
  }
  // Attach the replacement before releasing the latched attachment: move-assignment runs the old
  // detach callback, so the stale kernel wake object leaves the wait set exactly once and the wait
  // set is never left with no cancellation lane.
  auto replacement = wait_set_.attach_trigger();
  if (!replacement) {
    return make_unexpected(
        from_holomq_error(std::move(replacement).error(), kWaitTriggerRearmFailed));
  }
  trigger_attachment_ = std::move(replacement).value();
  trigger_ = trigger_attachment_.trigger();
  return {};
}

expected<void, Error> SubscriberWaitContext::wait_for(
    const std::chrono::nanoseconds timeout) noexcept {
  const holomq::WaitStrategy strategy{
      .mode = holomq::WaitMode::KernelWait,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = timeout,
  };
  auto waited = wait_set_.wait(strategy);
  if (!waited) {
    return make_unexpected(from_holomq_error(std::move(waited).error()));
  }
  // Both ready and timeout return success. A kernel wake can be spurious and report no ready
  // handle at all, so the caller rechecks readiness and the deadline either way.
  return {};
}

}  // namespace holoscan::channel_internal
