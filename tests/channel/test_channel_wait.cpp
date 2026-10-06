/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_loan.hpp>

#include "../../src/channel/channel_inline_payload_test_access.hpp"
#include "../../src/channel/channel_subscriber_wait_context.hpp"
#include "../common/allocation_counter.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "channel_test_support.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <dirent.h>
#if HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
#include <poll.h>
#endif
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace holoscan::channel_test;  // NOLINT(build/namespaces)

namespace {

// Published scalar used by every wait test in this file. A bare struct is a valid inline payload
// but not a PublishedPayload, which is what the sample-returning API is constrained on.
using WaitPayload = std::uint64_t;

// Event wake is service-static and defaults to false, so every waiting endpoint must opt in.
[[nodiscard]] holoscan::ChannelOptions event_wake_options() {
  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  return options;
}

[[nodiscard]] std::chrono::steady_clock::time_point in(const std::chrono::milliseconds delay) {
  return std::chrono::steady_clock::now() + delay;
}

// Cap how many new file descriptors this process may open, so a setup step that needs one can be
// made to fail on demand without disturbing the descriptors already in use. The cap is expressed
// relative to the highest descriptor open at construction, because RLIMIT_NOFILE bounds descriptor
// numbers rather than counts. Linux-only, which the event-wake transport already is.
class ScopedDescriptorBudget {
 public:
  ScopedDescriptorBudget() noexcept {
    if (::getrlimit(RLIMIT_NOFILE, &original_) != 0) {
      return;
    }
    DIR* const directory = ::opendir("/proc/self/fd");
    if (directory == nullptr) {
      return;
    }
    for (const dirent* entry = ::readdir(directory); entry != nullptr;
         entry = ::readdir(directory)) {
      highest_ = std::max(highest_, std::atoi(entry->d_name));
    }
    static_cast<void>(::closedir(directory));
    usable_ = true;
  }

  ScopedDescriptorBudget(const ScopedDescriptorBudget&) = delete;
  ScopedDescriptorBudget& operator=(const ScopedDescriptorBudget&) = delete;
  ScopedDescriptorBudget(ScopedDescriptorBudget&&) = delete;
  ScopedDescriptorBudget& operator=(ScopedDescriptorBudget&&) = delete;

  ~ScopedDescriptorBudget() { restore(); }

  /// @brief Report whether the process limit could be read, so the caller can skip rather than
  /// assert on a platform that does not expose it.
  [[nodiscard]] bool usable() const noexcept { return usable_; }

  /// @brief Allow at most \p spare descriptor numbers beyond those already open.
  [[nodiscard]] bool constrain(const int spare) const noexcept {
    if (!usable_) {
      return false;
    }
    struct rlimit tight = original_;
    tight.rlim_cur = static_cast<rlim_t>(highest_) + 1U + static_cast<rlim_t>(spare);
    return ::setrlimit(RLIMIT_NOFILE, &tight) == 0;
  }

  void restore() const noexcept {
    if (usable_) {
      static_cast<void>(::setrlimit(RLIMIT_NOFILE, &original_));
    }
  }

 private:
  struct rlimit original_{};
  int highest_{0};
  bool usable_{false};
};

}  // namespace

// A wake covers every sample queued at the time it fires, so a second queued sample must still be
// returned by a second call that has no notification left to consume.
//
// Scope: both samples are queued before either call, so each is served by the receive-first fast
// path and neither call reaches wait_for_readiness. This therefore proves ordered delivery of a
// batch and that the second call does not block -- it is NOT a witness for the pre-block readiness
// recheck, which cannot be reached from here because receive() short-circuits first.
// CatalogChannelTest.CatalogWaitRetriesAfterHealthOnlyOperation owns that path: it is the case
// where receive() reports kNotReady while the endpoint is genuinely ready.
// Topology (blocking, event wake): publisher --[WaitPayload x2]--> subscriber
// Key assertion: two consecutive calls return the two queued samples in order, promptly.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitAndReceiveReturnsQueuedSamplesWithoutBlocking) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  ASSERT_TRUE(publisher->emit(WaitPayload{1}));
  ASSERT_TRUE(publisher->emit(WaitPayload{2}));

  auto first = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());
  EXPECT_EQ(first->data, 1U);

  const auto started = std::chrono::steady_clock::now();
  auto second = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(second) << holoscan::runtime_error_message(second.error());
  EXPECT_EQ(second->data, 2U);

  // The second sample is already queued, so this call must be served without blocking at all. A
  // regression that stranded it would surface here as a full-deadline stall, not just a wrong
  // value.
  EXPECT_LT(elapsed, std::chrono::seconds(1));
}

// Timeout, cancellation, and end of stream must be distinguishable, and an already expired
// deadline must not become an unbounded wait (the negative-duration hazard).
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber
// Key assertion: three distinct codes, and the expired deadline returns promptly.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitAndReceiveDistinguishesTimeoutCancellationAndClose) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // An already expired deadline must clamp rather than wrap into a ~584-year wait.
  const auto started = std::chrono::steady_clock::now();
  auto expired = subscriber->wait_and_receive_sample(started - std::chrono::seconds(1));
  ASSERT_FALSE(expired);
  EXPECT_EQ(expired.error().code, holoscan::ErrorCode::kTimeout);
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(1));

  // A stop requested before the call is answered without entering a kernel wait.
  const std::stop_source source;
  source.request_stop();
  const auto cancel_started = std::chrono::steady_clock::now();
  auto cancelled =
      subscriber->wait_and_receive_sample(in(std::chrono::seconds(30)), source.get_token());
  const auto cancel_elapsed = std::chrono::steady_clock::now() - cancel_started;
  ASSERT_FALSE(cancelled);
  EXPECT_EQ(cancelled.error().code, holoscan::ErrorCode::kInterrupted);
  // Without this bound the same code would be returned after running the full 30 s deadline, so
  // the value assertion alone cannot tell "answered before blocking" from "blocked, then woken".
  EXPECT_LT(cancel_elapsed, std::chrono::seconds(1));

  // End of stream is a terminal outcome, distinct from both of the above.
  ASSERT_TRUE(publisher->close());
  auto closed = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
}

// Regression: the cancellation trigger latch is monotonic and the wait context is cached for the
// subscriber's lifetime, so without a re-arm on the next wait a single cancellation would make
// every later wait on that subscriber report kInterrupted forever, even with no token at all.
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber
// Key assertion: a wait after a cancelled wait still delivers a sample.
// REQ: CHAN-142
TEST(ChannelWaitTest, CancellationDoesNotDisableLaterWaits) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // Cancel a real, blocked wait. This latches the subscriber's trigger.
  const std::stop_source source;
  std::jthread canceller([&source] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    source.request_stop();
  });
  auto cancelled =
      subscriber->wait_and_receive_sample(in(std::chrono::seconds(30)), source.get_token());
  ASSERT_FALSE(cancelled);
  EXPECT_EQ(cancelled.error().code, holoscan::ErrorCode::kInterrupted);
  canceller.join();

  // The decisive assertion: with nothing queued, the next wait must genuinely block and report a
  // timeout. A latched trigger would end it immediately with a stale kInterrupted instead. This
  // must precede the emit below, because a queued sample would be served by the fast path and
  // never reach the wait at all.
  const auto started = std::chrono::steady_clock::now();
  auto timed_out = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(timed_out);
  EXPECT_EQ(timed_out.error().code, holoscan::ErrorCode::kTimeout);
  // A timeout cannot be reported early, so this lower bound is safe on a slow machine; it proves
  // the call actually waited rather than returning on the stale latch.
  EXPECT_GE(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(100));

  // The subscriber must still be usable: a fresh wait with no token has to deliver the sample
  // rather than report the stale cancellation.
  ASSERT_TRUE(publisher->emit(WaitPayload{7}));
  auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 7U);
}

// CHAN-066 forbids a silent busy-poll fallback, so an endpoint created without event wake must be
// refused rather than entering a wait only its deadline can end. The refusal must not lose data:
// the fast path still returns a sample that is already queued.
// Topology (blocking, event wake DISABLED on both ends): publisher --[WaitPayload]--> subscriber
// Key assertion: kNotSupported when a wait is required, and no sample is lost.
// REQ: CHAN-066
TEST(ChannelWaitTest, WaitAndReceiveRejectsPollOnlySubscriberWithNotSupported) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  // Both endpoints must opt out; a mismatch is rejected at endpoint creation instead.
  holoscan::ChannelOptions options{};
  options.enable_event_wake = false;

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // Empty endpoint: a wait is genuinely required, so the capability error surfaces.
  auto refused = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, holoscan::ErrorCode::kNotSupported);

  // The refusal must not have consumed or hidden anything: a plain receive still returns the
  // next sample, which is what proves there was no silent poll fallback behind the error.
  ASSERT_TRUE(publisher->emit(WaitPayload{3}));
  ASSERT_TRUE(wait_until([&subscriber] { return subscriber->has_samples().value_or(false); },
                         std::chrono::seconds(2)));
  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 3U);

  // And the queued sample is served by the fast path too, without ever reaching the refusal.
  ASSERT_TRUE(publisher->emit(WaitPayload{4}));
  auto fast_path = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(fast_path) << holoscan::runtime_error_message(fast_path.error());
  EXPECT_EQ(fast_path->data, 4U);
}

// wait_for_sample reports readiness without consuming, so the sample must still be there.
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber
// Key assertion: the wait succeeds and a following receive still returns the sample.
TEST(ChannelWaitTest, WaitForSampleReportsReadinessWithoutConsuming) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  ASSERT_TRUE(publisher->emit(WaitPayload{5}));

  auto ready = subscriber->wait_for_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(ready) << holoscan::runtime_error_message(ready.error());

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 5U);

  // Nothing left: an empty endpoint times out rather than reporting readiness.
  auto timed_out = subscriber->wait_for_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(timed_out);
  EXPECT_EQ(timed_out.error().code, holoscan::ErrorCode::kTimeout);
}

#if HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS

// The public descriptor stays level-readable while work is queued, and acknowledging it never
// consumes the sample. While the handle is live it owns the same slot as blocking waits;
// explicit detach and destruction each make the subscriber waitable again.
// Topology (event wake): publisher --[WaitPayload]--> readiness handle --> poll(2)
// Key assertion: queued and later samples wake poll, waits are exclusive, and both release paths
// detach.
// REQ: CHAN-148
// REQ: CHAN-149
TEST(ChannelReadinessTest, DescriptorReportsReceiveRelevantWorkAndClaimsTheWaitOwner) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));

  // Queue before creating the handle to cover readiness already present at attachment time.
  ASSERT_TRUE(publisher->emit(WaitPayload{41}));
  ASSERT_TRUE(wait_until([&subscriber] { return subscriber->has_samples().value_or(false); },
                         std::chrono::seconds(2)));
  {
    auto readiness = subscriber->readiness();
    ASSERT_TRUE(readiness) << holoscan::runtime_error_message(readiness.error());
    ASSERT_GE(readiness->descriptor(), 0);
    pollfd descriptor{.fd = readiness->descriptor(), .events = POLLIN, .revents = 0};

    ASSERT_EQ(::poll(&descriptor, 1, 2000), 1);
    EXPECT_NE(descriptor.revents & POLLIN, 0);
    descriptor.revents = 0;
    EXPECT_EQ(::poll(&descriptor, 1, 0), 1) << "readiness must remain until acknowledgement";

    ASSERT_TRUE(readiness->acknowledge());
    descriptor.revents = 0;
    EXPECT_EQ(::poll(&descriptor, 1, 0), 1) << "a queued sample keeps readiness level-readable";
    auto first = subscriber->receive_sample();
    ASSERT_TRUE(first) << holoscan::runtime_error_message(first.error());
    EXPECT_EQ(first->data, 41U);
    ASSERT_TRUE(readiness->acknowledge());
    descriptor.revents = 0;
    EXPECT_EQ(::poll(&descriptor, 1, 0), 0);

    auto repeated = subscriber->readiness();
    ASSERT_FALSE(repeated);
    EXPECT_EQ(repeated.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_NE(repeated.error().message.find("readiness handle"), std::string::npos);

    // Wait ownership is checked before the wait API's ordinary queued-sample fast path.
    ASSERT_TRUE(publisher->emit(WaitPayload{42}));
    ASSERT_TRUE(wait_until([&subscriber] { return subscriber->has_samples().value_or(false); },
                           std::chrono::seconds(2)));
    auto wait = subscriber->wait_for_sample(in(std::chrono::milliseconds(20)));
    ASSERT_FALSE(wait);
    EXPECT_EQ(wait.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_NE(wait.error().message.find("readiness handle"), std::string::npos);
    auto wait_and_receive = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(20)));
    ASSERT_FALSE(wait_and_receive);
    EXPECT_EQ(wait_and_receive.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_NE(wait_and_receive.error().message.find("readiness handle"), std::string::npos);

    // The queued publication also produces a fresh readable edge.
    descriptor.revents = 0;
    ASSERT_EQ(::poll(&descriptor, 1, 2000), 1);
    ASSERT_TRUE(readiness->acknowledge());
    auto second = subscriber->receive_sample();
    ASSERT_TRUE(second) << holoscan::runtime_error_message(second.error());
    EXPECT_EQ(second->data, 42U);

    readiness->detach();
    EXPECT_EQ(readiness->descriptor(), -1);
    auto acknowledged_after_detach = readiness->acknowledge();
    ASSERT_FALSE(acknowledged_after_detach);
    EXPECT_EQ(acknowledged_after_detach.error().code, holoscan::ErrorCode::kClosed);

    auto replacement = subscriber->readiness();
    ASSERT_TRUE(replacement) << holoscan::runtime_error_message(replacement.error());
  }

  ASSERT_TRUE(publisher->emit(WaitPayload{43}));
  auto after_destruction = subscriber->wait_and_receive_sample(in(std::chrono::seconds(2)));
  ASSERT_TRUE(after_destruction) << holoscan::runtime_error_message(after_destruction.error());
  EXPECT_EQ(after_destruction->data, 43U);
}

// A lossy inline receive may stage EOS privately while returning the preceding data sample. Once
// the old descriptor has drained its transport lanes, a replacement handle must still prime from
// that wrapper-owned terminal state or an event loop can sleep forever with kClosed already ready.
// Topology (drop-oldest): data lane + protected EOS lane --> staged EOS --> replacement poll(2)
// Key assertion: detach/reacquire preserves immediate readiness for the already staged terminal.
// REQ: CHAN-148
// REQ: CHAN-150
TEST(ChannelReadinessTest, ReacquiredInlineDescriptorReportsStagedEndOfStream) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options = event_wake_options();
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 1U;

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));

  auto first_readiness = subscriber->readiness();
  ASSERT_TRUE(first_readiness) << holoscan::runtime_error_message(first_readiness.error());
  pollfd descriptor{.fd = first_readiness->descriptor(), .events = POLLIN, .revents = 0};

  ASSERT_TRUE(publisher->emit(WaitPayload{71}));
  ASSERT_TRUE(publisher->close());
  ASSERT_EQ(::poll(&descriptor, 1, 2000), 1);
  ASSERT_TRUE(first_readiness->acknowledge());

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 71U);

  // The receive above stages EOS in the inline wrapper while returning data. Clear the transport
  // notifications and prove the first descriptor has no residual wake before replacing it.
  ASSERT_TRUE(first_readiness->acknowledge());
  descriptor.revents = 0;
  ASSERT_EQ(::poll(&descriptor, 1, 0), 0);
  first_readiness->detach();

  auto replacement = subscriber->readiness();
  ASSERT_TRUE(replacement) << holoscan::runtime_error_message(replacement.error());
  descriptor = pollfd{.fd = replacement->descriptor(), .events = POLLIN, .revents = 0};
  ASSERT_EQ(::poll(&descriptor, 1, 0), 1);
  EXPECT_NE(descriptor.revents & POLLIN, 0);
  ASSERT_TRUE(replacement->acknowledge());

  auto closed = subscriber->receive_sample();
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
}

// A poll-only name-based subscriber has no kernel listener to lend to an external event loop.
// Key assertion: fail closed with the same capability token as a blocking wait.
// REQ: CHAN-149
TEST(ChannelReadinessTest, NameBasedSubscriberWithoutEventWakeIsNotSupported) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(make_unique_channel_name());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  auto readiness = subscriber->readiness();
  ASSERT_FALSE(readiness);
  EXPECT_EQ(readiness.error().code, holoscan::ErrorCode::kNotSupported);

  auto active = std::move(*subscriber);
  readiness = subscriber->readiness();
  ASSERT_FALSE(readiness);
  EXPECT_EQ(readiness.error().code, holoscan::ErrorCode::kClosed);

  readiness = active.readiness();
  ASSERT_FALSE(readiness);
  EXPECT_EQ(readiness.error().code, holoscan::ErrorCode::kNotSupported);

  ASSERT_TRUE(active.close());
  readiness = active.readiness();
  ASSERT_FALSE(readiness);
  EXPECT_EQ(readiness.error().code, holoscan::ErrorCode::kClosed);
}

// The wait-owner claim begins before an endpoint checks or receives readiness and lasts until the
// complete public wait returns. A readiness handle therefore cannot steal lanes from a blocked
// wait; cancellation releases the claim and permits a later handle.
// Key assertion: readiness() reports kInvalidArgument during the in-flight wait only.
// REQ: CHAN-149
TEST(ChannelReadinessTest, InFlightWaitRejectsReadinessHandleCreation) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();
  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));

  std::stop_source source;
  std::atomic<holoscan::ErrorCode> wait_result{holoscan::ErrorCode::kSuccess};
  std::jthread waiter([&] {
    auto waited = subscriber->wait_for_sample(in(std::chrono::seconds(30)), source.get_token());
    wait_result.store(waited ? holoscan::ErrorCode::kSuccess : waited.error().code,
                      std::memory_order_release);
  });
  const bool wait_owner_claimed = wait_until(
      [&subscriber] {
        return holoscan::channel_internal::InlinePayloadTestAccess::wait_owner_claimed(*subscriber);
      },
      std::chrono::seconds(2));
  if (!wait_owner_claimed) {
    source.request_stop();
    waiter.join();
  }
  ASSERT_TRUE(wait_owner_claimed);

  auto refused = subscriber->readiness();
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(refused.error().message.find("in flight"), std::string::npos);

  source.request_stop();
  waiter.join();
  EXPECT_EQ(wait_result.load(std::memory_order_acquire), holoscan::ErrorCode::kInterrupted);
  auto after_wait = subscriber->readiness();
  ASSERT_TRUE(after_wait) << holoscan::runtime_error_message(after_wait.error());
}

// Transport peer presence is advisory and never a receive event. Attaching and then silently
// destroying a producer must leave the descriptor quiet and the empty subscriber nonterminal.
// Topology: subscriber/readiness --> producer connect --> producer destruction without close
// Key assertion: connect, disconnect, and silent loss create neither readiness nor kClosed.
// REQ: CHAN-150
TEST(ChannelReadinessTest, PeerChangesAndSilentProducerLossDoNotImplyReadiness) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  auto readiness = subscriber->readiness();
  ASSERT_TRUE(readiness) << holoscan::runtime_error_message(readiness.error());
  pollfd descriptor{.fd = readiness->descriptor(), .events = POLLIN, .revents = 0};
  ASSERT_EQ(::poll(&descriptor, 1, 0), 0);

  auto created_publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(created_publisher) << holoscan::runtime_error_message(created_publisher.error());
  std::optional<holoscan::ChannelPublisher<WaitPayload>> publisher{std::move(*created_publisher)};
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));
  ASSERT_TRUE(
      wait_until([&subscriber] { return subscriber->has_publishers(); }, std::chrono::seconds(2)));
  descriptor.revents = 0;
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);

  publisher.reset();
  ASSERT_TRUE(
      wait_until([&subscriber] { return !subscriber->has_publishers(); }, std::chrono::seconds(2)));
  descriptor.revents = 0;
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
  auto received = subscriber->receive_sample();
  ASSERT_FALSE(received);
  EXPECT_EQ(received.error().code, holoscan::ErrorCode::kNotReady);
}

#endif  // HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS

// wait_for_sample must answer CHAN-066 the same way wait_and_receive_sample does, with one
// documented exception: an endpoint that already holds a sample is answered from has_samples()
// before the event lane is consulted, so it reports readiness rather than the capability error.
// Without that carve-out a poll-only endpoint holding a sample would answer inconsistently --
// a sample from wait_and_receive_sample, but kNotSupported from wait_for_sample.
// Topology (blocking, event wake DISABLED on both ends): publisher --[WaitPayload]--> subscriber
// Key assertion: kNotSupported only when a wait is genuinely required; readiness when it is not.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitForSampleRefusesPollOnlySubscriberOnlyWhenAWaitIsRequired) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  // Both endpoints must opt out; a mismatch is rejected at endpoint creation instead.
  holoscan::ChannelOptions options{};
  options.enable_event_wake = false;

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // Empty endpoint: readiness could only come from the event lane this endpoint does not have, so
  // the capability error surfaces instead of a wait that only the deadline could end.
  auto refused = subscriber->wait_for_sample(in(std::chrono::seconds(5)));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, holoscan::ErrorCode::kNotSupported);

  // Queued sample: answered before the event lane is reached, so the same call now succeeds.
  ASSERT_TRUE(publisher->emit(WaitPayload{11}));
  ASSERT_TRUE(wait_until([&subscriber] { return subscriber->has_samples().value_or(false); },
                         std::chrono::seconds(2)));
  auto ready = subscriber->wait_for_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(ready) << holoscan::runtime_error_message(ready.error());

  // And it still reported readiness without consuming, exactly as on an event-wake endpoint.
  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 11U);
}

// wait_for_sample takes a stop token like wait_and_receive_sample, and cancellation must be scoped
// to the one call: the trigger latch is monotonic and the wait context is cached for the
// subscriber's lifetime, so a missing re-arm would make every later readiness query on this
// subscriber report kInterrupted forever.
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber
// Key assertion: kInterrupted for a stop requested before the call and for one landing during it,
// and the following readiness query still reaches its own timeout.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitForSampleReportsInterruptedAndStaysUsable) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // A stop already requested before the call must be seen before the first block, not after the
  // per-pass clamp, so this is bounded well below both the clamp and the deadline.
  const std::stop_source presignalled;
  presignalled.request_stop();
  const auto presignalled_started = std::chrono::steady_clock::now();
  auto pre = subscriber->wait_for_sample(in(std::chrono::seconds(30)), presignalled.get_token());
  ASSERT_FALSE(pre);
  EXPECT_EQ(pre.error().code, holoscan::ErrorCode::kInterrupted);
  EXPECT_LT(std::chrono::steady_clock::now() - presignalled_started, std::chrono::seconds(1));

  // A stop landing while the call is already blocked must end it too.
  std::stop_source source;
  std::jthread canceller([&source] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    source.request_stop();
  });
  auto cancelled = subscriber->wait_for_sample(in(std::chrono::seconds(30)), source.get_token());
  ASSERT_FALSE(cancelled);
  EXPECT_EQ(cancelled.error().code, holoscan::ErrorCode::kInterrupted);
  canceller.join();

  // Scoped to that call: with nothing queued the next readiness query must reach its own timeout
  // rather than returning a stale kInterrupted off the latched trigger. The lower bound is what
  // separates a genuine block from an immediate stale answer.
  const auto started = std::chrono::steady_clock::now();
  auto stale = subscriber->wait_for_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, holoscan::ErrorCode::kTimeout);
  EXPECT_GE(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(100));
}

// End of stream must stay terminal. A manually created inline endpoint surfaces kClosed from the
// transport only once and then reads as an empty, never-ready endpoint, so without a latch the
// second call would block for its whole deadline and report kTimeout instead.
// Topology (blocking, event wake): publisher --[close]--> subscriber
// Key assertion: repeated calls after end of stream keep reporting kClosed, promptly.
TEST(ChannelWaitTest, EndOfStreamStaysClosedAcrossRepeatedWaits) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  ASSERT_TRUE(publisher->close());

  auto first = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_FALSE(first);
  EXPECT_EQ(first.error().code, holoscan::ErrorCode::kClosed);

  const auto started = std::chrono::steady_clock::now();
  auto second = subscriber->wait_and_receive_sample(in(std::chrono::seconds(30)));
  ASSERT_FALSE(second);
  EXPECT_EQ(second.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(5));

  // The non-consuming wait reports the same terminal state rather than timing out.
  auto ready = subscriber->wait_for_sample(in(std::chrono::seconds(30)));
  ASSERT_FALSE(ready);
  EXPECT_EQ(ready.error().code, holoscan::ErrorCode::kClosed);
}

// End of stream is one-shot at the transport, so whichever call observes it has to latch it for
// the facade. Consuming it through the non-waiting receive_sample() must leave a later wait
// terminal: without the latch the endpoint reads as empty and never ready, so the wait would run
// out its whole deadline and report kTimeout for a stream that is already finished.
// Topology (blocking, event wake): publisher --[close]--> subscriber
// Key assertion: after receive_sample() observes kClosed, wait_for_sample reports kClosed promptly.
// REQ: CHAN-142
TEST(ChannelWaitTest, EndOfStreamConsumedByReceiveSampleStaysTerminalForLaterWaits) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  ASSERT_TRUE(publisher->close());

  // Consume the one-shot end of stream through the non-waiting path, which is the only way to
  // reach the state under test: wait_and_receive_sample would latch it on the way past.
  auto observed = holoscan::ErrorCode::kNotReady;
  ASSERT_TRUE(wait_until(
      [&subscriber, &observed] {
        auto received = subscriber->receive_sample();
        if (received) {
          return false;
        }
        observed = received.error().code;
        return observed == holoscan::ErrorCode::kClosed;
      },
      std::chrono::seconds(5)));
  ASSERT_EQ(observed, holoscan::ErrorCode::kClosed);

  const auto started = std::chrono::steady_clock::now();
  auto ready = subscriber->wait_for_sample(in(std::chrono::seconds(5)));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_FALSE(ready);
  EXPECT_EQ(ready.error().code, holoscan::ErrorCode::kClosed);

  // The bound is the part that fails without the latch: an unlatched facade has no queued sample
  // and no readiness, so it would spend the full five seconds before reporting kTimeout.
  EXPECT_LT(elapsed, std::chrono::milliseconds(500));
}

// A moved-from facade owns no endpoint, so every wait reports the same terminal code as receive.
// Key assertion: kClosed from both waiting methods, with no dereference of a null endpoint.
TEST(ChannelWaitTest, WaitOnMovedFromSubscriberReportsClosed) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const holoscan::ChannelOptions options = event_wake_options();

  auto subscriber = runtime->create_subscriber<WaitPayload>(make_unique_channel_name(), options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  const holoscan::ChannelSubscriber<WaitPayload> moved{std::move(subscriber.value())};

  auto received = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(received);
  EXPECT_EQ(received.error().code, holoscan::ErrorCode::kClosed);

  auto ready = subscriber->wait_for_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(ready);
  EXPECT_EQ(ready.error().code, holoscan::ErrorCode::kClosed);
}

// A wait that is already blocked must be woken by a stop requested from another thread, and must
// report cancellation rather than running to its deadline.
// Key assertion: kInterrupted well inside the 30 s deadline, under either interleaving.
TEST(ChannelWaitTest, CancellationDuringBlockedWaitReturnsInterrupted) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const std::stop_source source;
  std::jthread canceller([&source] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    source.request_stop();
  });

  const auto started = std::chrono::steady_clock::now();
  auto cancelled =
      subscriber->wait_and_receive_sample(in(std::chrono::seconds(30)), source.get_token());
  canceller.join();

  ASSERT_FALSE(cancelled);
  EXPECT_EQ(cancelled.error().code, holoscan::ErrorCode::kInterrupted);
  // Asserts only that the deadline was not reached; never a latency floor, which would be flaky.
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(30));
}

// The transport emits no connect or disconnect notification, so a producer that simply disappears
// is indistinguishable from an idle one. The honest outcome is a timeout: kStaleEndpoint is
// reserved for a producer incarnation change observed on a real received operation, and kClosed
// for an end of stream that was actually published. A destroyed publisher publishes neither.
// Topology (blocking, event wake): publisher destroyed before the subscriber waits
// Key assertion: kTimeout, and specifically neither kStaleEndpoint nor kClosed.
// REQ: CHAN-142
TEST(ChannelWaitTest, SilentProducerLossTimesOutRatherThanReportingStale) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  {
    auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
    ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

    // Prove the link was genuinely live before it disappears, so the timeout below cannot be a
    // never-connected endpoint reporting the same code for an unrelated reason.
    ASSERT_TRUE(publisher->emit(WaitPayload{42}));
    auto delivered = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
    ASSERT_TRUE(delivered) << holoscan::runtime_error_message(delivered.error());
    EXPECT_EQ(delivered->data, 42U);

    // Leaves scope without close(), so no end-of-stream record is ever published.
  }

  auto result = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, holoscan::ErrorCode::kTimeout);
  // The two codes a disconnect must never be reported as: kStaleEndpoint means a replacement
  // producer was observed, and kClosed means an end-of-stream record was actually published.
  EXPECT_NE(result.error().code, holoscan::ErrorCode::kStaleEndpoint);
  EXPECT_NE(result.error().code, holoscan::ErrorCode::kClosed);
}

// The warmed path must not allocate: the wait context, its lane attachments, and its trigger are
// built on the first wait and cached for the subscriber's lifetime. A regression that rebuilds any
// of them per call shows up here.
//
// Both the receive-first fast path and the blocking path are measured, because a sample published
// before the call is served by the first receive() and never reaches wait_for_readiness at all --
// which is why the fast-path regions below prove nothing about the wait. The blocking regions are
// the ones that exercise the kernel wait, the readiness recheck, and, for the live-token case, the
// std::stop_callback construction.
//
// One caveat is pinned rather than asserted away: the wait backend's ready-handle vector is never
// reserved, and it is still empty when the wait context is built, because that only happens when
// nothing is queued. The first blocking wake a lane actually satisfies therefore grows it once.
// That growth cannot recur -- the vector is cleared, not shrunk -- so the steady-state blocking
// regions still assert zero.
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber, published from a
// second thread for the blocking regions so this one is committed to the wait first
// Key assertion: zero allocations across a warmed wait_and_receive_sample on both paths, and the
// blocking regions really did block.
// REQ: CHAN-142
TEST(ChannelWaitTest, WarmWaitAndReceivePerformsNoAllocations) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  // Warm the receive path.
  ASSERT_TRUE(publisher->emit(WaitPayload{1}));
  auto warmup_received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(warmup_received) << holoscan::runtime_error_message(warmup_received.error());

  // Warm the wait itself. A receive that succeeds immediately never builds the wait context, so
  // this must be a genuine wait on an empty endpoint that runs out its deadline.
  auto warmup_wait = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(warmup_wait);
  ASSERT_EQ(warmup_wait.error().code, holoscan::ErrorCode::kTimeout);

  ASSERT_TRUE(publisher->emit(WaitPayload{2}));

  // No GTest macros inside the counted region: they allocate. Fold the outcome into a bool, and
  // use the single-argument overload so no std::stop_callback is registered.
  //
  // The sample is already queued, so this measures the receive-first fast path only: it returns
  // from the first receive() without entering wait_for_readiness.
  bool warmed_path_succeeded = false;
  std::size_t warmed_allocations = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
    warmed_path_succeeded = received.has_value() && received->data == 2U;
    warmed_allocations = allocations.count();
  }
  EXPECT_TRUE(warmed_path_succeeded);
  EXPECT_EQ(warmed_allocations, 0U);

  // Run the fast path again with a live token. The single-argument overload forwards a
  // default-constructed stop_token, whose stop_possible() is false, so it registers no callback at
  // all, whereas a real token makes borrowed_stop_token hand a live pointer down. This region still
  // returns from the first receive(), so it does not reach the std::stop_callback construction --
  // the live-token blocking region at the end of this test is what covers that. What it does prove
  // is that copying a live token into the facade costs nothing. The stop_source is built outside
  // the counted region because its shared state genuinely does allocate.
  const std::stop_source source;
  const std::stop_token token = source.get_token();
  ASSERT_TRUE(publisher->emit(WaitPayload{3}));

  bool token_path_succeeded = false;
  std::size_t token_allocations = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)), token);
    token_path_succeeded = received.has_value() && received->data == 3U;
    token_allocations = allocations.count();
  }
  EXPECT_TRUE(token_path_succeeded);
  EXPECT_EQ(token_allocations, 0U);

  // Everything above stayed on the receive-first path. Measure the blocking path by publishing
  // from a second thread only after this one is committed to the wait. The allocation counter is
  // thread-local, so the producer's own allocations are not counted.
  struct BlockedWaitMeasurement {
    bool succeeded{false};
    std::size_t allocations{0U};
    std::chrono::nanoseconds elapsed{};
  };
  constexpr auto kProducerLead = std::chrono::milliseconds(150);
  const auto measure_blocked_wait = [&](const WaitPayload expected,
                                        const std::stop_token* const live_token) {
    std::atomic<bool> release{false};
    std::thread producer([&] {
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      std::this_thread::sleep_for(kProducerLead);
      static_cast<void>(publisher->emit(expected));
    });

    BlockedWaitMeasurement measurement{};
    release.store(true, std::memory_order_release);
    {
      const holoscan::test::ScopedAllocationCounter allocations;
      const auto started = std::chrono::steady_clock::now();
      auto received =
          live_token == nullptr
              ? subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)))
              : subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)), *live_token);
      measurement.elapsed = std::chrono::steady_clock::now() - started;
      measurement.succeeded = received.has_value() && received->data == expected;
      measurement.allocations = allocations.count();
    }
    producer.join();
    return measurement;
  };
  // Well under kProducerLead, so this only ever fails when the call did not block at all.
  constexpr auto kBlockedFloor = std::chrono::milliseconds(50);

  // First blocking wake. This is the one that grows the unreserved ready-handle vector from empty
  // -- measured at exactly one allocation -- so it is bounded rather than required to be zero. The
  // bound is what fails if that one-time growth ever becomes per-wait, and it still holds if the
  // vector is given a reserve and the count drops to zero.
  const BlockedWaitMeasurement first_blocked = measure_blocked_wait(WaitPayload{4}, nullptr);
  EXPECT_TRUE(first_blocked.succeeded);
  EXPECT_GE(first_blocked.elapsed, kBlockedFloor) << "the measured call did not reach the wait";
  EXPECT_LE(first_blocked.allocations, 1U);

  // Steady state. Nothing on the blocking path may allocate from here on.
  const BlockedWaitMeasurement steady_blocked = measure_blocked_wait(WaitPayload{5}, nullptr);
  EXPECT_TRUE(steady_blocked.succeeded);
  EXPECT_GE(steady_blocked.elapsed, kBlockedFloor) << "the measured call did not reach the wait";
  EXPECT_EQ(steady_blocked.allocations, 0U);

  // The live-token blocking path is the only one that reaches arm.emplace(). TriggerFirer is a
  // non-capturing struct precisely so that the std::stop_callback stays allocation-free.
  const BlockedWaitMeasurement token_blocked = measure_blocked_wait(WaitPayload{6}, &token);
  EXPECT_TRUE(token_blocked.succeeded);
  EXPECT_GE(token_blocked.elapsed, kBlockedFloor) << "the measured call did not reach the wait";
  EXPECT_EQ(token_blocked.allocations, 0U);
}

// Readiness is owned by exactly one wait set. When the Runtime readiness pump already owns this
// subscriber, the private wait context cannot be built and the caller is told so rather than
// silently sharing or stealing the attachment.
// Topology: subscriber attached to a foreign wait set before its first wait
// Key assertion: kInvalidArgument from both waiting methods, and it is not sticky.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitAndReceiveOnAttachedSubscriberReportsInvalidArgument) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  // Build the raw endpoint first and keep a borrowed pointer; moving the unique_ptr into the
  // facade does not move the pointee.
  auto endpoint = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
      runtime.value(), channel_name, sizeof(WaitPayload), options);
  ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());
  auto* const raw = endpoint->get();

  // Claim its readiness with a foreign wait set, standing in for the Runtime pump. This wait set
  // must outlive the facade below.
  auto wait_set = holoscan::channel_internal::InlinePayloadTestAccess::create_wait_set(*runtime);
  ASSERT_TRUE(wait_set) << holoscan::runtime_error_message(wait_set.error());
  auto attached = wait_set.value()->attach(*raw);
  ASSERT_TRUE(attached) << holoscan::runtime_error_message(attached.error());

  auto subscriber =
      holoscan::channel_internal::InlinePayloadTestAccess::make_subscriber<WaitPayload>(
          std::move(endpoint).value());

  // The endpoint is empty, so the receive-first fast path falls through to the wait, and building
  // the private wait context hits the second attach.
  auto refused = subscriber.wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(refused.error().message.find("Runtime readiness pump"), std::string::npos);

  // The failure leaves no half-built context behind, so a later call reports the same thing rather
  // than degrading into a different error.
  auto ready = subscriber.wait_for_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(ready);
  EXPECT_EQ(ready.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(ready.error().message.find("Runtime readiness pump"), std::string::npos);
}

// An endpoint that still owes a payload loan cannot receive, and the caller must learn that
// immediately rather than blocking until the deadline on a queue it is not allowed to read.
// Topology (blocking, event wake): a live InlinePayloadReceiveGuard held across the wait
// Key assertion: kFailure, reported without approaching the deadline.
TEST(ChannelWaitTest, WaitAndReceiveWithActivePayloadLoanFailsWithoutWaiting) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  auto endpoint = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
      runtime.value(), channel_name, sizeof(WaitPayload), options);
  ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());
  auto* const raw = endpoint->get();
  auto subscriber =
      holoscan::channel_internal::InlinePayloadTestAccess::make_subscriber<WaitPayload>(
          std::move(endpoint).value());

  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));
  ASSERT_TRUE(publisher->emit(WaitPayload{11}));
  ASSERT_TRUE(publisher->emit(WaitPayload{12}));

  // Take a loan through the endpoint and hold it across the facade call.
  std::optional<holoscan::channel_internal::InlinePayloadReceiveGuard> borrowed;
  ASSERT_TRUE(wait_until(
      [&] {
        auto received = raw->receive_borrowed();
        if (!received) {
          return false;
        }
        borrowed.emplace(std::move(*received));
        return true;
      },
      std::chrono::seconds(2)));

  const auto started = std::chrono::steady_clock::now();
  auto refused = subscriber.wait_and_receive_sample(in(std::chrono::seconds(30)));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, holoscan::ErrorCode::kFailure);
  // Only asserts the deadline was never approached; never a latency floor. One second is well
  // under the 30 s deadline yet far above the microseconds this fail-fast path actually costs.
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(1));

  // Releasing the loan restores the endpoint rather than leaving it permanently refusing.
  borrowed.reset();
  auto received = subscriber.wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
}

namespace {

// Tensor end-of-stream is only reachable through the catalog path: ChannelPublisher<Tensor>::close
// is the sole EOS producer and that facade is only constructible from a catalog. The catalog also
// forces enable_event_wake true and tensor_transfer_mode kBufferHandle, which is what gives a
// kDropOldest endpoint its separate protected terminal lane.
[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error>
make_tensor_import_catalog(const std::string& endpoint_name,
                           const holoscan::QueuePolicy queue_policy) {
  holoscan::detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = queue_policy,
      .service_max_subscriber_depth = 1U,
      .service_max_subscriber_borrowed = 1U,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kTensor,
              .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
          },
      .tensor =
          holoscan::TensorBoundaryContractRecord{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 2U,
              .max_byte_span = 16U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
              // A copy-capable policy must declare a positive staging cap; the catalog rejects
              // the endpoint with ENDPOINT_TENSOR_POLICY_INVALID without one.
              .max_staging_byte_span = 16U,
          },
      .bounds = holoscan::EndpointBounds{.publisher =
                                             holoscan::PublisherEndpointBounds{
                                                 .max_loaned_samples = 2U,
                                             }},
  };
  return holoscan::detail::ChannelEndpointCatalogAccess::normalize("channel-wait-test",
                                                                   {std::move(endpoint)});
}

[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error>
make_tensor_export_catalog(const std::string& endpoint_name,
                           const holoscan::QueuePolicy queue_policy) {
  holoscan::detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kExport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = queue_policy,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kTensor,
              .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
          },
      .tensor =
          holoscan::TensorBoundaryContractRecord{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 2U,
              .max_byte_span = 16U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
              // A copy-capable policy must declare a positive staging cap; the catalog rejects
              // the endpoint with ENDPOINT_TENSOR_POLICY_INVALID without one.
              .max_staging_byte_span = 16U,
          },
      .bounds = holoscan::EndpointBounds{.subscriber =
                                             holoscan::SubscriberEndpointBounds{
                                                 .queue_depth = 1U,
                                                 .max_borrowed_samples = 1U,
                                             }},
  };
  return holoscan::detail::ChannelEndpointCatalogAccess::normalize("channel-wait-test",
                                                                   {std::move(endpoint)});
}

// The tensor tests refresh both sides, because a publisher-only connection snapshot can report a
// subscriber before that subscriber's own inbound connection is established.
[[nodiscard]] bool wait_for_tensor_subscriber(
    holoscan::ChannelPublisher<holoscan::Tensor>& publisher,
    holoscan::ChannelSubscriber<holoscan::Tensor>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto subscriber_refreshed = subscriber.has_samples();
    auto updated = publisher.update_connections();
    if (!updated || !subscriber_refreshed) {
      return false;
    }
    if (publisher.has_subscribers()) {
      ++consecutive_ready_refreshes;
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

[[nodiscard]] bool emit_one_tensor(holoscan::ChannelPublisher<holoscan::Tensor>& publisher,
                                   const std::byte fill) {
  static constexpr std::array<std::int64_t, 1U> kShape{8};
  auto loan = publisher.loan_tensor(holoscan::TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  if (!loan) {
    return false;
  }
  auto writer = loan->write_host();
  if (!writer) {
    return false;
  }
  std::memset(writer->data(), static_cast<int>(fill), writer->byte_size());
  if (!std::move(*writer).commit()) {
    return false;
  }
  return publisher.emit(std::move(*loan)).has_value();
}

}  // namespace

// The pimpl Tensor facade must honour the same wait contract as the header-only families. Both the
// data and the end-of-stream arrive while the caller is already blocked, so this exercises the wake
// path rather than the receive-first fast path.
// Topology (blocking catalog, event wake forced by the catalog):
//   ChannelPublisher<Tensor> --[Tensor]--> ChannelSubscriber<Tensor>
// Key assertion: a blocked wait is woken by the Tensor, and a second blocked wait by close().
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitAndReceiveTensorWakesOnDataAndEndOfStream) {
  const std::string endpoint_name = make_unique_channel_name();
  auto import_catalog = make_tensor_import_catalog(endpoint_name, holoscan::QueuePolicy::kBlocking);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  auto export_catalog = make_tensor_export_catalog(endpoint_name, holoscan::QueuePolicy::kBlocking);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<holoscan::Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(publisher.value(), subscriber.value()));

  // Emit only after the reader is already blocked, so the sample cannot be served without a wake.
  std::atomic<bool> emitted{false};
  std::chrono::steady_clock::duration data_elapsed{};
  {
    const std::jthread producer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      emitted.store(emit_one_tensor(publisher.value(), std::byte{0x6A}));
    });
    const auto started = std::chrono::steady_clock::now();
    auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
    data_elapsed = std::chrono::steady_clock::now() - started;
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    ASSERT_NE(received->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::byte*>(received->data.data()), std::byte{0x6A});
  }
  EXPECT_TRUE(emitted.load());

  // Same again for end of stream, which under kBlocking travels in band on the data lane.
  std::chrono::steady_clock::duration close_elapsed{};
  {
    const std::jthread closer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      static_cast<void>(publisher->close());
    });
    const auto started = std::chrono::steady_clock::now();
    auto closed = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
    close_elapsed = std::chrono::steady_clock::now() - started;
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
  }

  // Neither value assertion above can prove the wake reached the caller through the event lane:
  // each blocking pass is clamped to one second and re-checks has_samples(), so a wait that was
  // never woken at all would still observe both the sample and the close on its next pass and
  // report exactly the same values. Only the elapsed time separates a real wake from that
  // fallback. Both events land ~50 ms in, so a one second bound leaves a wide margin while still
  // failing if delivery had to wait for the clamp. DropOldestWaitWakesOnEndOfStreamTerminalLane
  // bounds its wait for the same reason.
  EXPECT_LT(data_elapsed, std::chrono::seconds(1));
  EXPECT_LT(close_elapsed, std::chrono::seconds(1));

  // The latch is sticky, so the non-consuming wait reports the terminal state rather than running
  // out its deadline.
  auto ready = subscriber->wait_for_sample(in(std::chrono::seconds(30)));
  ASSERT_FALSE(ready);
  EXPECT_EQ(ready.error().code, holoscan::ErrorCode::kClosed);
}

// Under a lossy replacement policy end of stream travels on a separate protected terminal lane, so
// a wait that watched only the data handle would never be woken by close(). Both lanes are
// attached to the private wait set, and has_samples() reports their union.
// Topology (drop-oldest catalog, buffer-handle transfer): two lanes, EOS on the terminal lane
// Key assertion: a wait already blocked with its data lane drained is woken by the terminal lane.
// REQ: CHAN-142
TEST(ChannelWaitTest, DropOldestWaitWakesOnEndOfStreamTerminalLane) {
  const std::string endpoint_name = make_unique_channel_name();
  auto import_catalog =
      make_tensor_import_catalog(endpoint_name, holoscan::QueuePolicy::kDropOldest);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  auto export_catalog =
      make_tensor_export_catalog(endpoint_name, holoscan::QueuePolicy::kDropOldest);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<holoscan::Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(publisher.value(), subscriber.value()));

  // Drain the data lane first, so the close() below can only be observed through the terminal lane.
  ASSERT_TRUE(emit_one_tensor(publisher.value(), std::byte{0x5C}));
  auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());

  std::chrono::steady_clock::duration elapsed{};
  {
    const std::jthread closer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      static_cast<void>(publisher->close());
    });
    const auto started = std::chrono::steady_clock::now();
    auto closed = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
    elapsed = std::chrono::steady_clock::now() - started;
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
  }

  // The value assertion above cannot on its own prove the terminal lane is attached: each blocking
  // pass is clamped to one second and re-checks has_samples(), which reports both lanes, so an
  // unattached terminal lane would still be noticed on the next pass. Only the elapsed time
  // separates a real terminal-lane wake from that fallback. The close lands ~50 ms in, so a one
  // second bound leaves a wide margin while still failing if the wake has to wait for the clamp.
  EXPECT_LT(elapsed, std::chrono::seconds(1));
}

#if HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS

// A Tensor readiness handle aggregates the data and protected terminal descriptors. With the data
// lane empty, close() can only make the aggregate readable through the terminal lane.
// Topology (drop-oldest catalog): Tensor data lane + protected EOS lane --> poll(2)
// Key assertion: a blocked external poller observes producer end of stream promptly.
// REQ: CHAN-150
TEST(ChannelReadinessTest, TensorDescriptorWakesOnProtectedEndOfStreamLane) {
  const std::string endpoint_name = make_unique_channel_name();
  auto import_catalog =
      make_tensor_import_catalog(endpoint_name, holoscan::QueuePolicy::kDropOldest);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  auto export_catalog =
      make_tensor_export_catalog(endpoint_name, holoscan::QueuePolicy::kDropOldest);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<holoscan::Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  // Prove the data lane works, then empty it before observing the terminal lane in isolation.
  auto readiness = subscriber->readiness();
  ASSERT_TRUE(readiness) << holoscan::runtime_error_message(readiness.error());
  pollfd descriptor{.fd = readiness->descriptor(), .events = POLLIN, .revents = 0};
  ASSERT_TRUE(emit_one_tensor(*publisher, std::byte{0x5D}));
  ASSERT_EQ(::poll(&descriptor, 1, 2000), 1);
  ASSERT_TRUE(readiness->acknowledge());
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  }
  ASSERT_TRUE(readiness->acknowledge());
  descriptor.revents = 0;
  ASSERT_EQ(::poll(&descriptor, 1, 0), 0);

  ASSERT_TRUE(publisher->close());
  descriptor.revents = 0;
  ASSERT_EQ(::poll(&descriptor, 1, 2000), 1);
  EXPECT_NE(descriptor.revents & POLLIN, 0);
  ASSERT_TRUE(readiness->acknowledge());
  auto closed = subscriber->receive_sample();
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
}

#endif  // HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS

#ifndef HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE
#define HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE ""
#endif

namespace {

// Forks channel_event_wait_child with the pipes and environment it expects, and returns the child
// pid. The child makes exactly one wait_and_receive_sample call; the parent drives its timing.
[[nodiscard]] pid_t spawn_event_wait_child(const char* const child_exe,
                                           const std::string& channel_name,
                                           const bool expect_sample,
                                           std::array<int, 2>& result_pipe,
                                           std::array<int, 2>& start_pipe) {
  const pid_t child = ::fork();
  if (child != 0) {
    return child;
  }
  ::close(result_pipe[0]);
  ::close(start_pipe[1]);
  const std::string result_fd = std::to_string(result_pipe[1]);
  const std::string start_fd = std::to_string(start_pipe[0]);
  static_cast<void>(::setenv("HOLOSCAN_CHANNEL_EVENT_WAIT_RESULT_FD", result_fd.c_str(), 1));
  static_cast<void>(::setenv("HOLOSCAN_CHANNEL_EVENT_WAIT_START_FD", start_fd.c_str(), 1));
  static_cast<void>(::setenv("HOLOSCAN_CHANNEL_EVENT_WAIT_NAME", channel_name.c_str(), 1));
  static_cast<void>(
      ::setenv("HOLOSCAN_CHANNEL_EVENT_WAIT_EXPECT_SAMPLE", expect_sample ? "1" : "0", 1));
  ::execl(child_exe, "channel_event_wait_child", static_cast<char*>(nullptr));
  _exit(127);
}

}  // namespace

// The acceptance criterion the whole feature exists for: waiting does not require periodic polling,
// and it works across a process boundary. The child calls wait_and_receive_sample exactly once,
// with no loop and no sleep, and additionally refuses to pass if that call burned CPU while
// blocked -- which is the only detector for a level-triggered hot spin.
// Topology (blocking, event wake): parent publisher --[WaitPayload]--> child subscriber process
// Key assertion: the child's single blocking call returns the parent's sample, cheaply.
// REQ: CHAN-142
TEST(ChannelWaitTest, ChildBlockedInWaitReceivesSampleWithoutPolling) {
  const char* const child_exe = HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE not configured";

  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options = event_wake_options();
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child =
      spawn_event_wait_child(child_exe, channel_name, true, result_pipe, start_pipe);
  ASSERT_GE(child, 0);
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  ASSERT_EQ(ready_code, 1U) << "child failed before its subscriber was ready, stage code "
                            << static_cast<unsigned>(ready_code);

  std::uint8_t blocked_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &blocked_code, sizeof(blocked_code)));
  ASSERT_EQ(blocked_code, 2U) << "child never reached its wait, stage code "
                              << static_cast<unsigned>(blocked_code);

  ASSERT_TRUE(wait_until(
      [&publisher] {
        static_cast<void>(publisher->update_connections());
        return publisher->has_subscribers();
      },
      std::chrono::seconds(5)))
      << "child subscriber never connected";

  // Leave the child blocked with nothing to receive. This window is what its idle-CPU check
  // measures, and it is also what a polling implementation would spend a core on.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  ASSERT_TRUE(wait_until(
      [&publisher] {
        auto published = publisher->emit(WaitPayload{4242});
        return published.has_value() && published.value() >= 1U;
      },
      std::chrono::seconds(5)))
      << "failed to publish to the waiting child";

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  // 16U is the idle-CPU refusal: the wait returned the right sample but spun to get it.
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// The same single call must report a timeout rather than hanging when no producer ever appears, so
// a cross-process client can bound its own shutdown without a watchdog.
// Topology (blocking, event wake): child subscriber process with no publisher at all
// Key assertion: the child's single blocking call returns kTimeout and the process exits cleanly.
TEST(ChannelWaitTest, ChildWaitWithNoProducerTimesOut) {
  const char* const child_exe = HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE not configured";

  const std::string channel_name = make_unique_channel_name();

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child =
      spawn_event_wait_child(child_exe, channel_name, false, result_pipe, start_pipe);
  ASSERT_GE(child, 0);
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  ASSERT_EQ(ready_code, 1U);
  std::uint8_t blocked_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &blocked_code, sizeof(blocked_code)));
  ASSERT_EQ(blocked_code, 2U);

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// Regression: building the private wait context is a transaction on the subscriber. The lane
// attachments it installs live inside the binding, and nothing in the Channel layer can retire
// them, so a setup step that fails after they are installed strands them: every later
// attach_wait_set rejects the repeated attachment with kInvalidArgument, the port stays flagged as
// owned by a wait set, and wait_for_readiness caches the context only on success -- so the
// subscriber could never wait again for the rest of its life.
//
// The injected fault is file-descriptor exhaustion, which the cancellation trigger's eventfd needs
// and the lane attachments do not. The budget is swept upward because the wait set itself also
// needs a descriptor and is created first; the window this test wants is the one where the wait
// set is created and the trigger is not, which is exactly where an ordering that attached the
// lanes first would strand them. If the environment never lands in that window the test skips
// rather than passing vacuously.
// Topology (blocking, event wake): publisher --[WaitPayload]--> subscriber, nothing queued
// Key assertion: after a failed setup the same subscriber still reaches a real wait.
// REQ: CHAN-142
TEST(ChannelWaitTest, WaitContextSetupFailureLeavesSubscriberUsable) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions options = event_wake_options();

  auto publisher = runtime->create_publisher<WaitPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const ScopedDescriptorBudget budget_guard;
  if (!budget_guard.usable()) {
    GTEST_SKIP() << "the process descriptor limit could not be read, so no fault can be injected";
  }

  // A failed setup caches nothing, so every attempt up to the first success re-enters
  // SubscriberWaitContext::create; only a success makes the setup path unreachable for the rest of
  // the sweep. Widen the budget until one attempt gets past wait-set creation and fails on the
  // trigger instead.
  bool reached_trigger_failure = false;
  for (int spare = 0; spare <= 8 && !reached_trigger_failure; ++spare) {
    if (!budget_guard.constrain(spare)) {
      break;
    }
    auto attempt = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(20)));
    budget_guard.restore();
    if (attempt) {
      break;  // the budget was already loose enough to build the context; nothing to prove
    }
    // "the wait set exists but its cancellation trigger does not" is the state that any ordering
    // which attached the lanes first would have reached with those lanes already installed.
    //
    // Gate on the token the trigger attachment itself produces, not on prose. Every step of setup
    // can report the same ErrorCode, so the code cannot name the window, and re-arming the trigger
    // fails with the same English wording -- unreachable here only because this wait carries no
    // stop token, which is not a property this test should depend on.
    reached_trigger_failure = std::string_view{attempt.error().message}.starts_with(
        holoscan::channel_internal::kWaitTriggerAttachFailed);
  }
  if (!reached_trigger_failure) {
    GTEST_SKIP() << "descriptor exhaustion never landed between wait-set creation and trigger "
                    "attachment; the transactional setup path was not exercised";
  }

  // The subscriber must still be able to build a context. An empty endpoint reports a timeout; the
  // stranded-attachment failure would instead report kInvalidArgument here, permanently.
  auto retried = subscriber->wait_and_receive_sample(in(std::chrono::milliseconds(200)));
  ASSERT_FALSE(retried);
  EXPECT_EQ(retried.error().code, holoscan::ErrorCode::kTimeout)
      << holoscan::runtime_error_message(retried.error());

  // And the recovered wait really works, rather than merely reporting a different error.
  ASSERT_TRUE(publisher->emit(WaitPayload{7}));
  auto received = subscriber->wait_and_receive_sample(in(std::chrono::seconds(5)));
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 7U);
}
