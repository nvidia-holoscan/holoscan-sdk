/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <latch>
#include <memory>
#include <thread>
#include <utility>

#include <holomq/cuda/stream_lifetime.hpp>

namespace {

using Lifetime = holomq::cuda::StreamLifetime;
using State = Lifetime::State;

struct OutcomeNode : Lifetime::DeferredRelease {
  OutcomeNode() {
    on_outcome = [](DeferredRelease& base, State state) noexcept {
      auto& node = static_cast<OutcomeNode&>(base);
      node.outcome.store(state, std::memory_order_relaxed);
      node.calls.fetch_add(1U, std::memory_order_release);
      if (state == State::kRetired) {
        node.owner.reset();
      }
    };
  }
  std::atomic_uint calls{};
  std::atomic<State> outcome{State::kOpen};
  std::shared_ptr<int> owner;
};

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, OpenPinsMoveAndTerminalStatesStayFinal) {
  Lifetime lifetime{nullptr, 1};
  auto admitted = lifetime.try_pin();
  ASSERT_TRUE(admitted.pin);
  EXPECT_EQ(admitted.state, State::kOpen);
  EXPECT_EQ(lifetime.device(), 1);
  auto moved = std::move(admitted.pin);
  EXPECT_FALSE(admitted.pin);
  lifetime.close_admission();
  EXPECT_EQ(lifetime.try_pin().state, State::kClosing);
  EXPECT_FALSE(lifetime.publish_retired());
  moved.reset();
  lifetime.wait_for_pins();
  EXPECT_TRUE(lifetime.publish_retired());
  EXPECT_FALSE(lifetime.try_pin().pin);
  EXPECT_EQ(lifetime.try_pin().state, State::kRetired);
  lifetime.close();
  EXPECT_FALSE(lifetime.publish_retired());
  EXPECT_FALSE(lifetime.publish_quarantined());
  EXPECT_EQ(lifetime.state(), State::kRetired);
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, CloseDrainsTheEnqueueBeforeItCanSynchronize) {
  Lifetime lifetime{nullptr, 0};
  auto admitted = lifetime.try_pin();
  ASSERT_TRUE(admitted.pin);
  std::latch entered{1};
  std::atomic_bool synchronized{};
  // Written under the pin and read after its drain: the acquire/release handshake must publish it.
  int submitted{};
  std::thread owner{[&] {
    entered.count_down();
    lifetime.close();
    EXPECT_EQ(submitted, 123);
    synchronized.store(true, std::memory_order_release);
    EXPECT_TRUE(lifetime.publish_retired());
  }};
  entered.wait();
  while (lifetime.state() == State::kOpen) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(synchronized.load(std::memory_order_acquire));
  const auto rejected = lifetime.try_pin();
  EXPECT_FALSE(rejected.pin);
  EXPECT_EQ(rejected.state, State::kClosing);
  submitted = 123;
  admitted.pin.reset();
  owner.join();
  EXPECT_TRUE(synchronized.load(std::memory_order_acquire));
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, ClosingDoesNotProveStorageSafeAndQuarantineNeverReleasesIt) {
  for (const auto outcome : {State::kRetired, State::kQuarantined}) {
    Lifetime lifetime{nullptr, 0};
    OutcomeNode node;
    node.owner = std::make_shared<int>(17);
    const std::weak_ptr<int> owner = node.owner;
    lifetime.close();
    ASSERT_EQ(lifetime.try_pin().state, State::kClosing);
    ASSERT_TRUE(lifetime.defer(node));
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(node.calls.load(), 0U);
    EXPECT_TRUE(outcome == State::kRetired ? lifetime.publish_retired()
                                           : lifetime.publish_quarantined());
    EXPECT_EQ(node.calls.load(), 1U);
    EXPECT_EQ(node.outcome.load(), outcome);
    EXPECT_EQ(owner.expired(), outcome == State::kRetired);
    EXPECT_FALSE(lifetime.defer(node));
    lifetime.close();
    EXPECT_EQ(lifetime.state(), outcome);
    EXPECT_FALSE(lifetime.publish_retired());
    EXPECT_FALSE(lifetime.publish_quarantined());
  }
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, ConcurrentDeferralAndPublicationDeliverExactlyOneOutcome) {
  for (unsigned iteration = 0U; iteration < 512U; ++iteration) {
    Lifetime lifetime{nullptr, 0};
    lifetime.close();
    OutcomeNode node;
    std::barrier start{2};
    const auto outcome = iteration % 2U == 0U ? State::kRetired : State::kQuarantined;
    std::thread holder{[&] {
      start.arrive_and_wait();
      if (!lifetime.defer(node)) {
        // A rejected registration must observe publication, even if it lost the seal race.
        const auto observed = lifetime.state();
        EXPECT_EQ(observed, outcome);
        node.on_outcome(node, observed);
      }
    }};
    start.arrive_and_wait();
    EXPECT_TRUE(outcome == State::kRetired ? lifetime.publish_retired()
                                           : lifetime.publish_quarantined());
    holder.join();
    EXPECT_EQ(node.calls.load(), 1U);
    EXPECT_EQ(node.outcome.load(), outcome);
  }
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, AllConcurrentNodesAreDeliveredAndCallbacksCanDestroyTheirNode) {
  Lifetime lifetime{nullptr, 0};
  lifetime.close();
  std::array<OutcomeNode, 128U> nodes;
  std::array<std::thread, 4U> holders;
  for (std::size_t worker = 0U; worker < holders.size(); ++worker) {
    holders[worker] = std::thread{[&, worker] {
      for (std::size_t index = worker; index < nodes.size(); index += holders.size()) {
        EXPECT_TRUE(lifetime.defer(nodes[index]));
      }
    }};
  }
  for (auto& holder : holders) {
    holder.join();
  }
  auto* self_destroying = new Lifetime::DeferredRelease;
  self_destroying->on_outcome = [](Lifetime::DeferredRelease& node, State) noexcept {
    delete &node;
  };
  ASSERT_TRUE(lifetime.defer(*self_destroying));
  ASSERT_TRUE(lifetime.publish_retired());
  for (const auto& node : nodes) {
    EXPECT_EQ(node.calls.load(), 1U);
    EXPECT_EQ(node.outcome.load(), State::kRetired);
  }
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, ReusedNativeHandleDoesNotReuseLifetimeIdentity) {
  // No CUDA call: model native handle reuse while both generations' control blocks survive.
  auto* handle = reinterpret_cast<CUstream_st*>(std::uintptr_t{0x1234U});
  auto old = std::make_shared<Lifetime>(handle, 0);
  old->close();
  ASSERT_TRUE(old->publish_retired());
  auto fresh = std::make_shared<Lifetime>(handle, 0);
  EXPECT_EQ(old->handle(), fresh->handle());
  EXPECT_NE(old.get(), fresh.get());
  EXPECT_EQ(old->try_pin().state, State::kRetired);
  auto admitted = fresh->try_pin();
  EXPECT_TRUE(admitted.pin);
  EXPECT_EQ(admitted.pin.stream(), handle);
  admitted.pin.reset();
  fresh->close();
  EXPECT_TRUE(fresh->publish_retired());
}

// REQ: LIFECYCLE-011
TEST(StreamLifetimeTest, EnumeratesLinearizedPinCloseDrainPublishAndDeferralHistories) {
  unsigned histories{};
  for (const auto outcome : {State::kRetired, State::kQuarantined}) {
    std::array<char, 6U> order{'C', 'D', 'F', 'P', 'R', 'U'};
    do {
      // Independent protocol model: admission and GPU completion are different facts.
      State expected = State::kOpen;
      bool pinned{};
      bool pin_attempted{};
      bool drained{};
      bool valid = true;
      bool queued{};
      for (const char operation : order) {
        if (operation == 'P') {
          pin_attempted = true;
          pinned = expected == State::kOpen;
        } else if (operation == 'U') {
          if (!pin_attempted) {
            valid = false;
            break;
          }
          pinned = false;
        } else if (operation == 'C') {
          expected = State::kClosing;
        } else if (operation == 'D') {
          if (expected != State::kClosing || pinned) {
            valid = false;
            break;
          }
          drained = true;
        } else if (operation == 'R') {
          if (!drained) {
            valid = false;
            break;
          }
          expected = outcome;
        } else if (operation == 'F') {
          if (expected == State::kOpen) {
            valid = false;
            break;
          }
          queued = expected == State::kClosing;
        }
      }
      if (!valid) {
        continue;
      }
      ++histories;
      Lifetime lifetime{nullptr, 0};
      Lifetime::Pin pin;
      OutcomeNode node;
      expected = State::kOpen;
      for (const char operation : order) {
        switch (operation) {
          case 'P': {
            auto admission = lifetime.try_pin();
            EXPECT_EQ(admission.state, expected);
            EXPECT_EQ(static_cast<bool>(admission.pin), expected == State::kOpen);
            pin = std::move(admission.pin);
            break;
          }
          case 'U':
            pin.reset();
            break;
          case 'C':
            lifetime.close_admission();
            expected = State::kClosing;
            break;
          case 'D':
            lifetime.wait_for_pins();
            break;
          case 'R':
            EXPECT_TRUE(outcome == State::kRetired ? lifetime.publish_retired()
                                                   : lifetime.publish_quarantined());
            expected = outcome;
            break;
          case 'F':
            EXPECT_EQ(lifetime.defer(node), expected == State::kClosing);
            break;
          default:
            FAIL();
        }
        EXPECT_EQ(lifetime.state(), expected);
      }
      EXPECT_EQ(node.calls.load(), queued ? 1U : 0U);
      EXPECT_EQ(lifetime.state(), outcome);
    } while (std::next_permutation(order.begin(), order.end()));
  }
  EXPECT_GT(histories, 20U);
}

}  // namespace
