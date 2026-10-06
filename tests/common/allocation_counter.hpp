/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace holoscan::test {

class ScopedAllocationCounter {
 public:
  ScopedAllocationCounter() noexcept;
  ~ScopedAllocationCounter() noexcept;

  std::size_t count() const noexcept;

 private:
  bool previous_enabled_ = false;
  std::size_t previous_count_ = 0;
};

/// Fail one allocation on the current test thread after a fixed number of successful allocations.
class ScopedAllocationFailure {
 public:
  explicit ScopedAllocationFailure(std::size_t successful_allocations_before_failure) noexcept;
  ~ScopedAllocationFailure() noexcept;

 private:
  bool previous_enabled_ = false;
  std::size_t previous_remaining_ = 0;
};

/// Fail every allocation on the current test thread after a fixed number of successes.
class ScopedAllocationExhaustion {
 public:
  explicit ScopedAllocationExhaustion(std::size_t successful_allocations_before_failure) noexcept;
  ~ScopedAllocationExhaustion() noexcept;

 private:
  bool previous_enabled_ = false;
  std::size_t previous_remaining_ = 0;
};

/// Fail one allocation on the current test thread after a fixed number of successes.
///
/// This unscoped form arms a worker-thread failure from an explicit test callback.
void fail_allocation_on_current_thread(std::size_t successful_allocations_before_failure) noexcept;

/// Fail the next explicitly aligned allocation on the current test thread.
class ScopedAlignedAllocationFailure {
 public:
  ScopedAlignedAllocationFailure() noexcept;
  ~ScopedAlignedAllocationFailure() noexcept;

 private:
  bool previous_enabled_ = false;
};

}  // namespace holoscan::test
