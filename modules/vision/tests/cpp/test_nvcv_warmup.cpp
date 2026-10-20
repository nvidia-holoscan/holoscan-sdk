// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <nvcv/Config.h>

#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>

#include "common/allocation_counter.hpp"
#include "detail/cvcuda_tensor.hpp"

namespace {
constinit thread_local bool count_nvcv_malloc = false;
constinit thread_local bool fail_next_nvcv_malloc = false;
constinit thread_local std::size_t nvcv_malloc_count = 0;
}  // namespace

#if defined(__GLIBC__)
// NVCV ships with a statically linked C++ allocator, bypassing the operator-new interposition
// used by the existing common/allocation_counter.hpp utility. Interpose malloc only in this
// test executable, with glibc's backing entry point to avoid recursive dlsym/bootstrap
// allocation. Never link this hook into production.
// The hook is live throughout holoscan_vision_nvcv_warmup_tests, including when counting is off;
// this dedicated executable isolates it from holoscan_vision_tests and its unrelated tests.
// This declaration names glibc's allocator entry point for the isolated test interposer.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" void* __libc_malloc(std::size_t size) noexcept;
extern "C" void* malloc(std::size_t size) noexcept {
  if (count_nvcv_malloc)
    ++nvcv_malloc_count;
  if (fail_next_nvcv_malloc) {
    fail_next_nvcv_malloc = false;
    errno = ENOMEM;
    return nullptr;
  }
  return __libc_malloc(size);
}
#endif

namespace holoscan::vision {
namespace {

class ScopedNvcvMallocProbe {
 public:
  explicit ScopedNvcvMallocProbe(bool fail_next = false) {
    nvcv_malloc_count = 0;
    count_nvcv_malloc = true;
    fail_next_nvcv_malloc = fail_next;
  }
  ~ScopedNvcvMallocProbe() {
    count_nvcv_malloc = false;
    fail_next_nvcv_malloc = false;
  }
  std::size_t count() const { return nvcv_malloc_count; }
};

std::pair<bool, std::size_t> count_wrapper_allocations(std::size_t count) {
  // Metadata-only wrapping, as in the existing adapter tests; never submit this host address.
  std::byte placeholder{};
  constexpr std::array<std::int64_t, 3> kShape{1, 1, 1};
  constexpr std::array<std::int64_t, 3> kStrides{1, 1, 1};
  bool success = true;
  std::size_t allocations{};
  {
    const ScopedNvcvMallocProbe counter;
    {
      std::array<nvcv::Tensor, 3> handles{};
      for (std::size_t index = 0; index != count; ++index) {
        auto wrapped =
            detail::wrap_nvcv_tensor(&placeholder, kShape, kStrides, {kDLUInt, 8, 1}, "HWC");
        if (!wrapped) {
          success = false;
          break;
        }
        handles[index] = std::move(*wrapped);
      }
    }
    allocations = counter.count();
  }
  return {success, allocations};
}

template <typename Op>
// Keep first and repeated native allocation checks beside the prepared operator lifecycle.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void check_started_wrapper_reuse() {
  constexpr std::size_t kHandles =
      std::is_same_v<Op, RemapOp> || std::is_same_v<Op, NonMaximumSuppressionOp> ? 3 : 2;
  Op operation;
  ASSERT_NO_THROW(operation.start());
  for (std::size_t iteration = 0; iteration != 64; ++iteration) {
    const auto [success, allocations] = count_wrapper_allocations(kHandles);
    ASSERT_TRUE(success);
    EXPECT_EQ(allocations, 0U) << "iteration " << iteration;
  }
  {
    const test::ScopedAllocationExhaustion failure(0);
    const ScopedNvcvMallocProbe native_failure(true);
    operation.start();
    operation.stop();
    operation.start();
  }
}

template <typename Op>
// Keep cold control, failure injection, and startup retry on the same native handle-pool path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void check_cold_start() {
  // Re-exec, not fork-only: no preceding test or operator may warm this process's NVCV pool.
  ASSERT_EXIT(
      {
        check_started_wrapper_reuse<Op>();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

// REQ: VISION-009
TEST(VisionNvcvWarmupDeathTest, EveryOperatorPreparesFirstAndRepeatedWrapperUse) {
#if !defined(__GLIBC__)
  GTEST_SKIP() << "Native allocator probe requires glibc";
#endif
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  check_cold_start<ResizeOp>();
  check_cold_start<ConvertToOp>();
  check_cold_start<CopyMakeBorderOp>();
  check_cold_start<CustomCropOp>();
  check_cold_start<NormalizeOp>();
  check_cold_start<ReformatOp>();
  check_cold_start<RemapOp>();
  check_cold_start<NonMaximumSuppressionOp>();
}

void check_counter_observes_cold_pool() {
  const auto cold = count_wrapper_allocations(3);
  ASSERT_TRUE(cold.first);
  EXPECT_GT(cold.second,
            0U);  // prove that the counter sees allocations from the installed NVCV DSO
  const auto warm = count_wrapper_allocations(3);
  ASSERT_TRUE(warm.first);
  EXPECT_EQ(warm.second, 0U);
}

// REQ: VISION-009
TEST(VisionNvcvWarmupDeathTest, CounterObservesColdPoolAndReusedStorage) {
#if !defined(__GLIBC__)
  GTEST_SKIP() << "Native allocator probe requires glibc";
#endif
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        check_counter_observes_cold_pool();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

void check_preparation_failure_retry() {
  expected<void, Error> prepared;
  {
    const ScopedNvcvMallocProbe failure(true);
    prepared = detail::prepare_nvcv_tensor_handles(3);
  }
  ASSERT_FALSE(prepared);
  EXPECT_EQ(prepared.error().code, ErrorCode::kResourceExhausted);
  ASSERT_TRUE(detail::prepare_nvcv_tensor_handles(3));
  const auto [success, allocations] = count_wrapper_allocations(3);
  ASSERT_TRUE(success);
  EXPECT_EQ(allocations, 0U);
}

// REQ: VISION-009
TEST(VisionNvcvWarmupDeathTest, ColdAllocationFailureIsRecoverable) {
#if !defined(__GLIBC__)
  GTEST_SKIP() << "Native allocator probe requires glibc";
#endif
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        check_preparation_failure_retry();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

// Keep fixed-capacity failure, partial cleanup, and retry in one test scenario so native pool
// state and expected retained handles can be audited together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void check_fixed_capacity_and_partial_failure() {
  // Configuration changes are confined to this fresh subprocess and never used by production.
  ASSERT_EQ(nvcvConfigSetMaxTensorCount(2), NVCV_SUCCESS);
  check_started_wrapper_reuse<ResizeOp>();
  RemapOp operation;
  try {
    operation.start();
    FAIL() << "Preparation must respect the application's two-handle limit";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kResourceExhausted);
  }
  // The failed third creation must return the two handles it had already acquired.
  const auto [success, allocations] = count_wrapper_allocations(2);
  ASSERT_TRUE(success);
  EXPECT_EQ(allocations, 0U);
  ASSERT_EQ(nvcvConfigSetMaxTensorCount(3), NVCV_SUCCESS);
  ASSERT_NO_THROW(operation.start());
  {
    const test::ScopedAllocationExhaustion failure(0);
    operation.stop();
    operation.start();
  }
}

// REQ: VISION-009
TEST(VisionNvcvWarmupDeathTest, FixedPolicyAndPartialStartupFailureArePreserved) {
#if !defined(__GLIBC__)
  GTEST_SKIP() << "Native allocator probe requires glibc";
#endif
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        check_fixed_capacity_and_partial_failure();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

}  // namespace
}  // namespace holoscan::vision
