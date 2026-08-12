/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "core/domain/tensor_access.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// Thread-local flags let the per-TU operator new intercepts charge allocations only to the active
// test. g_count_test_allocations acts as a gate; tests enable it through ScopedAllocationCounter.
constinit thread_local bool g_count_test_allocations = false;
constinit thread_local std::size_t g_test_allocation_count = 0;

static_assert(!holoscan::dlpack_version_requires_explicit_strides(DLPackVersion{1, 1}));
static_assert(holoscan::dlpack_version_requires_explicit_strides(DLPackVersion{1, 2}));
static_assert(holoscan::dlpack_version_requires_explicit_strides(DLPackVersion{2, 0}));
static_assert(holoscan::dlpack_dtype_item_size_bytes(DLDataType{kDLFloat, 32, 1}) == 4);
static_assert(holoscan::dlpack_dtype_item_size_bytes(DLDataType{kDLFloat, 32, 4}) == 16);
static_assert(holoscan::dlpack_dtype_item_size_bytes(DLDataType{kDLUInt, 1, 1}) == 1);
// REQ: TENSOR-CORE-015
static_assert(
    std::is_same_v<decltype(std::declval<const holoscan::Tensor&>().data()), const void*>);
static_assert(
    std::is_same_v<decltype(std::declval<holoscan::Tensor&>().data_as<float>()), const float*>);
static_assert(std::is_same_v<decltype(std::declval<const holoscan::Tensor&>().data_as<float>()),
                             const float*>);

// Concepts used in static_asserts below to verify that mutable_data() and mutable_data_as<T>()
// are inaccessible on const-qualified or rvalue Tensor references (TENSOR-CORE-015 const rule).
template <typename T>
concept HasMutableData = requires(T&& tensor) { std::forward<T>(tensor).mutable_data(); };

template <typename T>
concept HasMutableFloatData =
    requires(T&& tensor) { std::forward<T>(tensor).template mutable_data_as<float>(); };

// Ensures the internal DLPack context accessor is not inadvertently exposed as a public method.
template <typename T>
concept HasPublicDLPackContext = requires(T&& tensor) { std::forward<T>(tensor).dl_ctx(); };

static_assert(HasMutableData<holoscan::Tensor&>);
static_assert(!HasMutableData<const holoscan::Tensor&>);
static_assert(!HasMutableData<holoscan::Tensor>);
static_assert(HasMutableFloatData<holoscan::Tensor&>);
static_assert(!HasMutableFloatData<const holoscan::Tensor&>);
static_assert(!HasMutableFloatData<holoscan::Tensor>);
static_assert(!HasPublicDLPackContext<holoscan::Tensor&>);
static_assert(!HasPublicDLPackContext<const holoscan::Tensor&>);

TEST(DLPackUtils, DataTypeFromTypeStringAcceptsStringView) {
  constexpr std::string_view type_string{"<f4"};

  auto dtype = holoscan::dldatatype_from_typestr(type_string);

  ASSERT_TRUE(dtype) << dtype.error().message;
  EXPECT_EQ(dtype->code, kDLFloat);
  EXPECT_EQ(dtype->bits, 32);
  EXPECT_EQ(dtype->lanes, 1);
}

TEST(DLPackUtils, DataTypeFromTypeStringRejectsMalformedItemSize) {
  auto dtype = holoscan::dldatatype_from_typestr("<fnot-a-size");

  ASSERT_FALSE(dtype);
  EXPECT_EQ(dtype.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Replacement allocators that increment g_test_allocation_count whenever the counter is active.
// Paired with ScopedAllocationCounter to enforce zero-allocation invariants on hot metadata paths.
void* counted_malloc(std::size_t size) {
  if (g_count_test_allocations) {
    ++g_test_allocation_count;
  }
  if (void* ptr = std::malloc(size == 0 ? 1 : size)) {
    return ptr;
  }
  throw std::bad_alloc();
}

void* counted_aligned_malloc(std::size_t size, std::align_val_t alignment) {
  if (g_count_test_allocations) {
    ++g_test_allocation_count;
  }
  void* ptr = nullptr;
  const auto align = std::max(static_cast<std::size_t>(alignment), sizeof(void*));
  if (posix_memalign(&ptr, align, size == 0 ? 1 : size) == 0) {
    return ptr;
  }
  throw std::bad_alloc();
}

// RAII guard that activates the per-thread allocation counter for a delimited scope.
// On destruction it restores the previous activation state, allowing nested counters.
// Example: TensorMetadataViews/ShapeAndExplicitDLPackStridesDoNotAllocate calls count() after
// 1024 shape/stride accesses and asserts 0 to enforce the no-allocation invariant.
class ScopedAllocationCounter {
 public:
  ScopedAllocationCounter() noexcept
      : previous_enabled_(g_count_test_allocations), previous_count_(g_test_allocation_count) {
    g_test_allocation_count = 0;
    g_count_test_allocations = true;
  }

  ~ScopedAllocationCounter() noexcept {
    g_count_test_allocations = previous_enabled_;
    if (previous_enabled_) {
      g_test_allocation_count += previous_count_;
    }
  }

  std::size_t count() const noexcept { return g_test_allocation_count; }

 private:
  bool previous_enabled_ = false;
  std::size_t previous_count_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return counted_malloc(size);
}

void* operator new[](std::size_t size) {
  return counted_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

namespace holoscan {

namespace {

// A native (capable) TensorMemoryReference used to verify that Tensor forwards stream and
// readiness operations to its memory reference. It records what it was asked to do and reports a
// readiness-stream contract, in contrast to imported DLPack wrappers.
struct TestStorage : public TensorMemoryReference {
  explicit TestStorage(std::shared_ptr<std::atomic<int>> destroyed_counter)
      : destroyed_counter(std::move(destroyed_counter)) {}

  ~TestStorage() override { destroyed_counter->fetch_add(1, std::memory_order_relaxed); }

  bool set_deallocation_stream(cudaStream_t stream) override {
    last_deallocation_stream = stream;
    deallocation_stream_count.fetch_add(1, std::memory_order_relaxed);
    return accept_deallocation_stream;
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    last_readiness_stream = stream;
    readiness_stream_count.fetch_add(1, std::memory_order_relaxed);
    return accept_readiness_stream;
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(cudaStream_t stream) override {
    last_consumer_stream = stream;
    consumer_stream_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    host_sync_count.fetch_add(1, std::memory_order_relaxed);
    return {};
  }

  void retain_tensor_handle() noexcept override {
    live_handle_count.fetch_add(1, std::memory_order_relaxed);
  }

  void release_tensor_handle() noexcept override {
    live_handle_count.fetch_sub(1, std::memory_order_relaxed);
  }

  std::shared_ptr<std::atomic<int>> destroyed_counter;
  alignas(std::max_align_t) std::array<uint8_t, 32> data{{0, 1, 2, 3, 4, 5}};
  std::atomic<int> deallocation_stream_count{0};
  std::atomic<int> readiness_stream_count{0};
  std::atomic<int> consumer_stream_count{0};
  std::atomic<int> host_sync_count{0};
  std::atomic<int> live_handle_count{0};
  cudaStream_t last_deallocation_stream = nullptr;
  cudaStream_t last_readiness_stream = nullptr;
  cudaStream_t last_consumer_stream = nullptr;
  bool accept_deallocation_stream = false;
  bool accept_readiness_stream = false;
};

// Builds a reusable DLManagedTensorContext backed by TestStorage for a 2x3 uint8 CPU tensor.
// *data_out receives the raw backing-buffer pointer for data-identity assertions in callers.
// byte_offset is set to 4 so that most tests exercise the non-zero offset code path.
std::shared_ptr<DLManagedTensorContext> make_test_context(
    const std::shared_ptr<std::atomic<int>>& destroyed_counter, void** data_out) {
  auto storage = std::make_shared<TestStorage>(destroyed_counter);
  *data_out = storage->data.data();

  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = std::move(storage);
  context->dl_shape = {2, 3};
  context->dl_strides = {3, 1};
  context->tensor.dl_tensor.data = *data_out;
  context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 4;
  context->tensor.manager_ctx = nullptr;
  context->tensor.deleter = nullptr;
  return context;
}

// Shared assertion helper that confirms the canonical 2x3 uint8 CPU tensor fields produced by
// make_test_context (or a metadata snapshot thereof) including device, dtype, shape, strides,
// and byte_offset match the values established at context construction time.
void expect_test_tensor_metadata(const DLTensor& tensor, void* expected_data) {
  EXPECT_EQ(tensor.data, expected_data);
  EXPECT_EQ(tensor.device.device_type, kDLCPU);
  EXPECT_EQ(tensor.device.device_id, 0);
  EXPECT_EQ(tensor.dtype.code, kDLUInt);
  EXPECT_EQ(tensor.dtype.bits, 8);
  EXPECT_EQ(tensor.dtype.lanes, 1);
  EXPECT_EQ(tensor.ndim, 2);
  ASSERT_NE(tensor.shape, nullptr);
  EXPECT_EQ(tensor.shape[0], 2);
  EXPECT_EQ(tensor.shape[1], 3);
  ASSERT_NE(tensor.strides, nullptr);
  EXPECT_EQ(tensor.strides[0], 3);
  EXPECT_EQ(tensor.strides[1], 1);
  EXPECT_EQ(tensor.byte_offset, 4);
}

// Simulates a foreign DLManagedTensor producer using the legacy (unversioned) DLPack protocol.
// Owns shape/strides in inline arrays; its deleter calls operator delete on this object.
// Used by DLPackImport tests to verify Tensor takes a metadata snapshot and calls the deleter.
struct ExternalDLPackTensor {
  explicit ExternalDLPackTensor(std::shared_ptr<std::atomic<int>> destroyed_counter)
      : storage(std::make_shared<TestStorage>(std::move(destroyed_counter))) {
    tensor.dl_tensor.data = storage->data.data();
    tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
    tensor.dl_tensor.ndim = 2;
    tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
    tensor.dl_tensor.shape = shape.data();
    tensor.dl_tensor.strides = strides.data();
    tensor.dl_tensor.byte_offset = 4;
    tensor.manager_ctx = this;
    tensor.deleter = [](DLManagedTensor* self) {
      delete static_cast<ExternalDLPackTensor*>(self->manager_ctx);
    };
  }

  std::shared_ptr<TestStorage> storage;
  std::array<int64_t, 2> shape{{2, 3}};
  std::array<int64_t, 2> strides{{3, 1}};
  DLManagedTensor tensor{};
};

// Simulates a foreign DLManagedTensorVersioned producer (versioned DLPack >= 1.0 protocol).
// Mirrors ExternalDLPackTensor but exercises versioned-import paths including version and flag
// preservation; the deleter signature matches DLManagedTensorVersioned::deleter convention.
struct ExternalDLPackVersionedTensor {
  explicit ExternalDLPackVersionedTensor(std::shared_ptr<std::atomic<int>> destroyed_counter)
      : storage(std::make_shared<TestStorage>(std::move(destroyed_counter))) {
    tensor.version.major = HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR;
    tensor.version.minor = HOLOSCAN_DLPACK_IMPL_VERSION_MINOR;
    tensor.flags = 0;
    tensor.dl_tensor.data = storage->data.data();
    tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
    tensor.dl_tensor.ndim = 2;
    tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 1};
    tensor.dl_tensor.shape = shape.data();
    tensor.dl_tensor.strides = strides.data();
    tensor.dl_tensor.byte_offset = 4;
    tensor.manager_ctx = this;
    tensor.deleter = [](DLManagedTensorVersioned* self) {
      delete static_cast<ExternalDLPackVersionedTensor*>(self->manager_ctx);
    };
  }

  std::shared_ptr<TestStorage> storage;
  std::array<int64_t, 2> shape{{2, 3}};
  std::array<int64_t, 2> strides{{3, 1}};
  DLManagedTensorVersioned tensor{};
};

}  // namespace

TEST(TensorMetadataViews, ShapeAndExplicitDLPackStridesDoNotAllocate) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor(context);

  auto shape = tensor.shape_span();
  ASSERT_EQ(shape.size(), 2U);
  EXPECT_EQ(shape.data(), context->tensor.dl_tensor.shape);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);

  auto dlpack_strides = tensor.dlpack_strides_span();
  ASSERT_EQ(dlpack_strides.size(), 2U);
  EXPECT_EQ(dlpack_strides.data(), context->tensor.dl_tensor.strides);
  EXPECT_EQ(dlpack_strides[0], 3);
  EXPECT_EQ(dlpack_strides[1], 1);

  int64_t checksum = 0;
  {
    const ScopedAllocationCounter allocations;
    for (int iteration = 0; iteration < 1024; ++iteration) {
      auto shape_view = tensor.shape_span();
      auto stride_view = tensor.dlpack_strides_span();
      checksum += shape_view[0] + shape_view[1] + stride_view[0] + stride_view[1] +
                  static_cast<int64_t>(tensor.byte_offset());
    }
    EXPECT_EQ(allocations.count(), 0);
  }
  EXPECT_EQ(checksum, 1024 * (2 + 3 + 3 + 1 + 4));
}

TEST(TensorMetadataViews, ImplicitDLPackStridesReturnEmptySpan) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->dl_strides.clear();
  context->tensor.dl_tensor.strides = nullptr;
  const Tensor tensor(context);

  EXPECT_TRUE(tensor.dlpack_strides_span().empty());
  EXPECT_EQ(tensor.strides(), std::vector<int64_t>({3, 1}));
}

TEST(TensorHandleOperations, MoveTransfersTensorContextHandle) {
  static_assert(std::is_nothrow_move_constructible_v<Tensor>);
  static_assert(std::is_nothrow_move_assignable_v<Tensor>);

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);

  Tensor source(context);
  context.reset();
  Tensor moved_constructed(std::move(source));

  // The moved-from Tensor is expected to expose a null context.
  // NOLINTBEGIN(hicpp-invalid-access-moved)
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_EQ(detail::TensorAccess::context(source), nullptr);
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_EQ(source.data(), nullptr);
  // NOLINTEND(hicpp-invalid-access-moved)
  ASSERT_NE(detail::TensorAccess::context(moved_constructed), nullptr);
  expect_test_tensor_metadata(detail::TensorAccess::context(moved_constructed)->tensor.dl_tensor,
                              expected_data);

  Tensor move_assigned;
  move_assigned = std::move(moved_constructed);

  // The moved-from Tensor is expected to expose a null context.
  // NOLINTBEGIN(hicpp-invalid-access-moved)
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_EQ(detail::TensorAccess::context(moved_constructed), nullptr);
  // NOLINTEND(hicpp-invalid-access-moved)
  ASSERT_NE(detail::TensorAccess::context(move_assigned), nullptr);
  expect_test_tensor_metadata(detail::TensorAccess::context(move_assigned)->tensor.dl_tensor,
                              expected_data);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  move_assigned = Tensor();
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

TEST(TensorHandleOperations, RvalueContextConstructorConsumesHandle) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);

  const Tensor tensor(std::move(context));

  EXPECT_EQ(context, nullptr);
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);
  expect_test_tensor_metadata(detail::TensorAccess::context(tensor)->tensor.dl_tensor,
                              expected_data);
}

// REQ: TENSOR-STORAGE-010
TEST(TensorHandleOperations, CopyMoveAndSelfAssignmentBalanceBackingClaims) {
  // Key setup: one shared context is used to construct multiple Tensor handles; each live handle
  // increments live_handle_count via retain_tensor_handle() and decrements via
  // release_tensor_handle(). Self-copy and self-move must be no-ops with respect to claim count.
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);

  {
    Tensor first(context);
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 1);
    Tensor second(first);
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);
    Tensor third(std::move(second));
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);

    first = first;
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);
    third = std::move(third);
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);

    Tensor fourth;
    fourth = first;
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 3);
    fourth = Tensor{};
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);
  }
  // Key assertion: all Tensor handles destroyed → live_handle_count returns to 0.
  EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 0);
}

// REQ: TENSOR-STORAGE-010
TEST(TensorAccessAuthority, ReadLeaseTemporarilyBlocksMutationAndNarrowingIsPermanent) {
  // Key setup: acquire_read_lease() must block mutable_data() and legacy to_dlpack(); a versioned
  // export issued under an active lease must carry the READ_ONLY flag.
  // Key assertion: narrow_read_only() permanently prevents mutation even after the lease drops;
  // the read-only authority propagates to copies of the narrowed Tensor.
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  Tensor tensor(context);

  {
    auto lease = detail::TensorAccess::acquire_read_lease(tensor);
    ASSERT_TRUE(lease) << lease.error().message;
    auto denied = tensor.mutable_data();
    ASSERT_FALSE(denied);
    EXPECT_EQ(denied.error().code, ErrorCode::kPermissionDenied);

    auto legacy = tensor.to_dlpack();
    ASSERT_FALSE(legacy);
    EXPECT_EQ(legacy.error().code, ErrorCode::kNotSupported);
    auto versioned = tensor.to_dlpack_versioned();
    ASSERT_TRUE(versioned) << versioned.error().message;
    ASSERT_NE(*versioned, nullptr);
    EXPECT_NE((*versioned)->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0U);
    (*versioned)->deleter(*versioned);
  }
  auto writable = tensor.mutable_data();
  ASSERT_TRUE(writable) << writable.error().message;
  EXPECT_EQ(writable.value(), static_cast<std::byte*>(expected_data) + tensor.byte_offset());

  detail::TensorAccess::narrow_read_only(tensor);
  EXPECT_TRUE(tensor.is_read_only());
  auto permanently_denied = tensor.mutable_data();
  ASSERT_FALSE(permanently_denied);
  EXPECT_EQ(permanently_denied.error().code, ErrorCode::kPermissionDenied);

  Tensor copy = tensor;
  EXPECT_TRUE(copy.is_read_only());
  auto copy_denied = copy.mutable_data();
  ASSERT_FALSE(copy_denied);
  EXPECT_EQ(copy_denied.error().code, ErrorCode::kPermissionDenied);
}

// REQ: TENSOR-STORAGE-010
TEST(TensorAccessAuthority, IndependentWrappersOverOneContextShareReadLeaseState) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor first(context);
  Tensor second(context);

  auto lease = detail::TensorAccess::acquire_read_lease(first);
  ASSERT_TRUE(lease) << lease.error().message;

  auto denied = second.mutable_data();
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, ErrorCode::kPermissionDenied);

  lease = detail::TensorReadLease{};
  auto writable = second.mutable_data();
  ASSERT_TRUE(writable) << writable.error().message;
  EXPECT_EQ(writable.value(), static_cast<std::byte*>(expected_data) + second.byte_offset());
}

// REQ: TENSOR-STORAGE-010
TEST(TensorAccessAuthority, ReadLeaseRetainsBackingAfterTensorHandleRetires) {
  // Ownership diagram (live_handle_count at each step):
  //
  //   Tensor (count=1) ──acquire_read_lease()──► TensorReadLease (count=2)
  //       │
  //       └── scope exit ──► count=1, TestStorage NOT destroyed (lease still holds a claim)
  //
  //   retained_lease cleared ──► count=0 ──► ~TestStorage(), destroyed_counter becomes 1
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  std::shared_ptr<TestStorage> storage;
  detail::TensorReadLease retained_lease;
  {
    void* expected_data = nullptr;
    auto context = make_test_context(destroyed_counter, &expected_data);
    storage = std::static_pointer_cast<TestStorage>(context->memory_reference);
    const Tensor tensor(std::move(context));
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 1);
    auto lease = detail::TensorAccess::acquire_read_lease(tensor);
    ASSERT_TRUE(lease) << lease.error().message;
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);
    retained_lease = std::move(lease.value());
  }
  EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  retained_lease = detail::TensorReadLease{};
  EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 0);
  storage.reset();
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// REQ: TENSOR-CORE-015
TEST(TensorTypedAccess, RequiresExactScalarDtypeAndAlignment) {
  // Key setup: data_as<T>() and mutable_data_as<T>() return non-null only when T's bit-width and
  // DLDataType code match exactly and the data pointer is suitably aligned; lanes != 1 rejects
  // even when bits match. The lambda expect_exact mutates context->dtype in-place to test each
  // scalar type without rebuilding the context.
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.byte_offset = 0;
  Tensor tensor(context);

  EXPECT_EQ(tensor.data_as<std::uint8_t>(), expected_data);
  EXPECT_EQ(tensor.data_as<std::uint16_t>(), nullptr);
  auto mutable_bytes = tensor.mutable_data_as<std::uint8_t>();
  ASSERT_TRUE(mutable_bytes) << mutable_bytes.error().message;
  EXPECT_EQ(mutable_bytes.value(), expected_data);

  const auto expect_exact = [&]<typename Value>(DLDataType dtype) {
    context->tensor.dl_tensor.dtype = dtype;
    EXPECT_EQ(tensor.data_as<Value>(), static_cast<const Value*>(expected_data));
    auto mutable_value = tensor.mutable_data_as<Value>();
    ASSERT_TRUE(mutable_value) << mutable_value.error().message;
    EXPECT_EQ(mutable_value.value(), static_cast<Value*>(expected_data));
  };  // NOLINT(readability/braces)
  expect_exact.operator()<std::int8_t>(DLDataType{kDLInt, 8, 1});
  expect_exact.operator()<std::uint8_t>(DLDataType{kDLUInt, 8, 1});
  expect_exact.operator()<std::int16_t>(DLDataType{kDLInt, 16, 1});
  expect_exact.operator()<std::uint16_t>(DLDataType{kDLUInt, 16, 1});
  expect_exact.operator()<std::int32_t>(DLDataType{kDLInt, 32, 1});
  expect_exact.operator()<std::uint32_t>(DLDataType{kDLUInt, 32, 1});
  expect_exact.operator()<std::int64_t>(DLDataType{kDLInt, 64, 1});
  expect_exact.operator()<std::uint64_t>(DLDataType{kDLUInt, 64, 1});
  expect_exact.operator()<float>(DLDataType{kDLFloat, 32, 1});
  expect_exact.operator()<double>(DLDataType{kDLFloat, 64, 1});
  expect_exact.operator()<bool>(
      DLDataType{kDLBool, static_cast<std::uint8_t>(sizeof(bool) * 8), 1});
  expect_exact.operator()<std::complex<float>>(DLDataType{kDLComplex, 64, 1});
  expect_exact.operator()<std::complex<double>>(DLDataType{kDLComplex, 128, 1});

  context->tensor.dl_tensor.dtype = DLDataType{kDLInt, 8, 1};
  EXPECT_EQ(tensor.data_as<char>(), nullptr);
  auto unsupported_char = tensor.mutable_data_as<char>();
  ASSERT_FALSE(unsupported_char);
  EXPECT_EQ(unsupported_char.error().code, ErrorCode::kInvalidArgument);

  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 16, 1};
  context->tensor.dl_tensor.data = static_cast<std::uint8_t*>(expected_data) + 1;
  EXPECT_EQ(tensor.data_as<std::uint16_t>(), nullptr);
  auto misaligned = tensor.mutable_data_as<std::uint16_t>();
  ASSERT_FALSE(misaligned);
  EXPECT_EQ(misaligned.error().code, ErrorCode::kInvalidArgument);

  context->tensor.dl_tensor.data = expected_data;
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 2};
  EXPECT_EQ(tensor.data_as<std::uint8_t>(), nullptr);
}

// REQ: TENSOR-CORE-015
TEST(TensorTypedAccess, EmptyNullAndReadOnlyResultsFollowTheExactAccessTable) {
  Tensor empty;
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.data_as<std::uint8_t>(), nullptr);
  auto empty_mutable = empty.mutable_data_as<std::uint8_t>();
  ASSERT_FALSE(empty_mutable);
  EXPECT_EQ(empty_mutable.error().code, ErrorCode::kInvalidArgument);

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.data = nullptr;
  Tensor null_data(context);
  EXPECT_EQ(null_data.data_as<std::uint8_t>(), nullptr);
  auto null_mutable = null_data.mutable_data_as<std::uint8_t>();
  ASSERT_FALSE(null_mutable);
  EXPECT_EQ(null_mutable.error().code, ErrorCode::kInvalidArgument);

  context->tensor.dl_tensor.data = expected_data;
  Tensor read_only =
      detail::TensorAccess::from_context(std::move(context), detail::TensorAuthority::kReadOnly);
  EXPECT_EQ(read_only.data_as<std::uint8_t>(),
            static_cast<std::uint8_t*>(expected_data) + read_only.byte_offset());
  auto denied = read_only.mutable_data_as<std::uint8_t>();
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, ErrorCode::kPermissionDenied);
}

// REQ: TENSOR-CORE-015
TEST(TensorInternalAccess, SnapshotIsNoThrowAndRejectsMalformedOrOverflowingMetadata) {
  // Key setup: snapshot() must be noexcept; it computes byte_size (elements * itemsize) and
  // byte_span (the minimal contiguous region covering all stride extents). The shared context is
  // mutated in-place between calls to probe multiple stride/shape combinations without rebuilding.
  // Key assertion: overlapping strides, zero strides, and shape*stride arithmetic overflow all
  // produce valid=false while keeping initialized=true; integer-max shape overflows also rejected.
  static_assert(noexcept(detail::TensorAccess::snapshot(std::declval<const Tensor&>())));
  const Tensor empty;
  const auto empty_snapshot = detail::TensorAccess::snapshot(empty);
  EXPECT_FALSE(empty_snapshot.initialized);
  EXPECT_FALSE(empty_snapshot.valid);

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor(context);
  const auto valid_snapshot = detail::TensorAccess::snapshot(tensor);
  EXPECT_TRUE(valid_snapshot.initialized);
  EXPECT_TRUE(valid_snapshot.valid);
  EXPECT_EQ(valid_snapshot.data, expected_data);
  EXPECT_EQ(valid_snapshot.byte_offset, 4U);
  EXPECT_EQ(valid_snapshot.byte_size, 6U);
  EXPECT_EQ(valid_snapshot.byte_span, 6U);
  EXPECT_EQ(valid_snapshot.device.device_type, kDLCPU);

  context->dl_strides = {4, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  const auto sparse_snapshot = detail::TensorAccess::snapshot(tensor);
  EXPECT_TRUE(sparse_snapshot.valid);
  EXPECT_EQ(sparse_snapshot.byte_size, 6U);
  EXPECT_EQ(sparse_snapshot.byte_span, 7U);

  context->dl_strides = {1, 2};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  const auto transposed_snapshot = detail::TensorAccess::snapshot(tensor);
  EXPECT_TRUE(transposed_snapshot.valid);
  EXPECT_EQ(transposed_snapshot.byte_size, 6U);
  EXPECT_EQ(transposed_snapshot.byte_span, 6U);

  context->dl_shape = {2, 2};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->dl_strides = {1, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  EXPECT_FALSE(detail::TensorAccess::snapshot(tensor).valid);

  context->dl_strides = {0, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  EXPECT_FALSE(detail::TensorAccess::snapshot(tensor).valid);

  context->dl_shape = {0, 2};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->dl_strides = {2, -1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  EXPECT_FALSE(detail::TensorAccess::snapshot(tensor).valid);

  context->dl_strides = {-1, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  EXPECT_FALSE(detail::TensorAccess::snapshot(tensor).valid);

  context->dl_shape = {std::numeric_limits<std::int64_t>::max(), 3};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->dl_strides = {3, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  const auto overflow_snapshot = detail::TensorAccess::snapshot(tensor);
  EXPECT_TRUE(overflow_snapshot.initialized);
  EXPECT_FALSE(overflow_snapshot.valid);
}

TEST(TensorMetadataViews, TensorMatchesRequiresEveryDescriptorViewField) {
  static_assert(noexcept(
      tensor_matches(std::declval<const Tensor&>(), std::declval<const TensorViewMetadata&>())));
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor(context);
  const TensorViewMetadata matching{
      .rank = 2U,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .device = DLDevice{kDLCPU, 0},
      .has_explicit_byte_strides = true,
      .byte_size = 6U,
      .shape = {2, 3},
      .byte_strides = {3, 1},
  };

  EXPECT_TRUE(tensor_matches(tensor, matching));
  EXPECT_FALSE(tensor_matches(Tensor{}, matching));

  auto mismatched = matching;
  mismatched.rank = 1U;
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.dtype = DLDataType{kDLInt, 8U, 1U};
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.device = DLDevice{kDLCUDA, 0};
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.byte_size = 7U;
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.shape[1] = 4;
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.byte_strides[0] = 4;
  EXPECT_FALSE(tensor_matches(tensor, mismatched));

  mismatched = matching;
  mismatched.has_explicit_byte_strides = false;
  EXPECT_FALSE(tensor_matches(tensor, mismatched));
}

TEST(TensorMetadataViews, IsContiguousDoesNotAllocate) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;

  auto explicit_contiguous_context = make_test_context(destroyed_counter, &expected_data);
  const Tensor explicit_contiguous(explicit_contiguous_context);

  auto implicit_contiguous_context = make_test_context(destroyed_counter, &expected_data);
  implicit_contiguous_context->dl_strides.clear();
  implicit_contiguous_context->tensor.dl_tensor.strides = nullptr;
  const Tensor implicit_contiguous(implicit_contiguous_context);

  auto non_contiguous_context = make_test_context(destroyed_counter, &expected_data);
  non_contiguous_context->dl_strides = {4, 1};
  non_contiguous_context->tensor.dl_tensor.strides = non_contiguous_context->dl_strides.data();
  const Tensor non_contiguous(non_contiguous_context);

  int explicit_true_count = 0;
  int implicit_true_count = 0;
  int non_contiguous_true_count = 0;
  std::size_t allocation_count = 0;
  {
    const ScopedAllocationCounter allocations;
    for (int iteration = 0; iteration < 1024; ++iteration) {
      explicit_true_count += explicit_contiguous.is_contiguous() ? 1 : 0;
      implicit_true_count += implicit_contiguous.is_contiguous() ? 1 : 0;
      non_contiguous_true_count += non_contiguous.is_contiguous() ? 1 : 0;
    }
    allocation_count = allocations.count();
  }

  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(explicit_true_count, 1024);
  EXPECT_EQ(implicit_true_count, 1024);
  EXPECT_EQ(non_contiguous_true_count, 0);

  auto overflow_context = make_test_context(destroyed_counter, &expected_data);
  overflow_context->dl_shape = {std::numeric_limits<std::int64_t>::max(), 2};
  overflow_context->dl_strides = {2, 1};
  overflow_context->tensor.dl_tensor.shape = overflow_context->dl_shape.data();
  overflow_context->tensor.dl_tensor.strides = overflow_context->dl_strides.data();
  EXPECT_FALSE(Tensor(overflow_context).is_contiguous());
}

TEST(TensorHandleOperations, InternalAccessExposesConstContextReference) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);

  const Tensor tensor(context);

  static_assert(std::is_same_v<decltype(detail::TensorAccess::context(tensor)),
                               const std::shared_ptr<DLManagedTensorContext>&>);
  ASSERT_NE(detail::TensorAccess::context(tensor), nullptr);
  expect_test_tensor_metadata(detail::TensorAccess::context(tensor)->tensor.dl_tensor,
                              expected_data);
}

TEST(TensorDLPackImport, UnversionedImportOwnsMetadata) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackTensor(destroyed_counter);
  void* expected_data = external->tensor.dl_tensor.data;
  auto* source_state = external;

  {
    const Tensor tensor(&external->tensor);
    source_state->shape = {9, 9};
    source_state->strides = {9, 1};

    expect_test_tensor_metadata(detail::TensorAccess::context(tensor)->tensor.dl_tensor,
                                expected_data);
    EXPECT_EQ(tensor.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(tensor.strides(), std::vector<int64_t>({3, 1}));
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

TEST(TensorDLPackImport, VersionedImportOwnsMetadata) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);
  void* expected_data = external->tensor.dl_tensor.data;
  auto* source_state = external;

  {
    const Tensor tensor(&external->tensor);
    source_state->shape = {9, 9};
    source_state->strides = {9, 1};

    expect_test_tensor_metadata(detail::TensorAccess::context(tensor)->tensor.dl_tensor,
                                expected_data);
    EXPECT_EQ(tensor.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(tensor.strides(), std::vector<int64_t>({3, 1}));
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

TEST(TensorDLPackExport, UnversionedExportOwnsMetadataAfterTensorDestruction) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  DLManagedTensor* exported = nullptr;

  {
    auto context = make_test_context(destroyed_counter, &expected_data);
    const Tensor tensor(context);
    auto result = tensor.to_dlpack();
    ASSERT_TRUE(result) << result.error().message;
    exported = result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  expect_test_tensor_metadata(exported->dl_tensor, expected_data);

  exported->deleter(exported);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

TEST(TensorDLPackExport, VersionedExportOwnsMetadataAfterTensorDestruction) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  DLManagedTensorVersioned* exported = nullptr;

  {
    auto context = make_test_context(destroyed_counter, &expected_data);
    const Tensor tensor(context);
    auto result = tensor.to_dlpack_versioned();
    ASSERT_TRUE(result) << result.error().message;
    exported = result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  EXPECT_EQ(exported->version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
  EXPECT_EQ(exported->version.minor, HOLOSCAN_DLPACK_IMPL_VERSION_MINOR);
  EXPECT_EQ(exported->flags, 0U);
  expect_test_tensor_metadata(exported->dl_tensor, expected_data);

  exported->deleter(exported);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// REQ: TENSOR-CORE-016
TEST(TensorDLPackExport, ExportRetainsAndReleasesBackingHandleClaim) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);
  DLManagedTensorVersioned* exported = nullptr;

  {
    const Tensor tensor(context);
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 1);
    auto result = tensor.to_dlpack_versioned();
    ASSERT_TRUE(result) << result.error().message;
    exported = result.value();
    EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 2);
  }
  EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 1);
  exported->deleter(exported);
  EXPECT_EQ(storage->live_handle_count.load(std::memory_order_relaxed), 0);
}

// Mutable DLPack handoff is a dedicated writer claim. It must exclude runtime read leases until
// the external owner invokes the DLPack deleter.
// REQ: TENSOR-CORE-016
TEST(TensorDLPackExport, MutableExportsExcludeReadLeasesUntilConsumerRelease) {
  // Key setup: both legacy to_dlpack() and versioned to_dlpack_versioned() act as exclusive writer
  // claims while the external handle is live; acquire_read_lease() must be denied in that window.
  // Key assertion: after each deleter fires, acquire_read_lease() succeeds again (claim released).
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor(context);

  auto legacy = tensor.to_dlpack();
  ASSERT_TRUE(legacy) << legacy.error().message;
  auto legacy_denied = detail::TensorAccess::acquire_read_lease(tensor);
  ASSERT_FALSE(legacy_denied);
  EXPECT_EQ(legacy_denied.error().code, ErrorCode::kPermissionDenied);
  (*legacy)->deleter(*legacy);

  auto after_legacy = detail::TensorAccess::acquire_read_lease(tensor);
  ASSERT_TRUE(after_legacy) << after_legacy.error().message;
  detail::TensorReadLease released_legacy_lease = std::move(after_legacy.value());
  released_legacy_lease = detail::TensorReadLease{};

  auto versioned = tensor.to_dlpack_versioned();
  ASSERT_TRUE(versioned) << versioned.error().message;
  EXPECT_EQ((*versioned)->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0U);
  auto versioned_denied = detail::TensorAccess::acquire_read_lease(tensor);
  ASSERT_FALSE(versioned_denied);
  EXPECT_EQ(versioned_denied.error().code, ErrorCode::kPermissionDenied);
  (*versioned)->deleter(*versioned);

  auto after_versioned = detail::TensorAccess::acquire_read_lease(tensor);
  ASSERT_TRUE(after_versioned) << after_versioned.error().message;
}

// A versioned export formed while a read lease is active owns its own read claim. Releasing the
// activation lease therefore cannot expose writable legacy authority while the external read-only
// owner still exists.
// REQ: TENSOR-CORE-016
TEST(TensorDLPackExport, ReadOnlyVersionedExportRetainsReadClaimUntilConsumerRelease) {
  // Ownership diagram (read-claim count at each step):
  //
  //   TensorReadLease (claims=1) ──to_dlpack_versioned()──► READ_ONLY export (claims=2)
  //       │
  //       └── lease cleared ──► claims=1, legacy to_dlpack() still denied (export holds claim)
  //
  //   exported->deleter() ──► claims=0 ──► legacy to_dlpack() now succeeds
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor(context);

  auto lease = detail::TensorAccess::acquire_read_lease(tensor);
  ASSERT_TRUE(lease) << lease.error().message;
  auto versioned = tensor.to_dlpack_versioned();
  ASSERT_TRUE(versioned) << versioned.error().message;
  EXPECT_NE((*versioned)->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0U);

  lease = detail::TensorReadLease{};
  auto legacy_denied = tensor.to_dlpack();
  ASSERT_FALSE(legacy_denied);
  EXPECT_EQ(legacy_denied.error().code, ErrorCode::kNotSupported);

  (*versioned)->deleter(*versioned);
  auto legacy = tensor.to_dlpack();
  ASSERT_TRUE(legacy) << legacy.error().message;
  (*legacy)->deleter(*legacy);
}

// REQ: TENSOR-CORE-016
TEST(TensorDLPackExport, LegacyExportRejectsReadOnlyAuthority) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  const Tensor tensor =
      detail::TensorAccess::from_context(std::move(context), detail::TensorAuthority::kReadOnly);

  auto result = tensor.to_dlpack();

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kNotSupported);
}

// REQ: TENSOR-CORE-016
TEST(TensorDLPackExport, EmptyTensorReturnsCheckedInvalidArgument) {
  const Tensor tensor;

  auto legacy = tensor.to_dlpack();
  ASSERT_FALSE(legacy);
  EXPECT_EQ(legacy.error().code, ErrorCode::kInvalidArgument);

  auto versioned = tensor.to_dlpack_versioned();
  ASSERT_FALSE(versioned);
  EXPECT_EQ(versioned.error().code, ErrorCode::kInvalidArgument);
}

TEST(TensorDLPackFlags, MutableDataRejectsReadOnlyTensors) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
  Tensor tensor(context);

  EXPECT_TRUE(tensor.is_read_only());
  EXPECT_FALSE(tensor.is_dlpack_copied());
  context->tensor.flags = 0;
  EXPECT_TRUE(tensor.is_read_only());
  auto read_only_pointer = tensor.mutable_data();
  ASSERT_FALSE(read_only_pointer);
  EXPECT_EQ(read_only_pointer.error().code, ErrorCode::kPermissionDenied);
  EXPECT_EQ(tensor.data(), static_cast<std::byte*>(expected_data) + tensor.byte_offset());
}

// REQ: TENSOR-CORE-015
TEST(TensorTypedAccess, DataUsesTheLogicalByteOffsetAndRejectsAddressOverflow) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* base = nullptr;
  auto context = make_test_context(destroyed_counter, &base);
  Tensor tensor(context);

  EXPECT_EQ(tensor.data(), static_cast<std::byte*>(base) + 4U);
  EXPECT_EQ(tensor.data_as<std::uint8_t>(), static_cast<std::uint8_t*>(base) + 4U);
  auto mutable_pointer = tensor.mutable_data_as<std::uint8_t>();
  ASSERT_TRUE(mutable_pointer);
  EXPECT_EQ(*mutable_pointer, static_cast<std::uint8_t*>(base) + 4U);

  context->tensor.dl_tensor.data =
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1U);
  context->tensor.dl_tensor.byte_offset = 2U;
  EXPECT_EQ(tensor.data(), nullptr);
  auto rejected = tensor.mutable_data();
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  auto legacy_export = tensor.to_dlpack();
  ASSERT_FALSE(legacy_export);
  EXPECT_EQ(legacy_export.error().code, ErrorCode::kInvalidArgument);
  auto versioned_export = tensor.to_dlpack_versioned();
  ASSERT_FALSE(versioned_export);
  EXPECT_EQ(versioned_export.error().code, ErrorCode::kInvalidArgument);
}

TEST(TensorDLPackExport, VersionedExportMaterializesImplicitStridesForDLPack12AndNewer) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;

  auto context = make_test_context(destroyed_counter, &expected_data);
  context->dl_strides.clear();
  context->tensor.dl_tensor.strides = nullptr;
  const Tensor tensor(context);
  EXPECT_TRUE(tensor.dlpack_strides_span().empty());

  auto export_result = tensor.to_dlpack_versioned();
  ASSERT_TRUE(export_result) << export_result.error().message;
  DLManagedTensorVersioned* exported = export_result.value();
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(exported->deleter, nullptr);
  EXPECT_EQ(exported->version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
  EXPECT_EQ(exported->version.minor, HOLOSCAN_DLPACK_IMPL_VERSION_MINOR);
  EXPECT_EQ(exported->dl_tensor.data, expected_data);
  ASSERT_NE(exported->dl_tensor.shape, nullptr);
  EXPECT_EQ(exported->dl_tensor.shape[0], 2);
  EXPECT_EQ(exported->dl_tensor.shape[1], 3);

  constexpr bool kRequiresExplicitVersionedStrides =
      HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR > 1 ||
      (HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR == 1 && HOLOSCAN_DLPACK_IMPL_VERSION_MINOR >= 2);
  if constexpr (kRequiresExplicitVersionedStrides) {
    ASSERT_NE(exported->dl_tensor.strides, nullptr);
    EXPECT_EQ(exported->dl_tensor.strides[0], 3);
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);
  } else {
    EXPECT_EQ(exported->dl_tensor.strides, nullptr);
  }

  exported->deleter(exported);
}

// The expected-returning factories surface invalid producer handles instead of dereferencing them.
TEST(TensorDLPackImport, UnversionedFactoryRejectsNull) {
  auto maybe_tensor = make_tensor_from_dlpack(nullptr);

  EXPECT_FALSE(maybe_tensor);
  EXPECT_EQ(maybe_tensor.error().code, ErrorCode::kInvalidArgument);
}

TEST(TensorDLPackImport, VersionedFactoryRejectsNull) {
  auto maybe_tensor = make_tensor_from_dlpack_versioned(nullptr);

  EXPECT_FALSE(maybe_tensor);
  EXPECT_EQ(maybe_tensor.error().code, ErrorCode::kInvalidArgument);
}

// DLPack requires consumers to reject newer major versions because the versioned struct layout may
// have changed; rejection leaves ownership with the producer.
TEST(TensorDLPackImport, VersionedFactoryRejectsFutureMajorWithoutTakingOwnership) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);
  external->tensor.version.major = HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR + 1;

  auto maybe_tensor = make_tensor_from_dlpack_versioned(&external->tensor);

  EXPECT_FALSE(maybe_tensor);
  EXPECT_EQ(maybe_tensor.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  delete external;
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Raw-pointer constructors are retained for compatibility, but invalid imports now throw
// RuntimeError instead of crashing, and a rejected import leaves ownership with the producer.
TEST(TensorDLPackImport, RawPointerConstructorsThrowForInvalidImports) {
  EXPECT_THROW({ const Tensor tensor(static_cast<DLManagedTensor*>(nullptr)); }, RuntimeError);
  EXPECT_THROW(
      { const Tensor tensor(static_cast<DLManagedTensorVersioned*>(nullptr)); }, RuntimeError);

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);
  external->tensor.version.major = HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR + 1;

  EXPECT_THROW({ const Tensor tensor(&external->tensor); }, RuntimeError);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);

  delete external;
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Factory import follows the same metadata-ownership behavior as the constructor while exposing
// validation failures through expected<Tensor, Error>.
TEST(TensorDLPackImport, UnversionedFactoryOwnsMetadata) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackTensor(destroyed_counter);
  void* expected_data = external->tensor.dl_tensor.data;
  auto* source_state = external;

  {
    auto maybe_tensor = make_tensor_from_dlpack(&external->tensor);
    ASSERT_TRUE(maybe_tensor) << maybe_tensor.error().message;
    const Tensor tensor = std::move(maybe_tensor.value());
    source_state->shape = {9, 9};
    source_state->strides = {9, 1};

    const auto& imported_context = detail::TensorAccess::context(tensor);
    ASSERT_NE(imported_context->memory_reference, nullptr);
    expect_test_tensor_metadata(imported_context->tensor.dl_tensor, expected_data);
    EXPECT_EQ(imported_context->tensor.version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
    EXPECT_EQ(imported_context->tensor.version.minor, HOLOSCAN_DLPACK_IMPL_VERSION_MINOR);
    EXPECT_EQ(imported_context->tensor.flags & DLPACK_FLAG_BITMASK_IS_COPIED, 0U);
    EXPECT_EQ(tensor.dlpack_flags(), DLPACK_FLAG_BITMASK_READ_ONLY);
    EXPECT_TRUE(tensor.is_read_only());
    EXPECT_FALSE(tensor.is_dlpack_copied());
    EXPECT_EQ(tensor.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(tensor.strides(), std::vector<int64_t>({3, 1}));
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Versioned factory import validates the major version before taking ownership; current
// minor-version metadata imports through Tensor's versioned internal representation.
TEST(TensorDLPackImport, VersionedFactoryOwnsMetadata) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);
  void* expected_data = external->tensor.dl_tensor.data;
  auto* source_state = external;

  {
    auto maybe_tensor = make_tensor_from_dlpack_versioned(&external->tensor);
    ASSERT_TRUE(maybe_tensor) << maybe_tensor.error().message;
    const Tensor tensor = std::move(maybe_tensor.value());
    source_state->shape = {9, 9};
    source_state->strides = {9, 1};

    const auto& imported_context = detail::TensorAccess::context(tensor);
    ASSERT_NE(imported_context->memory_reference, nullptr);
    expect_test_tensor_metadata(imported_context->tensor.dl_tensor, expected_data);
    EXPECT_EQ(imported_context->tensor.version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
    EXPECT_EQ(imported_context->tensor.version.minor, HOLOSCAN_DLPACK_IMPL_VERSION_MINOR);
    EXPECT_EQ(imported_context->tensor.flags & DLPACK_FLAG_BITMASK_IS_COPIED, 0U);
    EXPECT_EQ(tensor.dlpack_flags(), DLPACK_FLAG_BITMASK_READ_ONLY);
    EXPECT_TRUE(tensor.is_read_only());
    EXPECT_FALSE(tensor.is_dlpack_copied());
    EXPECT_EQ(tensor.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(tensor.strides(), std::vector<int64_t>({3, 1}));
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Versioned import must not silently drop version-only metadata. The READ_ONLY and IS_COPIED flags
// in particular must survive import and re-export so consumers do not treat immutable or
// already-copied memory incorrectly.
// REQ: TENSOR-CORE-016
// REQ: TENSOR-CORE-017
TEST(TensorDLPackImport, VersionedFactoryPreservesVersionAndFlagsThroughExport) {
  // Key setup: external tensor carries version.minor=0, READ_ONLY|IS_COPIED flags, null strides.
  // Tensor must preserve all three pieces of metadata through an import -> export round-trip.
  // Key assertion: re-exported flags and stored minor version match the originals; null strides are
  // materialized on export to avoid advertising a protocol version with an invalid stride pointer.
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);
  external->tensor.version.minor = 0;
  external->tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED;
  external->tensor.dl_tensor.strides = nullptr;
  void* expected_data = external->tensor.dl_tensor.data;

  DLManagedTensorVersioned* exported = nullptr;
  {
    auto maybe_tensor = make_tensor_from_dlpack_versioned(&external->tensor);
    ASSERT_TRUE(maybe_tensor) << maybe_tensor.error().message;
    Tensor tensor = std::move(maybe_tensor.value());

    const auto& imported_context = detail::TensorAccess::context(tensor);
    ASSERT_NE(imported_context, nullptr);
    EXPECT_EQ(imported_context->tensor.version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
    EXPECT_EQ(imported_context->tensor.version.minor, 0U);
    EXPECT_EQ(imported_context->tensor.flags,
              DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED);
    EXPECT_EQ(tensor.dlpack_flags(), DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED);
    EXPECT_TRUE(tensor.is_read_only());
    EXPECT_TRUE(tensor.is_dlpack_copied());
    auto mutable_pointer = tensor.mutable_data();
    ASSERT_FALSE(mutable_pointer);
    EXPECT_EQ(mutable_pointer.error().code, ErrorCode::kPermissionDenied);

    auto result = tensor.to_dlpack_versioned();
    ASSERT_TRUE(result) << result.error().message;
    exported = result.value();
    ASSERT_NE(exported, nullptr);
    ASSERT_NE(exported->deleter, nullptr);
    EXPECT_EQ(exported->version.major, HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR);
    EXPECT_EQ(exported->version.minor, 0U);
    EXPECT_EQ(exported->flags, DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED);
    // Explicit strides remain valid for this older stored version and keep a later protocol layer
    // from advertising DLPack 1.2+ while retaining an invalid null stride pointer.
    ASSERT_NE(exported->dl_tensor.strides, nullptr);
    expect_test_tensor_metadata(exported->dl_tensor, expected_data);
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  exported->deleter(exported);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Imported DLPack tensors own the producer deleter but do not know which CUDA stream made the data
// ready, so consumer-stream synchronization reports the missing readiness contract instead of
// succeeding as a no-op.
TEST(TensorDLPackImport, UnversionedImportReportsMissingReadinessContract) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackTensor(destroyed_counter);

  {
    auto maybe_tensor = make_tensor_from_dlpack(&external->tensor);
    ASSERT_TRUE(maybe_tensor) << maybe_tensor.error().message;
    Tensor tensor = std::move(maybe_tensor.value());

    auto* stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1010});
    EXPECT_FALSE(tensor.has_readiness_stream_contract());
    EXPECT_FALSE(tensor.set_readiness_stream(stream));

    auto result = tensor.synchronize_for_consumer_stream(stream);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kNotImplemented);
    EXPECT_NE(std::string(result.error().message).find("producer-readiness"), std::string::npos);
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

TEST(TensorDLPackImport, VersionedImportReportsMissingReadinessContract) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto* external = new ExternalDLPackVersionedTensor(destroyed_counter);

  {
    auto maybe_tensor = make_tensor_from_dlpack_versioned(&external->tensor);
    ASSERT_TRUE(maybe_tensor) << maybe_tensor.error().message;
    Tensor tensor = std::move(maybe_tensor.value());

    auto* stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x2020});
    EXPECT_FALSE(tensor.has_readiness_stream_contract());
    EXPECT_FALSE(tensor.set_readiness_stream(stream));

    auto result = tensor.synchronize_for_consumer_stream(stream);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kNotImplemented);
    EXPECT_NE(std::string(result.error().message).find("__dlpack__(stream=...)"),
              std::string::npos);
    EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 1);
}

// Tensor forwards stream-aware deallocation to its native memory reference, which controls when an
// allocator may reuse the backing memory.
TEST(TensorMemoryReference, SetDeallocationStreamUsesNativeMemoryReference) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);
  storage->accept_deallocation_stream = true;

  Tensor tensor(context);
  auto* stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
  EXPECT_TRUE(tensor.set_deallocation_stream(stream));
  EXPECT_EQ(storage->deallocation_stream_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(storage->last_deallocation_stream, stream);
}

TEST(TensorMemoryReference, MakeTensorUsesTrustedDeviceOrdinal) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  auto storage = std::make_shared<TestStorage>(destroyed_counter);
  const std::array<int64_t, 1> shape{static_cast<int64_t>(storage->data.size())};
  constexpr std::int32_t kDeviceId = 7;

  auto tensor = make_tensor(shape,
                            DLDataType{kDLUInt, 8, 1},
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            storage->data.data(),
                            storage,
                            kDeviceId);

  ASSERT_TRUE(tensor) << runtime_error_message(tensor.error());
  EXPECT_EQ(tensor->data(), storage->data.data());
  EXPECT_EQ(tensor->device().device_type, kDLCUDA);
  EXPECT_EQ(tensor->device().device_id, kDeviceId);
  EXPECT_EQ(destroyed_counter->load(std::memory_order_relaxed), 0);
}

// The producer-readiness stream is distinct from the deallocation stream: DLPack consumers use it
// to wait for data to become readable, but allocators should not use it to decide reuse timing.
TEST(TensorMemoryReference, SetReadinessStreamUsesNativeMemoryReference) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);
  storage->accept_readiness_stream = true;

  Tensor tensor(context);
  auto* stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x2345});
  EXPECT_TRUE(tensor.set_readiness_stream(stream));
  EXPECT_EQ(storage->readiness_stream_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(storage->last_readiness_stream, stream);
  EXPECT_EQ(storage->deallocation_stream_count.load(std::memory_order_relaxed), 0);
}

// DLPack stream semantics require the producer to make the consumer-provided stream wait for the
// work that made the Tensor data ready; Tensor forwards that request to the memory reference.
TEST(TensorMemoryReference, SynchronizeForConsumerStreamUsesNativeMemoryReference) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);

  const Tensor tensor(context);
  auto* stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x4321});
  EXPECT_TRUE(tensor.has_readiness_stream_contract());
  auto result = tensor.synchronize_for_consumer_stream(stream);

  EXPECT_TRUE(result) << result.error().message;
  EXPECT_EQ(storage->consumer_stream_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(storage->last_consumer_stream, stream);
}

// CPU consumers need an explicit host-side readiness wait; Tensor forwards the request to the
// memory reference so synchronization remains tied to the actual buffer owner.
TEST(TensorMemoryReference, SynchronizeForHostUsesNativeMemoryReference) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);

  const Tensor tensor(context);
  auto result = tensor.synchronize_for_host();

  EXPECT_TRUE(result) << result.error().message;
  EXPECT_EQ(storage->host_sync_count.load(std::memory_order_relaxed), 1);
}

// Blocking host materialization should use caller-provided storage, wait for producer readiness,
// and avoid hiding an allocation inside Tensor. make_test_context uses byte_offset 4 for metadata
// coverage; the copy reads from data + byte_offset, so this case uses an offset of 0 to copy the
// whole 6-byte buffer in bounds.
TEST(TensorMemoryReference, CopyToHostCopiesCpuTensorAndSynchronizes) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.byte_offset = 0;
  auto storage = std::static_pointer_cast<TestStorage>(context->memory_reference);

  const Tensor tensor(context);
  std::array<uint8_t, 6> destination{};
  auto result = tensor.copy_to_host(destination.data(), destination.size());

  EXPECT_TRUE(result) << result.error().message;
  EXPECT_TRUE(std::equal(destination.begin(), destination.end(), storage->data.begin()));
  EXPECT_EQ(storage->host_sync_count.load(std::memory_order_relaxed), 1);
}

// The low-level copy helper intentionally supports only contiguous Tensor layouts; strided
// materialization belongs in a higher-level buffer policy that can define layout conversion.
TEST(TensorMemoryReference, CopyToHostRejectsNonContiguousTensor) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->dl_strides = {4, 1};
  context->tensor.dl_tensor.strides = context->dl_strides.data();

  const Tensor tensor(context);
  std::array<uint8_t, 6> destination{};
  auto result = tensor.copy_to_host(destination.data(), destination.size());

  EXPECT_FALSE(result);
  EXPECT_NE(std::string(result.error().message).find("contiguous"), std::string::npos);
}

// DLPack strides are in elements, so contiguity and byte-stride reporting must account for the full
// (lanes-aware) element size of vectorized dtypes rather than the per-lane bit width.
TEST(TensorMetadataViews, IsContiguousUsesElementStridesForVectorizedDTypes) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;

  auto contiguous_context = make_test_context(destroyed_counter, &expected_data);
  contiguous_context->tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 4};
  const Tensor contiguous(contiguous_context);

  EXPECT_TRUE(contiguous.is_contiguous());
  EXPECT_EQ(contiguous.itemsize(), 16);
  EXPECT_EQ(contiguous.strides(), (std::vector<int64_t>{48, 16}));

  auto non_contiguous_context = make_test_context(destroyed_counter, &expected_data);
  non_contiguous_context->tensor.dl_tensor.dtype = DLDataType{kDLFloat, 32, 4};
  non_contiguous_context->dl_strides = {4, 1};
  non_contiguous_context->tensor.dl_tensor.strides = non_contiguous_context->dl_strides.data();
  const Tensor non_contiguous(non_contiguous_context);

  EXPECT_FALSE(non_contiguous.is_contiguous());
  EXPECT_EQ(non_contiguous.strides(), (std::vector<int64_t>{64, 16}));
}

// Scalar tensors use the DLPack convention `ndim == 0` and contain exactly one element even though
// there is no shape array to expose.
TEST(TensorMetadataViews, ScalarTensorHasOneElementAndEmptyShapeViews) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->dl_shape.clear();
  context->dl_strides.clear();
  context->tensor.dl_tensor.ndim = 0;
  context->tensor.dl_tensor.shape = nullptr;
  context->tensor.dl_tensor.strides = nullptr;
  const Tensor tensor(context);

  EXPECT_EQ(tensor.size(), 1);
  EXPECT_EQ(tensor.nbytes(), tensor.itemsize());
  EXPECT_TRUE(tensor.shape().empty());
  EXPECT_TRUE(tensor.shape_span().empty());
  EXPECT_TRUE(tensor.strides().empty());
  EXPECT_TRUE(tensor.dlpack_strides_span().empty());
  EXPECT_TRUE(tensor.is_contiguous());
}

// Empty and malformed dtype metadata must remain safe to inspect and must not fabricate a byte
// count that callers could use to copy beyond the logical Tensor view.
TEST(TensorMetadataViews, EmptyAndInvalidDtypeMetadataReturnZeroByteSizes) {
  const Tensor empty;
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.itemsize(), 0U);
  EXPECT_EQ(empty.nbytes(), 0);

  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.dtype.bits = 0U;
  const Tensor zero_bits(context);
  EXPECT_EQ(zero_bits.size(), 6);
  EXPECT_EQ(zero_bits.itemsize(), 0U);
  EXPECT_EQ(zero_bits.nbytes(), 0);

  context->tensor.dl_tensor.dtype.bits = 8U;
  context->tensor.dl_tensor.dtype.lanes = 0U;
  const Tensor zero_lanes(context);
  EXPECT_EQ(zero_lanes.size(), 6);
  EXPECT_EQ(zero_lanes.itemsize(), 0U);
  EXPECT_EQ(zero_lanes.nbytes(), 0);
}

// Vectorized dtypes may occupy more than 255 bytes per logical element; itemsize must preserve the
// full DLPack bit-lane product and nbytes must use that untruncated value.
TEST(TensorMetadataViews, WideVectorLaneItemSizeIsNotTruncated) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8, 1024};
  const Tensor tensor(context);

  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.itemsize(), 1024U);
  EXPECT_EQ(tensor.nbytes(), 6144);
}

// Shape and byte-count overflow are malformed metadata, not an invitation to signed integer
// overflow. Both queries return the documented deterministic zero sentinel.
TEST(TensorMetadataViews, ShapeAndByteCountOverflowReturnZero) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;

  auto shape_overflow_context = make_test_context(destroyed_counter, &expected_data);
  shape_overflow_context->dl_shape = {std::numeric_limits<int64_t>::max(), 2};
  shape_overflow_context->tensor.dl_tensor.shape = shape_overflow_context->dl_shape.data();
  const Tensor shape_overflow(shape_overflow_context);
  EXPECT_EQ(shape_overflow.size(), 0);
  EXPECT_EQ(shape_overflow.nbytes(), 0);

  auto byte_overflow_context = make_test_context(destroyed_counter, &expected_data);
  byte_overflow_context->dl_shape = {std::numeric_limits<int64_t>::max()};
  byte_overflow_context->tensor.dl_tensor.ndim = 1;
  byte_overflow_context->tensor.dl_tensor.shape = byte_overflow_context->dl_shape.data();
  byte_overflow_context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 16, 1};
  const Tensor byte_overflow(byte_overflow_context);
  EXPECT_EQ(byte_overflow.size(), std::numeric_limits<int64_t>::max());
  EXPECT_EQ(byte_overflow.itemsize(), 2U);
  EXPECT_EQ(byte_overflow.nbytes(), 0);
}

// Malformed negative-rank metadata should not underflow vector sizes or crash metadata accessors.
TEST(TensorMetadataViews, NegativeRankTensorReturnsEmptyMetadataAndZeroSize) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->tensor.dl_tensor.ndim = -1;
  const Tensor tensor(context);

  EXPECT_EQ(tensor.size(), 0);
  EXPECT_TRUE(tensor.shape().empty());
  EXPECT_TRUE(tensor.shape_span().empty());
  EXPECT_TRUE(tensor.strides().empty());
  EXPECT_TRUE(tensor.dlpack_strides_span().empty());
  EXPECT_TRUE(tensor.is_contiguous());
}

// Ranked tensors must provide a shape pointer for contiguity and size queries; missing shape
// metadata is treated as invalid instead of being dereferenced.
TEST(TensorMetadataViews, RankedTensorWithMissingShapeIsNotContiguousAndHasZeroSize) {
  auto destroyed_counter = std::make_shared<std::atomic<int>>(0);
  void* expected_data = nullptr;
  auto context = make_test_context(destroyed_counter, &expected_data);
  context->dl_shape.clear();
  context->tensor.dl_tensor.shape = nullptr;
  const Tensor tensor(context);

  EXPECT_EQ(tensor.size(), 0);
  EXPECT_TRUE(tensor.shape().empty());
  EXPECT_TRUE(tensor.shape_span().empty());
  EXPECT_FALSE(tensor.is_contiguous());
}

}  // namespace holoscan
