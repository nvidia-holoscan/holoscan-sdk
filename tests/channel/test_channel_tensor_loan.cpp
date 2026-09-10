/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <holomq/context_builder.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/pubsub_options.hpp>

#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"
#include "channel/channel_common.hpp"
#include "channel/channel_tensor_loan_internal.hpp"
#include "channel/channel_tensor_test_access.hpp"
#include "channel/channel_tensor_transport_payload.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "channel/holomq_tensor.hpp"
#include "channel_test_support.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"

namespace holoscan {
namespace {

#ifndef HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE ""
#endif

// Named concepts for API-surface verification; each drives a static_assert below confirming
// that ChannelPublisher<Tensor> and ManualTensorPublisher expose exactly their documented verb
// sets (loan/write/commit path vs. bare-emit path) and nothing more.
template <typename Publisher>
concept HasBareTensorEmit =
    requires(Publisher& publisher, Tensor tensor) { publisher.emit(std::move(tensor)); };

template <typename Publisher>
concept HasTensorSampleEmit = requires(Publisher& publisher, Sample<Tensor> sample) {
  publisher.emit_sample(std::move(sample));
};

template <typename Publisher>
concept HasLoanTensor = requires(Publisher& publisher, TensorLoanRequest request) {
  { publisher.loan_tensor(request) } -> std::same_as<expected<TensorPublishLoan, Error>>;
};

template <typename Loan, typename Guard>
concept HasFusedTensorWrite = requires(Loan& loan, ::cuda::stream_ref stream) {
  {
    loan.write_and_commit(stream, [](const Guard&) {})
  } -> std::same_as<expected<void, Error>>;
  {
    loan.write_host_and_commit([](const Guard&) -> expected<void, Error> { return {}; })
  } -> std::same_as<expected<void, Error>>;
};

template <typename Publisher>
concept HasTensorCopyEmit =
    requires(Publisher& publisher, const Tensor& tensor, ::cuda::stream_ref stream) {
      { publisher.emit_copy(tensor, stream) } -> std::same_as<expected<std::uint32_t, Error>>;
    };

template <typename Publisher>
concept HasTensorMemoryOptions =
    requires(const Publisher& publisher) { publisher.tensor_memory_options(); };

template <typename Publisher>
concept HasMakeTensor =
    requires(Publisher& publisher, TensorLoanRequest request) { publisher.make_tensor(request); };

template <typename Publisher>
concept HasClockHealthReport = requires(Publisher& publisher, ClockHealthReport report) {
  publisher.report_clock_health(report);
};

template <typename Publisher>
concept HasExplicitClose = requires(Publisher& publisher) {
  { publisher.close() } -> std::same_as<expected<void, Error>>;
};

template <typename Publisher>
concept HasCaptureTimeFactory =
    requires(const Publisher& publisher) { publisher.make_capture_time(1); };

template <typename Enum>
[[nodiscard]] constexpr Enum unnamed_enum_value(const std::uint8_t value) noexcept {
  static_assert(sizeof(Enum) == sizeof(value));
  return std::bit_cast<Enum>(value);
}

template <typename Runtime>
concept HasCatalogTensorPublisherFactory =
    requires(Runtime& runtime, const ChannelEndpointCatalog& catalog,
             ChannelEndpointCatalog::EndpointView endpoint) {
      {
        runtime.template create_publisher<Tensor>(catalog, "tensor-input")
      } -> std::same_as<expected<ChannelPublisher<Tensor>, Error>>;
      {
        runtime.template create_publisher<Tensor>(endpoint)
      } -> std::same_as<expected<ChannelPublisher<Tensor>, Error>>;
    };

template <typename Runtime>
concept HasCatalogTensorSubscriberFactory =
    requires(Runtime& runtime, const ChannelEndpointCatalog& catalog,
             ChannelEndpointCatalog::EndpointView endpoint) {
      {
        runtime.template create_subscriber<Tensor>(catalog, "tensor-output")
      } -> std::same_as<expected<ChannelSubscriber<Tensor>, Error>>;
      {
        runtime.template create_subscriber<Tensor>(endpoint)
      } -> std::same_as<expected<ChannelSubscriber<Tensor>, Error>>;
    };

// Retains device backing while deliberately providing no producer-readiness contract, matching
// the stream semantics of an imported DLPack ownership wrapper. Representative tests:
// EmitCopyStagesDeviceSourceIntoDeviceCatalogPool and DeviceStagingUsesSeparateOrderedCopyStream.
class ContractlessDeviceTensorStorage final : public TensorMemoryReference {
 public:
  explicit ContractlessDeviceTensorStorage(Tensor backing) : backing_(std::move(backing)) {}

 private:
  Tensor backing_;
};

[[nodiscard]] Tensor make_contractless_device_tensor(const Tensor& backing) {
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = std::make_shared<ContractlessDeviceTensorStorage>(backing);
  context->dl_shape.assign(backing.shape_span().begin(), backing.shape_span().end());
  context->dl_strides.assign(backing.dlpack_strides_span().begin(),
                             backing.dlpack_strides_span().end());
  context->tensor.dl_tensor = DLTensor{
      .data = const_cast<void*>(backing.data()),
      .device = backing.device(),
      .ndim = static_cast<std::int32_t>(context->dl_shape.size()),
      .dtype = backing.dtype(),
      .shape = context->dl_shape.data(),
      .strides = context->dl_strides.empty() ? nullptr : context->dl_strides.data(),
      .byte_offset = 0U,
  };
  return Tensor{std::move(context)};
}

// TA-T01/TA-T02/TA-T45/TA-T46: the external and manual Tensor domains remain mechanically
// distinct, transport opaque, and constrained to their accepted verb sets.
static_assert(std::is_move_constructible_v<TensorPublishLoan>);
static_assert(!std::is_copy_constructible_v<TensorPublishLoan>);
static_assert(!std::is_copy_assignable_v<TensorPublishLoan>);
static_assert(!std::is_move_assignable_v<TensorPublishLoan>);
static_assert(std::is_nothrow_destructible_v<TensorPublishLoan>);
static_assert(std::is_move_constructible_v<TensorLoanWriteGuard>);
static_assert(!std::is_copy_constructible_v<TensorLoanWriteGuard>);
static_assert(!std::is_copy_assignable_v<TensorLoanWriteGuard>);
static_assert(!std::is_move_assignable_v<TensorLoanWriteGuard>);
static_assert(std::is_nothrow_destructible_v<TensorLoanWriteGuard>);
static_assert(!std::is_convertible_v<TensorPublishLoan, Tensor>);
static_assert(!std::is_convertible_v<Tensor, TensorPublishLoan>);
static_assert(!std::is_convertible_v<TensorPublishLoan, TensorOutputLoan>);
static_assert(!std::is_convertible_v<TensorOutputLoan, TensorPublishLoan>);
static_assert(std::is_move_constructible_v<ChannelPublisher<Tensor>>);
static_assert(!std::is_copy_constructible_v<ChannelPublisher<Tensor>>);
static_assert(!std::is_move_assignable_v<ChannelPublisher<Tensor>>);
static_assert(!std::is_default_constructible_v<ChannelPublisher<Tensor>>);
static_assert(noexcept(std::declval<ChannelPublisher<Tensor>&>().update_connections()));
static_assert(noexcept(std::declval<ManualTensorPublisher&>().update_connections()));

// REQ: OPERATOR-026
static_assert(HasLoanTensor<ChannelPublisher<Tensor>>);
// REQ: OPERATOR-027
static_assert(HasFusedTensorWrite<TensorPublishLoan, TensorLoanWriteGuard>);
static_assert(HasTensorCopyEmit<ChannelPublisher<Tensor>>);
static_assert(HasClockHealthReport<ChannelPublisher<Tensor>>);
static_assert(HasExplicitClose<ChannelPublisher<Tensor>>);
static_assert(!HasBareTensorEmit<ChannelPublisher<Tensor>>);
static_assert(!HasTensorSampleEmit<ChannelPublisher<Tensor>>);
static_assert(!HasTensorMemoryOptions<ChannelPublisher<Tensor>>);
static_assert(!HasMakeTensor<ChannelPublisher<Tensor>>);

static_assert(HasBareTensorEmit<ManualTensorPublisher>);
static_assert(HasTensorSampleEmit<ManualTensorPublisher>);
static_assert(!HasLoanTensor<ManualTensorPublisher>);
static_assert(!HasTensorCopyEmit<ManualTensorPublisher>);
static_assert(!HasClockHealthReport<ManualTensorPublisher>);
static_assert(!HasExplicitClose<ManualTensorPublisher>);
static_assert(!HasCaptureTimeFactory<ManualTensorPublisher>);
// REQ: CHAN-081
static_assert(HasCatalogTensorPublisherFactory<ChannelRuntime>);
static_assert(HasCatalogTensorSubscriberFactory<ChannelRuntime>);
static_assert(!CatalogChannelPayload<Tensor>);

// Mutable backing state for a synthetic loan/writer vtable pair used by ExternalTensorLoanSurface
// tests.  Records every transition (writing, committed, cancelled, abandon_count) so tests can
// verify TensorPublishLoan and TensorLoanWriteGuard state-machine contracts without a real
// HoloMQ publisher.  See MoveAndCommitInvalidateBorrowedAccessWithoutOwningTensor.
struct ExternalLoanState {
  std::array<std::int64_t, 2U> shape{2, 4};
  std::array<std::byte, 8U> storage{};
  DLDataType dtype{.code = kDLUInt, .bits = 8, .lanes = 1};
  bool live{true};
  bool writing{};
  bool committed{};
  bool cancelled{};
  std::uint32_t abandon_count{};
};

// TensorLoanWriteGuardOps callbacks: enforce the write-session state machine (open → committed
// or abandoned) and expose the in-place storage pointer for caller writes.  Combined into
// kWriterOps below; vended by begin_host_write to construct a TensorLoanWriteGuard.
void abandon_writer(void* state) noexcept {
  auto& loan = *static_cast<ExternalLoanState*>(state);
  loan.writing = false;
}

void* writer_data(const void* state) noexcept {
  return const_cast<std::byte*>(static_cast<const ExternalLoanState*>(state)->storage.data());
}

std::size_t writer_byte_size(const void* state) noexcept {
  return static_cast<const ExternalLoanState*>(state)->storage.size();
}

std::optional<::cuda::stream_ref> writer_stream(const void*) noexcept {
  return std::nullopt;
}

expected<void, Error> commit_writer(void* state) noexcept {
  auto& loan = *static_cast<ExternalLoanState*>(state);
  if (!loan.live || !loan.writing) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.writing = false;
  loan.committed = true;
  return {};
}

// Writer vtable wired to ExternalLoanState; commit is the only path that sets committed=true.
constexpr channel_internal::TensorLoanWriteGuardOps kWriterOps{
    .abandon = abandon_writer,
    .data = writer_data,
    .byte_size = writer_byte_size,
    .stream = writer_stream,
    .commit = commit_writer,
};

// TensorPublishLoanOps callbacks: expose ExternalLoanState shape metadata and enforce the
// single-active-write and single-cancel state machine required by TensorPublishLoan callers.
// CUDA write is intentionally unsupported (returns kNotSupported) to keep the surface host-only.
void abandon_loan(void* state) noexcept {
  auto& loan = *static_cast<ExternalLoanState*>(state);
  if (loan.live) {
    loan.live = false;
    ++loan.abandon_count;
  }
}

TensorShapeView loan_shape(const void* state) noexcept {
  return static_cast<const ExternalLoanState*>(state)->shape;
}

TensorByteStridesView loan_byte_strides(const void*) noexcept {
  return std::nullopt;
}

DLDataType loan_dtype(const void* state) noexcept {
  return static_cast<const ExternalLoanState*>(state)->dtype;
}

DeviceId loan_device_id(const void*) noexcept {
  return DeviceId{};
}

MemoryKind loan_memory_kind(const void*) noexcept {
  return MemoryKind::kHost;
}

std::size_t loan_byte_size(const void* state) noexcept {
  return static_cast<const ExternalLoanState*>(state)->storage.size();
}

expected<TensorLoanWriteGuard, Error> begin_cuda_write(void*, ::cuda::stream_ref) noexcept {
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<TensorLoanWriteGuard, Error> begin_host_write(void* state) noexcept {
  auto& loan = *static_cast<ExternalLoanState*>(state);
  if (!loan.live || loan.writing || loan.committed) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.writing = true;
  return channel_internal::TensorPublishLoanAccess::write_guard(state, &kWriterOps);
}

expected<void, Error> cancel_loan(void* state) noexcept {
  auto& loan = *static_cast<ExternalLoanState*>(state);
  if (!loan.live || loan.writing) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.live = false;
  loan.cancelled = true;
  return {};
}

// Loan vtable wired to ExternalLoanState; host-only (write always rejects a non-null stream).
constexpr channel_internal::TensorPublishLoanOps kLoanOps{
    .abandon = abandon_loan,
    .shape = loan_shape,
    .byte_strides = loan_byte_strides,
    .dtype = loan_dtype,
    .device_id = loan_device_id,
    .memory_kind = loan_memory_kind,
    .byte_size = loan_byte_size,
    .write = begin_cuda_write,
    .write_host = begin_host_write,
    .cancel = cancel_loan,
};

// TA-T43: every view borrows live loan/guard state and becomes defensively empty after a
// consuming or move operation.
TEST(ExternalTensorLoanSurface, MoveAndCommitInvalidateBorrowedAccessWithoutOwningTensor) {
  ExternalLoanState state;
  auto original = channel_internal::TensorPublishLoanAccess::loan(&state, &kLoanOps);
  ASSERT_EQ(original.shape().size(), 2U);
  EXPECT_EQ(original.byte_size(), state.storage.size());
  EXPECT_EQ(original.memory_kind(), MemoryKind::kHost);

  TensorPublishLoan moved{std::move(original)};
  // Moved-from loans and guards have a documented defensively empty state that this test must
  // inspect directly.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_TRUE(original.shape().empty());
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(original.byte_size(), 0U);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(original.memory_kind(), MemoryKind::kUnknown);

  auto writer_result = moved.write_host();
  ASSERT_TRUE(writer_result);
  TensorLoanWriteGuard writer{std::move(*writer_result)};
  EXPECT_EQ(writer.data(), state.storage.data());
  EXPECT_EQ(writer.byte_size(), state.storage.size());
  EXPECT_FALSE(writer.stream().has_value());

  TensorLoanWriteGuard moved_writer{std::move(writer)};
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(writer.data(), nullptr);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(writer.byte_size(), 0U);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(writer.stream().has_value());
  ASSERT_TRUE(std::move(moved_writer).commit());
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(moved_writer.data(), nullptr);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(moved_writer.byte_size(), 0U);
  EXPECT_TRUE(state.committed);

  ASSERT_TRUE(std::move(moved).cancel());
  EXPECT_TRUE(state.cancelled);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(moved.byte_size(), 0U);
  EXPECT_EQ(state.abandon_count, 0U);
}

TEST(ExternalTensorLoanSurface, DestructionRoutesLiveAuthorityThroughPrivateOpsOnce) {
  ExternalLoanState state;
  {
    auto loan = channel_internal::TensorPublishLoanAccess::loan(&state, &kLoanOps);
    EXPECT_TRUE(channel_internal::TensorPublishLoanAccess::loan_bound_to(loan, &state));
  }
  EXPECT_FALSE(state.live);
  EXPECT_EQ(state.abandon_count, 1U);
}

// Builds a single-endpoint import catalog (producer/publisher side) with a host-memory,
// rank-≤2, uint8 contiguous-stride contract.  All ExternalTensorLoanRuntime publisher tests
// call this; max_loaned_samples sets the fixed pool depth that drives pool-exhaustion coverage.
[[nodiscard]] expected<ChannelEndpointCatalog, Error> make_tensor_import_catalog(
    const std::string& endpoint_name, const std::uint32_t max_loaned_samples = 2U,
    const InterfaceTensorEmitPolicy foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
    const ClockHealthOptions clock_health = {},
    const std::uint32_t service_max_subscriber_depth = 1U,
    const std::uint32_t service_max_subscriber_borrowed = 1U,
    const MemoryKind memory_kind = MemoryKind::kHost,
    const QueuePolicy queue_policy = QueuePolicy::kBlocking) {
  detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = TopicDirection::kImport,
      .name_origin = EndpointNameOrigin::kExplicit,
      .queue_policy = queue_policy,
      .service_max_subscriber_depth = service_max_subscriber_depth,
      .service_max_subscriber_borrowed = service_max_subscriber_borrowed,
      .payload_identity =
          PayloadIdentityRecord{
              .kind = PayloadIdentityKind::kTensor,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .clock_health = clock_health,
      .tensor =
          TensorBoundaryContractRecord{
              .memory_kind = memory_kind,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 2U,
              .max_byte_span = 16U,
              .strides = TensorStridePolicy::kContiguousOnly,
              .foreign_emit_policy = foreign_emit_policy,
              .max_staging_byte_span =
                  foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable
                      ? std::nullopt
                      : std::optional<std::uint64_t>{16U},
          },
      .bounds = EndpointBounds{.publisher = PublisherEndpointBounds{.max_loaned_samples =
                                                                        max_loaned_samples}},
  };
  return detail::ChannelEndpointCatalogAccess::normalize("external-tensor-loan-test",
                                                         {std::move(endpoint)});
}

// Import catalog variant that permits arbitrary row/column byte strides within bounded extents
// (TensorStridePolicy::kBoundedStrided).  Used exclusively by RejectsOverlappingBoundedStrides
// to exercise the stride-aliasing validation path.
[[nodiscard]] expected<ChannelEndpointCatalog, Error> make_strided_tensor_import_catalog(
    const std::string& endpoint_name) {
  detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = TopicDirection::kImport,
      .name_origin = EndpointNameOrigin::kExplicit,
      .queue_policy = QueuePolicy::kBlocking,
      .payload_identity =
          PayloadIdentityRecord{
              .kind = PayloadIdentityKind::kTensor,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .tensor =
          TensorBoundaryContractRecord{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 2U,
              .max_byte_span = 16U,
              .axes = {{.max_extent = 16U, .max_abs_stride_bytes = 16U},
                       {.max_extent = 16U, .max_abs_stride_bytes = 16U}},
              .strides = TensorStridePolicy::kBoundedStrided,
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              .max_staging_byte_span = 16U,
          },
      .bounds = EndpointBounds{.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U}},
  };
  return detail::ChannelEndpointCatalogAccess::normalize("external-strided-tensor-loan-test",
                                                         {std::move(endpoint)});
}

// The source-tree-only subscriber peer must enforce the same finite publisher-loan contract as
// the production catalog publisher instead of repairing a malformed private catalog record.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, CatalogSubscriberRejectsMissingOrZeroLoanBound) {
  auto catalog = make_tensor_import_catalog("tensor-invalid-loan-bound", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  const auto* const canonical_state = detail::ChannelEndpointCatalogAccess::state(*catalog);
  ASSERT_NE(canonical_state, nullptr);
  ASSERT_EQ(canonical_state->endpoints.size(), 1U);
  ASSERT_TRUE(canonical_state->endpoints.front().publisher_request);

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  const auto expect_rejected = [&](const std::optional<std::uint32_t> max_loaned_samples) {
    auto malformed_state = std::make_shared<detail::ChannelEndpointCatalogState>(*canonical_state);
    malformed_state->endpoints.front().publisher_request->max_loaned_samples = max_loaned_samples;
    const ChannelEndpointCatalog malformed_catalog =
        detail::ChannelEndpointCatalogAccess::make(std::move(malformed_state));
    const auto endpoint = malformed_catalog.endpoint("tensor-invalid-loan-bound");
    ASSERT_TRUE(endpoint);
    auto subscriber =
        channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
    ASSERT_FALSE(subscriber);
    EXPECT_EQ(subscriber.error().code, ErrorCode::kInvalidArgument);
  };

  expect_rejected(std::nullopt);
  expect_rejected(0U);
}

// Builds the subscriber-side (export) catalog peer for tests that need a real
// ChannelSubscriber<Tensor>.  Queue depth, memory kind, attach policy, and borrow limit are
// each overridable; defaults pair with a depth-1 import catalog in most round-trip tests.
[[nodiscard]] expected<ChannelEndpointCatalog, Error> make_tensor_export_catalog(
    const std::string& endpoint_name, const std::uint32_t queue_depth = 1U,
    const MemoryKind memory_kind = MemoryKind::kHost, const TopicAttachPolicy attach = {},
    const std::optional<std::uint32_t> max_borrowed_samples = std::nullopt,
    const QueuePolicy queue_policy = QueuePolicy::kBlocking) {
  detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = TopicDirection::kExport,
      .name_origin = EndpointNameOrigin::kExplicit,
      .queue_policy = queue_policy,
      .payload_identity =
          PayloadIdentityRecord{
              .kind = PayloadIdentityKind::kTensor,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .attach = attach,
      .tensor =
          TensorBoundaryContractRecord{
              .memory_kind = memory_kind,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 2U,
              .max_byte_span = 16U,
              .strides = TensorStridePolicy::kContiguousOnly,
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              .max_staging_byte_span = 16U,
          },
      .bounds =
          EndpointBounds{.subscriber =
                             SubscriberEndpointBounds{
                                 .queue_depth = queue_depth,
                                 .max_borrowed_samples = max_borrowed_samples.value_or(queue_depth),
                             }},
  };
  return detail::ChannelEndpointCatalogAccess::normalize("external-tensor-loan-test",
                                                         {std::move(endpoint)});
}

using holoscan::channel_test::ChildProcessCleanup;
using holoscan::channel_test::read_exact;
using holoscan::channel_test::write_exact;

// Polls publisher.update_connections() with a 5-second deadline until at least one subscriber
// is visible.  The longer bound accommodates the interprocess child's exec() and attach latency.
[[nodiscard]] bool wait_for_tensor_subscribers(ChannelPublisher<Tensor>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated) {
      return false;
    }
    if (publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

template <typename T>
[[nodiscard]] T load_little(std::span<const std::byte> source, const std::size_t offset) {
  T value{};
  for (std::size_t index = 0U; index < sizeof(T); ++index) {
    value |= static_cast<T>(std::to_integer<std::uint8_t>(source[offset + index])) << (index * 8U);
  }
  return value;
}

// Decoded in-process representation of one clock-health report received on the inline control
// channel.  MonitoredHealthSharesOrderWithoutMutatingTensorLifetime unpacks the 34-byte wire
// frame into this struct to assert per-field sequencing and epoch stability.
struct TensorClockHealthOperation {
  SampleMetadata metadata;
  std::uint64_t producer_epoch{};
  std::uint64_t operation_sequence{};
  std::uint64_t report_sequence{};
  ClockHealthAssessment assessment{};
  ClockHealthReason reason{};
};

// Polls the inline health subscriber until a valid 34-byte control frame arrives or timeout
// expires.  Validates the fixed magic/version/kind header prefix before decoding payload fields
// into a TensorClockHealthOperation; returns kProtocolError on a header mismatch.
[[nodiscard]] expected<TensorClockHealthOperation, Error> receive_tensor_clock_health_until(
    channel_internal::InlinePayloadSubscriber& subscriber,
    const std::chrono::milliseconds timeout = std::chrono::seconds{2}) {
  constexpr std::size_t kClockHealthControlBytes = 34U;
  constexpr std::uint32_t kClockHealthControlMagic = 0x48434835U;
  constexpr std::uint16_t kClockHealthControlVersion = 1U;
  constexpr std::uint16_t kClockHealthControlKind = 1U;
  std::array<std::byte, kClockHealthControlBytes> wire{};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber.receive(wire);
    if (received) {
      if (load_little<std::uint32_t>(wire, 0U) != kClockHealthControlMagic ||
          load_little<std::uint16_t>(wire, 4U) != kClockHealthControlVersion ||
          load_little<std::uint16_t>(wire, 6U) != kClockHealthControlKind) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return TensorClockHealthOperation{
          .metadata = received->metadata,
          .producer_epoch = load_little<std::uint64_t>(wire, 8U),
          .operation_sequence = load_little<std::uint64_t>(wire, 16U),
          .report_sequence = load_little<std::uint64_t>(wire, 24U),
          .assessment =
              static_cast<ClockHealthAssessment>(std::to_integer<std::uint8_t>(wire[32U])),
          .reason = static_cast<ClockHealthReason>(std::to_integer<std::uint8_t>(wire[33U])),
      };
    }
    if (received.error().code != ErrorCode::kNotReady) {
      return make_unexpected(std::move(received).error());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// Requires two consecutive publisher-ready / subscriber-ready refresh cycles so subsequent
// loan/emit/receive calls start from a fully synchronized same-process connection view.
[[nodiscard]] bool wait_for_tensor_subscriber(ChannelPublisher<Tensor>& publisher,
                                              ChannelSubscriber<Tensor>& subscriber) {
  using namespace std::chrono_literals;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
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
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

// Retries loan_tensor until capacity becomes available (subscriber slot retired) or 2 seconds
// elapse.  Used after deferred CUDA stream callbacks where exact retirement timing is not
// predictable from the test side.
[[nodiscard]] expected<TensorPublishLoan, Error> wait_for_recycled_tensor_loan(
    ChannelPublisher<Tensor>& publisher, const TensorLoanRequest& request) {
  using namespace std::chrono_literals;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto loan = publisher.loan_tensor(request);
    if (loan || loan.error().code != ErrorCode::kResourceExhausted) {
      return loan;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(
      Error{ErrorCode::kTimeout, "timed out waiting for catalog Tensor pool retirement"});
}

// Waits for the publisher, data subscriber, and inline health subscriber to all reach a ready
// state simultaneously.  Two consecutive agreement cycles guard against transient-visibility
// false positives on the health channel.  Used by MonitoredHealthShares*.
[[nodiscard]] bool wait_for_tensor_catalog_subscribers(
    ChannelPublisher<Tensor>& publisher, ChannelSubscriber<Tensor>& data_subscriber,
    channel_internal::InlinePayloadSubscriber& health_subscriber) {
  using namespace std::chrono_literals;
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto data_refreshed = data_subscriber.has_samples();
    auto health_refreshed = health_subscriber.has_samples();
    auto publisher_refreshed = publisher.update_connections();
    if (data_refreshed && health_refreshed && publisher_refreshed &&
        health_subscriber.has_publishers() && publisher.has_subscribers()) {
      ++consecutive_ready_refreshes;
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

// Catalog sequencing must arbitrate a final data record ahead of protected EOS even when the
// terminal becomes visible after the receiver's first empty data probe.
// REQ: CHAN-136
TEST(ExternalTensorLoanRuntime, CatalogProtectedTerminalRechecksFinalDataAfterInitialProbe) {
  const std::string endpoint_name = channel_test::make_unique_channel_name();
  auto import_catalog = make_tensor_import_catalog(endpoint_name,
                                                   1U,
                                                   InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                   {},
                                                   1U,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   QueuePolicy::kDropOldest);
  auto export_catalog = make_tensor_export_catalog(
      endpoint_name, 1U, MemoryKind::kHost, {}, 1U, QueuePolicy::kDropOldest);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());
  auto publisher_endpoint = import_catalog->endpoint(endpoint_name);
  auto subscriber_endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(publisher_endpoint) << runtime_error_message(publisher_endpoint.error());
  ASSERT_TRUE(subscriber_endpoint) << runtime_error_message(subscriber_endpoint.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  channel_internal::TensorPublisherTestAccess::ReceiveProbeBarrier barrier;
  channel_internal::TensorPublisherTestAccess::
      attach_receive_probe_barrier_to_next_subscriber_for_test(barrier);
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*subscriber_endpoint);
  auto publisher = publisher_runtime->create_publisher<Tensor>(*publisher_endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  auto receive_future = std::async(std::launch::async, [&subscriber] {
    return detail::RuntimeChannelBindingAccess::receive_catalog_tensor_operation(*subscriber, true);
  });
  const bool first_data_probe_completed = barrier.reached.try_acquire_for(std::chrono::seconds(2));
  std::optional<expected<std::uint32_t, Error>> emitted;
  std::optional<expected<void, Error>> closed;
  if (first_data_probe_completed) {
    const std::array<std::int64_t, 1U> shape{8};
    auto loan = publisher->loan_tensor(TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
    });
    if (loan) {
      auto writer = loan->write_host();
      if (writer) {
        std::memset(writer->data(), 0x39, writer->byte_size());
        auto committed = std::move(*writer).commit();
        if (committed) {
          emitted.emplace(publisher->emit(std::move(*loan)));
        }
      }
    }
    closed.emplace(publisher->close());
  }
  barrier.resume.release();

  auto data = receive_future.get();
  ASSERT_TRUE(first_data_probe_completed);
  ASSERT_TRUE(emitted.has_value());
  ASSERT_TRUE(*emitted) << runtime_error_message(emitted->error());
  EXPECT_EQ(**emitted, 1U);
  ASSERT_TRUE(closed.has_value());
  ASSERT_TRUE(*closed) << runtime_error_message(closed->error());
  ASSERT_TRUE(data) << runtime_error_message(data.error());
  ASSERT_EQ(data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(data->metadata.sequence, 1U);
  data->tensor_sample.reset();

  auto terminal =
      detail::RuntimeChannelBindingAccess::receive_catalog_tensor_operation(*subscriber, true);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);
  EXPECT_EQ(terminal->metadata.sequence, 2U);
}

// REQ: CHAN-069
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, OrdinaryHealthModeRejectsWithoutMutatingLiveLoanState) {
  auto catalog = make_tensor_import_catalog("tensor-ordinary-health", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-ordinary-health");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  void* const slot_address = writer->data();

  auto rejected = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(writer->data(), slot_address);
  EXPECT_EQ(writer->byte_size(), 8U);
  ASSERT_TRUE(std::move(writer).value().commit());
  ASSERT_TRUE(std::move(loan).value().cancel());

  auto recycled = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  auto recycled_writer = recycled->write_host();
  ASSERT_TRUE(recycled_writer);
  EXPECT_EQ(recycled_writer->data(), slot_address);
}

// Q61-T12: the same-process Tensor publisher uses the common ten-reason report lane and one
// publisher operation sequence. Invalid reports consume neither sequence, and every health
// operation remains independent of a writing, queued, or received Tensor slot.
// REQ: CHAN-069
// REQ: CHAN-080
// REQ: CHAN-081
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, MonitoredHealthSharesOrderWithoutMutatingTensorLifetime) {
  //   publisher --[loan depth=1]-----------> data_subscriber
  //   publisher --[inline control channel]--> health_subscriber
  using namespace std::chrono_literals;

  constexpr ClockHealthReport kBraceDefault{};
  static_assert(kBraceDefault.assessment == ClockHealthAssessment::kHealthy);
  static_assert(kBraceDefault.reason == ClockHealthReason::kNone);
  const ClockHealthOptions monitored{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  auto catalog = make_tensor_import_catalog(
      "tensor-monitored-health", 1U, InterfaceTensorEmitPolicy::kCopyIfNeeded, monitored);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint("tensor-monitored-health");
  ASSERT_TRUE(endpoint);

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto data_subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(data_subscriber) << runtime_error_message(data_subscriber.error());
  auto health_subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_clock_health_subscriber(
          *runtime, *endpoint);
  ASSERT_TRUE(health_subscriber) << runtime_error_message(health_subscriber.error());
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(
      wait_for_tensor_catalog_subscribers(*publisher, *data_subscriber, **health_subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  // Key setup: the sole producer credit is held in writing state while health reports are sent;
  //            the slot address must remain constant across all report and invalid-report calls.
  auto loan = publisher->loan_tensor(request);
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  void* const slot_address = writer->data();
  std::memset(slot_address, 0x5A, writer->byte_size());

  auto exhausted_while_writing = publisher->loan_tensor(request);
  ASSERT_FALSE(exhausted_while_writing);
  EXPECT_EQ(exhausted_while_writing.error().code, ErrorCode::kResourceExhausted);

  auto reported_healthy = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_TRUE(reported_healthy) << runtime_error_message(reported_healthy.error());
  auto first_health = receive_tensor_clock_health_until(**health_subscriber);
  ASSERT_TRUE(first_health) << runtime_error_message(first_health.error());
  EXPECT_NE(first_health->producer_epoch, 0U);
  const std::uint64_t producer_generation = first_health->producer_epoch;
  EXPECT_EQ(first_health->operation_sequence, 1U);
  EXPECT_EQ(first_health->report_sequence, 1U);
  EXPECT_EQ(first_health->metadata.sequence, 1U);
  EXPECT_EQ(first_health->assessment, ClockHealthAssessment::kHealthy);
  EXPECT_EQ(first_health->reason, ClockHealthReason::kNone);

  // Unknown native enum values and an invalid healthy/non-none pair fail before either the common
  // operation sequence or the report sequence changes. The live writer remains the sole owner.
  const std::array invalid_reports{
      ClockHealthReport{
          .assessment = unnamed_enum_value<ClockHealthAssessment>(0xFFU),
          .reason = ClockHealthReason::kNone,
      },
      ClockHealthReport{
          .assessment = ClockHealthAssessment::kUnhealthy,
          .reason = unnamed_enum_value<ClockHealthReason>(0xFFU),
      },
      ClockHealthReport{
          .assessment = ClockHealthAssessment::kHealthy,
          .reason = ClockHealthReason::kMonitorFault,
      },
  };
  for (const ClockHealthReport report : invalid_reports) {
    auto rejected = publisher->report_clock_health(report);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(writer->data(), slot_address);
  }

  ASSERT_TRUE(std::move(writer).value().commit());
  auto emitted = publisher->emit(std::move(loan).value());
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);

  auto independent_while_queued = publisher->loan_tensor(request);
  ASSERT_TRUE(independent_while_queued) << runtime_error_message(independent_while_queued.error());
  auto independent_queued_writer = independent_while_queued->write_host();
  ASSERT_TRUE(independent_queued_writer)
      << runtime_error_message(independent_queued_writer.error());
  EXPECT_NE(independent_queued_writer->data(), slot_address);
  std::memset(independent_queued_writer->data(), 0xA5, independent_queued_writer->byte_size());
  ASSERT_TRUE(std::move(*independent_queued_writer).commit());
  ASSERT_TRUE(std::move(*independent_while_queued).cancel());

  constexpr std::array<ClockHealthReason, 9U> kNegativeReasons{
      ClockHealthReason::kSynchronizationLockLost,
      ClockHealthReason::kPhaseErrorLimitExceeded,
      ClockHealthReason::kRateErrorLimitExceeded,
      ClockHealthReason::kCalibrationExpired,
      ClockHealthReason::kClockStalled,
      ClockHealthReason::kTimestampRegression,
      ClockHealthReason::kTimestampSourceFault,
      ClockHealthReason::kMonitorFault,
      ClockHealthReason::kUnspecified,
  };
  for (std::size_t index = 0U; index < kNegativeReasons.size(); ++index) {
    const ClockHealthAssessment assessment =
        index % 2U == 0U ? ClockHealthAssessment::kDegraded : ClockHealthAssessment::kUnhealthy;
    const ClockHealthReport report{.assessment = assessment, .reason = kNegativeReasons[index]};
    ASSERT_TRUE(publisher->report_clock_health(report)) << index;
    auto observed = receive_tensor_clock_health_until(**health_subscriber);
    ASSERT_TRUE(observed) << index << ": " << runtime_error_message(observed.error());
    // Key assertion: operation_sequence (shared with data emits) advances independently of
    //                report_sequence; both counters stay monotone across nine negative reasons.
    EXPECT_EQ(observed->producer_epoch, producer_generation);
    EXPECT_EQ(observed->operation_sequence, index + 3U);
    EXPECT_EQ(observed->report_sequence, index + 2U);
    EXPECT_EQ(observed->metadata.sequence, observed->operation_sequence);
    EXPECT_EQ(observed->assessment, assessment);
    EXPECT_EQ(observed->reason, kNegativeReasons[index]);
  }

  {
    auto received = data_subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->metadata.sequence, 2U);
    ASSERT_NE(received->data.data(), nullptr);
    const auto* received_bytes = static_cast<const std::byte*>(received->data.data());
    for (std::size_t index = 0U; index < static_cast<std::size_t>(received->data.nbytes());
         ++index) {
      EXPECT_EQ(received_bytes[index], std::byte{0x5A});
    }
    EXPECT_TRUE(received->data.is_read_only());
    auto independent_while_borrowed = publisher->loan_tensor(request);
    ASSERT_TRUE(independent_while_borrowed)
        << runtime_error_message(independent_while_borrowed.error());
    auto independent_borrowed_writer = independent_while_borrowed->write_host();
    ASSERT_TRUE(independent_borrowed_writer)
        << runtime_error_message(independent_borrowed_writer.error());
    EXPECT_NE(independent_borrowed_writer->data(), slot_address);
    std::memset(
        independent_borrowed_writer->data(), 0x3C, independent_borrowed_writer->byte_size());
    ASSERT_TRUE(std::move(*independent_borrowed_writer).commit());
    ASSERT_TRUE(std::move(*independent_while_borrowed).cancel());
    for (std::size_t index = 0U; index < static_cast<std::size_t>(received->data.nbytes());
         ++index) {
      EXPECT_EQ(received_bytes[index], std::byte{0x5A});
    }
  }
  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
}

// An indeterminate health submit consumes the shared operation authority, so neither data nor EOS
// may continue from a health state that cannot be known to match the subscriber.
// REQ: CHAN-111
TEST(ExternalTensorLoanRuntime, IndeterminateHealthPublicationClosesCatalogPublisher) {
  const std::string endpoint_name = "tensor-health-indeterminate-" + std::to_string(::getpid());
  const ClockHealthOptions monitored{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = std::chrono::seconds{1},
  };
  auto catalog = make_tensor_import_catalog(
      endpoint_name, 1U, InterfaceTensorEmitPolicy::kCopyIfNeeded, monitored);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto data_subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  auto health_subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_clock_health_subscriber(
          *runtime, *endpoint);
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(data_subscriber) << runtime_error_message(data_subscriber.error());
  ASSERT_TRUE(health_subscriber) << runtime_error_message(health_subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(
      wait_for_tensor_catalog_subscribers(*publisher, *data_subscriber, **health_subscriber));

  // Key setup: inject an indeterminate result after the health operation has been submitted, so
  // the publisher cannot know whether its shared operation authority was consumed.
  ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::fail_next_catalog_post_submit_for_test(
      *publisher, ErrorCode::kPublicationIndeterminate));
  const auto health = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_FALSE(health);
  EXPECT_EQ(health.error().code, ErrorCode::kPublicationIndeterminate);

  // Key assertion: the uncertain submit closes every authority-bearing API rather than allowing
  // health, data, or EOS to continue from an unknowable sequence state.
  const auto later_health = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_FALSE(later_health);
  EXPECT_EQ(later_health.error().code, ErrorCode::kClosed);
  constexpr std::array<std::int64_t, 1U> kShape{8};
  const auto later_loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_FALSE(later_loan);
  EXPECT_EQ(later_loan.error().code, ErrorCode::kClosed);
  const auto terminal = publisher->close();
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
}

// TA-T03/TA-T06/TA-T07/TA-T43: a catalog endpoint owns a finite fixed pool and copies every
// request view before returning the loan.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, FinitePoolCopiesRequestAndCancelReturnsTheExactSlot) {
  auto catalog = make_tensor_import_catalog("tensor-pool", 2U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-pool");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  std::array<std::int64_t, 2U> first_shape{2, 4};
  std::array<std::int64_t, 2U> first_strides{4, 1};
  auto first = publisher->loan_tensor(TensorLoanRequest{
      .shape = first_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .byte_strides = TensorByteStridesSpan{first_strides},
  });
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  first_shape = {8, 8};
  first_strides = {8, 8};
  ASSERT_EQ(first->shape().size(), 2U);
  EXPECT_EQ(first->shape()[0], 2);
  EXPECT_EQ(first->shape()[1], 4);
  const auto copied_strides = first->byte_strides();
  ASSERT_TRUE(copied_strides.has_value());
  EXPECT_EQ(copied_strides.value()[0], 4);
  EXPECT_EQ(copied_strides.value()[1], 1);
  EXPECT_EQ(first->byte_size(), 8U);

  const std::array<std::int64_t, 1U> second_shape{16};
  auto second = publisher->loan_tensor(TensorLoanRequest{
      .shape = second_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto exhausted = publisher->loan_tensor(TensorLoanRequest{
      .shape = second_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);

  ASSERT_TRUE(std::move(first).value().cancel());
  auto reused = publisher->loan_tensor(TensorLoanRequest{
      .shape = second_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(reused) << runtime_error_message(reused.error());
}

// The catalog publisher credit limits unpublished producer loans, while its physical export pool
// also covers one recipient's queued and borrowed transport claims.
// REQ: CHAN-114
// REQ: CHAN-133
TEST(ExternalTensorLoanRuntime, PublisherCreditIsIndependentFromQueueAndBorrowedBacking) {
  const std::string endpoint_name = "tensor-publisher-credit-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name,
                                                   1U,
                                                   InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                   ClockHealthOptions{},
                                                   4U,
                                                   2U,
                                                   MemoryKind::kHost,
                                                   QueuePolicy::kDropOldest);
  auto export_catalog = make_tensor_export_catalog(
      endpoint_name, 4U, MemoryKind::kHost, TopicAttachPolicy{}, 2U, QueuePolicy::kDropOldest);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto subscriber = runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  auto publisher = runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };

  std::vector<Tensor> retained;
  retained.reserve(2U);
  for (std::uint8_t value = 1U; value <= 2U; ++value) {
    auto loan = publisher->loan_tensor(request);
    ASSERT_TRUE(loan) << runtime_error_message(loan.error());
    if (value == 1U) {
      // Key setup: one unpublished loan consumes the sole producer credit even though the larger
      // physical pool has capacity reserved for transport-owned backing.
      auto exceeds_credit = publisher->loan_tensor(request);
      ASSERT_FALSE(exceeds_credit);
      EXPECT_EQ(exceeds_credit.error().code, ErrorCode::kResourceExhausted);
    }
    auto writer = loan->write_host();
    ASSERT_TRUE(writer) << runtime_error_message(writer.error());
    std::memset(writer->data(), value, writer->byte_size());
    ASSERT_TRUE(std::move(writer).value().commit());
    ASSERT_TRUE(publisher->emit(std::move(loan).value()));
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    retained.push_back(std::move(*received));
  }

  auto queued = publisher->loan_tensor(request);
  ASSERT_TRUE(queued) << runtime_error_message(queued.error());
  auto queued_writer = queued->write_host();
  ASSERT_TRUE(queued_writer) << runtime_error_message(queued_writer.error());
  std::memset(queued_writer->data(), 3, queued_writer->byte_size());
  ASSERT_TRUE(std::move(queued_writer).value().commit());
  ASSERT_TRUE(publisher->emit(std::move(queued).value()));

  // Key assertion: two retained imports plus one drop-oldest unread sample do not consume the
  // independent producer credit; exactly one new unpublished loan remains available.
  auto active = publisher->loan_tensor(request);
  ASSERT_TRUE(active) << runtime_error_message(active.error());
  auto still_exceeds_credit = publisher->loan_tensor(request);
  ASSERT_FALSE(still_exceeds_credit);
  EXPECT_EQ(still_exceeds_credit.error().code, ErrorCode::kResourceExhausted);
  ASSERT_TRUE(std::move(active).value().cancel());
}

// A fail-closed slot is removed from the finite free inventory before validation, so it cannot be
// selected repeatedly or hide honest exhaustion after the remaining capacity is consumed.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, QuarantinedFreeSlotIsRemovedFromAvailableCapacity) {
  auto catalog = make_tensor_import_catalog("tensor-quarantined-free", 2U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-quarantined-free");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: quarantine three of the four physical slots independently of the two-loan
  //            producer-credit limit so only one valid backing remains available.
  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };

  for (std::size_t index = 0U; index < 3U; ++index) {
    ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::corrupt_next_catalog_free_slot_for_test(
        *publisher));
    auto quarantined = publisher->loan_tensor(request);
    ASSERT_FALSE(quarantined);
    EXPECT_EQ(quarantined.error().code, ErrorCode::kFailure);
  }

  auto remaining = publisher->loan_tensor(request);
  ASSERT_TRUE(remaining) << runtime_error_message(remaining.error());
  auto exhausted = publisher->loan_tensor(request);
  ASSERT_FALSE(exhausted);
  // Key assertion: only one physical slot remains; quarantined slots are not re-offered even
  //                though the independent producer-credit limit would allow a second loan.
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
}

// TA-T03/TA-T06/TA-T07/TA-T37: scalar and zero-extent layouts retain their distinct logical byte
// extents while each consumes one complete fixed-pool slot and control record.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, RankZeroAndZeroExtentEachConsumeOneFiniteSlot) {
  auto catalog = make_tensor_import_catalog("tensor-scalar-zero", 2U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-scalar-zero");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 0U> scalar_shape{};
  auto scalar = publisher->loan_tensor(TensorLoanRequest{
      .shape = scalar_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(scalar) << runtime_error_message(scalar.error());
  EXPECT_TRUE(scalar->shape().empty());
  EXPECT_EQ(scalar->byte_size(), 1U);

  const std::array<std::int64_t, 2U> zero_shape{0, 4};
  auto zero = publisher->loan_tensor(TensorLoanRequest{
      .shape = zero_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(zero) << runtime_error_message(zero.error());
  EXPECT_EQ(zero->byte_size(), 0U);

  auto exhausted = publisher->loan_tensor(TensorLoanRequest{
      .shape = scalar_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, ErrorCode::kResourceExhausted);
}

// TA-T33: after cold pool materialization, repeated host loan/write/commit/cancel cycles use only
// the endpoint's fixed slot and control records. The test deliberately excludes transport submit,
// whose queue and cross-process realization are owned by HSDK5-015.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, WarmedLoanWriteCommitCancelUsesNoHostHeapAllocation) {
  auto catalog = make_tensor_import_catalog("tensor-warm-allocation", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-warm-allocation");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };

  // Exercise every local transition once before measuring the bounded steady-state path.
  auto warm_loan = publisher->loan_tensor(request);
  ASSERT_TRUE(warm_loan) << runtime_error_message(warm_loan.error());
  auto warm_writer = warm_loan->write_host();
  ASSERT_TRUE(warm_writer) << runtime_error_message(warm_writer.error());
  ASSERT_TRUE(std::move(warm_writer).value().commit());
  ASSERT_TRUE(std::move(warm_loan).value().cancel());

  bool all_cycles_succeeded = true;
  std::size_t allocation_count = 0U;
  {
    const test::ScopedAllocationCounter allocations;
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
      auto loan = publisher->loan_tensor(request);
      if (!loan) {
        all_cycles_succeeded = false;
        break;
      }
      auto writer = loan->write_host();
      if (!writer) {
        all_cycles_succeeded = false;
        break;
      }
      auto committed = std::move(writer).value().commit();
      auto cancelled = std::move(loan).value().cancel();
      if (!committed || !cancelled) {
        all_cycles_succeeded = false;
        break;
      }
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  EXPECT_EQ(allocation_count, 0U);
}

// TA-T09/TA-T10/TA-T11/TA-T12: one loan grants at most one writer and only a consumed successful
// commit makes that loan publishable.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, WriterIsExclusiveAndCommitConsumesTheGuard) {
  auto catalog = make_tensor_import_catalog("tensor-writer", 1U);
  ASSERT_TRUE(catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-writer");
  ASSERT_TRUE(publisher);
  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan);
  // Key setup: a second write_host() on the same loan while the first guard is still live must
  //            fail immediately without acquiring a new writer token.
  auto writer = loan->write_host();
  ASSERT_TRUE(writer);
  auto duplicate = loan->write_host();
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);
  std::memset(writer->data(), 0x2A, writer->byte_size());
  ASSERT_TRUE(std::move(writer).value().commit());
  // Key assertion: commit leaves the guard as a defensively empty facade (null data, zero size),
  // and a subsequent write_host() on the same loan is rejected.
  // This assertion is the contract under test: commit leaves the deliberately consumed guard
  // queryable as an empty facade.
  EXPECT_EQ(writer->data(), nullptr);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(writer->byte_size(), 0U);  // NOLINT(bugprone-use-after-move)
  auto after_commit = loan->write_host();
  ASSERT_FALSE(after_commit);
  EXPECT_EQ(after_commit.error().code, ErrorCode::kInvalidArgument);
}

// The public convenience keeps endpoint-owned authority and retirement unchanged while accepting
// either a void fill callback or an exact checked callback result.
// REQ: OPERATOR-027
TEST(ExternalTensorLoanRuntime, FusedHostWriteCommitsAndPropagatesFillFailure) {
  auto catalog = make_tensor_import_catalog("tensor-fused-write", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-fused-write");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto loan = publisher->loan_tensor(request);
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto committed = loan->write_host_and_commit([](const TensorLoanWriteGuard& writer) {
    std::memset(writer.data(), 0x2A, writer.byte_size());
  });
  ASSERT_TRUE(committed) << runtime_error_message(committed.error());
  ASSERT_TRUE(std::move(*loan).cancel());

  {
    auto rejected_loan = publisher->loan_tensor(request);
    ASSERT_TRUE(rejected_loan) << runtime_error_message(rejected_loan.error());
    auto rejected = rejected_loan->write_host_and_commit(
        [](const TensorLoanWriteGuard&) -> expected<void, Error> {
          return make_unexpected(Error{ErrorCode::kFailure});
        });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kFailure);
  }

  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// TA-T12/TA-T16/TA-T17/TA-T48: abandoning the sole writer makes the loan non-publishable. The
// later consuming emit follows one cancellation path, advances no sequence, and returns capacity.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, DroppedWriterMakesLoanNonpublishableAndRecyclesOnce) {
  auto catalog = make_tensor_import_catalog("tensor-drop-writer", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-drop-writer");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto loan = publisher->loan_tensor(request);
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  {
    auto writer = loan->write_host();
    ASSERT_TRUE(writer) << runtime_error_message(writer.error());
    std::memset(writer->data(), 0x33, writer->byte_size());
  }

  auto rejected = publisher->emit(std::move(loan).value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// TA-T32/TA-T38/TA-T44/TA-T50: facade movement closes only the moved-from wrapper, while an
// outstanding old-control loan keeps its tombstone alive and can retire safely after facade
// destruction. It never silently rebinds to a later publisher.
// REQ: CHAN-081
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, MovedAndDestroyedFacadeLeavesOutstandingLoanSafelyRetirable) {
  auto catalog = make_tensor_import_catalog("tensor-facade-close", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  std::optional<TensorPublishLoan> outstanding;
  {
    auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-facade-close");
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    auto loan = publisher->loan_tensor(request);
    ASSERT_TRUE(loan) << runtime_error_message(loan.error());
    outstanding.emplace(std::move(loan).value());

    const ChannelPublisher<Tensor> moved{std::move(publisher).value()};
    // Moved-from publisher facades have a specified closed state, which this regression probes.
    auto moved_from = publisher->loan_tensor(request);  // NOLINT(bugprone-use-after-move)
    ASSERT_FALSE(moved_from);
    EXPECT_EQ(moved_from.error().code, ErrorCode::kClosed);
  }

  ASSERT_TRUE(outstanding.has_value());
  auto retired = std::move(*outstanding).cancel();
  ASSERT_TRUE(retired) << runtime_error_message(retired.error());
}

// TA-T13/TA-T17/TA-T18/TA-T31: publication transfers the exact endpoint-owned storage and the
// slot remains unavailable until the final received Tensor handle and read lease retire.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, PublicationRetainsSlotUntilReceivedTensorRetires) {
  //   publisher --[pool depth=1]--> subscriber
  auto catalog = make_tensor_import_catalog("tensor-publish", 1U);
  ASSERT_TRUE(catalog);
  auto endpoint = catalog->endpoint("tensor-publish");
  ASSERT_TRUE(endpoint);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  auto subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan);
  auto writer = loan->write_host();
  ASSERT_TRUE(writer);
  void* const published_pointer = writer->data();
  std::memset(published_pointer, 0x5A, writer->byte_size());
  ASSERT_TRUE(std::move(writer).value().commit());
  auto emitted = publisher->emit(std::move(loan).value());
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  Tensor received_tensor = std::move(*received);
  ASSERT_NE(received_tensor.data(), nullptr);
  const std::size_t published_byte_size = static_cast<std::size_t>(received_tensor.nbytes());
  const auto* received_bytes = static_cast<const std::byte*>(received_tensor.data());
  for (std::size_t index = 0U; index < received_tensor.nbytes(); ++index) {
    EXPECT_EQ(received_bytes[index], std::byte{0x5A});
  }
  EXPECT_TRUE(received_tensor.is_read_only());
  const auto exercise_distinct_backing = [&](const std::byte fill) {
    auto candidate = publisher->loan_tensor(TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
    });
    ASSERT_TRUE(candidate) << runtime_error_message(candidate.error());
    auto candidate_writer = candidate->write_host();
    ASSERT_TRUE(candidate_writer) << runtime_error_message(candidate_writer.error());
    EXPECT_NE(candidate_writer->data(), published_pointer);
    std::memset(
        candidate_writer->data(), std::to_integer<int>(fill), candidate_writer->byte_size());
    ASSERT_TRUE(std::move(candidate_writer).value().commit());
    ASSERT_TRUE(std::move(candidate).value().cancel());
    for (std::size_t index = 0U; index < published_byte_size; ++index) {
      EXPECT_EQ(received_bytes[index], std::byte{0x5A});
    }
  };

  // Key setup: the larger physical pool may serve another producer loan, but the received Tensor
  // keeps its exact published backing unavailable for reuse.
  exercise_distinct_backing(std::byte{0xA5});

  auto read_lease = detail::TensorAccess::acquire_read_lease(received_tensor);
  ASSERT_TRUE(read_lease) << runtime_error_message(read_lease.error());
  received_tensor = Tensor{};
  exercise_distinct_backing(std::byte{0x3C});

  // Key assertion: both the Tensor and the independent read lease protect the published bytes
  // while unrelated producer loans use separate preallocated backing.
  read_lease = detail::TensorReadLease{};
  ASSERT_TRUE(publisher->update_connections());
}

// The supported external subscriber uses only the immutable catalog and stable endpoint name.
// It receives the same read-only BufferHandle-backed storage as the private graph-side peer and
// retains that storage until the final Tensor claim retires.
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogSubscriberAttachesByStableNameAndRetainsReadOnlyStorage) {
  //   publisher_runtime/publisher --[import catalog]--> subscriber_runtime/subscriber
  const std::string endpoint_name = "tensor-public-subscriber-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  auto import_endpoint = import_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(import_endpoint) << runtime_error_message(import_endpoint.error());
  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  void* const published_data = writer->data();
  std::memset(writer->data(), 0x3C, writer->byte_size());
  ASSERT_TRUE(std::move(*writer).commit());
  auto emitted = publisher->emit(std::move(*loan), EmitOptions{.frame_id = 77U});
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_TRUE(received->data.is_read_only());
  EXPECT_EQ(received->metadata.frame_id, 77U);
  EXPECT_EQ(received->data.dtype().code, kDLUInt);
  EXPECT_EQ(received->data.dtype().bits, 8U);
  EXPECT_EQ(received->data.dtype().lanes, 1U);
  ASSERT_EQ(received->data.shape_span().size(), 1U);
  EXPECT_EQ(received->data.shape_span()[0], 8);
  EXPECT_TRUE(received->data.is_contiguous());
  const auto received_strides = received->data.strides();
  ASSERT_EQ(received_strides.size(), 1U);
  EXPECT_EQ(received_strides[0], 1);
  ASSERT_NE(received->data.data(), nullptr);
  const auto bytes = static_cast<const std::byte*>(received->data.data());
  for (std::size_t index = 0U; index < received->data.nbytes(); ++index) {
    EXPECT_EQ(bytes[index], std::byte{0x3C});
  }

  auto independent = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(independent) << runtime_error_message(independent.error());
  auto independent_writer = independent->write_host();
  ASSERT_TRUE(independent_writer) << runtime_error_message(independent_writer.error());
  EXPECT_NE(independent_writer->data(), published_data);
  std::memset(independent_writer->data(), 0xA5, independent_writer->byte_size());
  ASSERT_TRUE(std::move(*independent_writer).commit());
  ASSERT_TRUE(std::move(*independent).cancel());
  for (std::size_t index = 0U; index < received->data.nbytes(); ++index) {
    EXPECT_EQ(bytes[index], std::byte{0x3C});
  }

  received->data = Tensor{};
  std::optional<TensorPublishLoan> recycled;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!recycled && std::chrono::steady_clock::now() < deadline) {
    ASSERT_TRUE(publisher->update_connections());
    auto candidate = publisher->loan_tensor(TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
    });
    if (candidate) {
      recycled.emplace(std::move(*candidate));
      break;
    }
    ASSERT_EQ(candidate.error().code, ErrorCode::kResourceExhausted);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_TRUE(recycled.has_value());
}

// Warm catalog receive reuses one borrow-bounded DLPack/import control across distinct producer
// slots. Explicit strides exercise both pre-reserved metadata vectors inside the measured path.
// REQ: EXTERNAL-004
// REQ: EXTERNAL-006
TEST(ExternalTensorLoanRuntime, WarmedCatalogReceiveUsesNoHostHeapAllocation) {
  //   publisher --[pool depth=2, max_borrowed=1]--> subscriber [queue depth=2]
  const std::string endpoint_name = "tensor-warm-receive-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(
      endpoint_name, 2U, InterfaceTensorEmitPolicy::kCopyIfNeeded, ClockHealthOptions{}, 2U, 1U);
  auto export_catalog =
      make_tensor_export_catalog(endpoint_name, 2U, MemoryKind::kHost, TopicAttachPolicy{}, 1U);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());
  auto import_endpoint = import_catalog->endpoint(endpoint_name);
  auto export_endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(import_endpoint) << runtime_error_message(import_endpoint.error());
  ASSERT_TRUE(export_endpoint) << runtime_error_message(export_endpoint.error());
  EXPECT_EQ(detail::ChannelEndpointCatalogAccess::backend_service_binding(*import_endpoint),
            detail::ChannelEndpointCatalogAccess::backend_service_binding(*export_endpoint));
  const auto* const import_record = detail::ChannelEndpointCatalogAccess::record(*import_endpoint);
  const auto* const export_record = detail::ChannelEndpointCatalogAccess::record(*export_endpoint);
  ASSERT_NE(import_record, nullptr);
  ASSERT_NE(export_record, nullptr);
  EXPECT_EQ(import_record->service_static.max_subscriber_depth,
            export_record->service_static.max_subscriber_depth);
  EXPECT_EQ(import_record->service_static.max_subscriber_borrowed,
            export_record->service_static.max_subscriber_borrowed);

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  const std::array<std::int64_t, 1U> byte_strides{1};
  for (std::uint8_t value = 1U; value <= 2U; ++value) {
    auto loan = publisher->loan_tensor(TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
        .byte_strides =
            value == 2U ? TensorByteStridesView{TensorByteStridesSpan{byte_strides}} : std::nullopt,
    });
    ASSERT_TRUE(loan) << runtime_error_message(loan.error());
    auto writer = loan->write_host();
    ASSERT_TRUE(writer) << runtime_error_message(writer.error());
    std::memset(writer->data(), value, writer->byte_size());
    ASSERT_TRUE(std::move(writer).value().commit());
    ASSERT_TRUE(publisher->emit(std::move(loan).value()));
  }

  {
    auto warm = subscriber->receive_sample();
    ASSERT_TRUE(warm) << runtime_error_message(warm.error());
    ASSERT_NE(warm->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::uint8_t*>(warm->data.data()), 1U);
  }

  bool receive_succeeded = false;
  bool metadata_matches = false;
  std::size_t allocation_count = 0U;
  {
    const test::ScopedAllocationCounter allocations;
    {
      auto received = subscriber->receive_sample();
      receive_succeeded = received.has_value();
      if (received) {
        const auto strides = received->data.dlpack_strides_span();
        metadata_matches = received->data.data() != nullptr && strides.size() == 1U &&
                           strides[0] == 1 &&
                           *static_cast<const std::uint8_t*>(received->data.data()) == 2U;
      }
    }
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(receive_succeeded);
  EXPECT_TRUE(metadata_matches);
  EXPECT_EQ(allocation_count, 0U);
}

// A catalog subscriber owns the import context required to materialize BufferHandles. Destroying
// the factory facade before data arrives must not invalidate that retained attachment.
// REQ: EXTERNAL-001
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogSubscriberRetainsRuntimeImportContext) {
  //   subscriber (ChannelRuntime destroyed after create) <-- publisher (created later)
  const std::string endpoint_name = "tensor-retained-runtime-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  std::optional<ChannelSubscriber<Tensor>> subscriber;
  {
    auto subscriber_runtime = ChannelRuntime::create();
    ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
    auto created = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
    ASSERT_TRUE(created) << runtime_error_message(created.error());
    subscriber.emplace(std::move(created).value());
  }

  auto publisher_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  std::memset(writer->data(), 0x4D, writer->byte_size());
  ASSERT_TRUE(std::move(writer).value().commit());
  ASSERT_TRUE(publisher->emit(std::move(loan).value()));

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  subscriber.reset();
  ASSERT_NE(received->data.data(), nullptr);
  EXPECT_EQ(*static_cast<const std::byte*>(received->data.data()), std::byte{0x4D});
}

// A catalog subscriber validates every wire descriptor against the frozen logical Tensor
// contract before importing its BufferHandle. A structurally valid uint16 descriptor cannot cross
// an endpoint that admits only uint8 values, even when its rank and byte span remain in bounds.
// REQ: EXTERNAL-001
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogSubscriberRejectsPayloadOutsideFrozenTensorContract) {
  //   raw HoloMQ publisher (uint16/4-element payload) --> ChannelSubscriber (uint8 contract)
  const std::string endpoint_name = "tensor-contract-mismatch-" + std::to_string(::getpid());
  auto export_catalog = make_tensor_export_catalog(endpoint_name);
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());
  auto endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  ASSERT_NE(record, nullptr);

  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto publisher_context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  ASSERT_TRUE(publisher_context) << publisher_context.error().message;
  auto pool = publisher_context->create_host_buffer_pool(16U, 2U);
  ASSERT_TRUE(pool) << pool.error().message;

  const holomq::PubSubServiceOptions service_options{
      .queue_policy = holomq::QueuePolicy::kBlocking,
      .max_subscribers = 1U,
      .max_subscriber_queue_depth = record->service_static.max_subscriber_depth,
      .max_subscriber_borrowed_samples = record->service_static.max_subscriber_borrowed,
      .publisher_history_depth = record->service_static.publisher_history_depth,
      .enable_event_wake = record->service_static.event_wake,
  };
  auto publisher = publisher_context
                       ->publisher_builder(
                           detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint))
                       .service_options(service_options)
                       .publisher_options(holomq::PublisherPubSubOptions{.max_loaned_samples = 1U})
                       .payload_type<channel_internal::TensorTransportPayload>()
                       .create();
  ASSERT_TRUE(publisher) << publisher.error().message;
  const auto connection_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (consecutive_ready_refreshes < 2U &&
         std::chrono::steady_clock::now() < connection_deadline) {
    ASSERT_TRUE(subscriber->refresh_connections());
    ASSERT_TRUE(publisher->update_connections());
    consecutive_ready_refreshes =
        publisher->has_subscribers() ? consecutive_ready_refreshes + 1U : 0U;
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_EQ(consecutive_ready_refreshes, 2U);

  auto buffer = pool->acquire();
  ASSERT_TRUE(buffer) << buffer.error().message;
  auto handle = buffer->export_handle_unsynchronized();
  ASSERT_TRUE(handle) << handle.error().message;
  const std::array<std::int64_t, 1U> shape{4};
  auto descriptor = channel_internal::make_holomq_tensor_descriptor(
      shape,
      DLDataType{.code = kDLUInt, .bits = 16U, .lanes = 1U},
      std::nullopt,
      8U,
      *handle,
      /*read_only=*/true);
  ASSERT_TRUE(descriptor) << runtime_error_message(descriptor.error());

  auto message = publisher->loan();
  ASSERT_TRUE(message) << message.error().message;
  message->payload() = channel_internal::TensorTransportPayload{
      .metadata =
          SampleMetadata{
              .sequence = 1U,
              .byte_size = 8U,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .producer_generation = 1U,
      .payload_kind = ChannelPayloadKind::kBufferHandle,
      .reference = channel_internal::TensorTransportReference{.holomq = *descriptor},
  };
  auto sent =
      publisher->send(std::move(message).value(),
                      [retained = std::move(buffer).value()]() noexcept { (void)retained; });
  ASSERT_TRUE(sent) << sent.error().message;
  EXPECT_EQ(*sent, 1);

  expected<Sample<Tensor>, Error> rejected = make_unexpected(Error{ErrorCode::kNotReady});
  const auto receive_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!rejected && rejected.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < receive_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    rejected = subscriber->receive_sample();
  }
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kProtocolError);

  auto zero_generation_buffer = pool->acquire();
  ASSERT_TRUE(zero_generation_buffer) << zero_generation_buffer.error().message;
  auto zero_generation_handle = zero_generation_buffer->export_handle_unsynchronized();
  ASSERT_TRUE(zero_generation_handle) << zero_generation_handle.error().message;
  const std::array<std::int64_t, 1U> valid_shape{8};
  auto valid_descriptor = channel_internal::make_holomq_tensor_descriptor(
      valid_shape,
      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      std::nullopt,
      8U,
      *zero_generation_handle,
      /*read_only=*/true);
  ASSERT_TRUE(valid_descriptor) << runtime_error_message(valid_descriptor.error());

  auto zero_generation_message = publisher->loan();
  ASSERT_TRUE(zero_generation_message) << zero_generation_message.error().message;
  zero_generation_message->payload() = channel_internal::TensorTransportPayload{
      .metadata =
          SampleMetadata{
              .sequence = 2U,
              .byte_size = 8U,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .producer_generation = 0U,
      .payload_kind = ChannelPayloadKind::kBufferHandle,
      .reference = channel_internal::TensorTransportReference{.holomq = *valid_descriptor},
  };
  auto zero_generation_sent = publisher->send(
      std::move(*zero_generation_message),
      [retained = std::move(*zero_generation_buffer)]() noexcept { (void)retained; });
  ASSERT_TRUE(zero_generation_sent) << zero_generation_sent.error().message;
  EXPECT_EQ(*zero_generation_sent, 1);

  expected<Sample<Tensor>, Error> zero_generation_rejected =
      make_unexpected(Error{ErrorCode::kNotReady});
  const auto zero_generation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!zero_generation_rejected &&
         zero_generation_rejected.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < zero_generation_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    zero_generation_rejected = subscriber->receive_sample();
  }
  ASSERT_FALSE(zero_generation_rejected);
  EXPECT_EQ(zero_generation_rejected.error().code, ErrorCode::kProtocolError);
}

// Data and terminal records share one publisher operation sequence. Equal sequence heads are
// therefore ambiguous and must fail before either lane can expose a value.
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogSubscriberRejectsDataTerminalSequenceTie) {
  const std::string endpoint_name = "tensor-data-terminal-tie-" + std::to_string(::getpid());
  auto export_catalog = make_tensor_export_catalog(endpoint_name,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   TopicAttachPolicy{},
                                                   std::nullopt,
                                                   QueuePolicy::kDropOldest);
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());
  auto endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  ASSERT_NE(record, nullptr);

  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  auto publisher_context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  ASSERT_TRUE(publisher_context) << publisher_context.error().message;
  auto pool = publisher_context->create_host_buffer_pool(16U, 2U);
  ASSERT_TRUE(pool) << pool.error().message;

  ChannelOptions wire_options{};
  wire_options.queue_policy = ChannelQueuePolicy::kDropOldest;
  wire_options.max_subscriber_queue_depth = record->service_static.max_subscriber_depth;
  wire_options.subscriber_queue_depth = record->service_static.max_subscriber_depth;
  wire_options.max_subscriber_borrowed_samples = record->service_static.max_subscriber_borrowed;
  wire_options.max_borrowed_samples = record->service_static.max_subscriber_borrowed;
  wire_options.publisher_history_depth = record->service_static.publisher_history_depth;
  wire_options.expected_recipient_count = 1U;
  wire_options.enable_event_wake = record->service_static.event_wake;
  wire_options.tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle;
  auto data_publisher =
      publisher_context
          ->publisher_builder(
              detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint))
          .service_options(channel_internal::to_holomq_service_options(wire_options))
          .publisher_options(holomq::PublisherPubSubOptions{.max_loaned_samples = 1U})
          .payload_type<channel_internal::TensorTransportPayload>()
          .create();
  ASSERT_TRUE(data_publisher) << data_publisher.error().message;
  auto terminal_publisher =
      publisher_context
          ->publisher_builder(channel_internal::ordered_terminal_service_id(
              detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint)))
          .service_options(channel_internal::ordered_terminal_service_options(wire_options))
          .publisher_options(channel_internal::ordered_terminal_publisher_options(wire_options))
          .payload_type<channel_internal::TensorTransportPayload>()
          .create();
  ASSERT_TRUE(terminal_publisher) << terminal_publisher.error().message;

  const auto connection_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  bool connected = false;
  while (!connected && std::chrono::steady_clock::now() < connection_deadline) {
    ASSERT_TRUE(subscriber->has_samples());
    ASSERT_TRUE(data_publisher->update_connections());
    ASSERT_TRUE(terminal_publisher->update_connections());
    connected = data_publisher->has_subscribers() && terminal_publisher->has_subscribers();
    if (!connected) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  }
  ASSERT_TRUE(connected);

  auto buffer = pool->acquire();
  ASSERT_TRUE(buffer) << buffer.error().message;
  auto handle = buffer->export_handle_unsynchronized();
  ASSERT_TRUE(handle) << handle.error().message;
  const std::array<std::int64_t, 1U> shape{8};
  auto descriptor = channel_internal::make_holomq_tensor_descriptor(
      shape,
      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      std::nullopt,
      8U,
      *handle,
      /*read_only=*/true);
  ASSERT_TRUE(descriptor) << runtime_error_message(descriptor.error());

  // Key setup: queue valid data and EOS heads carrying the same generation and operation sequence.
  auto data = data_publisher->loan();
  ASSERT_TRUE(data) << data.error().message;
  data->payload() = channel_internal::TensorTransportPayload{
      .metadata =
          SampleMetadata{
              .sequence = 1U,
              .byte_size = 8U,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .producer_generation = 1U,
      .operation_kind = channel_internal::TensorTransportOperationKind::kData,
      .payload_kind = ChannelPayloadKind::kBufferHandle,
      .reference = channel_internal::TensorTransportReference{.holomq = *descriptor},
  };
  auto data_sent = data_publisher->send(
      std::move(*data), [retained = std::move(*buffer)]() noexcept { (void)retained; });
  ASSERT_TRUE(data_sent) << data_sent.error().message;

  auto terminal = terminal_publisher->loan();
  ASSERT_TRUE(terminal) << terminal.error().message;
  terminal->payload() = channel_internal::TensorTransportPayload{
      .metadata =
          SampleMetadata{
              .sequence = 1U,
              .payload_kind = PayloadKind::kDescriptorBacked,
          },
      .producer_generation = 1U,
      .operation_kind = channel_internal::TensorTransportOperationKind::kEndOfStream,
  };
  auto terminal_sent = terminal_publisher->send(std::move(*terminal));
  ASSERT_TRUE(terminal_sent) << terminal_sent.error().message;

  auto received = subscriber->receive_sample();

  // Key assertion: the subscriber rejects the tie without leaking the data-lane sample first.
  ASSERT_FALSE(received);
  EXPECT_EQ(received.error().code, ErrorCode::kProtocolError);
}

// A catalog Tensor subscriber latches the first producer incarnation independently of the public
// sample sequence. Recreating the publisher resets that sequence but must not make the replacement
// producer's data look like a continuation of the retired endpoint.
// REQ: EXTERNAL-004
// REQ: CHAN-142
TEST(ExternalTensorLoanRuntime, CatalogSubscriberFencesARecreatedProducerIncarnation) {
  //   first_publisher  --> subscriber  [epoch A, sequence 1]
  //   first_publisher destroyed, second_publisher created
  //   second_publisher --> subscriber  [epoch B, sequence 1 → expect kStaleEndpoint]
  const std::string endpoint_name = "tensor-recreated-producer-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  auto publish_byte = [](ChannelPublisher<Tensor>& publisher,
                         const std::byte value) -> expected<std::uint32_t, Error> {
    constexpr std::array<std::int64_t, 1U> kShape{8};
    auto loan = publisher.loan_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    std::memset(writer->data(), std::to_integer<int>(value), writer->byte_size());
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    return publisher.emit(std::move(*loan));
  };

  auto first_created = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(first_created) << runtime_error_message(first_created.error());
  std::optional<ChannelPublisher<Tensor>> first_publisher{std::move(*first_created)};
  ASSERT_TRUE(wait_for_tensor_subscriber(*first_publisher, *subscriber));
  auto first_emitted = publish_byte(*first_publisher, std::byte{0x31});
  ASSERT_TRUE(first_emitted) << runtime_error_message(first_emitted.error());
  {
    auto first_received = subscriber->receive_sample();
    ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
    EXPECT_EQ(first_received->metadata.sequence, 1U);
    ASSERT_NE(first_received->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::byte*>(first_received->data.data()), std::byte{0x31});
  }

  // Reclaim the first publisher's sole transport and physical slot before destroying its service.
  bool reclaimed = false;
  constexpr std::array<std::int64_t, 1U> kShape{8};
  const auto reclaim_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!reclaimed && std::chrono::steady_clock::now() < reclaim_deadline) {
    ASSERT_TRUE(first_publisher->update_connections());
    auto candidate = first_publisher->loan_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
    });
    if (candidate) {
      ASSERT_TRUE(std::move(candidate).value().cancel());
      reclaimed = true;
      break;
    }
    ASSERT_EQ(candidate.error().code, ErrorCode::kResourceExhausted);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_TRUE(reclaimed);
  first_publisher.reset();

  auto second_created = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(second_created) << runtime_error_message(second_created.error());
  auto second_publisher = std::move(*second_created);
  ASSERT_TRUE(wait_for_tensor_subscriber(second_publisher, *subscriber));
  auto second_emitted = publish_byte(second_publisher, std::byte{0x52});
  ASSERT_TRUE(second_emitted) << runtime_error_message(second_emitted.error());

  expected<Sample<Tensor>, Error> fenced = make_unexpected(Error{ErrorCode::kNotReady});
  const auto receive_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!fenced && fenced.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < receive_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    fenced = subscriber->receive_sample();
  }
  ASSERT_FALSE(fenced);
  // Key assertion: second producer's data is rejected as kStaleEndpoint, not silently consumed
  //                as a continuation of the first producer's stream.
  EXPECT_EQ(fenced.error().code, ErrorCode::kStaleEndpoint);
}

// Tensor EOS is an explicit operation in the same publisher-authored sequence as data. A
// successful close consumes publisher authority and the subscriber observes kClosed only after
// receiving the preceding data sample.
// REQ: CHAN-078
// REQ: CHAN-110
// REQ: EXTERNAL-003
// REQ: EXTERNAL-008
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogCloseIsOrderedAndConsumesPublisherAuthority) {
  //   publisher --[1 data sample]--> subscriber --[close marker]--> kClosed (ordered)
  const std::string endpoint_name = "tensor-explicit-eos-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name,
                                                   1U,
                                                   InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                   {},
                                                   1U,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   QueuePolicy::kDropOldest);
  auto export_catalog = make_tensor_export_catalog(
      endpoint_name, 1U, MemoryKind::kHost, {}, std::nullopt, QueuePolicy::kDropOldest);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  auto unobserved_close = publisher->close();
  ASSERT_FALSE(unobserved_close);
  EXPECT_EQ(unobserved_close.error().code, ErrorCode::kNotReady);

  auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  constexpr std::array<std::int64_t, 1U> kShape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  std::memset(writer->data(), 0x6A, writer->byte_size());
  ASSERT_TRUE(std::move(*writer).commit());
  ASSERT_TRUE(publisher->emit(std::move(*loan)));

  // Key setup: close while the kDropOldest data slot remains unread. The protected terminal
  // service must preserve the data record rather than replacing it.
  auto closed = publisher->close();
  ASSERT_TRUE(closed) << runtime_error_message(closed.error());

  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->metadata.sequence, 1U);
    ASSERT_NE(received->data.data(), nullptr);
    EXPECT_EQ(*static_cast<const std::byte*>(received->data.data()), std::byte{0x6A});
  }

  auto repeated = publisher->close();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ErrorCode::kClosed);
  auto loan_after_close = publisher->loan_tensor(TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_FALSE(loan_after_close);
  EXPECT_EQ(loan_after_close.error().code, ErrorCode::kClosed);

  expected<Sample<Tensor>, Error> end_of_stream = make_unexpected(Error{ErrorCode::kNotReady});
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!end_of_stream && end_of_stream.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    end_of_stream = subscriber->receive_sample();
  }
  ASSERT_FALSE(end_of_stream);
  // Key assertion: subscriber observes kClosed only after the data sample is drained, and every
  //                subsequent receive also returns kClosed without re-delivering any data.
  EXPECT_EQ(end_of_stream.error().code, ErrorCode::kClosed);
  auto repeated_receive = subscriber->receive_sample();
  ASSERT_FALSE(repeated_receive);
  EXPECT_EQ(repeated_receive.error().code, ErrorCode::kClosed);
  auto readiness_after_close = subscriber->has_samples();
  ASSERT_TRUE(readiness_after_close);
  EXPECT_FALSE(*readiness_after_close);
  EXPECT_FALSE(subscriber->has_publishers());
}

// Every post-submit terminal result is observed only after HoloMQ accepted the EOS. Each result
// therefore consumes both control-state and public-facade close authority.
// REQ: CHAN-078
// REQ: CHAN-110
TEST(ExternalTensorLoanRuntime, CatalogCloseConsumesAuthorityOnEveryTerminalPostSubmitOutcome) {
  constexpr std::array kTerminalCodes{
      ErrorCode::kPublicationIndeterminate,
      ErrorCode::kPublicationPartial,
      ErrorCode::kProtocolError,
      ErrorCode::kInvalidArgument,
  };
  for (std::size_t index = 0U; index < kTerminalCodes.size(); ++index) {
    const std::string endpoint_name =
        "tensor-terminal-close-" + std::to_string(::getpid()) + "-" + std::to_string(index);
    auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
    auto export_catalog = make_tensor_export_catalog(endpoint_name);
    ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
    ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

    auto publisher_runtime = ChannelRuntime::create();
    auto subscriber_runtime = ChannelRuntime::create();
    ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
    ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
    auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
    auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

    ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::fail_next_catalog_post_submit_for_test(
        *publisher, kTerminalCodes[index]));
    const auto terminal = publisher->close();
    ASSERT_FALSE(terminal);
    EXPECT_EQ(terminal.error().code, kTerminalCodes[index]);

    const auto repeated = publisher->close();
    ASSERT_FALSE(repeated);
    EXPECT_EQ(repeated.error().code, ErrorCode::kClosed);
    const auto later_loan = publisher->loan_tensor(TensorLoanRequest{});
    ASSERT_FALSE(later_loan);
    EXPECT_EQ(later_loan.error().code, ErrorCode::kClosed);
  }
}

// Every post-submit terminal result consumes the data publication's sequence exactly once while
// leaving the publisher available for the next logical sample.
// REQ: CHAN-078
// REQ: CHAN-111
TEST(ExternalTensorLoanRuntime, CatalogDataAdvancesSequenceOnEveryTerminalPostSubmitOutcome) {
  constexpr std::array kTerminalCodes{
      ErrorCode::kPublicationIndeterminate,
      ErrorCode::kPublicationPartial,
      ErrorCode::kProtocolError,
      ErrorCode::kInvalidArgument,
  };
  constexpr std::array<std::int64_t, 1U> kShape{8};
  const TensorLoanRequest request{
      .shape = kShape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  for (std::size_t index = 0U; index < kTerminalCodes.size(); ++index) {
    const std::string endpoint_name =
        "tensor-terminal-data-" + std::to_string(::getpid()) + "-" + std::to_string(index);
    auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
    auto export_catalog = make_tensor_export_catalog(endpoint_name);
    ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
    ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

    auto publisher_runtime = ChannelRuntime::create();
    auto subscriber_runtime = ChannelRuntime::create();
    ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
    ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
    auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
    auto subscriber = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
    ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

    auto first_loan = publisher->loan_tensor(request);
    ASSERT_TRUE(first_loan) << runtime_error_message(first_loan.error());
    auto first_writer = first_loan->write_host();
    ASSERT_TRUE(first_writer) << runtime_error_message(first_writer.error());
    std::memset(first_writer->data(), 0x41, first_writer->byte_size());
    ASSERT_TRUE(std::move(*first_writer).commit());
    ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::fail_next_catalog_post_submit_for_test(
        *publisher, kTerminalCodes[index]));
    const auto terminal = publisher->emit(std::move(*first_loan));
    ASSERT_FALSE(terminal);
    EXPECT_EQ(terminal.error().code, kTerminalCodes[index]);

    {
      auto first_received = subscriber->receive_sample();
      ASSERT_TRUE(first_received) << runtime_error_message(first_received.error());
      EXPECT_EQ(first_received->metadata.sequence, 1U);
    }

    auto second_loan = publisher->loan_tensor(request);
    ASSERT_TRUE(second_loan) << runtime_error_message(second_loan.error());
    auto second_writer = second_loan->write_host();
    ASSERT_TRUE(second_writer) << runtime_error_message(second_writer.error());
    std::memset(second_writer->data(), 0x42, second_writer->byte_size());
    ASSERT_TRUE(std::move(*second_writer).commit());
    const auto second = publisher->emit(std::move(*second_loan));
    ASSERT_TRUE(second) << runtime_error_message(second.error());
    auto second_received = subscriber->receive_sample();
    ASSERT_TRUE(second_received) << runtime_error_message(second_received.error());
    EXPECT_EQ(second_received->metadata.sequence, 2U);
  }
}

// A required Tensor subscriber refreshes its transport connection view while it waits. A
// publisher that is created after the subscriber's attach begins must therefore satisfy the
// finite wait instead of being hidden behind the subscriber's initial empty snapshot.
// REQ: EXTERNAL-001
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, RequiredSubscriberObservesPublisherCreatedDuringFiniteWait) {
  //   subscriber (blocked in required-peer wait) <-- publisher (created mid-wait via barrier)
  const std::string endpoint_name = "tensor-required-late-publisher-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   TopicAttachPolicy{
                                                       .peer = PeerRequirement::kRequired,
                                                       .max_wait = std::chrono::seconds{3},
                                                   });
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());

  channel_internal::TensorPublisherTestAccess::RequiredPublisherWaitBarrier barrier;
  channel_internal::TensorPublisherTestAccess::pause_next_required_publisher_wait_for_test(barrier);
  auto subscriber_future = std::async(std::launch::async, [&] {
    return subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  });
  const bool subscriber_waiting = barrier.reached.try_acquire_for(std::chrono::seconds{2});

  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  barrier.resume.release();
  ASSERT_TRUE(subscriber_waiting);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  auto subscriber = subscriber_future.get();
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  EXPECT_TRUE(subscriber->has_publishers());

  const std::array<std::int64_t, 1U> shape{1};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  *static_cast<std::byte*>(writer->data()) = std::byte{0x2A};
  ASSERT_TRUE(std::move(writer).value().commit());
  ASSERT_TRUE(publisher->emit(std::move(loan).value()));

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_NE(received->data.data(), nullptr);
  EXPECT_EQ(*static_cast<const std::byte*>(received->data.data()), std::byte{0x2A});
}

// A queued EOS is usable peer evidence even if the short-lived publisher retires before the
// required subscriber's connection loop observes a live publisher snapshot.
// REQ: CHAN-136
TEST(ExternalTensorLoanRuntime, RequiredSubscriberAcceptsRetainedEosAfterPublisherRetires) {
  const std::string endpoint_name = "tensor-required-retained-eos-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   TopicAttachPolicy{
                                                       .peer = PeerRequirement::kRequired,
                                                       .max_wait = std::chrono::seconds{3},
                                                   });
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto publisher_runtime = ChannelRuntime::create();
  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());

  channel_internal::TensorPublisherTestAccess::RequiredPublisherWaitBarrier barrier;
  barrier.pause_after_refresh = true;
  channel_internal::TensorPublisherTestAccess::pause_next_required_publisher_wait_for_test(barrier);
  auto subscriber_future = std::async(std::launch::async, [&] {
    return subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  });
  const bool subscriber_waiting = barrier.reached.try_acquire_for(std::chrono::seconds{2});

  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  barrier.resume.release();
  const bool subscriber_refreshed = barrier.refreshed.try_acquire_for(std::chrono::seconds{2});
  expected<void, Error> closed =
      publisher ? publisher->close() : make_unexpected(Error{ErrorCode::kFailure});
  barrier.resume_after_refresh.release();

  auto subscriber = subscriber_future.get();
  ASSERT_TRUE(subscriber_waiting);
  ASSERT_TRUE(subscriber_refreshed);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(closed) << runtime_error_message(closed.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  auto terminal = subscriber->receive_sample();
  ASSERT_FALSE(terminal);
  EXPECT_EQ(terminal.error().code, ErrorCode::kClosed);
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
  auto repeated = subscriber->receive_sample();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ErrorCode::kClosed);
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
}

// Catalog Tensor attachment fails before transport setup when the endpoint role, storage family,
// or caller-local wait bound is outside the admitted BufferHandle contract. A required peer uses
// the caller's shorter positive bound and never converts absence into an unbounded wait.
// REQ: EXTERNAL-001
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, CatalogSubscriberFailsClosedOutsideTheAttachmentContract) {
  const std::string endpoint_name = "tensor-public-negative-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto unsupported_export_catalog =
      make_tensor_export_catalog(endpoint_name, 1U, MemoryKind::kCudaManaged);
  auto required_export_catalog =
      make_tensor_export_catalog(endpoint_name + "-required",
                                 1U,
                                 MemoryKind::kHost,
                                 TopicAttachPolicy{
                                     .peer = PeerRequirement::kRequired,
                                     .max_wait = std::chrono::milliseconds{50},
                                 });
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(unsupported_export_catalog)
      << runtime_error_message(unsupported_export_catalog.error());
  ASSERT_TRUE(required_export_catalog) << runtime_error_message(required_export_catalog.error());

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());

  auto wrong_role = runtime->create_subscriber<Tensor>(*import_catalog, endpoint_name);
  ASSERT_FALSE(wrong_role);
  EXPECT_EQ(wrong_role.error().code, ErrorCode::kInvalidArgument);

  auto unsupported_storage =
      runtime->create_subscriber<Tensor>(*unsupported_export_catalog, endpoint_name);
  ASSERT_FALSE(unsupported_storage);
  EXPECT_EQ(unsupported_storage.error().code, ErrorCode::kNotSupported);

  auto excessive_bound = runtime->create_subscriber<Tensor>(
      *required_export_catalog,
      endpoint_name + "-required",
      ChannelAttachOptions{.timeout = std::chrono::milliseconds{100}});
  ASSERT_FALSE(excessive_bound);
  EXPECT_EQ(excessive_bound.error().code, ErrorCode::kInvalidArgument);

  const auto started = std::chrono::steady_clock::now();
  auto absent_required_peer = runtime->create_subscriber<Tensor>(
      *required_export_catalog,
      endpoint_name + "-required",
      ChannelAttachOptions{.timeout = std::chrono::milliseconds{2}});
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_FALSE(absent_required_peer);
  EXPECT_EQ(absent_required_peer.error().code, ErrorCode::kTimeout);
  EXPECT_LT(elapsed, std::chrono::milliseconds{250});
}

// BufferHandle-only attachment does not consume a process-local sidecar recipient. A timed-out
// required attach can therefore be retried after the publisher appears without exhausting a
// monotonic recipient slot that this transport mode never uses.
// REQ: EXTERNAL-001
// REQ: EXTERNAL-004
TEST(ExternalTensorLoanRuntime, RequiredSubscriberCanRetryAfterAttachTimeout) {
  const std::string endpoint_name = "tensor-required-retry-" + std::to_string(::getpid());
  auto import_catalog = make_tensor_import_catalog(endpoint_name, 1U);
  auto export_catalog = make_tensor_export_catalog(endpoint_name,
                                                   1U,
                                                   MemoryKind::kHost,
                                                   TopicAttachPolicy{
                                                       .peer = PeerRequirement::kRequired,
                                                       .max_wait = std::chrono::seconds{1},
                                                   });
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto subscriber_runtime = ChannelRuntime::create();
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  auto timed_out = subscriber_runtime->create_subscriber<Tensor>(
      *export_catalog,
      endpoint_name,
      ChannelAttachOptions{.timeout = std::chrono::milliseconds{2}});
  ASSERT_FALSE(timed_out);
  EXPECT_EQ(timed_out.error().code, ErrorCode::kTimeout);

  auto publisher_runtime = ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  auto publisher = publisher_runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  auto retried = subscriber_runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(retried) << runtime_error_message(retried.error());
  EXPECT_TRUE(retried->has_publishers());
}

// A catalog-created Tensor publisher must not quietly fall back to process-local pointer sharing.
// The child attaches after exec() through the frozen backend service identity and validates the
// imported read-only Tensor bytes, proving that the endpoint-owned host pool exports BufferHandles
// without exposing HoloMQ storage types through the public Channel API.
// REQ: CHAN-082
// REQ: CHAN-088
TEST(ExternalTensorLoanRuntime, CatalogLoanCrossesAnExecBoundaryThroughOpaqueBufferHandle) {
  //   parent (publisher) --[fork+exec, pipe handshake]--> child (raw HoloMQ subscriber)
  //   Proves the endpoint-owned host pool exports BufferHandles that survive an exec() boundary.
  const char* const child_exe = HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE not configured";

  constexpr std::uint32_t kSeed = 0x51000000U;
  constexpr std::size_t kByteCount = 16U;
  const std::string endpoint_name = "tensor-catalog-process-" + std::to_string(::getpid());
  auto catalog = make_tensor_import_catalog(endpoint_name, 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  ASSERT_NE(endpoint_record, nullptr);
  const std::string service_name =
      detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint);

  std::array<int, 2U> result_pipe{};
  std::array<int, 2U> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    const std::string expected_bytes = std::to_string(kByteCount);
    const std::string expected_seed = std::to_string(kSeed);
    const std::string max_queue_depth =
        std::to_string(endpoint_record->service_static.max_subscriber_depth);
    const std::string max_borrowed =
        std::to_string(endpoint_record->service_static.max_subscriber_borrowed);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_NAME", service_name.c_str(), 1));
    static_cast<void>(
        ::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_BYTES", expected_bytes.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_SEED", expected_seed.c_str(), 1));
    static_cast<void>(
        ::setenv("HOLOSCAN_CHANNEL_TENSOR_MAX_QUEUE_DEPTH", max_queue_depth.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_MAX_BORROWED", max_borrowed.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_TENSOR_EVENT_WAKE",
                               endpoint_record->service_static.event_wake ? "1" : "0",
                               1));
    ::execl(child_exe, "channel_tensor_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();
  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "Tensor child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }
  ASSERT_TRUE(wait_for_tensor_subscribers(*publisher));

  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  auto* words = static_cast<std::uint32_t*>(writer->data());
  for (std::size_t index = 0U; index < kByteCount / sizeof(std::uint32_t); ++index) {
    words[index] = kSeed + static_cast<std::uint32_t>(index);
  }
  ASSERT_TRUE(std::move(*writer).commit());
  auto emitted = publisher->emit(std::move(*loan));
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);

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

// TA-T31/TA-T48: a checked no-recipient observation occurs before transport submission. The
// by-value loan is still consumed exactly once, publishes no sequence, and its slot is recoverable.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, NoRecipientPreSubmitFailureConsumesAndRecyclesLoan) {
  auto catalog = make_tensor_import_catalog("tensor-no-recipient", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-no-recipient");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto loan = publisher->loan_tensor(request);
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  ASSERT_TRUE(std::move(writer).value().commit());

  auto rejected = publisher->emit(std::move(loan).value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kNotReady);

  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// TA-T31/TA-T48: an acknowledgement lost after the backend accepts the sample is not a safe
// retry signal. The caller observes the exact indeterminate category, while the one potentially
// visible publication retains the slot through queued and received claims. The next publication
// starts at sequence two, proving the uncertain attempt consumed its sequence without retry.
// REQ: CHAN-076
// REQ: CHAN-082
// REQ: EXTERNAL-003
TEST(ExternalTensorLoanRuntime,
     PostSubmitIndeterminateConsumesWithoutRetryAndRetainsClaimsUntilRetirement) {
  //   publisher --[ack injected to fail after backend accepts]--> subscriber
  //   Exact backing stays unavailable while queued and borrowed; sequence 2 proves no retry.
  auto catalog = make_tensor_import_catalog("tensor-post-submit-indeterminate", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint("tensor-post-submit-indeterminate");
  ASSERT_TRUE(endpoint);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto uncertain_loan = publisher->loan_tensor(request);
  ASSERT_TRUE(uncertain_loan) << runtime_error_message(uncertain_loan.error());
  auto uncertain_writer = uncertain_loan->write_host();
  ASSERT_TRUE(uncertain_writer) << runtime_error_message(uncertain_writer.error());
  void* const slot_address = uncertain_writer->data();
  std::memset(slot_address, 0x7B, uncertain_writer->byte_size());
  ASSERT_TRUE(std::move(uncertain_writer).value().commit());

  // Key setup: post-submit ack failure is injected so the exact backing remains held by the
  //            uncertain publication while queued and borrowed.
  ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::
                  fail_next_catalog_post_submit_acknowledgement_for_test(*publisher));
  auto uncertain = publisher->emit(std::move(uncertain_loan).value());
  ASSERT_FALSE(uncertain);
  EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
  // Even an indeterminate publication consumes the facade and leaves its observable view empty.
  EXPECT_EQ(uncertain_loan->byte_size(), 0U);  // NOLINT(bugprone-use-after-move)

  auto independent_while_queued = publisher->loan_tensor(request);
  ASSERT_TRUE(independent_while_queued) << runtime_error_message(independent_while_queued.error());
  auto independent_queued_writer = independent_while_queued->write_host();
  ASSERT_TRUE(independent_queued_writer)
      << runtime_error_message(independent_queued_writer.error());
  EXPECT_NE(independent_queued_writer->data(), slot_address);
  std::memset(independent_queued_writer->data(), 0xA5, independent_queued_writer->byte_size());
  ASSERT_TRUE(std::move(*independent_queued_writer).commit());
  ASSERT_TRUE(std::move(*independent_while_queued).cancel());

  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->metadata.sequence, 1U);
    ASSERT_NE(received->data.data(), nullptr);
    const auto* received_bytes = static_cast<const std::byte*>(received->data.data());
    for (std::size_t index = 0U; index < static_cast<std::size_t>(received->data.nbytes());
         ++index) {
      EXPECT_EQ(received_bytes[index], std::byte{0x7B});
    }
    EXPECT_TRUE(received->data.is_read_only());

    auto no_retry = subscriber->receive_sample();
    ASSERT_FALSE(no_retry);
    EXPECT_EQ(no_retry.error().code, ErrorCode::kNotReady);

    auto independent_while_borrowed = publisher->loan_tensor(request);
    ASSERT_TRUE(independent_while_borrowed)
        << runtime_error_message(independent_while_borrowed.error());
    auto independent_borrowed_writer = independent_while_borrowed->write_host();
    ASSERT_TRUE(independent_borrowed_writer)
        << runtime_error_message(independent_borrowed_writer.error());
    EXPECT_NE(independent_borrowed_writer->data(), slot_address);
    std::memset(
        independent_borrowed_writer->data(), 0x3C, independent_borrowed_writer->byte_size());
    ASSERT_TRUE(std::move(*independent_borrowed_writer).commit());
    ASSERT_TRUE(std::move(*independent_while_borrowed).cancel());
    for (std::size_t index = 0U; index < static_cast<std::size_t>(received->data.nbytes());
         ++index) {
      EXPECT_EQ(received_bytes[index], std::byte{0x7B});
    }
  }

  auto next_loan = publisher->loan_tensor(request);
  ASSERT_TRUE(next_loan) << runtime_error_message(next_loan.error());
  auto next_writer = next_loan->write_host();
  ASSERT_TRUE(next_writer) << runtime_error_message(next_writer.error());
  ASSERT_TRUE(std::move(next_writer).value().commit());
  auto next_emitted = publisher->emit(std::move(next_loan).value());
  ASSERT_TRUE(next_emitted) << runtime_error_message(next_emitted.error());
  EXPECT_EQ(*next_emitted, 1U);

  auto next_received = subscriber->receive_sample();
  ASSERT_TRUE(next_received) << runtime_error_message(next_received.error());
  // Key assertion: next successful emission carries sequence 2, proving the indeterminate attempt
  //                consumed its sequence number without triggering a silent retry at sequence 1.
  EXPECT_EQ(next_received->metadata.sequence, 2U);
}

// TA-T41/TA-T48/TA-T49: semantic metadata survives field-for-field, output-owned fields are
// replaced, and both malformed metadata and duplicate capture-time sources consume safely before
// publication without leaking the finite slot.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, SampleMetadataAndCaptureConflictFollowConsumingPrecedence) {
  //   publisher --[emit_sample / emit_captured variants]--> subscriber (field-for-field check)
  auto catalog = make_tensor_import_catalog("tensor-metadata", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint("tensor-metadata");
  ASSERT_TRUE(endpoint);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto make_ready = [&]() -> expected<TensorPublishLoan, Error> {
    auto loan = publisher->loan_tensor(request);
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    auto committed = std::move(writer).value().commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    return std::move(loan).value();
  };

  auto valid = make_ready();
  ASSERT_TRUE(valid) << runtime_error_message(valid.error());
  SampleMetadata metadata{
      .sequence = 99U,
      .publish_timestamp_ns = 1234,
      .publish_clock_id = ClockId{9U},
      .trace_id = 42U,
      .frame_id = 17U,
      .byte_size = 999U,
      .payload_kind = PayloadKind::kInlineTyped,
      .flags = SampleFlags::kDegraded,
  };
  // Key setup: caller-supplied metadata has intentionally wrong sequence, byte_size, and
  //            payload_kind to prove the publisher overwrites output-owned fields on the wire.
  auto emitted = publisher->emit_sample(std::move(valid).value(), metadata);
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    // Key assertion: pass-through fields (frame_id, trace_id, flags) survive unchanged while
    //                output-owned fields (sequence, byte_size, payload_kind, timestamps) are
    //                replaced by the publisher.
    EXPECT_EQ(received->metadata.sequence, 1U);
    EXPECT_EQ(received->metadata.frame_id, metadata.frame_id);
    EXPECT_EQ(received->metadata.trace_id, metadata.trace_id);
    EXPECT_EQ(received->metadata.flags, metadata.flags);
    EXPECT_EQ(received->metadata.publish_timestamp_ns, 0);
    EXPECT_EQ(received->metadata.publish_clock_id, ClockId{});
    EXPECT_EQ(received->metadata.byte_size, 8U);
    EXPECT_EQ(received->metadata.payload_kind, PayloadKind::kDescriptorBacked);
  }

  auto malformed = make_ready();
  ASSERT_TRUE(malformed) << runtime_error_message(malformed.error());
  metadata.version.major = 2U;
  auto invalid = publisher->emit_sample(std::move(malformed).value(), metadata);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);

  auto conflict = make_ready();
  ASSERT_TRUE(conflict) << runtime_error_message(conflict.error());
  const EmitOptions conflicting_options{
      .capture_time = TimePoint{.timestamp_ns = 5, .clock_id = ClockId{1U}},
  };
  auto conflicted = publisher->emit_captured(std::move(conflict).value(), 6, conflicting_options);
  ASSERT_FALSE(conflicted);
  EXPECT_EQ(conflicted.error().code, ErrorCode::kInvalidArgument);

  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// TA-T19/TA-T20/TA-T21: foreign storage is copied only through the explicit policy-gated verb.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, EmitCopyRejectsRequireExportablePolicyBeforeTakingPoolCapacity) {
  auto catalog = make_tensor_import_catalog(
      "tensor-copy-policy", 1U, InterfaceTensorEmitPolicy::kRequireExportable);
  ASSERT_TRUE(catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-copy-policy");
  ASSERT_TRUE(publisher);

  std::array<std::byte, 8U> storage{};
  const std::array<std::int64_t, 1U> shape{8};
  auto source = make_tensor(shape,
                            DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage.data());
  ASSERT_TRUE(source);
  auto copied = publisher->emit_copy(*source, ::cuda::stream_ref{cudaStream_t{nullptr}});
  ASSERT_FALSE(copied);
  EXPECT_EQ(copied.error().code, ErrorCode::kInvalidArgument);

  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan);
}

// TA-T19/TA-T20/TA-T21: the synchronous host-copy cell honors DLPack byte_offset, while a
// non-null producer stream rejects before reserving a destination slot.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, EmitCopyHonorsByteOffsetAndRejectsUnsupportedStreamBeforeCapacity) {
  //   publisher (host copy, byte_offset=2) --> subscriber (verifies 8-byte window after offset)
  auto catalog = make_tensor_import_catalog("tensor-copy-offset", 1U);
  ASSERT_TRUE(catalog);
  auto endpoint = catalog->endpoint("tensor-copy-offset");
  ASSERT_TRUE(endpoint);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher);
  auto subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(subscriber);
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  std::array<std::byte, 10U> storage{std::byte{0xEE},
                                     std::byte{0xEF},
                                     std::byte{0x01},
                                     std::byte{0x02},
                                     std::byte{0x03},
                                     std::byte{0x04},
                                     std::byte{0x05},
                                     std::byte{0x06},
                                     std::byte{0x07},
                                     std::byte{0x08}};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {8};
  context->tensor.dl_tensor.data = storage.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 2U;
  const Tensor source{std::move(context)};

  std::byte stream_sentinel{};
  auto unsupported = publisher->emit_copy(
      source, ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(&stream_sentinel)});
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().code, ErrorCode::kNotSupported);

  auto capacity = publisher->loan_tensor(TensorLoanRequest{
      .shape = source.shape_span(),
      .dtype = source.dtype(),
  });
  ASSERT_TRUE(capacity);
  ASSERT_TRUE(std::move(capacity).value().cancel());

  auto copied = publisher->emit_copy(source, ::cuda::stream_ref{cudaStream_t{nullptr}});
  ASSERT_TRUE(copied) << runtime_error_message(copied.error());
  EXPECT_EQ(*copied, 1U);
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_NE(received->data(), nullptr);
  EXPECT_EQ(std::memcmp(received->data(), storage.data() + 2U, 8U), 0);
}

// A CUDA source requires explicit producer-stream authority before pool acquisition. The accepted
// path orders one D2H materialization on that stream and exposes the host-pageable catalog backing.
// REQ: CHAN-091
TEST(ExternalTensorLoanRuntime, EmitCopyOrdersDeviceSourceBeforeHostCatalogPublication) {
  //   device source (producer_stream) --[D2H copy on stream]--> host catalog pool --> subscriber
  test::skip_without_cuda_device();

  auto catalog = make_tensor_import_catalog("tensor-copy-device", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint("tensor-copy-device");
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*endpoint);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  auto subscriber =
      channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime, *endpoint);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  constexpr std::size_t kByteCount = 8U;
  auto pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, 0);
  ASSERT_TRUE(pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  auto source = make_tensor(shape,
                            DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            pool,
                            ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(source) << runtime_error_message(source.error());
  auto destination = source->mutable_data();
  ASSERT_TRUE(destination) << runtime_error_message(destination.error());
  const std::array<std::byte, kByteCount> expected{std::byte{0x10},
                                                   std::byte{0x20},
                                                   std::byte{0x30},
                                                   std::byte{0x40},
                                                   std::byte{0x50},
                                                   std::byte{0x60},
                                                   std::byte{0x70},
                                                   std::byte{0x80}};
  ASSERT_EQ(cudaMemcpyAsync(*destination,
                            expected.data(),
                            expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer_stream.get()));

  auto missing_stream = publisher->emit_copy(*source, ::cuda::stream_ref{cudaStream_t{nullptr}});
  ASSERT_FALSE(missing_stream);
  EXPECT_EQ(missing_stream.error().code, ErrorCode::kNotSupported);
  auto preserved_capacity = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(preserved_capacity) << runtime_error_message(preserved_capacity.error());
  ASSERT_TRUE(std::move(preserved_capacity).value().cancel());

  auto copied = publisher->emit_copy(*source, ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(copied) << runtime_error_message(copied.error());
  EXPECT_EQ(*copied, 1U);
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_EQ(received->nbytes(), static_cast<std::int64_t>(expected.size()));
  EXPECT_EQ(std::memcmp(received->data(), expected.data(), expected.size()), 0);
}

// A CUDA catalog contract selects a HoloMQ device pool. The foreign source is copied once on its
// producer stream, and the imported BufferHandle remains device-backed for the consumer stream.
// REQ: CHAN-092
// REQ: EXTERNAL-006
TEST(ExternalTensorLoanRuntime, EmitCopyStagesDeviceSourceIntoDeviceCatalogPool) {
  //   device source (producer_stream) --[D2D copy]--> device catalog pool --> subscriber
  //   Source pool is released before producer_stream completes; emit_copy must retain backing.
  test::skip_without_cuda_device();

  // Key setup: wrap a device allocation in an imported-DLPack-like owner with no readiness
  // contract, while supplying the authoritative producer stream explicitly to emit_copy().
  const std::string endpoint_name = "tensor-copy-device-pool";
  auto import_catalog = make_tensor_import_catalog(endpoint_name,
                                                   1U,
                                                   InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                   {},
                                                   1U,
                                                   1U,
                                                   MemoryKind::kCudaDevice);
  auto export_catalog = make_tensor_export_catalog(endpoint_name, 1U, MemoryKind::kCudaDevice);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  auto subscriber = runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  constexpr std::size_t kByteCount = 8U;
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  const std::array<std::byte, kByteCount> expected{std::byte{0x11},
                                                   std::byte{0x22},
                                                   std::byte{0x33},
                                                   std::byte{0x44},
                                                   std::byte{0x55},
                                                   std::byte{0x66},
                                                   std::byte{0x77},
                                                   std::byte{0x88}};
  const void* source_address = nullptr;
  {
    auto source = make_tensor(shape,
                              request.dtype,
                              std::nullopt,
                              MemoryStorageType::kDevice,
                              source_pool,
                              ::cuda::stream_ref{producer_stream.get()});
    ASSERT_TRUE(source) << runtime_error_message(source.error());
    source_address = source->data();
    auto source_data = source->mutable_data();
    ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
    ASSERT_EQ(cudaMemcpyAsync(*source_data,
                              expected.data(),
                              expected.size(),
                              cudaMemcpyHostToDevice,
                              producer_stream.get()),
              cudaSuccess);

    const Tensor imported_source = make_contractless_device_tensor(*source);
    ASSERT_FALSE(imported_source.has_readiness_stream_contract());
    auto missing_stream =
        publisher->emit_copy(imported_source, ::cuda::stream_ref{cudaStream_t{nullptr}});
    ASSERT_FALSE(missing_stream);
    EXPECT_EQ(missing_stream.error().code, ErrorCode::kNotSupported);
    auto published =
        publisher->emit_copy(imported_source, ::cuda::stream_ref{producer_stream.get()});
    ASSERT_TRUE(published) << runtime_error_message(published.error());
    // Destruction before producer-stream completion proves emit_copy retains or orders the source
    // backing rather than transferring that lifetime obligation to the caller.
  }
  test::UniqueCudaStream consumer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);
  std::array<std::byte, kByteCount> actual{};
  {
    // This inner scope destroys the imported Tensor on scope exit, before consumer_stream is
    // destroyed, so deferred release never references an invalid CUDA stream.
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(received->data.device().device_id, device_ordinal);
    EXPECT_NE(received->data.data(), source_address);

    ASSERT_TRUE(received->data.synchronize_for_consumer_stream(consumer_stream.get()));
    ASSERT_EQ(cudaMemcpyAsync(actual.data(),
                              received->data.data(),
                              actual.size(),
                              cudaMemcpyDeviceToHost,
                              consumer_stream.get()),
              cudaSuccess);
    ASSERT_TRUE(received->data.set_deallocation_stream(consumer_stream.get()));
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);
  }
  // Key assertion: explicit stream authority orders the copy without requiring a readiness-aware
  // Tensor memory reference, and the source backing remains alive until that copy completes.
  EXPECT_EQ(actual, expected);
  // Imported-buffer destruction defers its HoloMQ ref release on consumer_stream. Wait for that
  // bounded backpressure to clear so no CUDA host callback escapes into the following test.
  auto recycled = wait_for_recycled_tensor_loan(*publisher, request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
  ASSERT_TRUE(publisher->close());
}

// A pinned catalog contract exposes CPU-addressable kDLCUDAHost storage while retaining a private
// CUDA device for generation synchronization. Both a completed CPU loan write and a
// producer-stream-ordered device copy publish the same pinned BufferHandle provider.
// REQ: CHAN-095
// REQ: EXTERNAL-007
TEST(ExternalTensorLoanRuntime, PinnedCatalogSupportsHostWritesAndOrderedDeviceCopies) {
  //   publisher (host write OR device emit_copy) --> kPinnedHost pool --> subscriber
  //   Exercises both the CPU-synchronous and producer-stream-ordered publication paths.
  test::skip_without_cuda_device();

  const std::string endpoint_name = "tensor-pinned-catalog";
  auto import_catalog = make_tensor_import_catalog(endpoint_name,
                                                   1U,
                                                   InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                   {},
                                                   1U,
                                                   1U,
                                                   MemoryKind::kPinnedHost);
  auto export_catalog = make_tensor_export_catalog(endpoint_name, 1U, MemoryKind::kPinnedHost);
  ASSERT_TRUE(import_catalog) << runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << runtime_error_message(export_catalog.error());

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*import_catalog, endpoint_name);
  auto subscriber = runtime->create_subscriber<Tensor>(*export_catalog, endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*publisher, *subscriber));

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  constexpr std::size_t kByteCount = 8U;
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  const std::array<std::byte, kByteCount> host_expected{std::byte{0x10},
                                                        std::byte{0x21},
                                                        std::byte{0x32},
                                                        std::byte{0x43},
                                                        std::byte{0x54},
                                                        std::byte{0x65},
                                                        std::byte{0x76},
                                                        std::byte{0x87}};
  auto host_loan = publisher->loan_tensor(request);
  ASSERT_TRUE(host_loan) << runtime_error_message(host_loan.error());
  EXPECT_EQ(host_loan->memory_kind(), MemoryKind::kPinnedHost);
  EXPECT_EQ(host_loan->device_id(), DeviceId{device_ordinal});
  auto host_writer = host_loan->write_host();
  ASSERT_TRUE(host_writer) << runtime_error_message(host_writer.error());
  std::memcpy(host_writer->data(), host_expected.data(), host_expected.size());
  ASSERT_TRUE(std::move(*host_writer).commit());
  ASSERT_TRUE(publisher->emit(std::move(*host_loan)));
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDAHost);
    EXPECT_EQ(received->data.device().device_id, 0);
    EXPECT_TRUE(received->data.is_read_only());
    ASSERT_TRUE(received->data.synchronize_for_host());
    EXPECT_EQ(std::memcmp(received->data.data(), host_expected.data(), host_expected.size()), 0);
  }

  // A host-written lease carries no producer readiness fence, so host synchronization does not
  // arm deferred CUDA release and the capacity-one pool recycles synchronously.
  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());

  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  const std::array<std::byte, kByteCount> device_expected{std::byte{0x91},
                                                          std::byte{0x82},
                                                          std::byte{0x73},
                                                          std::byte{0x64},
                                                          std::byte{0x55},
                                                          std::byte{0x46},
                                                          std::byte{0x37},
                                                          std::byte{0x28}};
  auto source = make_tensor(shape,
                            request.dtype,
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            source_pool,
                            ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(source) << runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  ASSERT_EQ(cudaMemcpyAsync(*source_data,
                            device_expected.data(),
                            device_expected.size(),
                            cudaMemcpyHostToDevice,
                            producer_stream.get()),
            cudaSuccess);
  ASSERT_TRUE(source->set_readiness_stream(producer_stream.get()));

  auto published = publisher->emit_copy(*source, ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(published) << runtime_error_message(published.error());
  {
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDAHost);
    EXPECT_EQ(received->data.device().device_id, 0);
    ASSERT_TRUE(received->data.synchronize_for_host());
    EXPECT_EQ(std::memcmp(received->data.data(), device_expected.data(), device_expected.size()),
              0);
  }
  auto final_recycled = wait_for_recycled_tensor_loan(*publisher, request);
  ASSERT_TRUE(final_recycled) << runtime_error_message(final_recycled.error());
  ASSERT_TRUE(std::move(*final_recycled).cancel());
  ASSERT_TRUE(publisher->close());
}

// A pinned-host source is copied synchronously on the CPU even when Runtime supplies a separate
// staging stream. The destination must not inherit a readiness fence for that unused stream.
// REQ: CHAN-095
TEST(ExternalTensorLoanRuntime, PinnedHostStagingDoesNotFenceSynchronousCopy) {
  //   foreign pinned source --[synchronous memcpy]--> pinned catalog staging slot
  //   blocked copy_stream --------------------------> (no work enqueued)
  test::skip_without_cuda_device();

  auto catalog = make_tensor_import_catalog("tensor-pinned-host-staging",
                                            1U,
                                            InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                            {},
                                            1U,
                                            1U,
                                            MemoryKind::kPinnedHost);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-pinned-host-staging");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  constexpr std::size_t kByteCount = 8U;
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto source_pool = std::make_shared<BlockMemoryPool>(MemoryStorageType::kHost, kByteCount, 1U);
  ASSERT_TRUE(source_pool->initialize_pool());
  auto source =
      make_tensor(shape, request.dtype, std::nullopt, MemoryStorageType::kHost, source_pool);
  ASSERT_TRUE(source) << runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  std::memset(*source_data, 0x5A, kByteCount);

  test::UniqueCudaStream copy_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(copy_stream), cudaSuccess);
  test::BlockingHostState blocked_copy;
  test::BlockingHostReleaseGuard release_guard{blocked_copy};
  ASSERT_EQ(cudaLaunchHostFunc(copy_stream.get(), test::blocking_host_callback, &blocked_copy),
            cudaSuccess);
  ASSERT_TRUE(test::wait_until_started(blocked_copy));

  // Key setup: offer a blocked staging stream even though pinned-to-pinned copy_to_host() performs
  // a completed CPU memcpy and enqueues no work on that stream.
  {
    auto staged = channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(
        *publisher, *source, nullptr, copy_stream.get());
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
    EXPECT_EQ(std::memcmp(staged->tensor.data(), *source_data, kByteCount), 0);
  }

  // Key assertion: dropping the staged Tensor recycles the capacity-one slot while copy_stream is
  // still blocked because the completed host copy has no CUDA producer work to fence.
  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
  test::release_blocking_host(blocked_copy);
  release_guard.dismiss();
  ASSERT_EQ(cudaStreamSynchronize(copy_stream.get()), cudaSuccess);
}

// A staged fanout copy owns one published pool claim independently of any particular endpoint
// submission. Destroying the staging result while another Tensor handle remains must protect that
// exact backing while independent producer loans use other preallocated slots.
// REQ: CHAN-082
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ExternalTensorLoanRuntime, StagedFanoutCopyRetainsOneSlotUntilLastTensorClaimReleases) {
  //   foreign source --[stage_foreign_tensor]--> provider pool
  //   No endpoint submission; slot recycles when the last retained Tensor handle is released.
  auto catalog = make_tensor_import_catalog("tensor-staged-fanout-claim", 1U);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-staged-fanout-claim");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  std::array<std::byte, 8U> storage{std::byte{0x01},
                                    std::byte{0x02},
                                    std::byte{0x03},
                                    std::byte{0x04},
                                    std::byte{0x05},
                                    std::byte{0x06},
                                    std::byte{0x07},
                                    std::byte{0x08}};
  const std::array<std::int64_t, 1U> shape{8};
  auto source = make_tensor(shape,
                            DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage.data());
  ASSERT_TRUE(source) << runtime_error_message(source.error());

  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };

  // With no endpoint submission and no retained Tensor handle, destroying the sole local fanout
  // claim must leave producer credit and physical backing available.
  {
    {
      auto staged =
          channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(*publisher, *source);
      ASSERT_TRUE(staged) << runtime_error_message(staged.error());
      ASSERT_NE(staged->tensor.data(), nullptr);
    }
    auto recycled = publisher->loan_tensor(request);
    ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
    ASSERT_TRUE(std::move(*recycled).cancel());
  }

  Tensor retained;
  const void* staged_data = nullptr;
  {
    auto staged =
        channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(*publisher, *source);
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
    staged_data = staged->tensor.data();
    ASSERT_NE(staged_data, nullptr);
    retained = staged->tensor;
  }

  auto independent = publisher->loan_tensor(request);
  ASSERT_TRUE(independent) << runtime_error_message(independent.error());
  auto independent_writer = independent->write_host();
  ASSERT_TRUE(independent_writer) << runtime_error_message(independent_writer.error());
  // Key assertion: the retained staged Tensor protects its exact backing while a producer loan
  //                uses a distinct preallocated slot.
  EXPECT_NE(independent_writer->data(), staged_data);
  std::memset(independent_writer->data(), 0xA5, independent_writer->byte_size());
  ASSERT_TRUE(std::move(*independent_writer).commit());
  ASSERT_TRUE(std::move(*independent).cancel());
  EXPECT_EQ(std::memcmp(retained.data(), storage.data(), storage.size()), 0);

  retained = Tensor{};
  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
}

// A provider can disappear from an optional fanout after its pool stages the payload. Publishing
// that staged Tensor through another endpoint must keep the provider's exact backing armed even
// when the secondary submit reports an indeterminate acknowledgement and the local fanout claim
// retires. Only the secondary recipient's final Tensor release may recycle the backing.
// REQ: CHAN-082
// REQ: SYSTEM-007
// REQ: SYSTEM-009
TEST(ExternalTensorLoanRuntime,
     StagedFanoutBackingSurvivesSkippedProviderAndIndeterminateSecondarySubmit) {
  //   foreign source --[stage_foreign_tensor]--> provider pool
  //                --[emit_prevalidated, ack injected to fail]--> recipient --> subscriber
  //   Provider slot stays held while subscriber borrows the copy; recycles on last Tensor retire.
  auto provider_catalog = make_tensor_import_catalog("tensor-staged-provider", 1U);
  auto recipient_catalog = make_tensor_import_catalog("tensor-staged-recipient", 1U);
  ASSERT_TRUE(provider_catalog) << runtime_error_message(provider_catalog.error());
  ASSERT_TRUE(recipient_catalog) << runtime_error_message(recipient_catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto provider = runtime->create_publisher<Tensor>(*provider_catalog, "tensor-staged-provider");
  auto recipient = runtime->create_publisher<Tensor>(*recipient_catalog, "tensor-staged-recipient");
  auto subscriber = channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(
      *runtime, *recipient_catalog->endpoint("tensor-staged-recipient"));
  ASSERT_TRUE(provider) << runtime_error_message(provider.error());
  ASSERT_TRUE(recipient) << runtime_error_message(recipient.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*recipient, *subscriber));

  std::array<std::byte, 8U> storage{std::byte{0x11},
                                    std::byte{0x22},
                                    std::byte{0x33},
                                    std::byte{0x44},
                                    std::byte{0x55},
                                    std::byte{0x66},
                                    std::byte{0x77},
                                    std::byte{0x88}};
  const std::array<std::int64_t, 1U> shape{8};
  auto source = make_tensor(shape,
                            DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage.data());
  ASSERT_TRUE(source) << runtime_error_message(source.error());

  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  const void* staged_data = nullptr;
  {
    auto staged =
        channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(*provider, *source);
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
    staged_data = staged->tensor.data();
    ASSERT_NE(staged_data, nullptr);
    ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::
                    fail_next_catalog_post_submit_acknowledgement_for_test(*recipient));
    auto submitted = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
        *recipient, staged->tensor, SampleMetadata{});
    ASSERT_FALSE(submitted);
    EXPECT_EQ(submitted.error().code, ErrorCode::kPublicationIndeterminate);
  }

  auto queued_claim = provider->loan_tensor(request);
  ASSERT_TRUE(queued_claim) << runtime_error_message(queued_claim.error());
  auto queued_writer = queued_claim->write_host();
  ASSERT_TRUE(queued_writer) << runtime_error_message(queued_writer.error());
  EXPECT_NE(queued_writer->data(), staged_data);
  std::memset(queued_writer->data(), 0xA5, queued_writer->byte_size());
  ASSERT_TRUE(std::move(*queued_writer).commit());
  ASSERT_TRUE(std::move(*queued_claim).cancel());

  {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    ASSERT_EQ(received->nbytes(), static_cast<std::int64_t>(storage.size()));
    EXPECT_EQ(std::memcmp(received->data(), storage.data(), storage.size()), 0);

    auto borrowed_claim = provider->loan_tensor(request);
    ASSERT_TRUE(borrowed_claim) << runtime_error_message(borrowed_claim.error());
    auto borrowed_writer = borrowed_claim->write_host();
    ASSERT_TRUE(borrowed_writer) << runtime_error_message(borrowed_writer.error());
    EXPECT_NE(borrowed_writer->data(), staged_data);
    std::memset(borrowed_writer->data(), 0x3C, borrowed_writer->byte_size());
    ASSERT_TRUE(std::move(*borrowed_writer).commit());
    ASSERT_TRUE(std::move(*borrowed_claim).cancel());
    EXPECT_EQ(std::memcmp(received->data(), storage.data(), storage.size()), 0);
  }

  ASSERT_TRUE(recipient->update_connections());
  auto recycled = provider->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
}

// A device staging destination is released through HoloMQ's writer-stream callback even when a
// secondary submit loses its acknowledgement. Retiring every Tensor claim while that stream is
// blocked must not make that exact destination backing reusable before the D2D copy completes.
// REQ: CHAN-092
TEST(ExternalTensorLoanRuntime,
     DeviceStagedFanoutFailureDefersDestinationReuseUntilWriterStreamCompletes) {
  //   device source --[D2D copy, blocked host callback on producer_stream]-->
  //       provider pool --[ack injected to fail]--> recipient --> subscriber
  //   The staged destination must remain unavailable until blocking_host_callback is released.
  test::skip_without_cuda_device();

  constexpr std::string_view kProviderName = "tensor-device-staged-provider";
  constexpr std::string_view kRecipientName = "tensor-device-staged-recipient";
  auto provider_catalog = make_tensor_import_catalog(std::string{kProviderName},
                                                     1U,
                                                     InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                     {},
                                                     1U,
                                                     1U,
                                                     MemoryKind::kCudaDevice);
  auto recipient_catalog = make_tensor_import_catalog(std::string{kRecipientName},
                                                      1U,
                                                      InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                                      {},
                                                      1U,
                                                      1U,
                                                      MemoryKind::kCudaDevice);
  ASSERT_TRUE(provider_catalog) << runtime_error_message(provider_catalog.error());
  ASSERT_TRUE(recipient_catalog) << runtime_error_message(recipient_catalog.error());

  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto provider = runtime->create_publisher<Tensor>(*provider_catalog, kProviderName);
  auto recipient = runtime->create_publisher<Tensor>(*recipient_catalog, kRecipientName);
  auto subscriber = channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(
      *runtime, *recipient_catalog->endpoint(kRecipientName));
  ASSERT_TRUE(provider) << runtime_error_message(provider.error());
  ASSERT_TRUE(recipient) << runtime_error_message(recipient.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_tensor_subscriber(*recipient, *subscriber));

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  constexpr std::size_t kByteCount = 8U;
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  auto source = make_tensor(shape,
                            request.dtype,
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            source_pool,
                            ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(source) << runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  ASSERT_EQ(cudaMemsetAsync(*source_data, 0x5A, kByteCount, producer_stream.get()), cudaSuccess);

  test::BlockingHostState blocked_stream;
  test::BlockingHostReleaseGuard release_guard{blocked_stream};
  ASSERT_EQ(
      cudaLaunchHostFunc(producer_stream.get(), test::blocking_host_callback, &blocked_stream),
      cudaSuccess);
  ASSERT_TRUE(test::wait_until_started(blocked_stream));

  const void* staged_data = nullptr;
  {
    // Stage after the blocked callback so the D2D copy and HoloMQ ready-generation record remain
    // pending while the secondary endpoint accepts the descriptor.
    auto staged = channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(
        *provider, *source, producer_stream.get());
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
    staged_data = staged->tensor.data();
    ASSERT_NE(staged_data, nullptr);
    ASSERT_TRUE(channel_internal::TensorPublishLoanAccess::
                    fail_next_catalog_post_submit_acknowledgement_for_test(*recipient));
    auto submitted = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
        *recipient, staged->tensor, SampleMetadata{});
    ASSERT_FALSE(submitted);
    EXPECT_EQ(submitted.error().code, ErrorCode::kPublicationIndeterminate);
  }

  {
    // Drop the imported recipient claim without waiting for the payload. The provider owner's
    // deferred stream release must now be the only reason its physical slot remains unavailable.
    auto received = subscriber->receive_sample();
    ASSERT_TRUE(received) << runtime_error_message(received.error());
    EXPECT_EQ(received->data.device().device_type, kDLCUDA);
    EXPECT_EQ(received->data.device().device_id, device_ordinal);
  }
  ASSERT_TRUE(recipient->update_connections());

  // Key assertion: another producer loan may use independent capacity, but the staged destination
  //                is not reused while the blocking host callback holds its writer stream.
  auto blocked_reuse = provider->loan_tensor(request);
  ASSERT_TRUE(blocked_reuse) << runtime_error_message(blocked_reuse.error());
  {
    auto blocked_writer = blocked_reuse->write(::cuda::stream_ref{producer_stream.get()});
    ASSERT_TRUE(blocked_writer) << runtime_error_message(blocked_writer.error());
    EXPECT_NE(blocked_writer->data(), staged_data);
  }
  ASSERT_TRUE(std::move(*blocked_reuse).cancel());

  test::release_blocking_host(blocked_stream);
  release_guard.dismiss();
  ASSERT_EQ(cudaStreamSynchronize(producer_stream.get()), cudaSuccess);

  auto recycled = provider->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
}

// REQ: CHAN-134
TEST(ExternalTensorLoanRuntime, DeviceStagingUsesSeparateOrderedCopyStream) {
  test::skip_without_cuda_device();

  // Key setup: block a producer stream after writing device memory, then stage the Tensor on a
  // separate copy stream that must wait for producer readiness.
  constexpr std::string_view kEndpointName = "tensor-device-service-stream-staging";
  auto catalog = make_tensor_import_catalog(std::string{kEndpointName},
                                            1U,
                                            InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                            {},
                                            1U,
                                            1U,
                                            MemoryKind::kCudaDevice);
  ASSERT_TRUE(catalog) << runtime_error_message(catalog.error());
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime) << runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<Tensor>(*catalog, kEndpointName);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  const std::size_t preallocated_event_count =
      channel_internal::TensorPublishLoanAccess::preallocated_copy_event_count_for_test(*publisher);
  EXPECT_GT(preallocated_event_count, 0U);

  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  constexpr std::size_t kByteCount = 8U;
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kByteCount)};
  const DLDataType dtype{.code = kDLUInt, .bits = 8U, .lanes = 1U};
  auto source_pool =
      std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, kByteCount, 1U, device_ordinal);
  ASSERT_TRUE(source_pool->initialize_pool());
  test::UniqueCudaStream producer_stream;
  test::UniqueCudaStream copy_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  ASSERT_EQ(test::create_non_blocking_cuda_stream(copy_stream), cudaSuccess);
  auto source = make_tensor(shape,
                            dtype,
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            source_pool,
                            ::cuda::stream_ref{producer_stream.get()});
  ASSERT_TRUE(source) << runtime_error_message(source.error());
  auto source_data = source->mutable_data();
  ASSERT_TRUE(source_data) << runtime_error_message(source_data.error());
  ASSERT_EQ(cudaMemsetAsync(*source_data, 0x6B, kByteCount, producer_stream.get()), cudaSuccess);

  test::BlockingHostState blocked_producer;
  test::BlockingHostReleaseGuard release_guard{blocked_producer};
  ASSERT_EQ(
      cudaLaunchHostFunc(producer_stream.get(), test::blocking_host_callback, &blocked_producer),
      cudaSuccess);
  ASSERT_TRUE(test::wait_until_started(blocked_producer));
  const Tensor contractless_source = make_contractless_device_tensor(*source);
  ASSERT_FALSE(contractless_source.has_readiness_stream_contract());

  auto staged = channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(
      *publisher, contractless_source, producer_stream.get(), copy_stream.get());
  ASSERT_TRUE(staged) << runtime_error_message(staged.error());
  EXPECT_EQ(
      channel_internal::TensorPublishLoanAccess::preallocated_copy_event_count_for_test(*publisher),
      preallocated_event_count);
  EXPECT_EQ(cudaStreamQuery(copy_stream.get()), cudaErrorNotReady);

  // Key assertion: releasing the producer lets the copy stream finish with the exact byte pattern.
  test::release_blocking_host(blocked_producer);
  release_guard.dismiss();
  ASSERT_EQ(cudaStreamSynchronize(copy_stream.get()), cudaSuccess);
  std::array<std::byte, kByteCount> observed{};
  ASSERT_TRUE(staged->tensor.copy_to_host(observed.data(), observed.size(), copy_stream.get()));
  EXPECT_TRUE(
      std::ranges::all_of(observed, [](std::byte value) { return value == std::byte{0x6B}; }));
}

// TA-T12/TA-T15/TA-T24: consuming a loan while its writer is live invalidates publication, and the
// final writer transition returns the exact slot rather than leaking one unit of pool capacity.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, ConsumedLoanWithLiveWriterRecyclesAfterWriterFinishes) {
  auto catalog = make_tensor_import_catalog("tensor-live-writer", 1U);
  ASSERT_TRUE(catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-live-writer");
  ASSERT_TRUE(publisher);
  const std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(loan);
  auto writer = loan->write_host();
  ASSERT_TRUE(writer);

  auto emitted = publisher->emit(std::move(loan).value());
  ASSERT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, ErrorCode::kInvalidArgument);
  auto committed = std::move(writer).value().commit();
  ASSERT_FALSE(committed);
  EXPECT_EQ(committed.error().code, ErrorCode::kInvalidArgument);

  auto recycled = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// A failed stream-readiness registration consumes the writer but leaves the loan cancellable;
// consuming that loan must return its sole finite slot instead of stranding it in kWriting.
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, ReadinessRegistrationFailureReturnsSlotAfterCancellation) {
  test::skip_without_cuda_device();

  auto catalog = make_tensor_import_catalog("tensor-readiness-failure",
                                            1U,
                                            InterfaceTensorEmitPolicy::kCopyIfNeeded,
                                            {},
                                            1U,
                                            1U,
                                            MemoryKind::kPinnedHost);
  ASSERT_TRUE(catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-readiness-failure");
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  const std::array<std::int64_t, 1U> shape{8};
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto loan = publisher->loan_tensor(request);
  ASSERT_TRUE(loan);
  test::UniqueCudaStream stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(stream), cudaSuccess);
  auto writer = loan->write(::cuda::stream_ref{stream.get()});
  ASSERT_TRUE(writer);
  ASSERT_TRUE(
      channel_internal::TensorPublishLoanAccess::fail_next_catalog_readiness_registration_for_test(
          *publisher));

  auto committed = std::move(*writer).commit();
  ASSERT_FALSE(committed);
  EXPECT_EQ(committed.error().code, ErrorCode::kFailure);
  ASSERT_TRUE(std::move(*loan).cancel());

  auto recycled = publisher->loan_tensor(request);
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
  ASSERT_TRUE(std::move(*recycled).cancel());
}

// TA-T04/TA-T14/TA-T25: request validation precedes capacity accounting, and a loan consumed by a
// foreign publisher cannot be retried but is returned safely to its owning pool.
// REQ: CHAN-081
// REQ: CHAN-082
TEST(ExternalTensorLoanRuntime, InvalidRequestAndForeignPublisherDoNotLeakCapacity) {
  auto first_catalog = make_tensor_import_catalog("tensor-owner-a", 1U);
  auto second_catalog = make_tensor_import_catalog("tensor-owner-b", 1U);
  ASSERT_TRUE(first_catalog);
  ASSERT_TRUE(second_catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto first = runtime->create_publisher<Tensor>(*first_catalog, "tensor-owner-a");
  auto second = runtime->create_publisher<Tensor>(*second_catalog, "tensor-owner-b");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  auto expect_invalid = [&first](const TensorLoanRequest request) {
    auto invalid = first->loan_tensor(request);
    EXPECT_FALSE(invalid);
    if (!invalid) {
      EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
    }
  };
  const std::array<std::int64_t, 1U> oversized_shape{17};
  expect_invalid(TensorLoanRequest{
      .shape = oversized_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  const std::array<std::int64_t, 3U> excessive_rank{1, 1, 1};
  expect_invalid(TensorLoanRequest{
      .shape = excessive_rank,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  const std::array<std::int64_t, 2U> matrix_shape{2, 4};
  const std::array<std::int64_t, 1U> missing_stride{1};
  expect_invalid(TensorLoanRequest{
      .shape = matrix_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .byte_strides = TensorByteStridesSpan{missing_stride},
  });
  const std::array<std::int64_t, 2U> noncontiguous_stride{5, 1};
  expect_invalid(TensorLoanRequest{
      .shape = matrix_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .byte_strides = TensorByteStridesSpan{noncontiguous_stride},
  });
  const std::array<std::int64_t, 1U> negative_shape{-1};
  expect_invalid(TensorLoanRequest{
      .shape = negative_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  const std::array<std::int64_t, 2U> zero_then_negative_shape{0, -1};
  expect_invalid(TensorLoanRequest{
      .shape = zero_then_negative_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  const std::array<std::int64_t, 2U> overflow_shape{std::numeric_limits<std::int64_t>::max(), 2};
  expect_invalid(TensorLoanRequest{
      .shape = overflow_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 64U, .lanes = 1U},
  });
  expect_invalid(TensorLoanRequest{
      .shape = matrix_shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 7U, .lanes = 1U},
  });

  const std::array<std::int64_t, 1U> shape{8};
  auto owned = first->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(owned);
  auto wrong_publisher = second->emit(std::move(owned).value());
  ASSERT_FALSE(wrong_publisher);
  EXPECT_EQ(wrong_publisher.error().code, ErrorCode::kInvalidArgument);

  auto recycled = first->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  ASSERT_TRUE(recycled) << runtime_error_message(recycled.error());
}

// Bounded-strided loans accept ordinary axis permutations but reject layouts where logical
// elements alias the same storage.
// REQ: CHAN-081
TEST(ExternalTensorLoanRuntime, RejectsOverlappingBoundedStrides) {
  auto catalog = make_strided_tensor_import_catalog("tensor-strided-layout");
  ASSERT_TRUE(catalog);
  auto runtime = ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<Tensor>(*catalog, "tensor-strided-layout");
  ASSERT_TRUE(publisher);

  const std::array<std::int64_t, 2U> shape{2, 2};
  const auto expect_invalid = [&publisher, &shape](const std::array<std::int64_t, 2U>& strides) {
    auto loan = publisher->loan_tensor(TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
        .byte_strides = TensorByteStridesSpan{strides},
    });
    ASSERT_FALSE(loan);
    EXPECT_EQ(loan.error().code, ErrorCode::kInvalidArgument);
  };
  expect_invalid({1, 1});
  expect_invalid({0, 1});

  const std::array<std::int64_t, 2U> transposed_strides{1, 2};
  auto transposed = publisher->loan_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .byte_strides = TensorByteStridesSpan{transposed_strides},
  });
  ASSERT_TRUE(transposed) << runtime_error_message(transposed.error());
  EXPECT_EQ(transposed->byte_size(), 4U);
}

}  // namespace
}  // namespace holoscan
