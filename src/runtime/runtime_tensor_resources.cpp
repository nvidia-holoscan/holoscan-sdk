/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_tensor_resources.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <holomq/buffer.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/context_builder.hpp>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include "../channel/channel_common.hpp"
#include "../channel/holomq_tensor.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/domain/tensor_access.hpp"
#include "../core/domain/tensor_layout_validation.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/tensor_allocation_record_validation.hpp"
#include "../core/tensor_output_loan_internal.hpp"
#include "graph_exec.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {

namespace {

constexpr std::size_t kTensorResourceErrorCapacity = 192U;

[[nodiscard]] Error tensor_resource_error(std::string_view context,
                                          std::string_view identity_kind = {},
                                          std::string_view identity = {}) noexcept {
  std::array<char, kTensorResourceErrorCapacity> buffer{};
  std::size_t size = 0U;
  const auto append = [&](std::string_view text) noexcept {
    const std::size_t available = buffer.size() - size;
    const std::size_t count = std::min(available, text.size());
    std::copy_n(text.data(), count, buffer.data() + size);
    size += count;
  };
  append("PROFILE_MATERIALIZATION_FAILED: ");
  append(context);
  if (!identity_kind.empty() && !identity.empty()) {
    append(" [");
    append(identity_kind);
    append("=");
    append(identity);
    append("]");
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return Error{ErrorCode::kFailure, std::string(buffer.data(), size)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Error reporting must not turn a bounded setup failure into process termination.
    return Error{ErrorCode::kFailure};
  }
#endif
}

[[nodiscard]] Error tensor_resource_error(std::string_view context, std::string_view identity_kind,
                                          std::uint64_t identity) noexcept {
  std::array<char, std::numeric_limits<std::uint64_t>::digits10 + 2U> number{};
  const auto converted = std::to_chars(number.data(), number.data() + number.size(), identity);
  const std::string_view text =
      converted.ec == std::errc()
          ? std::string_view(number.data(), static_cast<std::size_t>(converted.ptr - number.data()))
          : std::string_view("unknown");
  return tensor_resource_error(context, identity_kind, text);
}

[[nodiscard]] bool checked_add(std::uint32_t left, std::uint32_t right,
                               std::uint32_t& output) noexcept {
  if (right > std::numeric_limits<std::uint32_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] constexpr std::size_t claim_index(TensorSlotClaimReason reason) noexcept {
  return static_cast<std::size_t>(reason);
}

[[nodiscard]] bool canonical_claim_reasons(
    std::span<const TensorSlotClaimReason> reasons) noexcept {
  if (reasons.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < reasons.size(); ++index) {
    if (claim_index(reasons[index]) >= kTensorSlotClaimReasonCount ||
        (index != 0U && claim_index(reasons[index - 1U]) >= claim_index(reasons[index]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool contains_claim_reason(std::span<const TensorSlotClaimReason> reasons,
                                         TensorSlotClaimReason expected) noexcept {
  return std::find(reasons.begin(), reasons.end(), expected) != reasons.end();
}

struct OutputPlanContract {
  PublicationOutcomePlanRecord publication;
  TensorSlotClaimLedger::Admission admitted_claims{};
};

struct ClaimContributorValidation {
  TensorSlotClaimLedger::Admission admitted_claims{};
  std::uint32_t capacity{};
  bool producer_seen{};
  bool delivery_seen{};
  bool transport_retention_seen{};
  bool delivery_includes_transport_retention{};
  bool retention_seen{};
  bool retirement_seen{};
};

struct ClaimContributorIds {
  std::string producer;
  std::string delivery;
  std::string transport_retention;
  std::string retention;
  std::string retirement;
};

[[nodiscard]] bool valid_publication_errors(const PublicationOutcomePlanRecord& outcome) noexcept {
  return outcome.pre_submit_error == ErrorCode::kBackpressured &&
         outcome.post_submit_error == ErrorCode::kPublicationIndeterminate &&
         outcome.post_submit_partial_error == ErrorCode::kPublicationPartial;
}

[[nodiscard]] bool valid_publication_authority(
    const PublicationOutcomePlanRecord& outcome) noexcept {
  return outcome.pre_submit_authority == PreSubmitAuthorityDisposition::kConsumedToRetirement &&
         outcome.post_submit_consumes_authority && outcome.retain_claims_until_retirement;
}

[[nodiscard]] bool valid_publication_sequence(
    const PublicationOutcomePlanRecord& outcome) noexcept {
  return !outcome.pre_submit_assigns_sequence && outcome.post_submit_assigns_sequence &&
         !outcome.runtime_retry_enabled;
}

[[nodiscard]] expected<PublicationOutcomePlanRecord, Error> validate_publication_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) {
  const auto outcome = std::find_if(
      plan.publication_outcomes.begin(), plan.publication_outcomes.end(), [&](const auto& row) {
        return row.output == allocation.output;
      });
  const auto output_matches = [&](const auto& row) { return row.output == allocation.output; };
  if (outcome == plan.publication_outcomes.end() ||
      std::count_if(plan.publication_outcomes.begin(),
                    plan.publication_outcomes.end(),
                    output_matches) != 1 ||
      !outcome->tensor || !valid_publication_errors(*outcome) ||
      !valid_publication_authority(*outcome) || !valid_publication_sequence(*outcome)) {
    return make_unexpected(tensor_resource_error(
        "publication contract mismatch", "output", allocation.output.value()));
  }
  return *outcome;
}

[[nodiscard]] expected<void, Error> record_claim_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  if (contributor.maximum_distinct_slots == 0U || !canonical_claim_reasons(contributor.reasons) ||
      !checked_add(validation.capacity, contributor.maximum_distinct_slots, validation.capacity)) {
    return make_unexpected(
        tensor_resource_error("claim contributor capacity mismatch", "pool", allocation.pool_id));
  }
  for (const auto reason : contributor.reasons) {
    const std::size_t index = claim_index(reason);
    if (validation.admitted_claims[index]) {
      return make_unexpected(
          tensor_resource_error("duplicate claim contributor reason", "pool", allocation.pool_id));
    }
    validation.admitted_claims[index] = true;
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_producer_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  constexpr std::array expected_reasons{TensorSlotClaimReason::kProducerLoan,
                                        TensorSlotClaimReason::kProducerWriter,
                                        TensorSlotClaimReason::kProducerReadiness};
  if (validation.producer_seen ||
      contributor.maximum_distinct_slots != allocation.max_outstanding_loans ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.end())) {
    return make_unexpected(
        tensor_resource_error("producer claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.producer_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_delivery_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  std::array<TensorSlotClaimReason, 6U> expected_reasons{};
  std::size_t expected_count = 0U;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kAcceptedPublication;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kHolomqTransportOrFanout;
  const bool includes_transport_retention =
      contains_claim_reason(contributor.reasons, TensorSlotClaimReason::kSubscriberBorrowOrImport);
  if (includes_transport_retention) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
    if (allocation.memory_kind == MemoryKind::kCudaDevice ||
        allocation.memory_kind == MemoryKind::kCudaManaged) {
      expected_reasons[expected_count++] = TensorSlotClaimReason::kConsumerCudaRelease;
    }
    if (std::find(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  TensorSlotClaimReason::kRelayOrReexport) != contributor.reasons.end()) {
      expected_reasons[expected_count++] = TensorSlotClaimReason::kRelayOrReexport;
    }
    expected_reasons[expected_count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  }
  if (validation.delivery_seen || contributor.reasons.size() != expected_count ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.begin() + static_cast<std::ptrdiff_t>(expected_count))) {
    return make_unexpected(
        tensor_resource_error("delivery claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.delivery_seen = true;
  validation.delivery_includes_transport_retention = includes_transport_retention;
  return {};
}

[[nodiscard]] expected<void, Error> validate_transport_retention_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  std::array<TensorSlotClaimReason, 4U> expected_reasons{};
  std::size_t expected_count = 0U;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
  if (allocation.memory_kind == MemoryKind::kCudaDevice ||
      allocation.memory_kind == MemoryKind::kCudaManaged) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kConsumerCudaRelease;
  }
  if (std::find(contributor.reasons.begin(),
                contributor.reasons.end(),
                TensorSlotClaimReason::kRelayOrReexport) != contributor.reasons.end()) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kRelayOrReexport;
  }
  expected_reasons[expected_count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  if (validation.transport_retention_seen || contributor.reasons.size() != expected_count ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.begin() + static_cast<std::ptrdiff_t>(expected_count))) {
    return make_unexpected(tensor_resource_error(
        "transport-retention claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.transport_retention_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_retention_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  const bool has_unknown_reason =
      std::any_of(contributor.reasons.begin(), contributor.reasons.end(), [](const auto reason) {
        return reason != TensorSlotClaimReason::kTemporalCache &&
               reason != TensorSlotClaimReason::kPendingOrSelectedWindow &&
               reason != TensorSlotClaimReason::kReuseLast &&
               reason != TensorSlotClaimReason::kDrainAllActivation &&
               reason != TensorSlotClaimReason::kCopyOrRecorderMaterialization;
      });
  if (validation.retention_seen || has_unknown_reason) {
    return make_unexpected(
        tensor_resource_error("retention claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.retention_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_retirement_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  constexpr std::array expected_reasons{TensorSlotClaimReason::kCancellation,
                                        TensorSlotClaimReason::kQuarantine,
                                        TensorSlotClaimReason::kRetirementService};
  if (validation.retirement_seen || !std::equal(contributor.reasons.begin(),
                                                contributor.reasons.end(),
                                                expected_reasons.begin(),
                                                expected_reasons.end())) {
    return make_unexpected(
        tensor_resource_error("retirement claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.retirement_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_claim_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const ClaimContributorIds& ids, const TensorAllocationPlanRecord& allocation) {
  if (contributor.contributor_id == ids.producer) {
    return validate_producer_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.delivery) {
    return validate_delivery_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.transport_retention) {
    return validate_transport_retention_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.retention) {
    return validate_retention_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.retirement) {
    return validate_retirement_contributor(validation, contributor, allocation);
  }
  return make_unexpected(
      tensor_resource_error("unknown claim contributor", "pool", allocation.pool_id));
}

[[nodiscard]] expected<std::uint32_t, Error> release_stream_capacity(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept {
  std::uint32_t result = 0U;
  for (const auto& service : plan.process_local_channel_services) {
    if (service.source != allocation.output || !service.tensor_payload) {
      continue;
    }
    if (service.max_subscribers > std::numeric_limits<std::uint32_t>::max() - result) {
      return make_unexpected(tensor_resource_error(
          "release-stream capacity overflow", "output", allocation.output.value()));
    }
    result += service.max_subscribers;
  }
  // Runtime installs one shared device stream before compute. Reserve one additional distinct
  // stream per downstream subscriber for an application-selected working stream.
  if (result == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(tensor_resource_error(
        "release-stream capacity exhausted", "output", allocation.output.value()));
  }
  return result + 1U;
}

[[nodiscard]] bool requires_transport_retention(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept {
  const bool process_local_transport = std::any_of(
      plan.process_local_channel_services.begin(),
      plan.process_local_channel_services.end(),
      [&](const auto& service) {
        return service.source == allocation.output && service.tensor_payload &&
               (service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ ||
                (service.delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
                 service.queue_policy != QueuePolicy::kBlocking));
      });
  const bool cross_process_transport =
      std::any_of(plan.cross_process_channel_services.begin(),
                  plan.cross_process_channel_services.end(),
                  [&](const auto& service) {
                    return service.source == allocation.output && service.tensor_payload;
                  });
  const bool external_export_transport =
      std::any_of(plan.payload_edges.begin(), plan.payload_edges.end(), [&](const auto& edge) {
        return edge.source == allocation.output &&
               edge.locality == ChannelLocalityKind::kExternalExport;
      });
  return process_local_transport || cross_process_transport || external_export_transport;
}

[[nodiscard]] expected<OutputPlanContract, Error> validate_output_plan_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) {
  auto publication = validate_publication_contract(plan, allocation);
  if (!publication) {
    return make_unexpected(std::move(publication.error()));
  }

  ClaimContributorValidation validation;
  const ClaimContributorIds ids{.producer = allocation.pool_id + ".producer",
                                .delivery = allocation.pool_id + ".delivery",
                                .transport_retention = allocation.pool_id + ".transport-retention",
                                .retention = allocation.pool_id + ".retention",
                                .retirement = allocation.pool_id + ".retirement"};
  for (const auto& contributor : plan.pool_capacity_contributors) {
    if (contributor.pool_id != allocation.pool_id) {
      continue;
    }
    auto recorded = record_claim_contributor(validation, contributor, allocation);
    if (!recorded) {
      return make_unexpected(std::move(recorded.error()));
    }
    auto classified = validate_claim_contributor(validation, contributor, ids, allocation);
    if (!classified) {
      return make_unexpected(std::move(classified.error()));
    }
  }

  const bool transport_retention_required = requires_transport_retention(plan, allocation);
  const bool transport_retention_matches =
      transport_retention_required
          ? validation.delivery_seen && validation.transport_retention_seen &&
                !validation.delivery_includes_transport_retention
          : !validation.transport_retention_seen &&
                (!validation.delivery_seen || validation.delivery_includes_transport_retention);
  if (!validation.producer_seen || !validation.retirement_seen || !transport_retention_matches ||
      validation.capacity != allocation.slot_count) {
    return make_unexpected(
        tensor_resource_error("incomplete claim coverage", "pool", allocation.pool_id));
  }
  return OutputPlanContract{.publication = *publication,
                            .admitted_claims = validation.admitted_claims};
}

[[nodiscard]] bool dtype_equal(DLDataType left, DLDataType right) noexcept {
  return left.code == right.code && left.bits == right.bits && left.lanes == right.lanes;
}

[[nodiscard]] bool valid_dtype(DLDataType dtype) noexcept {
  if (dtype.bits == 0U || dtype.lanes == 0U) {
    return false;
  }
  switch (dtype.code) {
    case kDLInt:
    case kDLUInt:
    case kDLFloat:
    case kDLOpaqueHandle:
    case kDLBfloat:
    case kDLComplex:
    case kDLBool:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] expected<std::size_t, Error> tensor_byte_span(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides) noexcept {
  const std::optional<std::span<const std::int64_t>> strides =
      byte_strides ? std::optional<std::span<const std::int64_t>>{byte_strides.value()}
                   : std::nullopt;
  const auto layout = validate_tensor_layout(shape, dtype, strides, TensorStrideUnit::kBytes);
  if (!layout || layout->byte_span > std::numeric_limits<std::size_t>::max()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return static_cast<std::size_t>(layout->byte_span);
}

class ScopedCudaDevice final {
 public:
  explicit ScopedCudaDevice(DeviceId requested) noexcept {
    if (!requested.valid() || cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ~ScopedCudaDevice() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  bool active_{};
  bool changed_{};
};

class TensorBacking final {
 public:
  TensorBacking(const TensorBacking&) = delete;
  TensorBacking& operator=(const TensorBacking&) = delete;
  ~TensorBacking() {
    if (exportable_) {
      return;
    }
    if (pointer_ == nullptr) {
      return;
    }
    switch (kind_) {
      case MemoryKind::kHost:
        delete[] static_cast<std::byte*>(pointer_);
        break;
      case MemoryKind::kPinnedHost:
        static_cast<void>(cudaFreeHost(pointer_));
        break;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged: {
        const ScopedCudaDevice selected{device_};
        if (selected.active()) {
          static_cast<void>(cudaFree(pointer_));
        }
        break;
      }
      case MemoryKind::kUnknown:
        break;
    }
  }

  [[nodiscard]] static expected<std::shared_ptr<TensorBacking>, Error> create(
      MemoryKind kind, DeviceId device, std::size_t bytes, std::string_view pool_id) {
    if (kind == MemoryKind::kUnknown || bytes == 0U ||
        ((kind == MemoryKind::kCudaDevice || kind == MemoryKind::kCudaManaged) &&
         !device.valid())) {
      return make_unexpected(tensor_resource_error("invalid backing request", "pool", pool_id));
    }
    auto result = std::shared_ptr<TensorBacking>{new TensorBacking{kind, device, bytes}};
    switch (kind) {
      case MemoryKind::kHost:
        result->pointer_ = new (std::nothrow) std::byte[bytes];
        break;
      case MemoryKind::kPinnedHost:
        if (cudaMallocHost(&result->pointer_, bytes) != cudaSuccess) {
          return make_unexpected(
              tensor_resource_error("pinned-host allocation failed", "pool", pool_id));
        }
        break;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged: {
        const ScopedCudaDevice selected{device};
        if (!selected.active()) {
          return make_unexpected(
              tensor_resource_error("CUDA device selection failed", "pool", pool_id));
        }
        const cudaError_t status =
            kind == MemoryKind::kCudaDevice
                ? cudaMalloc(&result->pointer_, bytes)
                : cudaMallocManaged(&result->pointer_, bytes, cudaMemAttachGlobal);
        if (status != cudaSuccess) {
          return make_unexpected(
              tensor_resource_error("CUDA backing allocation failed", "pool", pool_id));
        }
        break;
      }
      case MemoryKind::kUnknown:
        break;
    }
    if (result->pointer_ == nullptr) {
      return make_unexpected(
          tensor_resource_error("backing allocation returned null", "pool", pool_id));
    }
    return result;
  }

  [[nodiscard]] static expected<std::shared_ptr<TensorBacking>, Error> create_exportable(
      MemoryKind kind, DeviceId device, std::size_t bytes,
      std::shared_ptr<holomq::BufferPool> pool_owner, std::string_view pool_id) {
    if ((kind != MemoryKind::kHost && kind != MemoryKind::kPinnedHost &&
         kind != MemoryKind::kCudaDevice) ||
        bytes == 0U ||
        ((kind == MemoryKind::kPinnedHost || kind == MemoryKind::kCudaDevice) && !device.valid()) ||
        (kind == MemoryKind::kHost && device.valid()) || pool_owner == nullptr) {
      return make_unexpected(
          tensor_resource_error("invalid exportable backing request", "pool", pool_id));
    }
    auto result = std::shared_ptr<TensorBacking>{new TensorBacking{kind, device, bytes}};
    result->pool_owner_ = std::move(pool_owner);
    result->exportable_ = true;
    return result;
  }

  [[nodiscard]] expected<void, Error> bind(holomq::Buffer buffer,
                                           std::string_view pool_id) noexcept {
    const holomq::BufferMemoryType expected =
        kind_ == MemoryKind::kCudaDevice
            ? holomq::BufferMemoryType::Device
            : (kind_ == MemoryKind::kPinnedHost ? holomq::BufferMemoryType::HostPinned
                                                : holomq::BufferMemoryType::HostPageable);
    if (!exportable_ || buffer_ || !buffer || buffer.memory_type() != expected ||
        buffer.size_bytes() < bytes_ ||
        (kind_ == MemoryKind::kCudaDevice &&
         buffer.cuda_device_ordinal() != std::optional<std::int32_t>{device_.value})) {
      return make_unexpected(
          tensor_resource_error("exportable buffer lease mismatch", "pool", pool_id));
    }
    pointer_ = buffer.ptr();
    buffer_.emplace(std::move(buffer));
    return {};
  }

  [[nodiscard]] bool has_buffer() const noexcept { return buffer_.has_value(); }

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle(cudaStream_t stream) noexcept {
    if (!exportable_ || !buffer_) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
          ErrorCode::kInvalidArgument, "direct Tensor output has no exportable buffer lease");
    }
    auto handle = stream == nullptr ? buffer_->export_handle_unsynchronized()
                                    : buffer_->export_handle(stream);
    if (!handle) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(handle).error(), "failed to export direct Tensor output buffer"));
    }
    return std::move(handle).value();
  }

  [[nodiscard]] bool prepare_for_reuse() noexcept {
    return !exportable_ || (buffer_ && buffer_->prepare_for_reuse());
  }

  [[nodiscard]] void* pointer() const noexcept { return pointer_; }

  [[nodiscard]] MemoryKind memory_kind() const noexcept { return kind_; }

 private:
  TensorBacking(MemoryKind kind, DeviceId device, std::size_t bytes) noexcept
      : kind_(kind), device_(device), bytes_(bytes) {}

  MemoryKind kind_{MemoryKind::kUnknown};
  DeviceId device_{};
  std::size_t bytes_{};
  void* pointer_{};
  // HoloMQ Buffer implementations reference their owning pool. Member order also releases the
  // optional Buffer before its final shared pool claim.
  std::shared_ptr<holomq::BufferPool> pool_owner_;
  std::optional<holomq::Buffer> buffer_;
  bool exportable_{};
};

class SlotLifetime final {
 public:
  SlotLifetime(const SlotLifetime&) = delete;
  SlotLifetime& operator=(const SlotLifetime&) = delete;
  ~SlotLifetime() {
    if (readiness_event_ == nullptr && release_events_.empty()) {
      return;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return;
    }
    for (cudaEvent_t event : release_events_) {
      if (event != nullptr && cudaEventDestroy(event) == cudaSuccess && test_control_ != nullptr) {
        test_control_->record_cuda_event_destroyed(test_pool_index_);
      }
    }
    if (readiness_event_ != nullptr) {
      if (cudaEventDestroy(readiness_event_) == cudaSuccess && test_control_ != nullptr) {
        test_control_->record_cuda_event_destroyed(test_pool_index_);
      }
    }
  }

  [[nodiscard]] static expected<std::shared_ptr<SlotLifetime>, Error> create(
      std::shared_ptr<TensorBacking> backing, DeviceId device, bool cuda_ordered,
      bool elide_same_stream_readiness, std::uint32_t release_stream_capacity,
      std::string_view pool_id, std::shared_ptr<RuntimeTensorOutputTestControl> test_control = {},
      std::size_t test_pool_index = 0U) {
    auto result = std::shared_ptr<SlotLifetime>{new SlotLifetime{std::move(backing),
                                                                 device,
                                                                 elide_same_stream_readiness,
                                                                 std::move(test_control),
                                                                 test_pool_index}};
    if (!cuda_ordered) {
      return result;
    }
    const ScopedCudaDevice selected{device};
    if (!selected.active() || cudaEventCreateWithFlags(&result->readiness_event_,
                                                       cudaEventDisableTiming) != cudaSuccess) {
      return make_unexpected(
          tensor_resource_error("readiness-event materialization failed", "pool", pool_id));
    }
    if (result->test_control_ != nullptr) {
      result->test_control_->record_cuda_events_created(result->test_pool_index_, 1U);
    }
    result->release_streams_.reserve(release_stream_capacity);
    result->release_events_.resize(release_stream_capacity, nullptr);
    for (cudaEvent_t& event : result->release_events_) {
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        return make_unexpected(
            tensor_resource_error("release-event materialization failed", "pool", pool_id));
      }
      if (result->test_control_ != nullptr) {
        result->test_control_->record_cuda_events_created(result->test_pool_index_, 1U);
      }
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> bind_exportable_buffer(holomq::Buffer buffer,
                                                             std::string_view pool_id) noexcept {
    return backing_ != nullptr ? backing_->bind(std::move(buffer), pool_id)
                               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

  [[nodiscard]] bool has_exportable_buffer() const noexcept {
    return backing_ != nullptr && backing_->has_buffer();
  }

  [[nodiscard]] void* data() const noexcept {
    return backing_ != nullptr ? backing_->pointer() : nullptr;
  }

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept {
    cudaStream_t stream = nullptr;
    {
      const std::lock_guard lock(mutex_);
      stream = producer_stream_;
    }
    return backing_ != nullptr
               ? backing_->export_handle(stream)
               : channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
                     ErrorCode::kFailure, "direct Tensor output backing is missing");
  }

  [[nodiscard]] bool prepare_exportable_buffer_for_reuse() noexcept {
    return backing_ != nullptr && backing_->prepare_for_reuse();
  }

  [[nodiscard]] bool record_readiness(cudaStream_t stream) noexcept {
    if (readiness_event_ == nullptr || stream == nullptr) {
      return false;
    }
    const std::lock_guard lock(mutex_);
    producer_stream_ = stream;
    if (elide_same_stream_readiness_) {
      readiness_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness_elision(test_pool_index_);
      }
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active() || cudaEventRecord(readiness_event_, stream) != cudaSuccess) {
      producer_stream_ = nullptr;
      return false;
    }
    readiness_recorded_ = true;
    readiness_event_recorded_ = true;
    if (test_control_ != nullptr) {
      test_control_->record_cuda_readiness(test_pool_index_);
    }
    return true;
  }

  [[nodiscard]] bool record_release() noexcept {
    const std::lock_guard lock(mutex_);
    if (release_streams_.empty()) {
      // A canceled allocation that never entered a CUDA write has no work to retire. A completed
      // device write is still protected by readiness_event_ in retirement_complete().
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return false;
    }
    release_record_count_ = 0U;
    for (std::size_t index = 0U; index < release_streams_.size(); ++index) {
      if (cudaEventRecord(release_events_[index], release_streams_[index]) != cudaSuccess) {
        return false;
      }
      ++release_record_count_;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_release(test_pool_index_);
      }
    }
    return true;
  }

  [[nodiscard]] bool retirement_complete() const noexcept {
    const std::lock_guard lock(mutex_);
    if (release_record_count_ == 0U && !readiness_event_recorded_) {
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return false;
    }
    bool complete = true;
    if (release_record_count_ != 0U) {
      for (std::size_t index = 0U; index < release_record_count_; ++index) {
        complete = complete && cudaEventQuery(release_events_[index]) == cudaSuccess;
      }
    } else if (readiness_event_recorded_) {
      complete = cudaEventQuery(readiness_event_) == cudaSuccess;
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_retirement_query(test_pool_index_, complete);
    }
    return complete;
  }

  [[nodiscard]] bool has_release_streams() const noexcept {
    const std::lock_guard lock(mutex_);
    return !release_streams_.empty();
  }

  [[nodiscard]] bool retirement_requires_query() const noexcept {
    const std::lock_guard lock(mutex_);
    return release_record_count_ != 0U || readiness_event_recorded_;
  }

  void reset_generation() noexcept {
    const std::lock_guard lock(mutex_);
    readiness_recorded_ = false;
    readiness_event_recorded_ = false;
    producer_stream_ = nullptr;
    release_record_count_ = 0U;
    release_streams_.clear();
  }

  [[nodiscard]] bool set_release_stream(cudaStream_t stream) noexcept {
    if (stream == nullptr) {
      return false;
    }
    const std::lock_guard lock(mutex_);
    if (release_events_.empty()) {
      return false;
    }
    if (elide_same_stream_readiness_ && producer_stream_ == stream) {
      // Reuse work is enqueued on this same fixed stream after logical handle retirement.
      if (test_control_ != nullptr) {
        test_control_->record_cuda_release_stream_elision(test_pool_index_);
      }
      return true;
    }
    if (std::find(release_streams_.begin(), release_streams_.end(), stream) !=
        release_streams_.end()) {
      return true;
    }
    if (release_streams_.size() == release_streams_.capacity()) {
      return false;
    }
    release_streams_.push_back(stream);
    if (test_control_ != nullptr) {
      test_control_->record_cuda_release_stream(test_pool_index_);
    }
    return true;
  }

  [[nodiscard]] bool has_readiness() const noexcept { return readiness_event_ != nullptr; }

  [[nodiscard]] cudaStream_t producer_stream() const noexcept {
    const std::lock_guard lock(mutex_);
    return readiness_recorded_ ? producer_stream_ : nullptr;
  }

  [[nodiscard]] expected<void, Error> synchronize(cudaStream_t consumer) const noexcept {
    const std::lock_guard lock(mutex_);
    if (consumer == nullptr || readiness_event_ == nullptr || !readiness_recorded_ ||
        producer_stream_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (consumer == producer_stream_) {
      if (test_control_ != nullptr) {
        test_control_->record_cuda_stream_wait_elision(test_pool_index_);
      }
      return {};
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (!readiness_event_recorded_) {
      if (cudaEventRecord(readiness_event_, producer_stream_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      readiness_event_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness(test_pool_index_);
      }
    }
    if (cudaStreamWaitEvent(consumer, readiness_event_, 0U) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_stream_wait(test_pool_index_);
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> synchronize_host() const noexcept {
    const std::lock_guard lock(mutex_);
    if (!readiness_recorded_ && backing_ != nullptr &&
        (backing_->memory_kind() == MemoryKind::kHost ||
         backing_->memory_kind() == MemoryKind::kPinnedHost)) {
      // A committed host write has no asynchronous producer work to fence.
      return {};
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active() || readiness_event_ == nullptr || !readiness_recorded_ ||
        producer_stream_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (!readiness_event_recorded_) {
      if (cudaEventRecord(readiness_event_, producer_stream_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      readiness_event_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness(test_pool_index_);
      }
    }
    if (cudaEventSynchronize(readiness_event_) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_host_wait(test_pool_index_);
    }
    return {};
  }

 private:
  SlotLifetime(std::shared_ptr<TensorBacking> backing, DeviceId device,
               bool elide_same_stream_readiness,
               std::shared_ptr<RuntimeTensorOutputTestControl> test_control,
               std::size_t test_pool_index) noexcept
      : backing_(std::move(backing)),
        device_(device),
        elide_same_stream_readiness_(elide_same_stream_readiness),
        test_control_(std::move(test_control)),
        test_pool_index_(test_pool_index) {}

  std::shared_ptr<TensorBacking> backing_;
  DeviceId device_{};
  bool elide_same_stream_readiness_{};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control_;
  std::size_t test_pool_index_{};
  cudaEvent_t readiness_event_{};
  std::vector<cudaEvent_t> release_events_;
  std::vector<cudaStream_t> release_streams_;
  mutable std::mutex mutex_;
  cudaStream_t producer_stream_{};
  std::size_t release_record_count_{};
  bool readiness_recorded_{};
  mutable bool readiness_event_recorded_{};
};

enum class SlotPhase : std::uint8_t {
  kFree,
  kLoaned,
  kWriterActive,
  kReady,
  kSubmitPending,
  kRetained,
  kPublicationTerminal,
  kReleaseRecording,
  kRetiring,
  kQuarantined,
};

enum class TensorOutputOperation : std::uint8_t {
  kAllocate,
  kBeginWrite,
  kCommitWrite,
  kDropWrite,
  kEmit,
  kCancel,
  kRetire,
  kRestart,
};

class OutputPool;

class OutputPoolReferenceControl final {
 public:
  explicit OutputPoolReferenceControl(OutputPool* pool) noexcept : pool_(pool) {}

  [[nodiscard]] OutputPool* acquire() noexcept {
    if (detaching_.load(std::memory_order_acquire)) {
      return nullptr;
    }
    in_flight_.fetch_add(1U, std::memory_order_acq_rel);
    if (detaching_.load(std::memory_order_acquire)) {
      release();
      return nullptr;
    }
    OutputPool* const result = pool_.load(std::memory_order_acquire);
    if (result == nullptr) {
      release();
    }
    return result;
  }

  void release() noexcept {
    if (in_flight_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      in_flight_.notify_all();
    }
  }

  void detach() noexcept {
    detaching_.store(true, std::memory_order_release);
    pool_.store(nullptr, std::memory_order_release);
    std::uint32_t current = in_flight_.load(std::memory_order_acquire);
    while (current != 0U) {
      in_flight_.wait(current, std::memory_order_acquire);
      current = in_flight_.load(std::memory_order_acquire);
    }
  }

 private:
  std::atomic<OutputPool*> pool_{};
  std::atomic<std::uint32_t> in_flight_{};
  std::atomic_bool detaching_{};
};

struct OutputSlot {
  OutputPool* pool{};
  std::shared_ptr<OutputPoolReferenceControl> pool_control;
  std::weak_ptr<OutputSlot> self;
  std::size_t index{};
  std::uint64_t generation{};
  PlanGenerationId loan_plan_generation{};
  RunGenerationId loan_run_generation{};
  std::uint64_t loan_pool_generation{};
  std::uint64_t loan_granted_ns{};
  Tag activation{};
  SlotPhase phase{SlotPhase::kFree};
  std::atomic<std::uint32_t> tensor_handles{};
  std::atomic_bool retirement_requested{};
  std::atomic_bool quarantine_requested{};
  std::atomic<OutputSlot*> deferred_next{};
  std::atomic_bool deferred_queued{};
  TensorSlotClaimLedger claims;
  bool on_free_list{};
  bool on_retirement_queue{};
  bool loan_live{};
  bool retry_cancellation_demand{};
  bool cancel_pending{};
  bool writer_cuda{};
  cudaStream_t writer_stream{};
  std::size_t byte_size{};
  DLDataType dtype{};
  DeviceId device;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> byte_strides;
  bool explicit_strides{};
  std::shared_ptr<SlotLifetime> lifetime;
  std::shared_ptr<DLManagedTensorContext> context;
};

class OutputPoolCallbackClaim final {
 public:
  explicit OutputPoolCallbackClaim(const OutputSlot& slot) noexcept
      : control_(slot.pool_control), pool_(control_ != nullptr ? control_->acquire() : nullptr) {}
  OutputPoolCallbackClaim(const OutputPoolCallbackClaim&) = delete;
  OutputPoolCallbackClaim& operator=(const OutputPoolCallbackClaim&) = delete;
  ~OutputPoolCallbackClaim() {
    if (pool_ != nullptr) {
      control_->release();
    }
  }

  [[nodiscard]] explicit operator bool() const noexcept { return pool_ != nullptr; }
  [[nodiscard]] OutputPool* pool() const noexcept { return pool_; }

 private:
  std::shared_ptr<OutputPoolReferenceControl> control_;
  OutputPool* pool_{};
};

template <typename Base>
class SlotMemoryReference final : public Base {
 public:
  SlotMemoryReference(std::shared_ptr<OutputPoolReferenceControl> control, std::size_t slot,
                      std::shared_ptr<SlotLifetime> lifetime) noexcept
      : control_(std::move(control)), slot_(slot), lifetime_(std::move(lifetime)) {}

  void retain_tensor_handle() noexcept override;
  void release_tensor_handle() noexcept override;

  bool set_deallocation_stream(cudaStream_t stream) override {
    return lifetime_ != nullptr && lifetime_->set_release_stream(stream);
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    return lifetime_ != nullptr && lifetime_->record_readiness(stream);
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return lifetime_ != nullptr && lifetime_->has_readiness();
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return lifetime_ != nullptr ? lifetime_->producer_stream() : nullptr;
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    return lifetime_ != nullptr
               ? lifetime_->synchronize(consumer_stream)
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kNotSupported})};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    return lifetime_ != nullptr
               ? lifetime_->synchronize_host()
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kNotSupported})};
  }

  // This is virtual only for the HoloMQ-exportable Base specialization.
  // NOLINTNEXTLINE(clang-diagnostic-inconsistent-missing-override)
  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept {
    return lifetime_ != nullptr
               ? lifetime_->export_handle()
               : channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
                     ErrorCode::kFailure, "direct Tensor output lifetime is missing");
  }

 private:
  std::shared_ptr<OutputPoolReferenceControl> control_;
  std::size_t slot_{};
  std::shared_ptr<SlotLifetime> lifetime_;
};

using NativeSlotMemoryReference = SlotMemoryReference<TensorMemoryReference>;
using ExportableSlotMemoryReference =
    SlotMemoryReference<channel_internal::HoloMQExportableTensorMemoryReference>;

struct DeviceResource {
  const TensorDeviceResourcePlanRecord* plan{};
  cudaStream_t stream{};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control;
  std::size_t test_pool_index{};

  DeviceResource(const TensorDeviceResourcePlanRecord* plan_record,
                 cudaStream_t materialized_stream,
                 std::shared_ptr<RuntimeTensorOutputTestControl> control = {},
                 std::size_t pool_index = 0U) noexcept
      : plan(plan_record),
        stream(materialized_stream),
        test_control(std::move(control)),
        test_pool_index(pool_index) {
    if (test_control != nullptr) {
      test_control->record_cuda_stream_created(test_pool_index);
    }
  }
  DeviceResource(const DeviceResource&) = delete;
  DeviceResource& operator=(const DeviceResource&) = delete;
  DeviceResource(DeviceResource&& other) noexcept
      : plan(std::exchange(other.plan, nullptr)),
        stream(std::exchange(other.stream, nullptr)),
        test_control(std::move(other.test_control)),
        test_pool_index(other.test_pool_index) {}
  DeviceResource& operator=(DeviceResource&&) = delete;
  ~DeviceResource() {
    if (plan == nullptr || stream == nullptr) {
      return;
    }
    const ScopedCudaDevice selected{plan->device};
    if (selected.active()) {
      if (cudaStreamDestroy(stream) == cudaSuccess && test_control != nullptr) {
        test_control->record_cuda_stream_destroyed(test_pool_index);
      }
    }
  }
};

[[nodiscard]] expected<bool, Error> output_requires_holomq_backing(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    const std::optional<std::uint32_t> tensor_index = std::nullopt) {
  bool required = false;
  for (const auto& service : plan.cross_process_channel_services) {
    if (service.source != allocation.output) {
      continue;
    }
    const bool direct_tensor = service.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema = descriptor_tensor_schema_service(service);
    bool storage_matches = service.tensor_max_byte_span == allocation.bounds.max_byte_span &&
                           service.tensor_memory_kind == allocation.memory_kind &&
                           service.tensor_device == allocation.device;
    if (tensor_index) {
      storage_matches =
          descriptor_tensor_schema && *tensor_index < service.tensor_fields.size() &&
          service.tensor_fields[*tensor_index].tensor_index == *tensor_index &&
          service.tensor_fields[*tensor_index].max_byte_span == allocation.bounds.max_byte_span &&
          service.tensor_fields[*tensor_index].memory_kind == allocation.memory_kind &&
          service.tensor_fields[*tensor_index].device == allocation.device;
    }
    if (!service.tensor_payload || (!direct_tensor && !descriptor_tensor_schema) ||
        (direct_tensor && service.tensor_pool_id != allocation.pool_id) || !storage_matches) {
      return make_unexpected(tensor_resource_error(
          "cross-process export contract mismatch", "output", allocation.output.value()));
    }
    required = true;
  }
  if (required && allocation.memory_kind != MemoryKind::kHost &&
      allocation.memory_kind != MemoryKind::kPinnedHost &&
      allocation.memory_kind != MemoryKind::kCudaDevice) {
    return make_unexpected(tensor_resource_error(
        "cross-process output placement is not exportable", "output", allocation.output.value()));
  }
  return required;
}

[[nodiscard]] expected<holomq::BufferPool, Error> create_holomq_output_pool(
    const TensorAllocationPlanRecord& allocation,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::size_t test_pool_index) {
  if (allocation.bounds.max_byte_span > std::numeric_limits<std::size_t>::max()) {
    return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
        ErrorCode::kResourceExhausted, "direct Tensor output byte span exceeds address space");
  }
  const auto create_pool = [&]() -> expected<holomq::BufferPool, Error> {
    if (cudaFree(nullptr) != cudaSuccess) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
          ErrorCode::kNotSupported, "direct Tensor output requires CUDA VMM support");
    }
    if (test_control != nullptr) {
      CUdevice current_device = -1;
      if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS) {
        return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
            ErrorCode::kNotSupported, "direct Tensor output could not observe its CUDA VMM device");
      }
      test_control->record_holomq_generation_vmm_device(test_pool_index, current_device);
    }
    auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
    if (!context) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(context).error(), "failed to create direct Tensor export context"));
    }
    auto pool =
        allocation.memory_kind == MemoryKind::kCudaDevice
            ? context->create_device_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span), allocation.slot_count)
        : allocation.memory_kind == MemoryKind::kPinnedHost
            ? context->create_host_pinned_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span),
                  allocation.slot_count,
                  CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP)
            : context->create_host_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span), allocation.slot_count);
    if (!pool) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(pool).error(), "failed to create direct Tensor export pool"));
    }
    return std::move(pool).value();
  };
  if (allocation.memory_kind == MemoryKind::kHost ||
      allocation.memory_kind == MemoryKind::kCudaDevice ||
      allocation.memory_kind == MemoryKind::kPinnedHost) {
    // Host payloads carry the DLPack-mandated device id 0 but have no allocation device. Their
    // split generation words are still CUDA VMM, so create that private synchronization pool on
    // the runtime's fixed host synchronization device instead of the caller's ambient device.
    const DeviceId pool_device =
        allocation.memory_kind == MemoryKind::kHost ? DeviceId{0} : allocation.device;
    const ScopedCudaDevice selected{pool_device};
    if (!selected.active()) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
          ErrorCode::kNotSupported, "direct Tensor output device is unavailable");
    }
    return create_pool();
  }
  return create_pool();
}

class OutputPool final {
 public:
  OutputPool(const OutputPool&) = delete;
  OutputPool& operator=(const OutputPool&) = delete;
  ~OutputPool() {
    detach();
    if (reference_control_ != nullptr) {
      reference_control_->detach();
    }
    record_test_snapshot(true);
  }

  [[nodiscard]] static expected<std::unique_ptr<OutputPool>, Error> create(
      const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
      OutputPlanContract contract, RuntimeOperatorInstance& instance, void* declaration,
      const RuntimePortTypeOps* type_ops, cudaStream_t stream, bool elide_same_stream_readiness,
      RunGenerationId run_generation,
      std::shared_ptr<RuntimeTensorOutputTestControl> test_control = {},
      std::size_t test_pool_index = 0U, bool bind = true, std::uint64_t pool_generation = 1U,
      std::optional<std::uint32_t> descriptor_tensor_index = std::nullopt) {
    if (!validate_tensor_allocation_record(allocation, plan.plan_generation) ||
        run_generation == 0U || pool_generation == 0U || declaration == nullptr ||
        type_ops == nullptr || type_ops->publish_tensor_loan == nullptr ||
        type_ops->bind_tensor_output == nullptr || type_ops->unbind_tensor_output == nullptr ||
        ((allocation.memory_kind == MemoryKind::kCudaDevice ||
          allocation.memory_kind == MemoryKind::kCudaManaged) &&
         (!allocation.device.valid() || stream == nullptr)) ||
        (allocation.memory_kind == MemoryKind::kPinnedHost && allocation.device.valid() &&
         stream == nullptr)) {
      return make_unexpected(tensor_resource_error(
          "invalid output-pool contract", "output", allocation.output.value()));
    }
    const auto max_release_streams = release_stream_capacity(plan, allocation);
    if (!max_release_streams) {
      return make_unexpected(std::move(max_release_streams.error()));
    }

    auto exportable = output_requires_holomq_backing(plan, allocation, descriptor_tensor_index);
    if (!exportable) {
      return make_unexpected(std::move(exportable.error()));
    }
    std::shared_ptr<holomq::BufferPool> export_pool;
    if (*exportable) {
      auto pool = create_holomq_output_pool(allocation, test_control, test_pool_index);
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      export_pool = std::make_shared<holomq::BufferPool>(std::move(*pool));
    }

    auto result = std::unique_ptr<OutputPool>{new OutputPool{allocation,
                                                             std::move(contract),
                                                             instance.vertex,
                                                             instance.operator_id,
                                                             declaration,
                                                             type_ops,
                                                             stream,
                                                             elide_same_stream_readiness,
                                                             plan.plan_generation,
                                                             run_generation,
                                                             pool_generation,
                                                             std::move(export_pool),
                                                             std::move(test_control),
                                                             test_pool_index,
                                                             descriptor_tensor_index}};
    result->slots_.reserve(allocation.slot_count);
    result->references_.reserve(allocation.slot_count);
    result->free_slots_.reserve(allocation.slot_count);
    result->retiring_slots_.reserve(allocation.retirement_capacity);
    result->diagnostic_storage_.resize(allocation.control_capacity);
    result->diagnostics_.reset(result->diagnostic_storage_);
    result->reference_control_ = std::make_shared<OutputPoolReferenceControl>(result.get());
    for (std::uint32_t index = 0U; index < allocation.slot_count; ++index) {
      auto slot = std::make_shared<OutputSlot>();
      slot->pool = result.get();
      slot->pool_control = result->reference_control_;
      slot->self = slot;
      slot->index = index;
      slot->generation = 1U;
      slot->on_free_list = true;
      slot->shape.reserve(allocation.bounds.max_rank);
      slot->byte_strides.reserve(allocation.bounds.max_rank);
      // Native and exportable slots own their physical allocation for the pool lifetime. A
      // cold-bound exportable lease advances only after its remote import ref-count returns to the
      // owner-only state, so hot draws need no Buffer implementation allocation.
      auto backing = result->export_pool_
                         ? TensorBacking::create_exportable(allocation.memory_kind,
                                                            allocation.device,
                                                            allocation.bounds.max_byte_span,
                                                            result->export_pool_,
                                                            allocation.pool_id)
                         : TensorBacking::create(allocation.memory_kind,
                                                 allocation.device,
                                                 allocation.bounds.max_byte_span,
                                                 allocation.pool_id);
      if (!backing) {
        return make_unexpected(std::move(backing.error()));
      }
      auto lifetime = SlotLifetime::create(std::move(*backing),
                                           allocation.device,
                                           stream != nullptr,
                                           elide_same_stream_readiness,
                                           *max_release_streams,
                                           allocation.pool_id,
                                           result->test_control_,
                                           result->test_pool_index_);
      if (!lifetime) {
        return make_unexpected(std::move(lifetime.error()));
      }
      slot->lifetime = std::move(*lifetime);
      if (result->export_pool_) {
        auto acquired = result->export_pool_->acquire();
        if (!acquired) {
          return make_unexpected(tensor_resource_error(
              "exportable buffer cold binding failed", "pool", allocation.pool_id));
        }
        auto bound =
            slot->lifetime->bind_exportable_buffer(std::move(*acquired), allocation.pool_id);
        if (!bound) {
          return make_unexpected(std::move(bound).error());
        }
      }
      std::shared_ptr<TensorMemoryReference> reference;
      if (result->export_pool_) {
        reference = std::make_shared<ExportableSlotMemoryReference>(
            result->reference_control_, index, slot->lifetime);
      } else {
        reference = std::make_shared<NativeSlotMemoryReference>(
            result->reference_control_, index, slot->lifetime);
      }
      slot->context = std::make_shared<DLManagedTensorContext>();
      slot->context->dl_shape.reserve(allocation.bounds.max_rank);
      slot->context->dl_strides.reserve(allocation.bounds.max_rank);
      slot->context->memory_reference = reference;
      slot->context->tensor.dl_tensor.data = slot->lifetime->data();
      slot->context->tensor.dl_tensor.device =
          runtime_tensor_dl_device(allocation.memory_kind, allocation.device);
      slot->device = allocation.device;
      slot->memory_kind = allocation.memory_kind;
      result->references_.push_back(std::move(reference));
      result->slots_.push_back(std::move(slot));
      result->free_slots_.push_back(allocation.slot_count - index - 1U);
    }
    if (bind) {
      result->bind();
      result->record_test_snapshot();
    }
    return result;
  }

  [[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> recreate_unbound(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance) const {
    if (pool_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(
          tensor_resource_error("pool generation overflow", "pool", plan_.pool_id));
    }
    return create(plan,
                  plan_,
                  contract_,
                  instance,
                  declaration_,
                  type_ops_,
                  stream_,
                  elide_same_stream_readiness_,
                  run_generation_,
                  test_control_,
                  test_pool_index_,
                  false,
                  pool_generation_ + 1U,
                  descriptor_tensor_index_);
  }

  [[nodiscard]] OperatorInstanceId owner() const noexcept { return owner_; }
  [[nodiscard]] PortId output() const noexcept { return plan_.output; }

  [[nodiscard]] expected<void, Error> begin_restart() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (restart_fenced_ || closed_ || lifecycle_transition_active_) {
      return make_unexpected(
          tensor_resource_error("restart fence rejected", "output", plan_.output.value()));
    }
    restart_fenced_ = true;
    closed_ = true;
    lifecycle_transition_active_ = true;
    for (auto& slot : slots_) {
      if (slot->phase == SlotPhase::kWriterActive) {
        slot->cancel_pending = true;
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
      } else if (slot->phase == SlotPhase::kLoaned || slot->phase == SlotPhase::kReady) {
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
        begin_retirement(*slot, lock);
      }
    }
    lifecycle_transition_active_ = false;
    return {};
  }

  [[nodiscard]] expected<void, Error> reopen(const bool bind_output = true) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!restart_fenced_ || lifecycle_transition_active_) {
      return make_unexpected(
          tensor_resource_error("restart reopen rejected", "output", plan_.output.value()));
    }
    restart_fenced_ = false;
    closed_ = false;
    if (bind_output) {
      bind();
    }
    record_test_snapshot();
    return {};
  }

  [[nodiscard]] bool restart_publication_ready() noexcept {
    const std::lock_guard lock(mutex_);
    return restart_fenced_ && closed_ && !lifecycle_transition_active_ &&
           !consume_test_fault(RuntimeTensorOutputTestFault::kRestartPublicationPreflightFailure);
  }

  // The controller calls this only after preflighting every affected pool. No other controller
  // mutation may interleave at a lifecycle safe point, so publication contains no remaining
  // fallible step after another restart participant has committed.
  void publish_reopen(const bool bind_output = true) noexcept {
    const std::lock_guard lock(mutex_);
    restart_fenced_ = false;
    closed_ = false;
    if (bind_output) {
      bind();
    }
    record_test_snapshot();
  }

  void bind() noexcept {
    if (!bound_) {
      type_ops_->bind_tensor_output(declaration_,
                                    TensorOutputBinding{.state = this, .ops = &kBindingOps});
      bound_ = true;
    }
  }

  void close() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      return;
    }
    closed_ = true;
    lifecycle_transition_active_ = true;
    for (auto& slot : slots_) {
      if (slot->phase == SlotPhase::kWriterActive) {
        slot->cancel_pending = true;
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
      } else if (slot->phase == SlotPhase::kLoaned || slot->phase == SlotPhase::kReady) {
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
        begin_retirement(*slot, lock);
      }
    }
    lifecycle_transition_active_ = false;
    record_test_snapshot();
  }

  void detach() noexcept {
    if (bound_) {
      type_ops_->unbind_tensor_output(declaration_);
      bound_ = false;
    }
  }

  void retain_tensor(std::size_t index) noexcept {
    if (index >= slots_.size()) {
      return;
    }
    auto& slot = *slots_[index];
    std::uint32_t current = slot.tensor_handles.load(std::memory_order_acquire);
    while (current != std::numeric_limits<std::uint32_t>::max() &&
           !slot.tensor_handles.compare_exchange_weak(
               current, current + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    if (current == std::numeric_limits<std::uint32_t>::max()) {
      // A no-throw Tensor copy cannot report saturation. Quarantine the backing permanently rather
      // than wrap the counter and make a still-referenced slot reusable.
      slot.quarantine_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
    }
  }

  void release_tensor(std::size_t index) noexcept {
    if (index >= slots_.size()) {
      return;
    }
    auto& slot = *slots_[index];
    std::uint32_t current = slot.tensor_handles.load(std::memory_order_acquire);
    while (current != 0U &&
           !slot.tensor_handles.compare_exchange_weak(
               current, current - 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    if (current == 0U) {
      slot.quarantine_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
      return;
    }
    if (current == 1U) {
      slot.retirement_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
    }
  }

  struct PreparedBundleField {
    OutputSlot* slot{};
    Tensor tensor;
  };

  [[nodiscard]] bool accepts_bundle_request(const TensorLoanRequest& request) const noexcept {
    if (!valid_dtype(request.dtype) || request.shape.size() > plan_.bounds.max_rank ||
        (plan_.layout && plan_.layout->memory_kind &&
         *plan_.layout->memory_kind != plan_.memory_kind) ||
        (plan_.layout && plan_.layout->rank && request.shape.size() != *plan_.layout->rank) ||
        (plan_.layout && plan_.layout->dtype &&
         !dtype_equal(request.dtype, *plan_.layout->dtype))) {
      return false;
    }
    auto span = tensor_byte_span(request.shape, request.dtype, request.byte_strides);
    return span && *span <= plan_.bounds.max_byte_span;
  }

  [[nodiscard]] std::optional<DLDataType> exact_bundle_dtype() const noexcept {
    if (plan_.layout && plan_.layout->dtype) {
      return plan_.layout->dtype;
    }
    return std::nullopt;
  }

  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_bundle_field(
      TensorLoanRequest request) noexcept {
    return allocate(request);
  }

  [[nodiscard]] bool preflight_bundle_field(const TensorOutputLoan& loan) noexcept {
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    if (selected == nullptr) {
      return false;
    }
    const OutputPoolCallbackClaim selected_pool{*selected};
    if (!selected_pool || selected_pool.pool() != this || selected->index >= slots_.size() ||
        slots_[selected->index].get() != selected) {
      return false;
    }
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    return !closed_ && plan_identity_current(*selected) && callback_activation_current(*selected) &&
           selected->loan_live && selected->phase == SlotPhase::kReady && identity_valid(*selected);
  }

  [[nodiscard]] expected<PreparedBundleField, Error> prepare_bundle_field(
      const TensorOutputLoan& loan) noexcept {
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    if (selected == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto context = prepare_emit(*selected, false);
    if (!context) {
      return make_unexpected(std::move(context.error()));
    }
    return PreparedBundleField{
        .slot = selected,
        .tensor = TensorAccess::from_context(std::move(*context), TensorAuthority::kLocalWritable),
    };
  }

  [[nodiscard]] expected<void, Error> finish_bundle_field(
      OutputSlot* slot, const expected<void, Error>& publication) noexcept {
    if (slot == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (publication) {
      return finish_publication(*slot, {});
    }
    return finish_publication(*slot,
                              make_unexpected(Error{.code = publication.error().code,
                                                    .message = publication.error().message}));
  }

 private:
  OutputPool(TensorAllocationPlanRecord plan, OutputPlanContract contract, VertexId vertex,
             OperatorInstanceId owner, void* declaration, const RuntimePortTypeOps* type_ops,
             cudaStream_t stream, bool elide_same_stream_readiness,
             PlanGenerationId plan_generation, RunGenerationId run_generation,
             std::uint64_t pool_generation, std::shared_ptr<holomq::BufferPool> export_pool,
             std::shared_ptr<RuntimeTensorOutputTestControl> test_control,
             std::size_t test_pool_index,
             std::optional<std::uint32_t> descriptor_tensor_index) noexcept
      : plan_(std::move(plan)),
        contract_(std::move(contract)),
        vertex_(vertex),
        owner_(owner),
        declaration_(declaration),
        type_ops_(type_ops),
        stream_(stream),
        elide_same_stream_readiness_(elide_same_stream_readiness),
        plan_generation_(plan_generation),
        run_generation_(run_generation),
        pool_generation_(pool_generation),
        export_pool_(std::move(export_pool)),
        test_control_(std::move(test_control)),
        test_pool_index_(test_pool_index),
        descriptor_tensor_index_(descriptor_tensor_index) {}

  [[nodiscard]] bool pool_plan_identity_current() const noexcept {
    return vertex_ && plan_.plan_generation == plan_generation_ && plan_.run_generation == 0U &&
           run_generation_ != 0U;
  }

  [[nodiscard]] bool plan_identity_current(const OutputSlot& slot) const noexcept {
    return pool_plan_identity_current() && slot.loan_plan_generation == plan_generation_ &&
           slot.loan_run_generation == run_generation_;
  }

  [[nodiscard]] bool callback_activation_current(const OutputSlot& slot) const noexcept {
    const RuntimeCallbackIdentity identity = runtime_callback_identity();
    return identity.context == RuntimeCallbackContext::kCompute && identity.operator_id == owner_ &&
           identity.tag.valid() && identity.tag == slot.activation;
  }

  [[nodiscard]] bool current_activation(const OutputSlot& slot) const noexcept {
    return plan_identity_current(slot) && callback_activation_current(slot);
  }

  [[nodiscard]] bool callback_can_allocate(Tag& activation) const noexcept {
    const RuntimeCallbackIdentity identity = runtime_callback_identity();
    if (!vertex_ || plan_.plan_generation != plan_generation_ || plan_.run_generation != 0U ||
        run_generation_ == 0U || identity.context != RuntimeCallbackContext::kCompute ||
        identity.operator_id != owner_ || !identity.tag.valid()) {
      return false;
    }
    activation = identity.tag;
    return true;
  }

  [[nodiscard]] bool claim_admitted(TensorSlotClaimReason reason) const noexcept {
    return contract_.admitted_claims[claim_index(reason)];
  }

  [[nodiscard]] std::uint32_t admitted_claim_mask() const noexcept {
    std::uint32_t result = 0U;
    for (std::size_t index = 0U; index < contract_.admitted_claims.size(); ++index) {
      result |= static_cast<std::uint32_t>(contract_.admitted_claims[index]) << index;
    }
    return result;
  }

  [[nodiscard]] std::uint64_t metric_time_ns() const noexcept {
    if (test_control_ != nullptr) {
      const std::uint64_t injected = test_control_->metric_time_ns();
      if (injected != std::numeric_limits<std::uint64_t>::max()) {
        return injected;
      }
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
  }

  [[nodiscard]] static std::uint64_t elapsed_ns(std::uint64_t start, std::uint64_t end) noexcept {
    return end >= start ? end - start : 0U;
  }

  [[nodiscard]] static std::uint64_t saturating_add(std::uint64_t left,
                                                    std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
  }

  void update_primitive_metrics() noexcept {
    const auto physical_free_depth = static_cast<std::uint32_t>(free_slots_.size());
    const std::uint32_t publisher_loan_credit =
        live_loans_ < plan_.max_outstanding_loans ? plan_.max_outstanding_loans - live_loans_ : 0U;
    const std::uint32_t effective_free_depth = std::min(physical_free_depth, publisher_loan_credit);
    metrics_.physical_free_slot_depth = physical_free_depth;
    metrics_.publisher_loan_credit = publisher_loan_credit;
    metrics_.effective_free_slot_depth = closed_ ? 0U : effective_free_depth;
    if (!metrics_initialized_) {
      metrics_.minimum_physical_free_slot_depth = physical_free_depth;
      metrics_.minimum_publisher_loan_credit = publisher_loan_credit;
      metrics_.minimum_effective_free_slot_depth = metrics_.effective_free_slot_depth;
      metrics_initialized_ = true;
    } else {
      metrics_.minimum_physical_free_slot_depth =
          std::min(metrics_.minimum_physical_free_slot_depth, physical_free_depth);
      metrics_.minimum_publisher_loan_credit =
          std::min(metrics_.minimum_publisher_loan_credit, publisher_loan_credit);
      metrics_.minimum_effective_free_slot_depth =
          std::min(metrics_.minimum_effective_free_slot_depth, metrics_.effective_free_slot_depth);
    }

    const std::uint64_t now = metric_time_ns();
    if (metrics_.effective_free_slot_depth == 0U) {
      if (!zero_depth_active_) {
        zero_depth_active_ = true;
        zero_depth_started_ = now;
        metrics_.zero_depth_since_ns = now;
        ++metrics_.zero_depth_interval_count;
      }
      const std::uint64_t active = elapsed_ns(zero_depth_started_, now);
      metrics_.zero_depth_total_ns = saturating_add(zero_depth_accumulated_ns_, active);
      metrics_.zero_depth_longest_ns = std::max(metrics_.zero_depth_longest_ns, active);
    } else if (zero_depth_active_) {
      const std::uint64_t completed = elapsed_ns(zero_depth_started_, now);
      zero_depth_accumulated_ns_ = saturating_add(zero_depth_accumulated_ns_, completed);
      metrics_.zero_depth_total_ns = zero_depth_accumulated_ns_;
      metrics_.zero_depth_longest_ns = std::max(metrics_.zero_depth_longest_ns, completed);
      metrics_.zero_depth_since_ns = std::numeric_limits<std::uint64_t>::max();
      zero_depth_active_ = false;
    }

    std::uint64_t oldest_grant = std::numeric_limits<std::uint64_t>::max();
    metrics_.oldest_retained_slot = std::numeric_limits<std::uint32_t>::max();
    metrics_.oldest_retained_phase = 0U;
    metrics_.oldest_retained_claim_mask = 0U;
    metrics_.oldest_retained_age_ns = 0U;
    for (const auto& slot : slots_) {
      if (slot->phase != SlotPhase::kFree &&
          (slot->loan_granted_ns < oldest_grant ||
           (slot->loan_granted_ns == oldest_grant &&
            static_cast<std::uint32_t>(slot->index) < metrics_.oldest_retained_slot))) {
        oldest_grant = slot->loan_granted_ns;
        metrics_.oldest_retained_slot = static_cast<std::uint32_t>(slot->index);
        metrics_.oldest_retained_phase = static_cast<std::uint8_t>(slot->phase);
        metrics_.oldest_retained_claim_mask = slot->claims.mask();
        metrics_.oldest_retained_age_ns = elapsed_ns(slot->loan_granted_ns, now);
      }
    }
  }

  void record_test_snapshot(bool terminal = false, bool invariant_failure = false) noexcept {
    if (test_control_ == nullptr) {
      return;
    }
    update_primitive_metrics();
    std::uint32_t active_claims = 0U;
    std::size_t active_slots = 0U;
    for (const auto& slot : slots_) {
      const std::uint32_t mask = slot->claims.mask();
      active_claims |= mask;
      active_slots += static_cast<std::size_t>(mask != 0U);
    }
    test_control_->record(test_pool_index_,
                          admitted_claim_mask(),
                          active_claims,
                          active_slots,
                          pool_generation_,
                          metrics_,
                          invariant_failure,
                          terminal);
  }

  void record_failed_loan(bool physical_exhausted, bool credit_exhausted) noexcept {
    update_primitive_metrics();
    ++metrics_.failed_loan_count;
    metrics_.failed_loan_physical_free_slot_depth = static_cast<std::uint32_t>(free_slots_.size());
    metrics_.failed_loan_publisher_credit =
        live_loans_ < plan_.max_outstanding_loans ? plan_.max_outstanding_loans - live_loans_ : 0U;
    metrics_.failed_loan_effective_free_slot_depth = std::min(
        metrics_.failed_loan_physical_free_slot_depth, metrics_.failed_loan_publisher_credit);
    metrics_.failed_loan_physical_limited = physical_exhausted;
    metrics_.failed_loan_credit_limited = credit_exhausted;
    metrics_.failed_loan_blocking_claim_mask = 0U;
    for (const auto& slot : slots_) {
      metrics_.failed_loan_blocking_claim_mask |= slot->claims.mask();
    }
    metrics_.failed_loan_oldest_retained_slot = metrics_.oldest_retained_slot;
    metrics_.failed_loan_oldest_retained_phase = metrics_.oldest_retained_phase;
    metrics_.failed_loan_oldest_retained_claim_mask = metrics_.oldest_retained_claim_mask;
    metrics_.failed_loan_oldest_retained_age_ns = metrics_.oldest_retained_age_ns;
    record_test_snapshot();
  }

  void record_retry_cancellation(OutputSlot& slot) noexcept {
    if (slot.retry_cancellation_demand) {
      return;
    }
    slot.retry_cancellation_demand = true;
    ++metrics_.retry_cancellation_total_started;
    ++metrics_.current_retry_cancellation_demand;
    metrics_.peak_retry_cancellation_demand = std::max(metrics_.peak_retry_cancellation_demand,
                                                       metrics_.current_retry_cancellation_demand);
    record_test_snapshot();
  }

  [[nodiscard]] bool consume_test_fault(RuntimeTensorOutputTestFault fault) noexcept {
    return test_control_ != nullptr && test_control_->consume_fault(fault);
  }

  [[nodiscard]] RuntimeTensorOutputTestCorruption consume_test_corruption() noexcept {
    return test_control_ != nullptr ? test_control_->consume_corruption()
                                    : RuntimeTensorOutputTestCorruption::kNone;
  }

  void inject_test_corruption(OutputSlot& slot) noexcept {
    switch (consume_test_corruption()) {
      case RuntimeTensorOutputTestCorruption::kPlanGeneration:
        slot.loan_plan_generation = plan_generation_ + 1U;
        break;
      case RuntimeTensorOutputTestCorruption::kPoolGeneration:
        slot.loan_pool_generation = pool_generation_ + 1U;
        break;
      case RuntimeTensorOutputTestCorruption::kSlotGeneration:
        slot.generation = 0U;
        break;
      case RuntimeTensorOutputTestCorruption::kNone:
      default:
        break;
    }
  }

  [[nodiscard]] TensorOutputDiagnosticReason identity_failure_reason(
      const OutputSlot& slot) const noexcept {
    return slot.loan_pool_generation != pool_generation_
               ? TensorOutputDiagnosticReason::kLoanPoolGenerationMismatch
               : TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch;
  }

  [[nodiscard]] bool identity_valid(const OutputSlot& slot) const noexcept {
    return slot.pool == this && slot.index < slots_.size() && slots_[slot.index].get() == &slot &&
           pool_generation_ != 0U && slot.loan_pool_generation == pool_generation_ &&
           slot.generation != 0U && slot.generation != std::numeric_limits<std::uint64_t>::max();
  }

  void fail_identity(OutputSlot& slot, TensorOutputOperation operation) noexcept {
    record_diagnostic(&slot, operation, ErrorCode::kFailure, identity_failure_reason(slot));
    if (slot.phase == SlotPhase::kWriterActive) {
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    }
    consume_loan(slot);
    quarantine(slot);
    record_test_snapshot(false, true);
  }

  void record_diagnostic(const OutputSlot* slot, TensorOutputOperation operation, ErrorCode error,
                         TensorOutputDiagnosticReason reason) noexcept {
    const TensorOutputDiagnosticRecord record{
        .output = plan_.output,
        .plan_generation = plan_generation_,
        .run_generation = run_generation_,
        .activation = slot != nullptr ? slot->activation : Tag{},
        .pool_generation = pool_generation_,
        .slot_generation = slot != nullptr ? slot->generation : 0U,
        .slot = slot != nullptr ? static_cast<std::uint32_t>(slot->index) : 0U,
        .from_state = slot != nullptr ? static_cast<std::uint8_t>(slot->phase) : std::uint8_t{0U},
        .attempted_operation = static_cast<std::uint8_t>(operation),
        .writer_count = slot != nullptr && slot->phase == SlotPhase::kWriterActive
                            ? std::uint8_t{1U}
                            : std::uint8_t{0U},
        .error = error,
        .reason = reason,
    };
    diagnostics_.append(record);
    if (test_control_ != nullptr) {
      test_control_->record_diagnostic(test_pool_index_, record);
    }
  }

  void add_claim(OutputSlot& slot, TensorSlotClaimReason reason) noexcept {
    if (!claim_admitted(reason)) {
      slot.phase = SlotPhase::kQuarantined;
      if (claim_admitted(TensorSlotClaimReason::kQuarantine)) {
        static_cast<void>(
            slot.claims.add(TensorSlotClaimReason::kQuarantine, contract_.admitted_claims));
      }
      record_test_snapshot();
      return;
    }
    if (!slot.claims.add(reason, contract_.admitted_claims)) {
      slot.phase = SlotPhase::kQuarantined;
    }
    record_test_snapshot();
  }

  void remove_claim(OutputSlot& slot, TensorSlotClaimReason reason) noexcept {
    slot.claims.remove(reason);
    record_test_snapshot();
  }

  void quarantine(OutputSlot& slot) noexcept {
    slot.phase = SlotPhase::kQuarantined;
    add_claim(slot, TensorSlotClaimReason::kQuarantine);
  }

  void enqueue_deferred(OutputSlot& slot) noexcept {
    if (slot.deferred_queued.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    OutputSlot* head = deferred_head_.load(std::memory_order_acquire);
    do {
      slot.deferred_next.store(head, std::memory_order_relaxed);
    } while (!deferred_head_.compare_exchange_weak(
        head, &slot, std::memory_order_release, std::memory_order_acquire));
  }

  void reconcile_deferred_releases(std::unique_lock<std::mutex>& lock) noexcept {
    OutputSlot* slot = deferred_head_.exchange(nullptr, std::memory_order_acquire);
    while (slot != nullptr) {
      OutputSlot* const next = slot->deferred_next.exchange(nullptr, std::memory_order_relaxed);
      slot->deferred_queued.store(false, std::memory_order_release);
      if (slot->quarantine_requested.exchange(false, std::memory_order_acq_rel)) {
        quarantine(*slot);
      }
      if (slot->retirement_requested.load(std::memory_order_acquire) &&
          slot->tensor_handles.load(std::memory_order_acquire) == 0U) {
        if (slot->phase == SlotPhase::kRetained) {
          slot->retirement_requested.store(false, std::memory_order_release);
          begin_retirement(*slot, lock);
        } else if (slot->phase == SlotPhase::kPublicationTerminal) {
          slot->retirement_requested.store(false, std::memory_order_release);
          quarantine(*slot);
        }
      }
      slot = next;
    }
  }

  void erase_retiring_slot(std::size_t position) noexcept {
    retiring_slots_[position] = retiring_slots_.back();
    retiring_slots_.pop_back();
  }

  void quarantine_retiring_slot(OutputSlot& slot, std::size_t position) noexcept {
    slot.on_retirement_queue = false;
    quarantine(slot);
    erase_retiring_slot(position);
  }

  void recycle_retired_slot(OutputSlot& slot, std::size_t position) noexcept {
    slot.phase = SlotPhase::kFree;
    slot.claims.clear();
    slot.loan_plan_generation = 0U;
    slot.loan_run_generation = 0U;
    slot.loan_pool_generation = 0U;
    slot.loan_granted_ns = 0U;
    if (slot.retry_cancellation_demand) {
      slot.retry_cancellation_demand = false;
      if (metrics_.current_retry_cancellation_demand != 0U) {
        --metrics_.current_retry_cancellation_demand;
      }
    }
    slot.cancel_pending = false;
    slot.writer_cuda = false;
    slot.writer_stream = nullptr;
    slot.byte_size = 0U;
    slot.activation = {};
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.quarantine_requested.store(false, std::memory_order_release);
    slot.on_free_list = true;
    free_slots_.push_back(static_cast<std::uint32_t>(slot.index));
    if (test_control_ != nullptr) {
      test_control_->record_retired_slot(test_pool_index_, static_cast<std::uint32_t>(slot.index));
    }
    record_test_snapshot();
    erase_retiring_slot(position);
  }

  void reclaim_completed_retirements(std::unique_lock<std::mutex>& lock) noexcept {
    if (retirement_query_active_) {
      // Allocation remains nonblocking while another controller queries the active inventory.
      return;
    }
    retirement_query_active_ = true;
    std::size_t position = 0U;
    while (position < retiring_slots_.size()) {
      const std::uint32_t index = retiring_slots_[position];
      if (index >= slots_.size()) {
        erase_retiring_slot(position);
        continue;
      }
      auto& slot = *slots_[index];
      if (!slot.on_retirement_queue || slot.phase != SlotPhase::kRetiring) {
        quarantine_retiring_slot(slot, position);
        continue;
      }
      const std::uint64_t generation = slot.generation;
      bool complete = true;
      if (slot.lifetime->retirement_requires_query()) {
        lock.unlock();
        complete = slot.lifetime->retirement_complete();
        lock.lock();
      }
      if (slot.generation != generation || !slot.on_retirement_queue ||
          slot.phase != SlotPhase::kRetiring) {
        quarantine_retiring_slot(slot, position);
        continue;
      }
      if (slot.tensor_handles.load(std::memory_order_acquire) != 0U || !complete) {
        ++position;
        continue;
      }
      if (export_pool_ && !slot.lifetime->prepare_exportable_buffer_for_reuse()) {
        ++position;
        continue;
      }
      slot.on_retirement_queue = false;
      if (!advance_tensor_slot_generation(slot.generation)) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kRetire,
                          ErrorCode::kFailure,
                          TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch);
        quarantine(slot);
        continue;
      }
      recycle_retired_slot(slot, position);
    }
    retirement_query_active_ = false;
  }

  void consume_loan(OutputSlot& slot) noexcept {
    if (slot.loan_live) {
      slot.loan_live = false;
      if (live_loans_ != 0U) {
        --live_loans_;
      } else {
        quarantine(slot);
      }
      remove_claim(slot, TensorSlotClaimReason::kProducerLoan);
    }
  }

  void begin_retirement(OutputSlot& slot, std::unique_lock<std::mutex>& lock) noexcept {
    if (slot.tensor_handles.load(std::memory_order_acquire) != 0U ||
        slot.phase == SlotPhase::kWriterActive || slot.phase == SlotPhase::kSubmitPending ||
        slot.phase == SlotPhase::kPublicationTerminal ||
        slot.phase == SlotPhase::kReleaseRecording || slot.phase == SlotPhase::kQuarantined ||
        slot.phase == SlotPhase::kRetiring || slot.on_retirement_queue) {
      return;
    }
    add_claim(slot, TensorSlotClaimReason::kRetirementService);
    if (slot.phase == SlotPhase::kQuarantined) {
      return;
    }
    if (retiring_slots_.size() + retirement_records_in_flight_ >= plan_.retirement_capacity) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kRetire,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanRetirementStalled);
      quarantine(slot);
      return;
    }
    if (!slot.lifetime->has_release_streams()) {
      slot.phase = SlotPhase::kRetiring;
      slot.on_retirement_queue = true;
      retiring_slots_.push_back(static_cast<std::uint32_t>(slot.index));
      reclaim_completed_retirements(lock);
      return;
    }
    const std::uint64_t generation = slot.generation;
    // Publish an in-progress phase before dropping the pool lock. Other controllers may continue,
    // but this slot cannot enter the retirement inventory or free list until recording completes.
    slot.phase = SlotPhase::kReleaseRecording;
    ++retirement_records_in_flight_;
    lock.unlock();
    const bool release_recorded = slot.lifetime->record_release();
    lock.lock();
    --retirement_records_in_flight_;
    if (slot.generation != generation || slot.phase != SlotPhase::kReleaseRecording) {
      quarantine(slot);
      return;
    }
    if (!release_recorded) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kRetire,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanReleaseEvidenceMissing);
      quarantine(slot);
      return;
    }
    slot.phase = SlotPhase::kRetiring;
    slot.on_retirement_queue = true;
    retiring_slots_.push_back(static_cast<std::uint32_t>(slot.index));
    reclaim_completed_retirements(lock);
  }

  // Keep allocation admission, diagnostics, and the bounded slot transition in one transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate(TensorLoanRequest request) noexcept {
    Tag activation{};
    if (!callback_can_allocate(activation)) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kAllocationCapabilityMissing);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!valid_dtype(request.dtype) || request.shape.size() > plan_.bounds.max_rank ||
        (plan_.layout && plan_.layout->memory_kind &&
         *plan_.layout->memory_kind != plan_.memory_kind) ||
        (plan_.layout && plan_.layout->rank && request.shape.size() != *plan_.layout->rank) ||
        (plan_.layout && plan_.layout->dtype &&
         !dtype_equal(request.dtype, *plan_.layout->dtype))) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanRequestLayoutMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto span = tensor_byte_span(request.shape, request.dtype, request.byte_strides);
    if (!span || *span > plan_.bounds.max_byte_span) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanRequestLayoutMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kAllocationCapabilityMissing);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    reclaim_completed_retirements(lock);
    const bool credit_exhausted = live_loans_ >= plan_.max_outstanding_loans;
    const bool physical_exhausted = free_slots_.empty();
    if (credit_exhausted || physical_exhausted) {
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kResourceExhausted,
                        TensorOutputDiagnosticReason::kAllocationReadinessRecordExhausted);
      if (test_control_ != nullptr) {
        record_failed_loan(physical_exhausted, credit_exhausted);
      }
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const std::uint32_t index = free_slots_.back();
    free_slots_.pop_back();
    auto& slot = *slots_[index];
    if (slot.pool != this || slot.index != index || slot.generation == 0U ||
        slot.generation == std::numeric_limits<std::uint64_t>::max() ||
        slot.phase != SlotPhase::kFree || !slot.on_free_list || slot.loan_live ||
        slot.on_retirement_queue || slot.deferred_queued.load(std::memory_order_acquire) ||
        slot.tensor_handles.load(std::memory_order_acquire) != 0U || slot.claims.any()) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (export_pool_ &&
        (!slot.lifetime->has_exportable_buffer() || slot.lifetime->data() == nullptr)) {
      slot.on_free_list = false;
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    slot.on_free_list = false;
    slot.activation = activation;
    slot.loan_plan_generation = plan_generation_;
    slot.loan_run_generation = run_generation_;
    slot.loan_pool_generation = pool_generation_;
    slot.loan_granted_ns = test_control_ != nullptr ? metric_time_ns() : 0U;
    slot.phase = SlotPhase::kLoaned;
    slot.loan_live = true;
    slot.cancel_pending = false;
    slot.writer_cuda = false;
    slot.writer_stream = nullptr;
    slot.byte_size = *span;
    slot.dtype = request.dtype;
    slot.shape.assign(request.shape.begin(), request.shape.end());
    slot.byte_strides.clear();
    slot.explicit_strides = request.byte_strides.has_value();
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.quarantine_requested.store(false, std::memory_order_release);
    ++live_loans_;
    add_claim(slot, TensorSlotClaimReason::kProducerLoan);
    add_claim(slot, TensorSlotClaimReason::kProducerReadiness);
    if (slot.phase == SlotPhase::kQuarantined) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (request.byte_strides) {
      slot.byte_strides.assign(request.byte_strides.begin(), request.byte_strides.end());
    }
    slot.lifetime->reset_generation();

    auto& context = *slot.context;
    context.dl_shape.assign(slot.shape.begin(), slot.shape.end());
    context.dl_strides.clear();
    const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(request.dtype);
    if (slot.explicit_strides) {
      for (const std::int64_t stride : slot.byte_strides) {
        context.dl_strides.push_back(stride / static_cast<std::int64_t>(item_bytes));
      }
    }
    context.tensor.flags = 0U;
    context.tensor.dl_tensor.ndim = static_cast<std::int32_t>(context.dl_shape.size());
    context.tensor.dl_tensor.dtype = request.dtype;
    context.tensor.dl_tensor.shape = context.dl_shape.empty() ? nullptr : context.dl_shape.data();
    context.tensor.dl_tensor.strides =
        context.dl_strides.empty() ? nullptr : context.dl_strides.data();
    context.tensor.dl_tensor.byte_offset = 0U;
    record_test_snapshot();
    return TensorOutputLoanAccess::loan(&slot, &kLoanOps, slots_[index]);
  }

  [[nodiscard]] expected<TensorOutputWriteGuard, Error> begin_write(OutputSlot& slot,
                                                                    cudaStream_t stream,
                                                                    bool cuda_write) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (!current_activation(slot)) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    inject_test_corruption(slot);
    if (!identity_valid(slot)) {
      fail_identity(slot, TensorOutputOperation::kBeginWrite);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (slot.phase != SlotPhase::kLoaned || !slot.loan_live) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kInvalidArgument,
                        slot.phase == SlotPhase::kWriterActive
                            ? TensorOutputDiagnosticReason::kLoanWriterActive
                            : TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (cuda_write) {
      if (stream_ == nullptr || stream == nullptr || stream != stream_) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kBeginWrite,
                          ErrorCode::kNotSupported,
                          TensorOutputDiagnosticReason::kLoanStreamDeviceMismatch);
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
    } else if (plan_.memory_kind == MemoryKind::kCudaDevice) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kNotSupported,
                        TensorOutputDiagnosticReason::kLoanStreamDeviceMismatch);
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    slot.writer_cuda = cuda_write;
    slot.writer_stream = stream;
    slot.phase = SlotPhase::kWriterActive;
    add_claim(slot, TensorSlotClaimReason::kProducerWriter);
    if (slot.phase == SlotPhase::kQuarantined) {
      consume_loan(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return TensorOutputLoanAccess::write_guard(&slot, &kWriteOps, slot.self.lock());
  }

  [[nodiscard]] expected<void, Error> commit_write(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kGuardDoubleCommit);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool initially_invalid =
        closed_ || !current_activation(slot) || slot.cancel_pending || !slot.loan_live;
    if (!initially_invalid) {
      inject_test_corruption(slot);
      if (!identity_valid(slot)) {
        fail_identity(slot, TensorOutputOperation::kCommitWrite);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    cudaStream_t readiness_stream = slot.writer_cuda ? slot.writer_stream : stream_;
    const bool readiness_required =
        slot.writer_cuda || (!initially_invalid && slot.lifetime->has_readiness());
    const bool force_readiness_failure =
        !initially_invalid &&
        consume_test_fault(RuntimeTensorOutputTestFault::kCommitReadinessFailure);
    bool readiness_ok = !force_readiness_failure;
    if (readiness_ok && readiness_required) {
      lock.unlock();
      readiness_ok = slot.lifetime->record_readiness(readiness_stream);
      lock.lock();
    }
    if (slot.phase != SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (closed_ || !current_activation(slot) || slot.cancel_pending || !slot.loan_live) {
      const ErrorCode invalid_code = closed_ ? ErrorCode::kClosed : ErrorCode::kInvalidArgument;
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        invalid_code,
                        !slot.loan_live
                            ? TensorOutputDiagnosticReason::kGuardOutlivedLoan
                            : TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      if (!readiness_ok) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kCommitWrite,
                          ErrorCode::kFailure,
                          TensorOutputDiagnosticReason::kLoanReadinessRecordFailed);
        quarantine(slot);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      slot.phase = SlotPhase::kReady;
      begin_retirement(slot, lock);
      return make_unexpected(Error{invalid_code});
    }
    if (!readiness_ok) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanReadinessRecordFailed);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    slot.phase = SlotPhase::kReady;
    slot.writer_stream = nullptr;
    return {};
  }

  void abandon_write(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kWriterActive) {
      return;
    }
    record_diagnostic(&slot,
                      TensorOutputOperation::kDropWrite,
                      ErrorCode::kInvalidArgument,
                      TensorOutputDiagnosticReason::kGuardDroppedWithoutCommit);
    const bool writer_cuda = slot.writer_cuda;
    cudaStream_t writer_stream = slot.writer_stream;
    bool readiness_ok = true;
    if (writer_cuda) {
      lock.unlock();
      readiness_ok = slot.lifetime->record_readiness(writer_stream);
      lock.lock();
    }
    if (!readiness_ok) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kDropWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanCancelSafetyRecordFailed);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      quarantine(slot);
      return;
    }
    remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    slot.phase = SlotPhase::kReady;
    begin_retirement(slot, lock);
  }

  void abandon_loan(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!slot.loan_live) {
      return;
    }
    consume_loan(slot);
    if (slot.phase == SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanDestroyedWithActiveWriter);
      slot.cancel_pending = true;
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      return;
    }
    if (slot.phase == SlotPhase::kLoaned || slot.phase == SlotPhase::kReady) {
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      begin_retirement(slot, lock);
    }
  }

  [[nodiscard]] expected<void, Error> cancel(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!slot.loan_live || (slot.phase != SlotPhase::kLoaned && slot.phase != SlotPhase::kReady &&
                            slot.phase != SlotPhase::kWriterActive)) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanDoubleCancel);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    if (slot.phase == SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanWriterActive);
      slot.cancel_pending = true;
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    begin_retirement(slot, lock);
    return {};
  }

  void add_publication_claims(OutputSlot& slot) noexcept {
    constexpr std::array publication_reasons{
        TensorSlotClaimReason::kAcceptedPublication,
        TensorSlotClaimReason::kHolomqTransportOrFanout,
        TensorSlotClaimReason::kSubscriberBorrowOrImport,
        TensorSlotClaimReason::kConsumerCudaRelease,
        TensorSlotClaimReason::kRelayOrReexport,
        TensorSlotClaimReason::kTemporalCache,
        TensorSlotClaimReason::kPendingOrSelectedWindow,
        TensorSlotClaimReason::kReuseLast,
        TensorSlotClaimReason::kDrainAllActivation,
        TensorSlotClaimReason::kDlpackOrFrameworkBorrow,
        TensorSlotClaimReason::kCopyOrRecorderMaterialization,
    };
    for (const auto reason : publication_reasons) {
      if (claim_admitted(reason)) {
        add_claim(slot, reason);
      }
    }
  }

  [[nodiscard]] expected<void, Error> finish_publication(OutputSlot& slot,
                                                         expected<void, Error> result) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kSubmitPending) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    const std::uint32_t handles = slot.tensor_handles.load(std::memory_order_acquire);
    if (result) {
      add_publication_claims(slot);
      if (slot.phase == SlotPhase::kQuarantined) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      slot.phase = SlotPhase::kRetained;
      if (handles == 0U) {
        slot.retirement_requested.store(false, std::memory_order_release);
        begin_retirement(slot, lock);
      }
      return {};
    }

    const ErrorCode code = result.error().code;
    if (publication_error_consumes_authority(code)) {
      const TensorOutputDiagnosticReason reason =
          code == ErrorCode::kPublicationPartial
              ? TensorOutputDiagnosticReason::kPublicationPartial
              : TensorOutputDiagnosticReason::kPublicationIndeterminate;
      record_diagnostic(&slot, TensorOutputOperation::kEmit, code, reason);
      add_publication_claims(slot);
      slot.phase = SlotPhase::kPublicationTerminal;
      if (handles == 0U) {
        slot.retirement_requested.store(false, std::memory_order_release);
        quarantine(slot);
      }
      return make_unexpected(std::move(result).error());
    }

    // A proven pre-submit result cannot retain a recipient-owned Tensor handle. A live handle at
    // this boundary contradicts the frozen publication phase, so fail closed without retry/reuse.
    if (handles != 0U) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.phase = SlotPhase::kReady;
    if (test_control_ != nullptr) {
      record_retry_cancellation(slot);
    }
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    begin_retirement(slot, lock);
    if (code == contract_.publication.pre_submit_error || code == ErrorCode::kClosed) {
      return make_unexpected(Error{code});
    }
    if (code == ErrorCode::kNotReady || code == ErrorCode::kStaleEndpoint) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return make_unexpected(Error{code});
  }

  void cancel_invalid_emit(OutputSlot& slot, std::unique_lock<std::mutex>& lock) noexcept {
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    if (slot.phase == SlotPhase::kWriterActive) {
      slot.cancel_pending = true;
    } else {
      begin_retirement(slot, lock);
    }
  }

  [[nodiscard]] TensorOutputDiagnosticReason emit_readiness_reason(
      const OutputSlot& slot) const noexcept {
    if (!callback_activation_current(slot)) {
      return TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch;
    }
    if (slot.phase == SlotPhase::kWriterActive) {
      return TensorOutputDiagnosticReason::kLoanWriterActive;
    }
    return !slot.loan_live ? TensorOutputDiagnosticReason::kLoanDoubleEmit
                           : TensorOutputDiagnosticReason::kLoanReadinessNotCommitted;
  }

  [[nodiscard]] expected<std::shared_ptr<DLManagedTensorContext>, Error> prepare_emit(
      OutputSlot& selected, bool force_closed) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_ || force_closed) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    inject_test_corruption(selected);
    if (!plan_identity_current(selected)) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanPlanGenerationMismatch);
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!callback_activation_current(selected) || !selected.loan_live ||
        selected.phase != SlotPhase::kReady) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        emit_readiness_reason(selected));
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!identity_valid(selected)) {
      fail_identity(selected, TensorOutputOperation::kEmit);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    consume_loan(selected);
    selected.phase = SlotPhase::kSubmitPending;
    return selected.context;
  }

  [[nodiscard]] expected<void, Error> emit(TensorOutputLoan loan, const void* descriptor,
                                           EmitOptions options) noexcept {
    const bool force_closed = consume_test_fault(RuntimeTensorOutputTestFault::kEmitClosedBinding);
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    std::optional<OutputPoolCallbackClaim> selected_pool;
    if (selected != nullptr) {
      selected_pool.emplace(*selected);
    }
    if (!selected_pool || !*selected_pool || selected_pool->pool() != this ||
        selected->index >= slots_.size() || slots_[selected->index].get() != selected) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanOutputMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool descriptor_schema =
        type_ops_->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
    if (descriptor_schema != (descriptor != nullptr)) {
      std::unique_lock lock(mutex_);
      reconcile_deferred_releases(lock);
      cancel_invalid_emit(*selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    auto context = prepare_emit(*selected, force_closed);
    if (!context) {
      return make_unexpected(std::move(context.error()));
    }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      Tensor tensor =
          TensorAccess::from_context(std::move(*context), TensorAuthority::kLocalWritable);
      auto published =
          type_ops_->publish_tensor_loan(declaration_, descriptor, std::move(tensor), options);
      if (published && consume_test_fault(RuntimeTensorOutputTestFault::kPostSubmitIndeterminate)) {
        published = make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      } else if (published &&
                 consume_test_fault(RuntimeTensorOutputTestFault::kPostSubmitPartial)) {
        published = make_unexpected(make_publication_partial_error(0U, 1U));
      }
      return finish_publication(*selected, std::move(published));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      std::unique_lock lock(mutex_);
      reconcile_deferred_releases(lock);
      quarantine(*selected);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<TensorOutputLoan, Error> allocate_callback(void* state,
                                                             TensorLoanRequest request) noexcept {
    return static_cast<OutputPool*>(state)->allocate(request);
  }

  static expected<void, Error> emit_callback(void* state, TensorOutputLoan loan,
                                             const void* descriptor, EmitOptions options) noexcept {
    return static_cast<OutputPool*>(state)->emit(std::move(loan), descriptor, options);
  }

  static void abandon_loan_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    if (pool) {
      pool.pool()->abandon_loan(slot);
    }
  }

  static TensorShapeView shape_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->shape;
  }

  static TensorByteStridesView strides_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    return slot.explicit_strides ? TensorByteStridesView{slot.byte_strides}
                                 : TensorByteStridesView{std::nullopt};
  }

  static DLDataType dtype_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->dtype;
  }

  static DeviceId device_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->device;
  }

  static MemoryKind memory_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->memory_kind;
  }

  static std::size_t bytes_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->byte_size;
  }

  static expected<TensorOutputWriteGuard, Error> write_callback(
      void* state, ::cuda::stream_ref stream) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->begin_write(slot, stream.get(), true)
                : expected<TensorOutputWriteGuard, Error>{
                      make_unexpected(Error{ErrorCode::kClosed})};
  }

  static expected<TensorOutputWriteGuard, Error> write_host_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->begin_write(slot, nullptr, false)
                : expected<TensorOutputWriteGuard, Error>{
                      make_unexpected(Error{ErrorCode::kClosed})};
  }

  static expected<void, Error> cancel_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->cancel(slot)
                : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
  }

  static void abandon_write_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    if (pool) {
      pool.pool()->abandon_write(slot);
    }
  }

  static void* data_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? slot.context->tensor.dl_tensor.data : nullptr;
  }

  static std::optional<::cuda::stream_ref> stream_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool && slot.writer_cuda
               ? std::optional<::cuda::stream_ref>{::cuda::stream_ref{slot.writer_stream}}
               : std::nullopt;
  }

  static expected<void, Error> commit_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->commit_write(slot)
                : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
  }

  inline static const TensorOutputBindingOps kBindingOps{
      .allocate = &allocate_callback,
      .emit_loan = &emit_callback,
  };

  inline static const TensorOutputLoanOps kLoanOps{
      .abandon = &abandon_loan_callback,
      .shape = &shape_callback,
      .byte_strides = &strides_callback,
      .dtype = &dtype_callback,
      .device_id = &device_callback,
      .memory_kind = &memory_callback,
      .byte_size = &bytes_callback,
      .write = &write_callback,
      .write_host = &write_host_callback,
      .cancel = &cancel_callback,
  };

  inline static const TensorOutputWriteGuardOps kWriteOps{
      .abandon = &abandon_write_callback,
      .data = &data_callback,
      .byte_size = &bytes_callback,
      .stream = &stream_callback,
      .commit = &commit_callback,
  };

  TensorAllocationPlanRecord plan_;
  OutputPlanContract contract_;
  VertexId vertex_{};
  OperatorInstanceId owner_{};
  void* declaration_{};
  const RuntimePortTypeOps* type_ops_{};
  cudaStream_t stream_{};
  bool elide_same_stream_readiness_{};
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  std::uint64_t pool_generation_{};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control_;
  std::size_t test_pool_index_{};
  std::optional<std::uint32_t> descriptor_tensor_index_;
  std::mutex mutex_;
  std::vector<std::shared_ptr<OutputSlot>> slots_;
  std::vector<std::shared_ptr<TensorMemoryReference>> references_;
  std::shared_ptr<OutputPoolReferenceControl> reference_control_;
  std::shared_ptr<holomq::BufferPool> export_pool_;
  std::vector<std::uint32_t> free_slots_;
  std::vector<std::uint32_t> retiring_slots_;
  std::atomic<OutputSlot*> deferred_head_{};
  std::vector<TensorOutputDiagnosticRecord> diagnostic_storage_;
  TensorOutputDiagnosticBuffer diagnostics_;
  RuntimeTensorOutputTestControl::PrimitiveMetrics metrics_;
  std::uint64_t zero_depth_started_{};
  std::uint64_t zero_depth_accumulated_ns_{};
  std::uint32_t live_loans_{};
  std::uint32_t retirement_records_in_flight_{};
  bool metrics_initialized_{};
  bool zero_depth_active_{};
  bool retirement_query_active_{};
  bool closed_{};
  bool bound_{};
  bool restart_fenced_{};
  bool lifecycle_transition_active_{};
};

template <typename Base>
void SlotMemoryReference<Base>::retain_tensor_handle() noexcept {
  if (OutputPool* const pool = control_ != nullptr ? control_->acquire() : nullptr;
      pool != nullptr) {
    pool->retain_tensor(slot_);
    control_->release();
  }
}

template <typename Base>
void SlotMemoryReference<Base>::release_tensor_handle() noexcept {
  if (OutputPool* const pool = control_ != nullptr ? control_->acquire() : nullptr;
      pool != nullptr) {
    pool->release_tensor(slot_);
    control_->release();
  }
}

[[nodiscard]] bool canonical_resource_order(const TensorDeviceResourcePlanRecord& left,
                                            const TensorDeviceResourcePlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value) <
         std::tie(right.process_group.value, right.device.value);
}

struct ResourceTotals {
  std::uint32_t pools{};
  std::uint32_t slots{};
  std::uint32_t loans{};
  std::uint32_t controls{};
  std::uint32_t readiness{};
  std::uint32_t releases{};
  std::uint32_t retirements{};
  std::uint32_t quarantines{};
};

[[nodiscard]] bool append_totals(ResourceTotals& totals, std::uint32_t slots, std::uint32_t loans,
                                 std::uint32_t controls, std::uint32_t readiness,
                                 std::uint32_t releases, std::uint32_t retirements,
                                 std::uint32_t quarantines) noexcept {
  return checked_add(totals.pools, 1U, totals.pools) &&
         checked_add(totals.slots, slots, totals.slots) &&
         checked_add(totals.loans, loans, totals.loans) &&
         checked_add(totals.controls, controls, totals.controls) &&
         checked_add(totals.readiness, readiness, totals.readiness) &&
         checked_add(totals.releases, releases, totals.releases) &&
         checked_add(totals.retirements, retirements, totals.retirements) &&
         checked_add(totals.quarantines, quarantines, totals.quarantines);
}

struct OperatorStreamSelection {
  cudaStream_t stream{};
  DeviceId device{};
};

[[nodiscard]] bool valid_fused_linear_edges(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource) noexcept {
  if (std::adjacent_find(resource.fused_linear_edges.begin(),
                         resource.fused_linear_edges.end(),
                         [](EdgeId left, EdgeId right) { return left.value >= right.value; }) !=
      resource.fused_linear_edges.end()) {
    return false;
  }
  for (const EdgeId edge_id : resource.fused_linear_edges) {
    const auto edge = std::find_if(plan.payload_edges.begin(),
                                   plan.payload_edges.end(),
                                   [&](const auto& row) { return row.edge == edge_id; });
    const bool tensor_backed =
        edge != plan.payload_edges.end() &&
        (edge->payload_identity.kind == PayloadIdentityKind::kTensor ||
         (edge->payload_identity.kind == PayloadIdentityKind::kSchema &&
          edge->payload_identity.payload_kind == PayloadKind::kDescriptorBacked));
    if (!edge_id || edge == plan.payload_edges.end() ||
        edge->disposition != PlanDisposition::kAccepted ||
        edge->locality != ChannelLocalityKind::kProcessLocal || !tensor_backed ||
        edge->fanout != 1U || !edge->source || !edge->target ||
        edge->target.value() > plan.ports.size()) {
      return false;
    }
    const auto allocation = std::find_if(
        plan.tensor_allocations.begin(), plan.tensor_allocations.end(), [&](const auto& row) {
          return row.output == edge->source &&
                 row.allocator_process_group == resource.process_group &&
                 row.device == resource.device;
        });
    const auto field_allocation =
        std::find_if(plan.tensor_field_allocations.begin(),
                     plan.tensor_field_allocations.end(),
                     [&](const auto& row) {
                       return row.output == edge->source &&
                              row.allocator_process_group == resource.process_group &&
                              row.device == resource.device;
                     });
    const bool matching_allocation =
        allocation != plan.tensor_allocations.end() ||
        (edge->payload_identity.kind == PayloadIdentityKind::kSchema &&
         edge->payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         field_allocation != plan.tensor_field_allocations.end());
    const auto& target = plan.ports[static_cast<std::size_t>(edge->target.value() - 1U)];
    const auto placement =
        std::find_if(plan.vertex_process_placements.begin(),
                     plan.vertex_process_placements.end(),
                     [&](const auto& row) { return row.vertex == target.owner; });
    if (!matching_allocation || target.id != edge->target ||
        target.owner_kind != PortOwnerKind::kOperator ||
        placement == plan.vertex_process_placements.end() ||
        placement->process_group != resource.process_group) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_device_resource_contract(const ExecutionPlanState& plan,
                                                  std::size_t index) noexcept {
  const auto& resource = plan.tensor_device_resources[index];
  const bool common =
      resource.plan_generation == plan.plan_generation && resource.run_generation == 0U &&
      resource.process_group && resource.device.valid() && !resource.allocator_root_id.empty() &&
      !resource.stream_id.empty() && resource.materialize_before == LifecycleStage::kArm &&
      !resource.per_partition_stream && !resource.per_partition_event &&
      !resource.per_partition_green_context && !resource.gpu_resident_backend_selected &&
      resource.fallback_rejected && valid_fused_linear_edges(plan, resource) &&
      (index == 0U || canonical_resource_order(plan.tensor_device_resources[index - 1U], resource));
  if (!common) {
    return false;
  }
  if (resource.pool_count != 0U) {
    return resource.slot_capacity != 0U && resource.control_capacity == resource.slot_capacity &&
           resource.readiness_capacity == resource.slot_capacity &&
           resource.release_capacity == resource.slot_capacity &&
           resource.retirement_capacity == resource.slot_capacity &&
           resource.quarantine_capacity == resource.slot_capacity;
  }
  const bool capacities_empty =
      resource.slot_capacity == 0U && resource.loan_credit_capacity == 0U &&
      resource.control_capacity == 0U && resource.readiness_capacity == 0U &&
      resource.release_capacity == 0U && resource.retirement_capacity == 0U &&
      resource.quarantine_capacity == 0U && resource.fused_linear_edges.empty();
  return capacities_empty &&
         std::any_of(plan.cross_process_channel_services.begin(),
                     plan.cross_process_channel_services.end(),
                     [&](const auto& service) {
                       return cross_process_tensor_service_requires_stream_for_device_in_group(
                           service, resource.process_group, resource.device);
                     });
}

[[nodiscard]] expected<void, Error> append_direct_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  for (const auto& pool : plan.tensor_allocations) {
    if (pool.device != resource.device || pool.allocator_process_group != resource.process_group) {
      continue;
    }
    if (allocator_root.empty()) {
      allocator_root = pool.allocator_root_id;
    }
    if (allocator_root != pool.allocator_root_id || !append_totals(totals,
                                                                   pool.slot_count,
                                                                   pool.max_outstanding_loans,
                                                                   pool.control_capacity,
                                                                   pool.readiness_capacity,
                                                                   pool.release_capacity,
                                                                   pool.retirement_capacity,
                                                                   pool.quarantine_capacity)) {
      return make_unexpected(tensor_resource_error(
          "direct-pool resource totals overflow", "resource", resource.allocator_root_id));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_field_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  for (const auto& pool : plan.tensor_field_allocations) {
    if (pool.device != resource.device || pool.allocator_process_group != resource.process_group) {
      continue;
    }
    if (allocator_root.empty()) {
      allocator_root = pool.allocator_root_id;
    }
    if (allocator_root != pool.allocator_root_id || !append_totals(totals,
                                                                   pool.slot_count,
                                                                   pool.max_outstanding_loans,
                                                                   pool.control_capacity,
                                                                   pool.readiness_capacity,
                                                                   pool.release_capacity,
                                                                   pool.retirement_capacity,
                                                                   pool.quarantine_capacity)) {
      return make_unexpected(tensor_resource_error(
          "field-pool resource totals overflow", "resource", resource.allocator_root_id));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_interface_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  for (const auto& pool : plan.tensor_interface_pools) {
    if (pool.device != resource.device || pool.allocator_process_group != resource.process_group) {
      continue;
    }
    if (allocator_root.empty()) {
      allocator_root = pool.allocator_root_id;
    }
    if (allocator_root != pool.allocator_root_id || !append_totals(totals,
                                                                   pool.slot_count,
                                                                   pool.publisher_loan_capacity,
                                                                   pool.control_capacity,
                                                                   pool.readiness_capacity,
                                                                   pool.release_capacity,
                                                                   pool.retirement_capacity,
                                                                   pool.quarantine_capacity)) {
      return make_unexpected(tensor_resource_error(
          "interface-pool resource totals overflow", "resource", resource.allocator_root_id));
    }
  }
  return {};
}

[[nodiscard]] bool resource_totals_match(const TensorDeviceResourcePlanRecord& resource,
                                         const ResourceTotals& totals,
                                         std::string_view allocator_root) noexcept {
  const bool root_matches = resource.pool_count == 0U
                                ? allocator_root.empty()
                                : allocator_root == resource.allocator_root_id;
  return root_matches && totals.pools == resource.pool_count &&
         totals.slots == resource.slot_capacity && totals.loans == resource.loan_credit_capacity &&
         totals.controls == resource.control_capacity &&
         totals.readiness == resource.readiness_capacity &&
         totals.releases == resource.release_capacity &&
         totals.retirements == resource.retirement_capacity &&
         totals.quarantines == resource.quarantine_capacity;
}

[[nodiscard]] expected<void, Error> materialize_device_resources(
    const ExecutionPlanState& plan,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    ProcessGroupId process_group, std::vector<DeviceResource>& devices) {
  devices.reserve(plan.tensor_device_resources.size());
  for (std::size_t index = 0U; index < plan.tensor_device_resources.size(); ++index) {
    const auto& resource = plan.tensor_device_resources[index];
    if (!valid_device_resource_contract(plan, index)) {
      return make_unexpected(
          tensor_resource_error("device resource contract mismatch", "resource_index", index));
    }
    ResourceTotals totals;
    std::string_view allocator_root;
    auto direct = append_direct_pool_totals(plan, resource, totals, allocator_root);
    if (!direct) {
      return make_unexpected(std::move(direct.error()));
    }
    auto fields = append_field_pool_totals(plan, resource, totals, allocator_root);
    if (!fields) {
      return make_unexpected(std::move(fields.error()));
    }
    auto interface = append_interface_pool_totals(plan, resource, totals, allocator_root);
    if (!interface) {
      return make_unexpected(std::move(interface.error()));
    }
    if (!resource_totals_match(resource, totals, allocator_root)) {
      return make_unexpected(tensor_resource_error(
          "device resource totals mismatch", "resource", resource.allocator_root_id));
    }
    if (resource.process_group != process_group) {
      continue;
    }

    const ScopedCudaDevice selected{resource.device};
    cudaStream_t stream = nullptr;
    if (!selected.active() ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      return make_unexpected(tensor_resource_error(
          "device stream materialization failed", "resource", resource.allocator_root_id));
    }
    devices.emplace_back(&resource, stream, test_control, 0U);
  }
  return {};
}

[[nodiscard]] cudaStream_t resource_stream(std::span<const DeviceResource> devices,
                                           ProcessGroupId group, DeviceId device) noexcept {
  const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& row) {
    return row.plan->process_group == group && row.plan->device == device;
  });
  return found != devices.end() ? found->stream : nullptr;
}

[[nodiscard]] ProcessGroupId vertex_group(const ExecutionPlanState& plan,
                                          VertexId vertex) noexcept {
  const auto found = std::find_if(plan.vertex_process_placements.begin(),
                                  plan.vertex_process_placements.end(),
                                  [&](const auto& row) { return row.vertex == vertex; });
  return found != plan.vertex_process_placements.end() ? found->process_group : ProcessGroupId{};
}

[[nodiscard]] VertexId port_owner(const ExecutionPlanState& plan, PortId id) noexcept {
  const auto found = std::find_if(
      plan.ports.begin(), plan.ports.end(), [&](const auto& row) { return row.id == id; });
  return found != plan.ports.end() ? found->owner : VertexId{};
}

[[nodiscard]] bool can_elide_native_readiness(const ExecutionPlanState& plan,
                                              std::span<const DeviceResource> devices,
                                              const TensorAllocationPlanRecord& allocation,
                                              cudaStream_t producer_stream) noexcept {
  if (producer_stream == nullptr) {
    return false;
  }
  const auto resource =
      std::find_if(plan.tensor_device_resources.begin(),
                   plan.tensor_device_resources.end(),
                   [&](const auto& row) {
                     return row.process_group == allocation.allocator_process_group &&
                            row.device == allocation.device;
                   });
  if (resource == plan.tensor_device_resources.end() || resource->fused_linear_edges.empty()) {
    return false;
  }
  bool found_edge = false;
  for (const auto& edge : plan.payload_edges) {
    if (edge.source != allocation.output || edge.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    found_edge = true;
    if (!std::binary_search(resource->fused_linear_edges.begin(),
                            resource->fused_linear_edges.end(),
                            edge.edge,
                            [](EdgeId left, EdgeId right) { return left.value < right.value; })) {
      return false;
    }
    if (edge.locality != ChannelLocalityKind::kProcessLocal || !edge.target ||
        edge.target.value() > plan.ports.size()) {
      return false;
    }
    const auto& target = plan.ports[static_cast<std::size_t>(edge.target.value() - 1U)];
    if (target.id != edge.target || target.owner_kind != PortOwnerKind::kOperator) {
      return false;
    }
    const ProcessGroupId target_group = vertex_group(plan, target.owner);
    if (!target_group ||
        resource_stream(devices, target_group, allocation.device) != producer_stream) {
      return false;
    }
  }
  return found_edge;
}

[[nodiscard]] bool interface_pool_belongs_to_vertex(const ExecutionPlanState& plan,
                                                    const TensorInterfacePoolPlanRecord& pool,
                                                    VertexId vertex) noexcept {
  PortId resolved = pool.graph_port;
  for (std::size_t depth = 0U; depth <= plan.interface_port_bindings.size(); ++depth) {
    if (port_owner(plan, resolved) == vertex) {
      return true;
    }
    const auto alias = std::find_if(plan.interface_port_bindings.begin(),
                                    plan.interface_port_bindings.end(),
                                    [&](const auto& row) { return row.public_port == resolved; });
    if (alias == plan.interface_port_bindings.end()) {
      return false;
    }
    resolved = alias->internal_port;
  }
  return false;
}

// Resolve every alias, allocation, and schema field before accepting one inline CUDA device.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<std::optional<DeviceId>, Error> inline_descriptor_subscriber_device(
    const ExecutionPlanState& plan, const ProcessLocalChannelSubscriberPlanRecord& subscriber,
    const ProcessLocalChannelServicePlanRecord& service, ProcessGroupId process_group) {
  const bool descriptor_service =
      service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kSchema &&
      service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (!plan.inline_partition_execution || !descriptor_service) {
    return std::optional<DeviceId>{};
  }
  const auto edge =
      std::find_if(plan.payload_edges.begin(), plan.payload_edges.end(), [&](const auto& row) {
        return row.edge == subscriber.edge && row.source == subscriber.source &&
               row.target == subscriber.target;
      });
  if (edge == plan.payload_edges.end() ||
      !std::binary_search(plan.inline_partition_execution->edges.begin(),
                          plan.inline_partition_execution->edges.end(),
                          edge->edge,
                          [](EdgeId left, EdgeId right) { return left.value < right.value; })) {
    return std::optional<DeviceId>{};
  }

  const TensorAllocationPlanRecord* allocation = nullptr;
  for (const auto& candidate : plan.tensor_allocations) {
    if (candidate.output != subscriber.source) {
      continue;
    }
    if (allocation != nullptr) {
      return make_unexpected(
          tensor_resource_error("inline descriptor subscriber has duplicate allocations",
                                "output",
                                subscriber.source.value()));
    }
    allocation = &candidate;
  }
  std::vector<const TensorFieldAllocationPlanRecord*> field_allocations;
  for (const auto& candidate : plan.tensor_field_allocations) {
    if (candidate.output == subscriber.source) {
      field_allocations.push_back(&candidate);
    }
  }
  std::sort(field_allocations.begin(),
            field_allocations.end(),
            [](const auto* left, const auto* right) { return left->field_id < right->field_id; });
  if ((allocation == nullptr && field_allocations.empty()) ||
      (allocation != nullptr && !field_allocations.empty())) {
    return make_unexpected(tensor_resource_error(
        "inline descriptor subscriber allocation mismatch", "output", subscriber.source.value()));
  }
  DeviceId device;
  if (allocation != nullptr) {
    if (!validate_tensor_allocation_record(*allocation, plan.plan_generation) ||
        allocation->allocator_process_group != process_group ||
        (allocation->memory_kind != MemoryKind::kHost &&
         allocation->memory_kind != MemoryKind::kCudaDevice)) {
      return make_unexpected(tensor_resource_error(
          "inline descriptor subscriber allocation mismatch", "output", subscriber.source.value()));
    }
    device = allocation->device;
  } else {
    if (service.tensor_fields.size() != field_allocations.size()) {
      return make_unexpected(
          tensor_resource_error("inline descriptor subscriber field inventory mismatch",
                                "output",
                                subscriber.source.value()));
    }
    for (std::size_t index = 0U; index < field_allocations.size(); ++index) {
      const auto& field = *field_allocations[index];
      const auto& storage = service.tensor_fields[index];
      const bool layout_matches =
          field.layout && storage.rank == field.layout->rank &&
          storage.dtype.has_value() == field.layout->dtype.has_value() &&
          (!storage.dtype || dtype_equal(*storage.dtype, *field.layout->dtype));
      if (!validate_tensor_field_allocation_record(field, plan.plan_generation) ||
          field.output != subscriber.source || field.allocator_process_group != process_group ||
          (field.memory_kind != MemoryKind::kHost &&
           field.memory_kind != MemoryKind::kCudaDevice) ||
          field.tensor_index != index ||
          (index != 0U && field_allocations[index - 1U]->field_id >= field.field_id) ||
          storage.field_id != field.field_id || storage.tensor_index != field.tensor_index ||
          storage.max_byte_span != field.bounds.max_byte_span ||
          storage.memory_kind != field.memory_kind || storage.device != field.device ||
          !layout_matches || (field.device.valid() && device.valid() && field.device != device)) {
        return make_unexpected(
            tensor_resource_error("inline descriptor subscriber field allocation mismatch",
                                  "output",
                                  subscriber.source.value()));
      }
      if (field.device.valid()) {
        device = field.device;
      }
    }
  }
  if (!device.valid()) {
    return std::optional<DeviceId>{device};
  }
  const auto resource =
      std::find_if(plan.tensor_device_resources.begin(),
                   plan.tensor_device_resources.end(),
                   [&](const auto& row) {
                     return row.process_group == process_group && row.device == device &&
                            std::binary_search(
                                row.fused_linear_edges.begin(),
                                row.fused_linear_edges.end(),
                                edge->edge,
                                [](EdgeId left, EdgeId right) { return left.value < right.value; });
                   });
  if (resource == plan.tensor_device_resources.end()) {
    return make_unexpected(
        tensor_resource_error("inline descriptor subscriber device resource mismatch",
                              "output",
                              subscriber.source.value()));
  }
  return std::optional<DeviceId>{device};
}

// Select a stream by joining the plan's exact process, device, and interface obligations.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<OperatorStreamSelection, Error> select_operator_stream(
    const ExecutionPlanState& plan, std::span<const DeviceResource> devices,
    const RuntimeOperatorInstance& instance) {
  const ProcessGroupId group = vertex_group(plan, instance.vertex);
  if (!group) {
    return make_unexpected(tensor_resource_error(
        "operator process placement missing", "operator", instance.operator_id));
  }
  cudaStream_t selected_stream = nullptr;
  DeviceId selected_device;
  const auto admit_device = [&](DeviceId device) noexcept -> bool {
    if (!device.valid()) {
      return true;
    }
    cudaStream_t candidate = resource_stream(devices, group, device);
    if (candidate == nullptr || (selected_stream != nullptr && selected_stream != candidate)) {
      return false;
    }
    selected_stream = candidate;
    selected_device = device;
    return true;
  };
  const auto admit_service_devices = [&](const auto& service) noexcept -> bool {
    if (service.tensor_fields.empty()) {
      return admit_device(service.tensor_device);
    }
    return std::all_of(service.tensor_fields.begin(),
                       service.tensor_fields.end(),
                       [&](const auto& field) { return admit_device(field.device); });
  };

  for (const auto& allocation : plan.tensor_allocations) {
    if (port_owner(plan, allocation.output) == instance.vertex &&
        !admit_device(allocation.device)) {
      return make_unexpected(tensor_resource_error(
          "output device resource mismatch", "output", allocation.output.value()));
    }
  }
  for (const auto& allocation : plan.tensor_field_allocations) {
    if (port_owner(plan, allocation.output) == instance.vertex &&
        !admit_device(allocation.device)) {
      return make_unexpected(tensor_resource_error(
          "field output device resource mismatch", "output", allocation.output.value()));
    }
  }
  for (const auto& subscriber : plan.process_local_channel_subscribers) {
    if (port_owner(plan, subscriber.target) != instance.vertex) {
      continue;
    }
    const auto service = std::find_if(plan.process_local_channel_services.begin(),
                                      plan.process_local_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == subscriber.source &&
                                               row.service_id == subscriber.service_id;
                                      });
    if (service == plan.process_local_channel_services.end()) {
      return make_unexpected(tensor_resource_error(
          "subscriber device resource mismatch", "output", subscriber.source.value()));
    }
    auto inline_device = inline_descriptor_subscriber_device(plan, subscriber, *service, group);
    if (!inline_device) {
      return make_unexpected(std::move(inline_device).error());
    }
    if (service->tensor_payload &&
        (*inline_device ? !admit_device(**inline_device) : !admit_service_devices(*service))) {
      return make_unexpected(tensor_resource_error(
          "subscriber device resource mismatch", "output", subscriber.source.value()));
    }
  }
  for (const auto& subscriber : plan.cross_process_channel_subscribers) {
    if (port_owner(plan, subscriber.target) != instance.vertex) {
      continue;
    }
    const auto service = std::find_if(plan.cross_process_channel_services.begin(),
                                      plan.cross_process_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == subscriber.source &&
                                               row.service_id == subscriber.service_id;
                                      });
    if (service == plan.cross_process_channel_services.end() ||
        (service->tensor_payload && !admit_service_devices(*service))) {
      return make_unexpected(
          tensor_resource_error("cross-process subscriber device resource mismatch",
                                "output",
                                subscriber.source.value()));
    }
  }
  for (const auto& service : plan.cross_process_channel_services) {
    if (descriptor_tensor_schema_service(service) &&
        port_owner(plan, service.source) == instance.vertex && !admit_service_devices(service)) {
      return make_unexpected(
          tensor_resource_error("descriptor-schema publisher device resource mismatch",
                                "output",
                                service.source.value()));
    }
  }
  for (const auto& pool : plan.tensor_interface_pools) {
    if (pool.imported && pool.device.valid() &&
        interface_pool_belongs_to_vertex(plan, pool, instance.vertex) &&
        !admit_device(pool.device)) {
      return make_unexpected(
          tensor_resource_error("interface device resource mismatch", "pool", pool.pool_id));
    }
  }
  return OperatorStreamSelection{.stream = selected_stream, .device = selected_device};
}

[[nodiscard]] expected<void, Error> bind_operator_streams(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    std::span<const DeviceResource> devices) {
  if (devices.empty()) {
    return {};
  }
  for (auto& instance : operators) {
    auto stream = select_operator_stream(plan, devices, instance);
    if (!stream) {
      return make_unexpected(std::move(stream.error()));
    }
    instance.compute_stream = stream->stream;
    instance.compute_device = stream->device;
  }
  return {};
}

struct TensorBackedOutputDeclaration {
  void* declaration{};
  const RuntimePortTypeOps* type_ops{};
};

[[nodiscard]] TensorBackedOutputDeclaration find_output_declaration(
    RuntimeOperatorInstance& instance, std::string_view local_name) noexcept {
  if (instance.replay_spec == nullptr) {
    return {};
  }
  for (std::size_t index = 0U;
       index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
       ++index) {
    const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
    if (replay.direction == PortDirection::kOutput && replay.name == local_name &&
        replay.declaration != nullptr && replay.runtime_type_ops != nullptr &&
        replay.runtime_type_ops->publish_tensor_loan != nullptr &&
        replay.runtime_type_ops->bind_tensor_output != nullptr &&
        replay.runtime_type_ops->unbind_tensor_output != nullptr) {
      return TensorBackedOutputDeclaration{
          .declaration = const_cast<void*>(replay.declaration),
          .type_ops = replay.runtime_type_ops,
      };
    }
  }
  return {};
}

[[nodiscard]] expected<cudaStream_t, Error> output_stream(
    std::span<const DeviceResource> devices, const TensorAllocationPlanRecord& allocation) {
  if (!allocation.device.valid()) {
    return nullptr;
  }
  const auto resource = std::find_if(devices.begin(), devices.end(), [&](const auto& candidate) {
    return candidate.plan->process_group == allocation.allocator_process_group &&
           candidate.plan->device == allocation.device &&
           candidate.plan->allocator_root_id == allocation.allocator_root_id;
  });
  if (resource == devices.end()) {
    return make_unexpected(tensor_resource_error(
        "output device resource missing", "output", allocation.output.value()));
  }
  return resource->stream;
}

[[nodiscard]] TensorAllocationPlanRecord tensor_field_output_allocation(
    const TensorFieldAllocationPlanRecord& field) {
  return TensorAllocationPlanRecord{
      .output = field.output,
      .plan_generation = field.plan_generation,
      .run_generation = field.run_generation,
      .pool_id = field.pool_id,
      .allocator_root_id = field.allocator_root_id,
      .allocator_process_group = field.allocator_process_group,
      .bounds = field.bounds,
      .layout = field.layout,
      .memory_kind = field.memory_kind,
      .device = field.device,
      .slot_count = field.slot_count,
      .aggregate_byte_capacity = field.aggregate_byte_capacity,
      .max_outstanding_loans = field.max_outstanding_loans,
      .max_emits_per_compute = field.max_emits_per_compute,
      .control_capacity = field.control_capacity,
      .readiness_capacity = field.readiness_capacity,
      .release_capacity = field.release_capacity,
      .retirement_capacity = field.retirement_capacity,
      .quarantine_capacity = field.quarantine_capacity,
      .materialize_before = field.materialize_before,
      .effective_profile = field.effective_profile,
      .disposition = field.disposition,
      .fallback_rejected = field.fallback_rejected,
  };
}

// One Operator owns this output binding, and its single active compute callback is the only
// publication authority. The count-sized request/Tensor/slot scratch below is therefore safe to
// reuse without additional same-output synchronization under the current Holoscan compute model.
class TensorSchemaOutputPool final {
 public:
  TensorSchemaOutputPool(const TensorSchemaOutputPool&) = delete;
  TensorSchemaOutputPool& operator=(const TensorSchemaOutputPool&) = delete;

  ~TensorSchemaOutputPool() {
    detach();
    close();
  }

  [[nodiscard]] static expected<std::unique_ptr<TensorSchemaOutputPool>, Error> create(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance, void* declaration,
      const RuntimePortTypeOps* type_ops, std::span<const DeviceResource> devices,
      const std::span<const TensorFieldAllocationPlanRecord> allocations,
      const RunGenerationId run_generation,
      const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
      const std::size_t test_pool_offset) {
    if (declaration == nullptr || type_ops == nullptr ||
        type_ops->descriptor_tensor_static_count != allocations.size() ||
        type_ops->publish_tensor_loans == nullptr || type_ops->bind_tensor_output == nullptr ||
        type_ops->unbind_tensor_output == nullptr || allocations.empty() ||
        allocations.size() > kMaximumTensorSchemaLoanFields) {
      return make_unexpected(tensor_resource_error("invalid Tensor schema output-pool contract"));
    }
    auto result = std::unique_ptr<TensorSchemaOutputPool>{new TensorSchemaOutputPool{
        instance.operator_id, allocations.front().output, declaration, type_ops}};
    result->fields_.reserve(allocations.size());
    result->field_ids_.resize(allocations.size());
    result->field_dtypes_.resize(allocations.size());
    result->canonical_requests_.resize(allocations.size());
    result->tensor_scratch_.resize(allocations.size());
    result->slot_scratch_.resize(allocations.size());
    for (std::size_t index = 0U; index < allocations.size(); ++index) {
      const auto& field = allocations[index];
      if (field.output != result->output_ || field.tensor_index != index ||
          (index != 0U && field.field_id <= allocations[index - 1U].field_id)) {
        return make_unexpected(
            tensor_resource_error("noncanonical Tensor schema field allocation"));
      }
      const TensorAllocationPlanRecord allocation = tensor_field_output_allocation(field);
      auto contract = validate_output_plan_contract(plan, allocation);
      if (!contract) {
        return make_unexpected(std::move(contract.error()));
      }
      auto stream = output_stream(devices, allocation);
      if (!stream) {
        return make_unexpected(std::move(stream.error()));
      }
      const bool elide_same_stream_readiness =
          can_elide_native_readiness(plan, devices, allocation, *stream);
      auto pool = OutputPool::create(plan,
                                     allocation,
                                     std::move(*contract),
                                     instance,
                                     declaration,
                                     type_ops,
                                     *stream,
                                     elide_same_stream_readiness,
                                     run_generation,
                                     test_control,
                                     test_pool_offset + index,
                                     false,
                                     1U,
                                     static_cast<std::uint32_t>(index));
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      result->field_ids_[index] = field.field_id;
      result->field_dtypes_[index] = (*pool)->exact_bundle_dtype();
      result->fields_.push_back(std::move(*pool));
    }
    result->bind();
    return result;
  }

  [[nodiscard]] OperatorInstanceId owner() const noexcept { return owner_; }
  [[nodiscard]] PortId output() const noexcept { return output_; }

  [[nodiscard]] expected<std::unique_ptr<TensorSchemaOutputPool>, Error> recreate_unbound(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance) const {
    auto result = std::unique_ptr<TensorSchemaOutputPool>{
        new TensorSchemaOutputPool{owner_, output_, declaration_, type_ops_}};
    result->fields_.reserve(fields_.size());
    result->field_ids_.resize(fields_.size());
    result->field_dtypes_ = field_dtypes_;
    result->canonical_requests_.resize(fields_.size());
    result->tensor_scratch_.resize(fields_.size());
    result->slot_scratch_.resize(fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto field = fields_[index]->recreate_unbound(plan, instance);
      if (!field) {
        return make_unexpected(std::move(field.error()));
      }
      result->field_ids_[index] = field_ids_[index];
      result->fields_.push_back(std::move(*field));
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> begin_restart() noexcept {
    std::size_t prepared = 0U;
    for (; prepared < fields_.size(); ++prepared) {
      auto fenced = fields_[prepared]->begin_restart();
      if (!fenced) {
        for (std::size_t index = 0U; index < prepared; ++index) {
          static_cast<void>(fields_[index]->reopen(false));
        }
        return make_unexpected(std::move(fenced.error()));
      }
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> reopen() noexcept {
    for (auto& field : fields_) {
      auto reopened = field->reopen(false);
      if (!reopened) {
        return make_unexpected(std::move(reopened.error()));
      }
    }
    bind();
    return {};
  }

  [[nodiscard]] bool restart_publication_ready() noexcept {
    return std::ranges::all_of(
        fields_, [](const auto& field) { return field->restart_publication_ready(); });
  }

  void publish_reopen() noexcept {
    for (auto& field : fields_) {
      field->publish_reopen(false);
    }
    bind();
  }

  void close() noexcept {
    for (auto& field : fields_) {
      field->close();
    }
  }

  void detach() noexcept {
    if (bound_) {
      type_ops_->unbind_tensor_output(declaration_);
      bound_ = false;
    }
  }

  void bind_output() noexcept { bind(); }

 private:
  TensorSchemaOutputPool(OperatorInstanceId owner, PortId output, void* declaration,
                         const RuntimePortTypeOps* type_ops) noexcept
      : owner_(owner), output_(output), declaration_(declaration), type_ops_(type_ops) {}

  void bind() noexcept {
    if (!bound_) {
      type_ops_->bind_tensor_output(declaration_,
                                    TensorOutputBinding{
                                        .state = this,
                                        .ops = &kBindingOps,
                                        .tensor_field_dtypes = field_dtypes_,
                                    });
      bound_ = true;
    }
  }

  [[nodiscard]] expected<TensorSchemaLoanBundleStorage, Error> allocate_tensors(
      const TensorSchemaLoanRequestList& requests) noexcept {
    if (requests.field_count != fields_.size()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_COUNT_MISMATCH"});
    }
    auto& canonical = canonical_requests_;
    std::fill(canonical.begin(), canonical.end(), nullptr);
    for (std::size_t index = 0U; index < requests.field_count; ++index) {
      const auto& request = requests.fields[index];
      if (request.tensor_index >= fields_.size() ||
          field_ids_[request.tensor_index] != request.field_id) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_SELECTOR_INVALID"});
      }
      if (canonical[request.tensor_index] != nullptr) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_DUPLICATE"});
      }
      canonical[request.tensor_index] = &request;
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      if (canonical[index] == nullptr ||
          !fields_[index]->accepts_bundle_request(canonical[index]->tensor)) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_REQUEST_INVALID"});
      }
    }

    TensorSchemaLoanBundleStorage result;
    result.field_count = static_cast<std::uint32_t>(fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto loan = fields_[index]->allocate_bundle_field(canonical[index]->tensor);
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }
      result.field_ids[index] = field_ids_[index];
      result.fields[index].emplace(std::move(*loan));
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> emit_tensors(TensorSchemaLoanBundleStorage bundle,
                                                   const void* descriptor,
                                                   EmitOptions options) noexcept {
    if (descriptor == nullptr || bundle.field_count != fields_.size()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_PUBLICATION_INVALID"});
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      if (bundle.field_ids[index] != field_ids_[index] || !bundle.fields[index] ||
          !fields_[index]->preflight_bundle_field(*bundle.fields[index])) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_PREFLIGHT_FAILED"});
      }
    }

    auto& tensors = tensor_scratch_;
    auto& slots = slot_scratch_;
    std::fill(slots.begin(), slots.end(), nullptr);
    std::size_t prepared_count = 0U;
    for (; prepared_count < fields_.size(); ++prepared_count) {
      auto prepared = fields_[prepared_count]->prepare_bundle_field(*bundle.fields[prepared_count]);
      if (!prepared) {
        const expected<void, Error> failure = make_unexpected(
            Error{.code = prepared.error().code, .message = prepared.error().message});
        for (std::size_t index = 0U; index < prepared_count; ++index) {
          tensors[index] = Tensor{};
          static_cast<void>(fields_[index]->finish_bundle_field(slots[index], failure));
        }
        return make_unexpected(std::move(prepared.error()));
      }
      slots[prepared_count] = prepared->slot;
      tensors[prepared_count] = std::move(prepared->tensor);
    }

    const std::span<const Tensor> authoritative{tensors.data(), fields_.size()};
    const expected<void, Error> published =
        type_ops_->publish_tensor_loans(declaration_, descriptor, authoritative, options);
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      tensors[index] = Tensor{};
    }
    expected<void, Error> finished = published;
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto field_finished = fields_[index]->finish_bundle_field(slots[index], published);
      if (!field_finished && finished) {
        finished = make_unexpected(std::move(field_finished.error()));
      }
    }
    return finished;
  }

  static expected<TensorSchemaLoanBundleStorage, Error> allocate_tensors_callback(
      void* state, const TensorSchemaLoanRequestList& requests) noexcept {
    return static_cast<TensorSchemaOutputPool*>(state)->allocate_tensors(requests);
  }

  static expected<void, Error> emit_tensors_callback(void* state,
                                                     TensorSchemaLoanBundleStorage bundle,
                                                     const void* descriptor,
                                                     EmitOptions options) noexcept {
    return static_cast<TensorSchemaOutputPool*>(state)->emit_tensors(
        std::move(bundle), descriptor, options);
  }

  inline static const TensorOutputBindingOps kBindingOps{
      .allocate = nullptr,
      .emit_loan = nullptr,
      .allocate_tensors = &allocate_tensors_callback,
      .emit_tensors = &emit_tensors_callback,
  };

  OperatorInstanceId owner_;
  PortId output_;
  void* declaration_{};
  const RuntimePortTypeOps* type_ops_{};
  std::vector<std::unique_ptr<OutputPool>> fields_;
  std::vector<std::uint16_t> field_ids_;
  std::vector<std::optional<DLDataType>> field_dtypes_;
  std::vector<const TensorSchemaLoanRequest*> canonical_requests_;
  std::vector<Tensor> tensor_scratch_;
  std::vector<OutputSlot*> slot_scratch_;
  bool bound_{};
};

[[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> materialize_output_pool(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    std::span<const DeviceResource> devices, const TensorAllocationPlanRecord& allocation,
    RunGenerationId run_generation, ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control, std::size_t pool_index) {
  if (std::count_if(
          plan.tensor_allocations.begin(), plan.tensor_allocations.end(), [&](const auto& row) {
            return row.output == allocation.output || row.pool_id == allocation.pool_id;
          }) != 1) {
    return make_unexpected(tensor_resource_error(
        "duplicate output-pool identity", "output", allocation.output.value()));
  }
  auto contract = validate_output_plan_contract(plan, allocation);
  if (!contract) {
    return make_unexpected(std::move(contract.error()));
  }
  const auto port = std::find_if(plan.ports.begin(), plan.ports.end(), [&](const auto& row) {
    return row.id == allocation.output && row.owner_kind == PortOwnerKind::kOperator &&
           row.direction == PortDirection::kOutput;
  });
  if (port == plan.ports.end()) {
    return make_unexpected(tensor_resource_error(
        "output port inventory mismatch", "output", allocation.output.value()));
  }
  if (!worker_owns_vertex(plan, process_group, port->owner)) {
    return std::unique_ptr<OutputPool>{};
  }
  const auto instance =
      std::find_if(operators.begin(), operators.end(), [&](const auto& candidate) {
        return candidate.vertex == port->owner;
      });
  if (instance == operators.end() || instance->replay_spec == nullptr) {
    return make_unexpected(tensor_resource_error(
        "output operator inventory mismatch", "output", allocation.output.value()));
  }
  const auto declaration = find_output_declaration(*instance, port->local_name);
  if (declaration.declaration == nullptr || declaration.type_ops == nullptr) {
    return make_unexpected(
        tensor_resource_error("output declaration mismatch", "output", allocation.output.value()));
  }
  auto stream = output_stream(devices, allocation);
  if (!stream) {
    return make_unexpected(std::move(stream.error()));
  }
  const bool elide_same_stream_readiness =
      can_elide_native_readiness(plan, devices, allocation, *stream);
  return OutputPool::create(plan,
                            allocation,
                            std::move(*contract),
                            *instance,
                            declaration.declaration,
                            declaration.type_ops,
                            *stream,
                            elide_same_stream_readiness,
                            run_generation,
                            test_control,
                            pool_index);
}

[[nodiscard]] expected<void, Error> materialize_output_pools(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    std::span<const DeviceResource> devices, RunGenerationId run_generation,
    ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::vector<std::unique_ptr<OutputPool>>& outputs) {
  outputs.reserve(plan.tensor_allocations.size());
  for (const auto& allocation : plan.tensor_allocations) {
    auto pool = materialize_output_pool(plan,
                                        operators,
                                        devices,
                                        allocation,
                                        run_generation,
                                        process_group,
                                        test_control,
                                        outputs.size());
    if (!pool) {
      return make_unexpected(std::move(pool.error()));
    }
    if (*pool != nullptr) {
      outputs.push_back(std::move(*pool));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> materialize_tensor_schema_output_pools(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    std::span<const DeviceResource> devices, RunGenerationId run_generation,
    ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::vector<std::unique_ptr<TensorSchemaOutputPool>>& outputs) {
  std::size_t begin = 0U;
  while (begin < plan.tensor_field_allocations.size()) {
    std::size_t end = begin + 1U;
    while (end < plan.tensor_field_allocations.size() &&
           plan.tensor_field_allocations[end].output ==
               plan.tensor_field_allocations[begin].output) {
      ++end;
    }
    const auto allocations =
        std::span<const TensorFieldAllocationPlanRecord>{plan.tensor_field_allocations}.subspan(
            begin, end - begin);
    const PortId output = allocations.front().output;
    const auto port = std::find_if(plan.ports.begin(), plan.ports.end(), [&](const auto& row) {
      return row.id == output && row.owner_kind == PortOwnerKind::kOperator &&
             row.direction == PortDirection::kOutput;
    });
    if (port == plan.ports.end()) {
      return make_unexpected(
          tensor_resource_error("field output port inventory mismatch", "output", output.value()));
    }
    if (worker_owns_vertex(plan, process_group, port->owner)) {
      const auto instance =
          std::find_if(operators.begin(), operators.end(), [&](const auto& candidate) {
            return candidate.vertex == port->owner;
          });
      if (instance == operators.end() || instance->replay_spec == nullptr) {
        return make_unexpected(tensor_resource_error(
            "field output operator inventory mismatch", "output", output.value()));
      }
      const auto declaration = find_output_declaration(*instance, port->local_name);
      if (declaration.declaration == nullptr || declaration.type_ops == nullptr) {
        return make_unexpected(
            tensor_resource_error("field output declaration mismatch", "output", output.value()));
      }
      auto pool = TensorSchemaOutputPool::create(plan,
                                                 *instance,
                                                 declaration.declaration,
                                                 declaration.type_ops,
                                                 devices,
                                                 allocations,
                                                 run_generation,
                                                 test_control,
                                                 plan.tensor_allocations.size() + begin);
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      outputs.push_back(std::move(*pool));
    }
    begin = end;
  }
  return {};
}

}  // namespace

class RuntimeTensorResources::Impl final {
 public:
  struct RestartOutput {
    std::size_t output_index{};
    std::unique_ptr<OutputPool> candidate;
  };
  struct RestartTensorSchemaOutput {
    std::size_t output_index{};
    std::unique_ptr<TensorSchemaOutputPool> candidate;
  };

  const ExecutionPlanState* plan{};
  std::span<RuntimeOperatorInstance> operators;
  std::vector<DeviceResource> devices;
  std::vector<std::unique_ptr<OutputPool>> outputs;
  std::vector<std::unique_ptr<TensorSchemaOutputPool>> tensor_schema_outputs;
  std::vector<std::unique_ptr<OutputPool>> retired_outputs;
  std::vector<std::unique_ptr<TensorSchemaOutputPool>> retired_tensor_schema_outputs;
  std::vector<RestartOutput> restart_outputs;
  std::vector<RestartTensorSchemaOutput> restart_tensor_schema_outputs;
  bool restart_active{};
};

RuntimeTensorResources::RuntimeTensorResources(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeTensorResources::~RuntimeTensorResources() {
  close();
  detach_ports();
}

expected<std::unique_ptr<RuntimeTensorResources>, Error> RuntimeTensorResources::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, ProcessGroupId process_group) {
  if (plan.plan_generation == 0U || run_generation == 0U ||
      (!process_group && !plan.graph.vertices().empty()) ||
      !std::is_sorted(plan.tensor_device_resources.begin(),
                      plan.tensor_device_resources.end(),
                      canonical_resource_order)) {
    return make_unexpected(tensor_resource_error(
        "tensor resource inventory invalid", "plan_generation", plan.plan_generation));
  }

  auto impl = std::make_unique<Impl>();
  impl->plan = &plan;
  impl->operators = operators;
  const auto test_control = plan.executable_factories.test_tensor_output_control;
  if (test_control != nullptr) {
    test_control->prepare(plan.tensor_allocations.size() + plan.tensor_field_allocations.size());
  }
  auto device_resources =
      materialize_device_resources(plan, test_control, process_group, impl->devices);
  if (!device_resources) {
    return make_unexpected(std::move(device_resources.error()));
  }
  auto operator_streams = bind_operator_streams(plan, operators, impl->devices);
  if (!operator_streams) {
    return make_unexpected(std::move(operator_streams.error()));
  }

  impl->restart_outputs.reserve(plan.tensor_allocations.size());
  impl->restart_tensor_schema_outputs.reserve(plan.tensor_field_allocations.size());
  if (plan.lifecycle_transaction_capacity) {
    const std::size_t receipt_capacity =
        plan.lifecycle_transaction_capacity->configured.max_receipts;
    if (receipt_capacity != 0U &&
        (plan.tensor_allocations.size() >
             std::numeric_limits<std::size_t>::max() / receipt_capacity ||
         plan.tensor_field_allocations.size() >
             std::numeric_limits<std::size_t>::max() / receipt_capacity)) {
      return make_unexpected(tensor_resource_error("restart retirement capacity overflow"));
    }
    impl->retired_outputs.reserve(plan.tensor_allocations.size() * receipt_capacity);
    impl->retired_tensor_schema_outputs.reserve(plan.tensor_field_allocations.size() *
                                                receipt_capacity);
  }
  auto output_pools = materialize_output_pools(
      plan, operators, impl->devices, run_generation, process_group, test_control, impl->outputs);
  if (!output_pools) {
    return make_unexpected(std::move(output_pools.error()));
  }
  auto tensor_schema_output_pools =
      materialize_tensor_schema_output_pools(plan,
                                             operators,
                                             impl->devices,
                                             run_generation,
                                             process_group,
                                             test_control,
                                             impl->tensor_schema_outputs);
  if (!tensor_schema_output_pools) {
    return make_unexpected(std::move(tensor_schema_output_pools.error()));
  }

  return std::unique_ptr<RuntimeTensorResources>{new RuntimeTensorResources{std::move(impl)}};
}

void RuntimeTensorResources::close() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  for (auto& output : impl_->outputs) {
    output->close();
  }
  for (auto& output : impl_->tensor_schema_outputs) {
    output->close();
  }
  for (auto& output : impl_->retired_outputs) {
    output->close();
  }
  for (auto& output : impl_->retired_tensor_schema_outputs) {
    output->close();
  }
}

void RuntimeTensorResources::detach_ports() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  for (auto& output : impl_->outputs) {
    output->detach();
  }
  for (auto& output : impl_->tensor_schema_outputs) {
    output->detach();
  }
  for (auto& output : impl_->retired_outputs) {
    output->detach();
  }
  for (auto& output : impl_->retired_tensor_schema_outputs) {
    output->detach();
  }
}

// Restart fencing is one transaction across active, retired, and schema Tensor resources.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeTensorResources::begin_restart(
    std::span<const OperatorInstanceId> members, LifecycleStage from_stage) noexcept {
  if (impl_ == nullptr || impl_->plan == nullptr || impl_->restart_active || members.empty() ||
      !is_lifecycle_forward_stage(from_stage)) {
    return make_unexpected(tensor_resource_error("restart request state invalid"));
  }
  impl_->restart_active = true;
  impl_->restart_outputs.clear();
  impl_->restart_tensor_schema_outputs.clear();
  const bool recreate =
      static_cast<std::uint8_t>(from_stage) <= static_cast<std::uint8_t>(LifecycleStage::kAllocate);
  std::size_t affected_count = 0U;
  for (const auto& output : impl_->outputs) {
    if (std::find(members.begin(), members.end(), output->owner()) != members.end()) {
      ++affected_count;
    }
  }
  std::size_t affected_tensor_schema_count = 0U;
  for (const auto& output : impl_->tensor_schema_outputs) {
    if (std::find(members.begin(), members.end(), output->owner()) != members.end()) {
      ++affected_tensor_schema_count;
    }
  }
  if (recreate &&
      (affected_count > impl_->retired_outputs.capacity() - impl_->retired_outputs.size() ||
       affected_tensor_schema_count > impl_->retired_tensor_schema_outputs.capacity() -
                                          impl_->retired_tensor_schema_outputs.size())) {
    impl_->restart_active = false;
    return make_unexpected(tensor_resource_error("restart retirement capacity exhausted"));
  }

  const auto rollback_prepared = [&]() noexcept {
    for (const auto& prepared : impl_->restart_outputs) {
      static_cast<void>(impl_->outputs[prepared.output_index]->reopen());
    }
    for (const auto& prepared : impl_->restart_tensor_schema_outputs) {
      static_cast<void>(impl_->tensor_schema_outputs[prepared.output_index]->reopen());
    }
    impl_->restart_outputs.clear();
    impl_->restart_tensor_schema_outputs.clear();
    impl_->restart_active = false;
  };
  for (std::size_t index = 0U; index < impl_->outputs.size(); ++index) {
    auto& output = impl_->outputs[index];
    if (std::find(members.begin(), members.end(), output->owner()) == members.end()) {
      continue;
    }
    auto fenced = output->begin_restart();
    if (!fenced) {
      rollback_prepared();
      return make_unexpected(std::move(fenced.error()));
    }
    std::unique_ptr<OutputPool> candidate;
    if (recreate) {
      const auto instance =
          std::find_if(impl_->operators.begin(), impl_->operators.end(), [&](const auto& row) {
            return row.operator_id == output->owner();
          });
      if (instance == impl_->operators.end()) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(tensor_resource_error(
            "restart operator inventory mismatch", "output", output->output().value()));
      }
      expected<std::unique_ptr<OutputPool>, Error> created = make_unexpected(tensor_resource_error(
          "restart output recreation not attempted", "output", output->output().value()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        created = output->recreate_unbound(*impl_->plan, *instance);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(tensor_resource_error(
            "restart output recreation threw", "output", output->output().value()));
      }
#endif
      if (!created) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(std::move(created.error()));
      }
      candidate = std::move(*created);
    }
    impl_->restart_outputs.push_back(
        Impl::RestartOutput{.output_index = index, .candidate = std::move(candidate)});
  }
  for (std::size_t index = 0U; index < impl_->tensor_schema_outputs.size(); ++index) {
    auto& output = impl_->tensor_schema_outputs[index];
    if (std::find(members.begin(), members.end(), output->owner()) == members.end()) {
      continue;
    }
    auto fenced = output->begin_restart();
    if (!fenced) {
      rollback_prepared();
      return make_unexpected(std::move(fenced.error()));
    }
    std::unique_ptr<TensorSchemaOutputPool> candidate;
    if (recreate) {
      const auto instance =
          std::find_if(impl_->operators.begin(), impl_->operators.end(), [&](const auto& row) {
            return row.operator_id == output->owner();
          });
      if (instance == impl_->operators.end()) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(tensor_resource_error(
            "restart operator inventory mismatch", "output", output->output().value()));
      }
      expected<std::unique_ptr<TensorSchemaOutputPool>, Error> created = make_unexpected(
          tensor_resource_error("restart Tensor schema output recreation not attempted",
                                "output",
                                output->output().value()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        created = output->recreate_unbound(*impl_->plan, *instance);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(tensor_resource_error(
            "restart Tensor schema output recreation threw", "output", output->output().value()));
      }
#endif
      if (!created) {
        static_cast<void>(output->reopen());
        rollback_prepared();
        return make_unexpected(std::move(created.error()));
      }
      candidate = std::move(*created);
    }
    impl_->restart_tensor_schema_outputs.push_back(
        Impl::RestartTensorSchemaOutput{.output_index = index, .candidate = std::move(candidate)});
  }
  return {};
}

expected<void, Error> RuntimeTensorResources::publish_restart() noexcept {
  auto preflight = preflight_restart_publication();
  if (!preflight) {
    return preflight;
  }
  for (auto& prepared : impl_->restart_outputs) {
    auto& predecessor = impl_->outputs[prepared.output_index];
    if (prepared.candidate != nullptr) {
      predecessor->detach();
      prepared.candidate->bind();
      impl_->retired_outputs.push_back(std::move(predecessor));
      predecessor = std::move(prepared.candidate);
    } else {
      predecessor->publish_reopen();
    }
  }
  for (auto& prepared : impl_->restart_tensor_schema_outputs) {
    auto& predecessor = impl_->tensor_schema_outputs[prepared.output_index];
    if (prepared.candidate != nullptr) {
      predecessor->detach();
      prepared.candidate->bind_output();
      impl_->retired_tensor_schema_outputs.push_back(std::move(predecessor));
      predecessor = std::move(prepared.candidate);
    } else {
      predecessor->publish_reopen();
    }
  }
  impl_->restart_outputs.clear();
  impl_->restart_tensor_schema_outputs.clear();
  impl_->restart_active = false;
  return {};
}

expected<void, Error> RuntimeTensorResources::preflight_restart_publication() noexcept {
  if (impl_ == nullptr || !impl_->restart_active) {
    return make_unexpected(tensor_resource_error("restart publication state invalid"));
  }
  for (const auto& prepared : impl_->restart_outputs) {
    if (!impl_->outputs[prepared.output_index]->restart_publication_ready()) {
      return make_unexpected(
          tensor_resource_error("restart publication preflight failed",
                                "output",
                                impl_->outputs[prepared.output_index]->output().value()));
    }
  }
  for (const auto& prepared : impl_->restart_tensor_schema_outputs) {
    if (!impl_->tensor_schema_outputs[prepared.output_index]->restart_publication_ready()) {
      return make_unexpected(tensor_resource_error(
          "restart publication preflight failed",
          "output",
          impl_->tensor_schema_outputs[prepared.output_index]->output().value()));
    }
  }
  return {};
}

expected<void, Error> RuntimeTensorResources::rollback_restart() noexcept {
  if (impl_ == nullptr || !impl_->restart_active) {
    return make_unexpected(tensor_resource_error("restart rollback state invalid"));
  }
  for (auto& prepared : impl_->restart_outputs) {
    auto reopened = impl_->outputs[prepared.output_index]->reopen();
    if (!reopened) {
      return make_unexpected(std::move(reopened.error()));
    }
    prepared.candidate.reset();
  }
  for (auto& prepared : impl_->restart_tensor_schema_outputs) {
    auto reopened = impl_->tensor_schema_outputs[prepared.output_index]->reopen();
    if (!reopened) {
      return make_unexpected(std::move(reopened.error()));
    }
    prepared.candidate.reset();
  }
  impl_->restart_outputs.clear();
  impl_->restart_tensor_schema_outputs.clear();
  impl_->restart_active = false;
  return {};
}

std::size_t RuntimeTensorResources::device_resource_count() const noexcept {
  return impl_ != nullptr ? impl_->devices.size() : 0U;
}

std::size_t RuntimeTensorResources::output_pool_count() const noexcept {
  return impl_ != nullptr ? impl_->outputs.size() : 0U;
}

}  // namespace holoscan::detail
