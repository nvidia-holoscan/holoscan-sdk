/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_update_coordinator.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/logger/logger.hpp>

#include "parameter_catalog_projection.hpp"
#include "runtime/runtime_termination.hpp"

namespace holoscan::detail {
namespace {

constexpr std::uint32_t kInvalidSlot = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kRetiredClaim = std::uint64_t{1} << 63U;
constexpr std::uint64_t kPinCountMask = kRetiredClaim - 1U;

enum class SlotRole : std::uint8_t {
  kFree,
  kPublished,
  kPending,
  kLifecycleCandidate,
  kRetired,
};

struct GenerationSlot {
  PinnedParameterHeader header{};
  std::atomic<std::uint64_t> claims{kRetiredClaim};
  std::atomic<std::uint64_t> version{};
  std::atomic_bool observed{};
  ParamUpdateSequence update_sequence{};
  LifecycleTransactionId lifecycle_transaction{};
  std::uint64_t lifecycle_receipt{};
  SlotRole role{SlotRole::kFree};
  bool live{};
};

struct PendingGeneration {
  std::uint32_t slot{kInvalidSlot};
  ParamUpdateSequence sequence{};
};

struct LifecyclePublication {
  ParamUpdateSequence published_sequence{};
  ParamUpdateSequence retired_sequence{};
  bool retired_was_observed{};
};

enum class ObservationStatus : std::uint8_t {
  kInvalid,
  kSatisfied,
  kPending,
};

struct ResultLedgerEntry {
  ParamUpdateResult result{};
  OperatorRef owner{};
  ParamFieldId sole_field{};
  ParamReasonId audit_reason{};
  std::uint64_t audit_actor{};
  std::uint32_t audit_patch_count{};
  std::uint32_t event_reservations{};
  bool single_field{};
  bool requires_internal{};
  bool requires_sensitive{};
  bool external_tuning_visible{};
  bool valid{};
  bool reserved{};
  bool acknowledged{};
};

class FixedEventLedger {
 public:
  static constexpr std::uint32_t kAcceptedTransitionReservation = 8U;

  void initialize(std::uint32_t capacity, std::uint32_t cursor_capacity) {
    capacity_ = capacity;
    entries_ = EventStorage{new ParamEvent[capacity_]{}};
    cursor_capacity_ = cursor_capacity;
    cursors_ = CursorStorage{new CursorEntry[cursor_capacity_]{}};
  }

  [[nodiscard]] bool can_append_unreserved(std::uint32_t count = 1U) const noexcept {
    const ParamEventSequence available =
        std::numeric_limits<ParamEventSequence>::max() - next_sequence_;
    return reserved_sequences_ <= available && count <= available - reserved_sequences_;
  }

  [[nodiscard]] bool replace_accepted_transition_reservation(
      std::uint32_t released_count) noexcept {
    if (released_count > reserved_sequences_) {
      runtime_log_and_terminate_noexcept([] {
        HOLOSCAN_LOG_ERROR(
            "Parameter event reservation replacement exceeded the retained reservation count");
      });
    }
    const ParamEventSequence available =
        std::numeric_limits<ParamEventSequence>::max() - next_sequence_;
    const std::uint64_t retained_reservations = reserved_sequences_ - released_count;
    if (retained_reservations > available ||
        kAcceptedTransitionReservation > available - retained_reservations) {
      return false;
    }
    reserved_sequences_ = retained_reservations;
    reserved_sequences_ += kAcceptedTransitionReservation;
    return true;
  }

  void release_accepted_transitions(std::uint32_t count) noexcept {
    if (count > reserved_sequences_) {
      runtime_log_and_terminate_noexcept([] {
        HOLOSCAN_LOG_ERROR(
            "Parameter event reservation release exceeded the retained reservation count");
      });
    }
    reserved_sequences_ -= count;
  }

  [[nodiscard]] std::uint64_t reserved_sequences() const noexcept { return reserved_sequences_; }

  void set_next_sequence_for_test(ParamEventSequence sequence) noexcept {
    next_sequence_ = sequence;
  }

  template <typename Visible>
  [[nodiscard]] bool append(ParamEvent event, bool use_reservation,
                            const Visible& visible) noexcept {
    if (capacity_ == 0U || next_sequence_ == std::numeric_limits<ParamEventSequence>::max() ||
        (use_reservation ? reserved_sequences_ == 0U : !can_append_unreserved())) {
      return false;
    }
    if (count_ == capacity_) {
      const ParamEvent& overwritten = entries_[next_];
      for (std::uint32_t index = 0U; index < cursor_capacity_; ++index) {
        CursorEntry& cursor = cursors_[index];
        if (!cursor.active || cursor.source_exhausted ||
            cursor.source_sequence > overwritten.event_sequence) {
          continue;
        }
        cursor.gap_pending = cursor.gap_pending || visible(cursor.policy, overwritten);
        advance_source(cursor, overwritten.event_sequence);
      }
    }
    if (use_reservation) {
      --reserved_sequences_;
    }
    event.kind = ParamEventKind::kTransition;
    event.event_sequence = ++next_sequence_;
    entries_[next_] = event;
    next_ = (next_ + 1U) % capacity_;
    count_ = std::min<std::uint32_t>(count_ + 1U, capacity_);
    return true;
  }

  [[nodiscard]] expected<std::uint32_t, Error> register_cursor(
      const ParamAccessPolicy& policy) noexcept {
    for (std::uint32_t index = 0U; index < cursor_capacity_; ++index) {
      CursorEntry& cursor = cursors_[index];
      if (!cursor.active) {
        cursor = CursorEntry{.policy = policy, .projected_sequence = 1U, .active = true};
        if (count_ == 0U) {
          advance_source(cursor, next_sequence_);
        } else {
          cursor.source_sequence = next_sequence_ - count_ + 1U;
        }
        return index + 1U;
      }
    }
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }

  void unregister_cursor(std::uint32_t cursor) noexcept {
    if (cursor == 0U || cursor > cursor_capacity_) {
      return;
    }
    cursors_[cursor - 1U] = CursorEntry{};
  }

  template <typename Visible>
  expected<std::size_t, Error> drain(std::uint32_t cursor_id, std::span<ParamEvent> output,
                                     const Visible& visible) noexcept {
    if (cursor_id == 0U || cursor_id > cursor_capacity_) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    CursorEntry& cursor = cursors_[cursor_id - 1U];
    if (!cursor.active) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    if (output.empty()) {
      return 0U;
    }
    std::size_t written = 0U;
    if (cursor.gap_pending) {
      if (cursor.projected_exhausted) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      output[written++] = ParamEvent{
          .kind = ParamEventKind::kGap,
          .event_sequence = cursor.projected_sequence,
          .first_missing_sequence = cursor.projected_sequence,
          .last_missing_sequence = cursor.projected_sequence,
      };
      advance_projected(cursor);
      cursor.gap_pending = false;
      if (written == output.size()) {
        return written;
      }
    }
    while (!cursor.source_exhausted && cursor.source_sequence <= next_sequence_ &&
           written < output.size()) {
      const ParamEventSequence distance = next_sequence_ - cursor.source_sequence;
      const std::uint32_t index = static_cast<std::uint32_t>(
          (next_ + capacity_ - 1U - static_cast<std::uint32_t>(distance)) % capacity_);
      const ParamEvent& event = entries_[index];
      advance_source(cursor, cursor.source_sequence);
      if (visible(cursor.policy, event)) {
        if (cursor.projected_exhausted) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        ParamEvent projected = event;
        projected.event_sequence = cursor.projected_sequence;
        advance_projected(cursor);
        projected.first_missing_sequence = 0U;
        projected.last_missing_sequence = 0U;
        output[written++] = projected;
      }
    }
    return written;
  }

 private:
  struct CursorEntry {
    ParamAccessPolicy policy{};
    ParamEventSequence source_sequence{};
    ParamEventSequence projected_sequence{};
    bool active{};
    bool gap_pending{};
    bool source_exhausted{};
    bool projected_exhausted{};
  };

  static void advance_source(CursorEntry& cursor, ParamEventSequence consumed) noexcept {
    if (consumed == std::numeric_limits<ParamEventSequence>::max()) {
      cursor.source_exhausted = true;
      return;
    }
    cursor.source_sequence = consumed + 1U;
  }

  static void advance_projected(CursorEntry& cursor) noexcept {
    if (cursor.projected_sequence == std::numeric_limits<ParamEventSequence>::max()) {
      cursor.projected_exhausted = true;
      return;
    }
    ++cursor.projected_sequence;
  }

  // These arrays are fixed-capacity ledgers allocated once at run reconstruction. A vector would
  // expose capacity-changing operations that are forbidden after reconstruction.
  using EventStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ParamEvent[]>;
  using CursorStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<CursorEntry[]>;
  EventStorage entries_;
  CursorStorage cursors_;
  std::uint32_t capacity_{};
  std::uint32_t cursor_capacity_{};
  std::uint32_t count_{};
  std::uint32_t next_{};
  ParamEventSequence next_sequence_{};
  std::uint64_t reserved_sequences_{};
};

[[nodiscard]] bool terminal_state(ParamUpdateState state) noexcept {
  switch (state) {
    case ParamUpdateState::kRejected:
    case ParamUpdateState::kAcceptedNoChange:
    case ParamUpdateState::kPublished:
    case ParamUpdateState::kObserved:
    case ParamUpdateState::kSuperseded:
    case ParamUpdateState::kCanceled:
    case ParamUpdateState::kRolledBack:
    case ParamUpdateState::kFailedStopped:
      return true;
    case ParamUpdateState::kSubmitted:
    case ParamUpdateState::kAccepted:
    case ParamUpdateState::kQuiescing:
    case ParamUpdateState::kRestarting:
      return false;
  }
  return false;
}

[[nodiscard]] bool reusable_result_slot(const ResultLedgerEntry& entry) noexcept {
  return (!entry.valid && !entry.reserved) ||
         (entry.valid && terminal_state(entry.result.state) &&
          entry.result.state != ParamUpdateState::kPublished && entry.acknowledged);
}

template <typename Entry>
[[nodiscard]] Entry* select_reusable_result_slot(Entry* entries, std::uint32_t capacity) noexcept {
  Entry* oldest_acknowledged = nullptr;
  for (std::uint32_t index = 0U; index < capacity; ++index) {
    Entry& entry = entries[index];
    if (!reusable_result_slot(entry)) {
      continue;
    }
    if (!entry.valid) {
      return &entry;
    }
    if (oldest_acknowledged == nullptr ||
        entry.result.update_sequence < oldest_acknowledged->result.update_sequence) {
      oldest_acknowledged = &entry;
    }
  }
  return oldest_acknowledged;
}

[[nodiscard]] bool final_evidence_state(ParamUpdateState state) noexcept {
  return terminal_state(state) && state != ParamUpdateState::kPublished;
}

[[nodiscard]] std::optional<ParamUpdateState> parameter_state(
    LifecycleTransactionState state) noexcept {
  switch (state) {
    case LifecycleTransactionState::kQueued:
      return ParamUpdateState::kAccepted;
    case LifecycleTransactionState::kQuiescing:
      return ParamUpdateState::kQuiescing;
    case LifecycleTransactionState::kRestarting:
      return ParamUpdateState::kRestarting;
    case LifecycleTransactionState::kPublished:
      return ParamUpdateState::kPublished;
    case LifecycleTransactionState::kSuperseded:
      return ParamUpdateState::kSuperseded;
    case LifecycleTransactionState::kCanceled:
      return ParamUpdateState::kCanceled;
    case LifecycleTransactionState::kRolledBack:
      return ParamUpdateState::kRolledBack;
    case LifecycleTransactionState::kFailedStopped:
      return ParamUpdateState::kFailedStopped;
    case LifecycleTransactionState::kAcceptedNoChange:
      return ParamUpdateState::kAcceptedNoChange;
    case LifecycleTransactionState::kRejected:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

[[nodiscard]] bool checked_multiply(std::size_t lhs, std::size_t rhs,
                                    std::size_t& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

[[nodiscard]] bool aligned_stride(std::size_t size, std::size_t alignment,
                                  std::size_t& result) noexcept {
  if (!is_power_of_two(alignment) ||
      size > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    return false;
  }
  result = (size + alignment - 1U) & ~(alignment - 1U);
  return true;
}

[[nodiscard]] ParamUpdateStatus validation_status(ParamValidationResult result) noexcept {
  switch (result.code) {
    case ParamValidationCode::kOk:
      return ParamUpdateStatus::kAccepted;
    case ParamValidationCode::kOutOfRange:
    case ParamValidationCode::kNotAllowed:
    case ParamValidationCode::kCardinalityExceeded:
      return ParamUpdateStatus::kOutOfBounds;
    case ParamValidationCode::kInvalidEncoding:
    case ParamValidationCode::kValidatorRejected:
      return ParamUpdateStatus::kValidationFailed;
  }
  return ParamUpdateStatus::kValidationFailed;
}

[[nodiscard]] std::string scalar_suffix(ParamLogicalType type, std::uint16_t bit_width) {
  switch (type) {
    case ParamLogicalType::kBool:
      return "bool";
    case ParamLogicalType::kSignedInteger:
      return "i" + std::to_string(bit_width);
    case ParamLogicalType::kUnsignedInteger:
      return "u" + std::to_string(bit_width);
    case ParamLogicalType::kFloatingPoint:
      return "f" + std::to_string(bit_width);
    case ParamLogicalType::kString:
      return "string";
    case ParamLogicalType::kSequence:
      return "sequence";
    case ParamLogicalType::kFilesystemPath:
      return "filesystem-path";
    case ParamLogicalType::kDuration:
      return "duration";
    case ParamLogicalType::kObject:
      return "object";
  }
  return "object";
}

[[nodiscard]] ParamSnapshotTypeIdentity snapshot_type_identity(
    const ParameterFieldPlanRecord& field) {
  return ParamSnapshotTypeIdentity{
      .id = "holoscan." + scalar_suffix(field.logical_type, field.type_shape.bit_width),
      .version = 1U,
      .shape = field.type_shape,
  };
}

[[nodiscard]] std::optional<ParamSnapshotCodecIdentity> snapshot_codec_identity(
    const ParameterFieldPlanRecord& field) {
  if (field.value_route != ParamValueRouteDisposition::kPortableScalar) {
    return std::nullopt;
  }
  return ParamSnapshotCodecIdentity{
      .id = "holoscan.config." + scalar_suffix(field.logical_type, field.type_shape.bit_width),
      .version = 1U,
  };
}

[[nodiscard]] bool operator_path_less(const OperatorPath& left,
                                      const OperatorPath& right) noexcept {
  return std::lexicographical_compare(left.segments().begin(),
                                      left.segments().end(),
                                      right.segments().begin(),
                                      right.segments().end());
}

[[nodiscard]] bool has_portable_scalar_route(const ParamFieldDescriptor& field) noexcept {
  if (field.type_shape.optional || !field.type_shape.has_stable_shape ||
      (field.local_operations.read_member_as_scalar == nullptr &&
       field.local_operations.read_hosted_member_as_scalar == nullptr)) {
    return false;
  }
  switch (field.logical_type) {
    case ParamLogicalType::kBool:
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
    case ParamLogicalType::kFloatingPoint:
      return true;
    case ParamLogicalType::kDuration:
      return (field.type_shape.element_or_rep_type == ParamLogicalType::kSignedInteger ||
              field.type_shape.element_or_rep_type == ParamLogicalType::kUnsignedInteger) &&
             field.type_shape.element_or_rep_bit_width > 0U &&
             field.type_shape.element_or_rep_bit_width <= 64U;
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] ParamUpdateStatus disposition_status(
    ParameterFieldUpdateDisposition disposition) noexcept {
  switch (disposition) {
    case ParameterFieldUpdateDisposition::kAdmitted:
      return ParamUpdateStatus::kAccepted;
    case ParameterFieldUpdateDisposition::kPolicyDisabled:
      return ParamUpdateStatus::kUpdatesDisabled;
    case ParameterFieldUpdateDisposition::kCompileTime:
    case ParameterFieldUpdateDisposition::kRecompileRequired:
      return ParamUpdateStatus::kRequiresRecompile;
    case ParameterFieldUpdateDisposition::kLifecycleDeferred:
      return ParamUpdateStatus::kRequiresLifecycleBarrier;
    case ParameterFieldUpdateDisposition::kUnsupportedLogicalType:
    case ParameterFieldUpdateDisposition::kReplacementUnavailable:
      return ParamUpdateStatus::kTypeMismatch;
    case ParameterFieldUpdateDisposition::kProfileDisallowed:
      return ParamUpdateStatus::kUnsupportedInProfile;
    case ParameterFieldUpdateDisposition::kCapacityUnavailable:
      return ParamUpdateStatus::kQueueFull;
  }
  return ParamUpdateStatus::kRejected;
}

[[nodiscard]] ParamUpdateStatus lifecycle_submission_status(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kResourceExhausted:
    case ErrorCode::kLifecycleRestartQueueFull:
      return ParamUpdateStatus::kQueueFull;
    case ErrorCode::kLifecycleTransactionsDisabled:
      return ParamUpdateStatus::kRequiresLifecycleBarrier;
    case ErrorCode::kLifecycleRestartUnsupportedInProfile:
      return ParamUpdateStatus::kUnsupportedInProfile;
    case ErrorCode::kClosed:
      return ParamUpdateStatus::kClosed;
    default:
      return ParamUpdateStatus::kRejected;
  }
}

[[nodiscard]] std::optional<ParamScalarValue> enum_label_value(const ParamFieldDescriptor& field,
                                                               std::string_view label) noexcept {
  for (std::size_t index = 0U; index < field.enum_choices.count; ++index) {
    if (field.enum_choices.values[index].name == label) {
      return field.enum_choices.values[index].value;
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool valid_integer_width(std::uint16_t width) noexcept {
  return width == 8U || width == 16U || width == 32U || width == 64U;
}

[[nodiscard]] bool decode_signed(ParamScalarValue value, std::int64_t& result) noexcept {
  if (value.kind != ParamScalarKind::kSignedInteger || !valid_integer_width(value.bit_width)) {
    return false;
  }
  if (value.bit_width == 64U) {
    result = std::bit_cast<std::int64_t>(value.raw_bits);
    return true;
  }
  const std::uint64_t mask = (std::uint64_t{1} << value.bit_width) - 1U;
  if ((value.raw_bits & ~mask) != 0U) {
    return false;
  }
  const std::uint64_t sign = std::uint64_t{1} << (value.bit_width - 1U);
  result = std::bit_cast<std::int64_t>((value.raw_bits & sign) != 0U ? value.raw_bits | ~mask
                                                                     : value.raw_bits);
  return true;
}

[[nodiscard]] bool decode_unsigned(ParamScalarValue value, std::uint64_t& result) noexcept {
  if (value.kind != ParamScalarKind::kUnsignedInteger || !valid_integer_width(value.bit_width) ||
      (value.bit_width != 64U && value.raw_bits >= (std::uint64_t{1} << value.bit_width))) {
    return false;
  }
  result = value.raw_bits;
  return true;
}

[[nodiscard]] ParamScalarValue signed_value(std::int64_t value, std::uint16_t width) noexcept {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const std::uint64_t mask =
      width == 64U ? std::numeric_limits<std::uint64_t>::max() : (std::uint64_t{1} << width) - 1U;
  return ParamScalarValue{.kind = ParamScalarKind::kSignedInteger,
                          .bit_width = static_cast<std::uint8_t>(width),
                          .raw_bits = bits & mask};
}

// Interpret Python-style wide numeric carriers against the exact declared scalar width.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool coerce_patch_scalar(ParamScalarValue input, ParamLogicalType type,
                                       std::uint16_t width, ParamScalarValue& output) noexcept {
  if (type == ParamLogicalType::kBool) {
    if (input.kind == ParamScalarKind::kBool && input.bit_width == 1U && input.raw_bits <= 1U) {
      output = input;
      return true;
    }
    return false;
  }
  if (type == ParamLogicalType::kDuration) {
    if (input.kind == ParamScalarKind::kDurationNanoseconds && input.bit_width == 64U) {
      output = input;
      return true;
    }
    return false;
  }
  if (type == ParamLogicalType::kSignedInteger || type == ParamLogicalType::kUnsignedInteger) {
    if (!valid_integer_width(width)) {
      return false;
    }
    std::int64_t signed_input{};
    std::uint64_t unsigned_input{};
    if (type == ParamLogicalType::kSignedInteger) {
      if (decode_signed(input, signed_input)) {
        const bool fits = width == 64U || (signed_input >= -(std::int64_t{1} << (width - 1U)) &&
                                           signed_input < (std::int64_t{1} << (width - 1U)));
        if (fits) {
          output = signed_value(signed_input, width);
          return true;
        }
      }
      if (decode_unsigned(input, unsigned_input) &&
          unsigned_input <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        const auto converted = static_cast<std::int64_t>(unsigned_input);
        const bool fits = width == 64U || converted < (std::int64_t{1} << (width - 1U));
        if (fits) {
          output = signed_value(converted, width);
          return true;
        }
      }
      return false;
    }
    if (decode_signed(input, signed_input) && signed_input >= 0) {
      unsigned_input = static_cast<std::uint64_t>(signed_input);
    } else if (!decode_unsigned(input, unsigned_input)) {
      return false;
    }
    if (width != 64U && unsigned_input >= (std::uint64_t{1} << width)) {
      return false;
    }
    output = ParamScalarValue{.kind = ParamScalarKind::kUnsignedInteger,
                              .bit_width = static_cast<std::uint8_t>(width),
                              .raw_bits = unsigned_input};
    return true;
  }
  if (type != ParamLogicalType::kFloatingPoint) {
    return false;
  }
  double decoded{};
  if (input.kind == ParamScalarKind::kFloatingPoint && input.bit_width == 64U) {
    decoded = std::bit_cast<double>(input.raw_bits);
  } else if (input.kind == ParamScalarKind::kFloatingPoint && input.bit_width == 32U &&
             input.raw_bits <= std::numeric_limits<std::uint32_t>::max()) {
    decoded = std::bit_cast<float>(static_cast<std::uint32_t>(input.raw_bits));
  } else {
    std::int64_t signed_input{};
    std::uint64_t unsigned_input{};
    if (decode_signed(input, signed_input)) {
      decoded = static_cast<double>(signed_input);
    } else if (decode_unsigned(input, unsigned_input)) {
      decoded = static_cast<double>(unsigned_input);
    } else {
      return false;
    }
  }
  if (width == 32U) {
    const float narrowed = static_cast<float>(decoded);
    if (std::isfinite(decoded) && !std::isfinite(narrowed)) {
      return false;
    }
    output = ParamScalarValue::from(narrowed);
    return true;
  }
  if (width == 64U) {
    output = ParamScalarValue::from(decoded);
    return true;
  }
  return false;
}

[[nodiscard]] bool exact_patch_scalar(ParamScalarValue input, ParamLogicalType type,
                                      std::uint16_t width) noexcept {
  switch (type) {
    case ParamLogicalType::kBool:
      return input.kind == ParamScalarKind::kBool && input.bit_width == 1U && input.raw_bits <= 1U;
    case ParamLogicalType::kSignedInteger:
      return input.kind == ParamScalarKind::kSignedInteger && input.bit_width == width;
    case ParamLogicalType::kUnsignedInteger:
      return input.kind == ParamScalarKind::kUnsignedInteger && input.bit_width == width;
    case ParamLogicalType::kFloatingPoint:
      return input.kind == ParamScalarKind::kFloatingPoint && input.bit_width == width;
    case ParamLogicalType::kDuration:
      return input.kind == ParamScalarKind::kDurationNanoseconds && input.bit_width == 64U;
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] std::optional<ParamScalarValue> patch_scalar_value(
    const ParamFieldDescriptor& field, const ParamPatchValue& value,
    bool sequence_element = false) noexcept {
  ParamScalarValue input{};
  if (value.kind() == ParamPatchValueKind::kScalar) {
    input = value.scalar();
  } else if (value.kind() == ParamPatchValueKind::kString && field.enum_choices.count != 0U) {
    const auto choice = enum_label_value(field, value.string());
    if (!choice) {
      return std::nullopt;
    }
    input = *choice;
  } else {
    return std::nullopt;
  }
  const ParamLogicalType type =
      sequence_element ? field.type_shape.element_or_rep_type : field.logical_type;
  const std::uint16_t width =
      sequence_element ? field.type_shape.element_or_rep_bit_width : field.type_shape.bit_width;
  if (!detail::ParamPatchValueAccess::permits_numeric_conversion(value)) {
    return exact_patch_scalar(input, type, width) ? std::optional{input} : std::nullopt;
  }
  ParamScalarValue converted{};
  return coerce_patch_scalar(input, type, width, converted) ? std::optional{converted}
                                                            : std::nullopt;
}

[[nodiscard]] const ParamFieldDescriptor* find_nested_field(const ParamSchemaDescriptor& schema,
                                                            std::string_view name) noexcept {
  const auto found = std::find_if(schema.fields.begin(),
                                  schema.fields.end(),
                                  [&](const auto& item) { return item.canonical_name == name; });
  return found == schema.fields.end() ? nullptr : &*found;
}

// Decode the owning control-plane carrier directly into one cloned aggregate. A failure can leave
// that private candidate changed; the caller discards it before any publication is possible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamUpdateStatus replace_patch_value(void* aggregate,
                                                    const ParamFieldDescriptor& field,
                                                    const ParamPatchValue& value,
                                                    std::uint32_t depth = 0U) {
  if (value.kind() == ParamPatchValueKind::kUnset) {
    return ParamUpdateStatus::kTypeMismatch;
  }
  if (value.kind() == ParamPatchValueKind::kNone) {
    if (!field.type_shape.optional || field.local_operations.clear_optional_member == nullptr) {
      return ParamUpdateStatus::kTypeMismatch;
    }
    field.local_operations.clear_optional_member(aggregate);
    return ParamUpdateStatus::kAccepted;
  }
  if (const auto scalar = patch_scalar_value(field, value)) {
    return field.local_operations.replace_member_from_scalar != nullptr &&
                   field.local_operations.replace_member_from_scalar(aggregate, *scalar)
               ? ParamUpdateStatus::kAccepted
               : ParamUpdateStatus::kTypeMismatch;
  }
  if (value.kind() == ParamPatchValueKind::kString) {
    if (field.logical_type != ParamLogicalType::kString ||
        field.local_operations.replace_member_from_string == nullptr) {
      return ParamUpdateStatus::kTypeMismatch;
    }
    if (field.constraints.max_utf8_bytes != 0U &&
        value.string().size() > field.constraints.max_utf8_bytes) {
      return ParamUpdateStatus::kOutOfBounds;
    }
    field.local_operations.replace_member_from_string(aggregate, value.string());
    return ParamUpdateStatus::kAccepted;
  }
  if (value.kind() == ParamPatchValueKind::kSequence) {
    const auto elements = value.sequence();
    if (field.logical_type != ParamLogicalType::kSequence ||
        (field.constraints.max_elements != 0U &&
         elements.size() > field.constraints.max_elements)) {
      return field.logical_type == ParamLogicalType::kSequence ? ParamUpdateStatus::kOutOfBounds
                                                               : ParamUpdateStatus::kTypeMismatch;
    }
    if (field.type_shape.element_or_rep_type == ParamLogicalType::kString) {
      if (field.local_operations.replace_member_from_string_sequence == nullptr) {
        return ParamUpdateStatus::kTypeMismatch;
      }
      std::vector<std::string_view> strings;
      strings.reserve(elements.size());
      for (const ParamPatchValue& element : elements) {
        if (element.kind() != ParamPatchValueKind::kString ||
            (field.constraints.max_element_utf8_bytes != 0U &&
             element.string().size() > field.constraints.max_element_utf8_bytes)) {
          return element.kind() == ParamPatchValueKind::kString ? ParamUpdateStatus::kOutOfBounds
                                                                : ParamUpdateStatus::kTypeMismatch;
        }
        strings.push_back(element.string());
      }
      field.local_operations.replace_member_from_string_sequence(aggregate, strings);
      return ParamUpdateStatus::kAccepted;
    }
    if (field.local_operations.replace_member_from_scalar_sequence == nullptr) {
      return ParamUpdateStatus::kTypeMismatch;
    }
    std::vector<ParamScalarValue> scalars;
    scalars.reserve(elements.size());
    for (const ParamPatchValue& element : elements) {
      const auto scalar = patch_scalar_value(field, element, true);
      if (!scalar) {
        return ParamUpdateStatus::kTypeMismatch;
      }
      scalars.push_back(*scalar);
    }
    return field.local_operations.replace_member_from_scalar_sequence(aggregate, scalars)
               ? ParamUpdateStatus::kAccepted
               : ParamUpdateStatus::kTypeMismatch;
  }
  if (value.kind() != ParamPatchValueKind::kObject ||
      field.logical_type != ParamLogicalType::kObject || !field.nested || depth >= 32U ||
      field.local_operations.mutable_nested_member == nullptr ||
      field.local_operations.nested_schema == nullptr) {
    return ParamUpdateStatus::kTypeMismatch;
  }
  const ParamSchemaDescriptor* const schema = field.local_operations.nested_schema();
  if (schema == nullptr || schema->coverage != ParamSchemaCoverage::kComplete) {
    return ParamUpdateStatus::kTypeMismatch;
  }
  void* const nested = field.local_operations.mutable_nested_member(aggregate);
  if (nested == nullptr) {
    return ParamUpdateStatus::kTypeMismatch;
  }
  const auto assignments = value.object();
  for (std::size_t index = 0U; index < assignments.size(); ++index) {
    if (assignments[index].name.empty() ||
        std::any_of(assignments.begin(),
                    assignments.begin() + static_cast<std::ptrdiff_t>(index),
                    [&](const auto& prior) { return prior.name == assignments[index].name; })) {
      return ParamUpdateStatus::kRejected;
    }
    const ParamFieldDescriptor* const child = find_nested_field(*schema, assignments[index].name);
    if (child == nullptr) {
      return ParamUpdateStatus::kUnknownParameter;
    }
    const ParamUpdateStatus status =
        replace_patch_value(nested, *child, assignments[index].value, depth + 1U);
    if (status != ParamUpdateStatus::kAccepted) {
      return status;
    }
  }
  return ParamUpdateStatus::kAccepted;
}

// The nested switches deliberately enumerate the admitted scalar kind/width pairs. Keeping this
// conversion table explicit makes unsupported encodings fail closed and auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<ParamValue> value_from_scalar(ParamScalarValue scalar) noexcept {
  switch (scalar.kind) {
    case ParamScalarKind::kBool: {
      bool value{};
      return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
    }
    case ParamScalarKind::kSignedInteger:
      switch (scalar.bit_width) {
        case 8: {
          std::int8_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 16: {
          std::int16_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 32: {
          std::int32_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 64: {
          std::int64_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        default:
          return std::nullopt;
      }
    case ParamScalarKind::kUnsignedInteger:
      switch (scalar.bit_width) {
        case 8: {
          std::uint8_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 16: {
          std::uint16_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 32: {
          std::uint32_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 64: {
          std::uint64_t value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        default:
          return std::nullopt;
      }
    case ParamScalarKind::kFloatingPoint:
      switch (scalar.bit_width) {
        case 32: {
          float value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        case 64: {
          double value{};
          return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
        }
        default:
          return std::nullopt;
      }
    case ParamScalarKind::kDurationNanoseconds: {
      std::chrono::nanoseconds value{};
      return scalar.try_as(value) ? std::optional{ParamValue::from(value)} : std::nullopt;
    }
    case ParamScalarKind::kNone:
      return std::nullopt;
  }
  return std::nullopt;
}

template <typename T>
[[nodiscard]] bool decoded_scalar_equal(ParamScalarValue left, ParamScalarValue right) noexcept {
  T left_value{};
  T right_value{};
  if (left.raw_bits == right.raw_bits) {
    return left.try_as(left_value);
  }
  return left.try_as(left_value) && right.try_as(right_value) && left_value == right_value;
}

[[nodiscard]] bool scalar_semantically_equal(ParamScalarValue left,
                                             ParamScalarValue right) noexcept {
  if (left.kind != right.kind || left.bit_width != right.bit_width) {
    return false;
  }
  switch (left.kind) {
    case ParamScalarKind::kBool:
      return decoded_scalar_equal<bool>(left, right);
    case ParamScalarKind::kSignedInteger:
      switch (left.bit_width) {
        case 8:
          return decoded_scalar_equal<std::int8_t>(left, right);
        case 16:
          return decoded_scalar_equal<std::int16_t>(left, right);
        case 32:
          return decoded_scalar_equal<std::int32_t>(left, right);
        case 64:
          return decoded_scalar_equal<std::int64_t>(left, right);
        default:
          return false;
      }
    case ParamScalarKind::kUnsignedInteger:
      switch (left.bit_width) {
        case 8:
          return decoded_scalar_equal<std::uint8_t>(left, right);
        case 16:
          return decoded_scalar_equal<std::uint16_t>(left, right);
        case 32:
          return decoded_scalar_equal<std::uint32_t>(left, right);
        case 64:
          return decoded_scalar_equal<std::uint64_t>(left, right);
        default:
          return false;
      }
    case ParamScalarKind::kFloatingPoint:
      if (left.bit_width == 32U) {
        return decoded_scalar_equal<float>(left, right);
      }
      return left.bit_width == 64U && decoded_scalar_equal<double>(left, right);
    case ParamScalarKind::kDurationNanoseconds:
      return decoded_scalar_equal<std::chrono::nanoseconds>(left, right);
    case ParamScalarKind::kNone:
      return false;
  }
  return false;
}

[[nodiscard]] bool field_semantically_equal(const ParamFieldDescriptor& field, const void* left,
                                            const void* right) noexcept {
  ParamScalarValue left_scalar{};
  ParamScalarValue right_scalar{};
  const bool left_has_scalar = read_param_field_scalar(field, left, left_scalar);
  const bool right_has_scalar = read_param_field_scalar(field, right, right_scalar);
  if (left_has_scalar || right_has_scalar) {
    return left_has_scalar && right_has_scalar &&
           scalar_semantically_equal(left_scalar, right_scalar);
  }

  const ParamFieldLocalOperations& operations = field.local_operations;
  if (operations.read_hosted_member_as_scalar_sequence != nullptr) {
    const std::span<const ParamScalarValue> left_values =
        read_param_field_scalar_sequence(field, left);
    const std::span<const ParamScalarValue> right_values =
        read_param_field_scalar_sequence(field, right);
    return left_values.size() == right_values.size() && std::equal(left_values.begin(),
                                                                   left_values.end(),
                                                                   right_values.begin(),
                                                                   scalar_semantically_equal);
  }
  if (operations.nested_member != nullptr && operations.nested_schema != nullptr) {
    const void* const left_nested = operations.nested_member(left);
    const void* const right_nested = operations.nested_member(right);
    if (left_nested == nullptr || right_nested == nullptr) {
      return left_nested == right_nested;
    }
    const ParamSchemaDescriptor* const nested_schema = operations.nested_schema();
    if (nested_schema == nullptr) {
      return false;
    }
    return std::all_of(nested_schema->fields.begin(),
                       nested_schema->fields.end(),
                       [left_nested, right_nested](const ParamFieldDescriptor& nested_field) {
                         return field_semantically_equal(nested_field, left_nested, right_nested);
                       });
  }

  if (operations.semantic_equal_members == nullptr) {
    return false;
  }
  try {
    return operations.semantic_equal_members(left, right);
  } catch (...) {
    return false;
  }
}

[[nodiscard]] bool valid_audit(ParamAudit audit) noexcept {
  return audit == ParamAudit::kProfileDefault || audit == ParamAudit::kMinimal ||
         audit == ParamAudit::kDetailed;
}

[[nodiscard]] bool profile_allows_active_parameter_control(
    std::optional<Profile> profile) noexcept {
  return profile == Profile::kStandard || profile == Profile::kResearch;
}

[[nodiscard]] bool profile_allows_detailed_parameter_audit(
    std::optional<Profile> profile) noexcept {
  return profile_allows_active_parameter_control(profile);
}

}  // namespace

class ParameterGenerationStore {
 public:
  ParameterGenerationStore() = default;
  ParameterGenerationStore(const ParameterGenerationStore&) = delete;
  ParameterGenerationStore& operator=(const ParameterGenerationStore&) = delete;

  ~ParameterGenerationStore() {
    if (slots_ != nullptr && type_ != nullptr) {
      for (std::uint32_t index = 0U; index < slot_count_; ++index) {
        if (slots_[index].live) {
          type_->destroy(const_cast<void*>(slots_[index].header.value));
        }
      }
    }
    if (owned_static_value_ != nullptr && type_ != nullptr) {
      type_->destroy(owned_static_value_);
      ::operator delete(owned_static_value_, std::align_val_t{owned_static_alignment_});
    }
    if (arena_ != nullptr) {
      ::operator delete(arena_, std::align_val_t{arena_alignment_});
    }
  }

  // Reconstruction validates the complete cross-record contract before publishing any storage.
  // The checks remain together so no partially validated generation store becomes observable.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> initialize(
      const ParameterRuntimeOperatorBinding& binding,
      const ParameterRuntimeCapacityPlanRecord& capacity) {
    if (!binding.run_bound_owner.run_bound() || binding.plan == nullptr ||
        binding.type == nullptr || binding.initial_value == nullptr ||
        !binding.plan->owner.plan_bound() || binding.plan->owner.id != binding.run_bound_owner.id ||
        binding.plan->owner.plan_generation != binding.run_bound_owner.plan_generation ||
        binding.plan->object_size != binding.type->object_size ||
        binding.plan->object_alignment != binding.type->object_alignment ||
        binding.type->local_type_token == nullptr || binding.type->object_size == 0U ||
        !is_power_of_two(binding.type->object_alignment) || binding.type->destroy == nullptr ||
        binding.type->validate_aggregate == nullptr) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (binding.plan->schema_availability == ParamSchemaAvailability::kDeclared) {
      if (binding.schema == nullptr || binding.schema->local_type != binding.type ||
          binding.schema->stable_id != binding.plan->schema_id ||
          binding.schema->version != binding.plan->schema_version ||
          binding.schema->structural_hash != binding.plan->structural_hash ||
          binding.schema->contract_hash != binding.plan->contract_hash) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
    } else if (binding.schema != nullptr) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (binding.plan->no_params && (binding.schema != nullptr || !binding.fields.empty())) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if ((binding.plan->runtime_updates_enabled || binding.plan->lifecycle_updates_enabled) &&
        !profile_allows_active_parameter_control(binding.plan->effective_profile)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }

    owner_ = binding.run_bound_owner;
    plan_ = binding.plan;
    fields_ = binding.fields;
    derived_terms_ = binding.derived_terms;
    type_ = binding.type;
    schema_ = binding.schema;
    static_header_ = PinnedParameterHeader{
        .value = binding.initial_value,
        .local_type_token = binding.type->local_type_token,
        .generation = 0U,
        .schema_hash = binding.plan->contract_hash,
        .slot = 0U,
    };

    if (const auto initial_validation = type_->validate_aggregate(binding.initial_value);
        !initial_validation.ok()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (auto verified = verify_fields(); !verified) {
      return verified;
    }
    if (auto verified = verify_derived_terms(); !verified) {
      return verified;
    }
    if (!plan_->runtime_updates_enabled && !plan_->lifecycle_updates_enabled) {
      // GraphExec owns the reconstructed Operator aggregate, but the coordinator can outlive
      // GraphExec so terminal handles and RunSession::capture_params() remain inspectable after
      // wait(). Keep generation-zero-only values in distinct immutable storage; this is not an
      // update-generation arena and therefore does not consume parameter_arena_bytes capacity.
      if (!plan_->no_params) {
        if (type_->copy_construct == nullptr) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        owned_static_alignment_ = std::max(type_->object_alignment, alignof(std::max_align_t));
        void* const storage =
            ::operator new(type_->object_size, std::align_val_t{owned_static_alignment_});
        try {
          type_->copy_construct(storage, binding.initial_value);
        } catch (...) {
          ::operator delete(storage, std::align_val_t{owned_static_alignment_});
          throw;
        }
        owned_static_value_ = storage;
        static_header_.value = owned_static_value_;
      }
      return initialize_derived_term_storage(1U);
    }
    if (capacity.generation_slots_per_operator == 0U || type_->copy_construct == nullptr ||
        (plan_->materialization != ParameterMaterializationStrategy::kFixedObject &&
         plan_->materialization != ParameterMaterializationStrategy::kColdOwnedTyped) ||
        !type_->destroy_is_nothrow ||
        (plan_->runtime_updates_enabled &&
         (capacity.resolved_update_policy != ParamRuntimeUpdatePolicy::kEnabledIfAdmitted ||
          !capacity.runtime_updates_enabled || capacity.max_pending_updates_per_operator == 0U)) ||
        (plan_->lifecycle_updates_enabled && !capacity.lifecycle_updates_enabled)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }

    std::size_t arena_bytes{};
    if (!aligned_stride(type_->object_size, type_->object_alignment, stride_) ||
        !checked_multiply(stride_, capacity.generation_slots_per_operator, arena_bytes)) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (capacity.max_pending_updates_per_operator >
            std::numeric_limits<std::uint32_t>::max() - 2U ||
        capacity.generation_slots_per_operator < capacity.max_pending_updates_per_operator + 2U) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }

    slot_count_ = capacity.generation_slots_per_operator;
    pending_capacity_ =
        plan_->runtime_updates_enabled ? capacity.max_pending_updates_per_operator : 0U;
    slots_ = SlotStorage{new GenerationSlot[slot_count_]{}};
    if (pending_capacity_ != 0U) {
      pending_ = PendingStorage{new PendingGeneration[pending_capacity_]{}};
    }
    arena_alignment_ = std::max(type_->object_alignment, alignof(std::max_align_t));
    arena_ =
        static_cast<std::byte*>(::operator new(arena_bytes, std::align_val_t{arena_alignment_}));
    arena_bytes_ = arena_bytes;

    if (auto initialized = initialize_derived_term_storage(slot_count_); !initialized) {
      return initialized;
    }

    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      slots_[index].header = PinnedParameterHeader{
          .value = arena_ + stride_ * index,
          .local_type_token = type_->local_type_token,
          .generation = 0U,
          .schema_hash = plan_->contract_hash,
          .slot = index + 1U,
      };
    }
    type_->copy_construct(const_cast<void*>(slots_[0].header.value), binding.initial_value);
    slots_[0].live = true;
    slots_[0].role = SlotRole::kPublished;
    slots_[0].observed.store(true, std::memory_order_release);
    slots_[0].claims.store(0U, std::memory_order_release);
    slots_[0].version.store(1U, std::memory_order_release);
    published_.store(&slots_[0].header, std::memory_order_release);
    latest_slot_ = 0U;
    return {};
  }

  [[nodiscard]] std::size_t arena_bytes() const noexcept { return arena_bytes_; }
  [[nodiscard]] OperatorRef owner() const noexcept { return owner_; }
  [[nodiscard]] bool updates_enabled() const noexcept {
    return plan_->runtime_updates_enabled || plan_->lifecycle_updates_enabled;
  }
  [[nodiscard]] bool runtime_updates_enabled() const noexcept {
    return plan_->runtime_updates_enabled;
  }
  [[nodiscard]] bool lifecycle_updates_enabled() const noexcept {
    return plan_->lifecycle_updates_enabled;
  }

  [[nodiscard]] const ParameterFieldPlanRecord* find_plan_field(ParamFieldId field) const noexcept {
    const auto found = std::find_if(fields_.begin(), fields_.end(), [field](const auto& value) {
      return value.field == field;
    });
    return found == fields_.end() ? nullptr : &*found;
  }

  [[nodiscard]] const ParamFieldDescriptor* find_field(ParamFieldId field) const noexcept {
    if (schema_ == nullptr) {
      return nullptr;
    }
    const auto found = std::find_if(schema_->fields.begin(),
                                    schema_->fields.end(),
                                    [field](const auto& value) { return value.id == field; });
    return found == schema_->fields.end() ? nullptr : &*found;
  }

  [[nodiscard]] ParamGeneration latest_generation() const noexcept {
    return slots_ == nullptr ? ParamGeneration{0U} : slots_[latest_slot_].header.generation;
  }

  void set_published_generation_for_test(ParamGeneration generation) noexcept {
    if (slots_ == nullptr) {
      static_header_.generation = generation;
      return;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    slots_[current->slot - 1U].header.generation = generation;
  }

  void set_published_pin_count_for_test(std::uint64_t pin_count) noexcept {
    if (slots_ == nullptr) {
      return;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    slots_[current->slot - 1U].claims.store(pin_count & kPinCountMask, std::memory_order_release);
  }

  [[nodiscard]] std::uint64_t published_claim_word_for_test() const noexcept {
    if (slots_ == nullptr) {
      return 0U;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    return slots_[current->slot - 1U].claims.load(std::memory_order_acquire);
  }

  [[nodiscard]] const void* latest_value() const noexcept {
    return slots_ == nullptr ? static_header_.value : slots_[latest_slot_].header.value;
  }

  [[nodiscard]] ParamUpdateSequence latest_accepted_sequence() const noexcept {
    return latest_accepted_sequence_;
  }

  [[nodiscard]] const void* published_value() const noexcept {
    if (slots_ == nullptr) {
      return static_header_.value;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    return current == nullptr ? nullptr : current->value;
  }

  [[nodiscard]] ParamGeneration published_generation() const noexcept {
    if (slots_ == nullptr) {
      return 0U;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    return current == nullptr ? ParamGeneration{} : current->generation;
  }

  [[nodiscard]] const OperatorParameterPlanRecord& plan() const noexcept { return *plan_; }
  [[nodiscard]] std::span<const ParameterFieldPlanRecord> fields() const noexcept {
    return fields_;
  }

  [[nodiscard]] bool pending_full() const noexcept {
    return pending_capacity_ == 0U || pending_count_ == pending_capacity_;
  }

  [[nodiscard]] bool publication_pending() const noexcept {
    return publication_pending_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool lifecycle_candidate_active() const noexcept {
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      if (slots_[index].role == SlotRole::kLifecycleCandidate) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] expected<std::uint32_t, Error> prepare_candidate() {
    reclaim_retired();
    std::uint32_t selected = kInvalidSlot;
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      if (slots_[index].role == SlotRole::kFree) {
        selected = index;
        break;
      }
    }
    if (selected == kInvalidSlot) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    GenerationSlot& slot = slots_[selected];
    try {
      type_->copy_construct(const_cast<void*>(slot.header.value), latest_value());
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (derived_term_count_ != 0U) {
      std::copy_n(derived_terms_for_slot(latest_slot_),
                  derived_term_count_,
                  derived_terms_for_slot(selected));
    }
    slot.live = true;
    slot.observed.store(false, std::memory_order_relaxed);
    return selected;
  }

  void discard_candidate(std::uint32_t slot_index) noexcept {
    GenerationSlot& slot = slots_[slot_index];
    if (slot.live) {
      type_->destroy(const_cast<void*>(slot.header.value));
    }
    slot.live = false;
    slot.role = SlotRole::kFree;
    slot.update_sequence = 0U;
    slot.lifecycle_transaction = 0U;
    slot.lifecycle_receipt = 0U;
    slot.claims.store(kRetiredClaim, std::memory_order_release);
  }

  [[nodiscard]] ParamUpdateStatus validate_candidate(std::uint32_t slot_index) const noexcept {
    return validation_status(type_->validate_aggregate(slots_[slot_index].header.value));
  }

  [[nodiscard]] ParamUpdateStatus replace_candidate_value(std::uint32_t slot_index,
                                                          const ParamFieldDescriptor& field,
                                                          const ParamPatchValue& value) {
    return replace_patch_value(const_cast<void*>(slots_[slot_index].header.value), field, value);
  }

  [[nodiscard]] ParamUpdateStatus validate_derived_term_patch(
      ParamFieldId field, ParamScalarValue value) const noexcept {
    std::chrono::nanoseconds decoded{};
    for (const ParameterDerivedTermPlanRecord& term : derived_terms_) {
      if (term.field != field) {
        continue;
      }
      if (term.update_disposition != ParameterDerivedTermUpdateDisposition::kRuntimeAtomic) {
        return ParamUpdateStatus::kRequiresRecompile;
      }
      if (!value.try_as(decoded)) {
        return ParamUpdateStatus::kTypeMismatch;
      }
      if (decoded.count() < term.minimum_value_ns || decoded.count() > term.maximum_value_ns) {
        return ParamUpdateStatus::kOutOfBounds;
      }
    }
    return ParamUpdateStatus::kAccepted;
  }

  [[nodiscard]] bool replace_candidate_derived_terms(std::uint32_t slot_index, ParamFieldId field,
                                                     ParamScalarValue value) noexcept {
    std::chrono::nanoseconds decoded{};
    bool decoded_value = false;
    ParameterRuntimeDerivedTermValue* const values = derived_terms_for_slot(slot_index);
    for (std::uint32_t index = 0U; index < derived_term_count_; ++index) {
      if (derived_terms_[index].field != field) {
        continue;
      }
      if (!decoded_value) {
        decoded_value = value.try_as(decoded);
        if (!decoded_value) {
          return false;
        }
      }
      values[index].value_ns = decoded.count();
    }
    return true;
  }

  [[nodiscard]] bool candidate_equivalent_to_published(std::uint32_t slot_index) const noexcept {
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    if (current == nullptr) {
      return false;
    }
    const void* const candidate = slots_[slot_index].header.value;
    for (const ParameterFieldPlanRecord& planned : fields_) {
      if (planned.update_disposition != ParameterFieldUpdateDisposition::kAdmitted) {
        continue;
      }
      const ParamFieldDescriptor* const field = find_field(planned.field);
      if (field == nullptr || !field_semantically_equal(*field, candidate, current->value)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool candidate_field_equivalent_to_latest(std::uint32_t slot_index,
                                                          ParamFieldId field_id) const noexcept {
    const ParamFieldDescriptor* const field = find_field(field_id);
    return field != nullptr &&
           field_semantically_equal(*field, slots_[slot_index].header.value, latest_value());
  }

  void commit_candidate(std::uint32_t slot_index, ParamUpdateSequence sequence,
                        ParamGeneration generation) noexcept {
    release_no_change_lineage();
    GenerationSlot& slot = slots_[slot_index];
    slot.header.generation = generation;
    slot.update_sequence = sequence;
    slot.role = SlotRole::kPending;
    slot.version.fetch_add(1U, std::memory_order_release);
    slot.claims.store(0U, std::memory_order_release);
    const std::uint32_t tail = (pending_head_ + pending_count_) % pending_capacity_;
    pending_[tail] = PendingGeneration{.slot = slot_index, .sequence = sequence};
    ++pending_count_;
    publication_pending_.store(true, std::memory_order_release);
    latest_slot_ = slot_index;
    latest_accepted_sequence_ = sequence;
  }

  void commit_lifecycle_candidate(std::uint32_t slot_index, ParamUpdateSequence sequence,
                                  ParamGeneration generation,
                                  const ParameterLifecycleSubmitOutcome& lifecycle) noexcept {
    release_no_change_lineage();
    GenerationSlot& slot = slots_[slot_index];
    slot.header.generation = generation;
    slot.update_sequence = sequence;
    slot.lifecycle_transaction = lifecycle.transaction;
    slot.lifecycle_receipt = lifecycle.receipt.receipt_id;
    slot.role = SlotRole::kLifecycleCandidate;
    slot.version.fetch_add(1U, std::memory_order_release);
    slot.claims.store(0U, std::memory_order_release);
    latest_slot_ = slot_index;
    latest_accepted_sequence_ = sequence;
  }

  [[nodiscard]] expected<void, Error> accept_latest_without_candidate(
      ParamUpdateSequence sequence) noexcept {
    if (slots_ == nullptr) {
      return retain_no_change_lineage(0U, sequence);
    }
    refresh_latest_slot();
    return retain_no_change_lineage(latest_slot_, sequence);
  }

  [[nodiscard]] expected<void, Error> accept_published_without_candidate(
      ParamUpdateSequence sequence) noexcept {
    if (slots_ == nullptr) {
      latest_accepted_sequence_ = sequence;
      return {};
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    if (current == nullptr || current->slot == 0U || current->slot > slot_count_) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return retain_no_change_lineage(current->slot - 1U, sequence);
  }

  [[nodiscard]] expected<void, Error> retain_no_change_lineage(
      std::uint32_t slot_index, ParamUpdateSequence sequence) noexcept {
    if (slots_ == nullptr) {
      latest_accepted_sequence_ = sequence;
      return {};
    }
    if (no_change_lineage_slot_ != slot_index) {
      // A no-change acceptance owns no new generation, but latest-accepted reads must still
      // return its sequence/state with the exact immutable value it validated. Apply the same
      // retired-bit and no-wrap rule as an external generation pin before replacing the prior
      // lineage claim; rejection therefore leaves both the old lineage and sequence untouched.
      GenerationSlot& target = slots_[slot_index];
      std::uint64_t claims = target.claims.load(std::memory_order_acquire);
      for (;;) {
        if ((claims & kRetiredClaim) != 0U) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        if ((claims & kPinCountMask) == kPinCountMask) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        if (target.claims.compare_exchange_weak(
                claims, claims + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
          break;
        }
      }
      release_no_change_lineage();
      no_change_lineage_slot_ = slot_index;
    }
    latest_accepted_sequence_ = sequence;
    return {};
  }

  [[nodiscard]] GenerationSlot* lifecycle_candidate(ParamGeneration generation,
                                                    LifecycleTransactionId transaction) noexcept {
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      GenerationSlot& slot = slots_[index];
      if (slot.role == SlotRole::kLifecycleCandidate && slot.header.generation == generation &&
          slot.lifecycle_transaction == transaction) {
        return &slot;
      }
    }
    return nullptr;
  }

  [[nodiscard]] GenerationSlot* lifecycle_candidate(LifecycleTransactionId transaction) noexcept {
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      GenerationSlot& slot = slots_[index];
      if (slot.role == SlotRole::kLifecycleCandidate && slot.lifecycle_transaction == transaction) {
        return &slot;
      }
    }
    return nullptr;
  }

  [[nodiscard]] ParamUpdateSequence retire_lifecycle_candidate(
      LifecycleTransactionId transaction) noexcept {
    GenerationSlot* const candidate = lifecycle_candidate(transaction);
    if (candidate == nullptr) {
      return 0U;
    }
    const ParamUpdateSequence sequence = candidate->update_sequence;
    candidate->role = SlotRole::kRetired;
    candidate->lifecycle_transaction = 0U;
    candidate->lifecycle_receipt = 0U;
    candidate->claims.fetch_or(kRetiredClaim, std::memory_order_acq_rel);
    refresh_latest_slot();
    return sequence;
  }

  [[nodiscard]] LifecyclePublication publish_lifecycle_candidate(
      ParamGeneration generation, LifecycleTransactionId transaction) noexcept {
    GenerationSlot* const candidate = lifecycle_candidate(generation, transaction);
    if (candidate == nullptr) {
      return {};
    }
    const PinnedParameterHeader* const old_header =
        published_.exchange(&candidate->header, std::memory_order_acq_rel);
    candidate->role = SlotRole::kPublished;
    candidate->lifecycle_transaction = 0U;
    candidate->lifecycle_receipt = 0U;
    GenerationSlot& previous = slots_[old_header->slot - 1U];
    const ParamUpdateSequence retired_sequence = previous.update_sequence;
    const bool retired_was_observed = previous.observed.load(std::memory_order_acquire);
    previous.role = SlotRole::kRetired;
    previous.claims.fetch_or(kRetiredClaim, std::memory_order_acq_rel);
    refresh_latest_slot();
    return LifecyclePublication{
        .published_sequence = candidate->update_sequence,
        .retired_sequence = retired_sequence,
        .retired_was_observed = retired_was_observed,
    };
  }

  [[nodiscard]] std::optional<ParamUpdateResult> publish_next(
      ParamUpdateSequence accepted_cutoff) noexcept {
    if (pending_count_ == 0U || pending_[pending_head_].sequence > accepted_cutoff) {
      return std::nullopt;
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    if (current != nullptr) {
      const GenerationSlot& published_slot = slots_[current->slot - 1U];
      // A callback boundary may advance only after the generation selected by the preceding
      // boundary has actually been pinned and observed. Otherwise a burst of accepted updates
      // could replace an unobserved published slot, leaving its observation reservation live and
      // making the transient generation impossible for operator code to observe.
      if (published_slot.update_sequence != 0U &&
          !published_slot.observed.load(std::memory_order_acquire)) {
        return std::nullopt;
      }
    }
    const PendingGeneration next = pending_[pending_head_];
    pending_head_ = (pending_head_ + 1U) % pending_capacity_;
    --pending_count_;
    publication_pending_.store(pending_count_ != 0U, std::memory_order_release);

    GenerationSlot& candidate = slots_[next.slot];
    const PinnedParameterHeader* const old_header =
        published_.exchange(&candidate.header, std::memory_order_acq_rel);
    candidate.role = SlotRole::kPublished;
    GenerationSlot& previous = slots_[old_header->slot - 1U];
    previous.role = SlotRole::kRetired;
    previous.claims.fetch_or(kRetiredClaim, std::memory_order_acq_rel);
    refresh_latest_slot();
    return ParamUpdateResult{
        .status = ParamUpdateStatus::kAccepted,
        .state = ParamUpdateState::kPublished,
        .update_sequence = next.sequence,
        .target_generation = candidate.header.generation,
    };
  }

  template <typename CancelFunction>
  void cancel_pending(const CancelFunction& cancel) noexcept {
    if (slots_ == nullptr) {
      return;
    }
    while (pending_count_ != 0U) {
      const PendingGeneration entry = pending_[pending_head_];
      pending_head_ = (pending_head_ + 1U) % pending_capacity_;
      --pending_count_;
      GenerationSlot& slot = slots_[entry.slot];
      cancel(entry.sequence, slot.header.generation);
      // A no-change receipt may retain this exact immutable value as its
      // latest-accepted lineage. Mark the pending slot retired and defer
      // destruction until every external or coordinator-owned claim releases.
      slot.role = SlotRole::kRetired;
      slot.claims.fetch_or(kRetiredClaim, std::memory_order_acq_rel);
    }
    publication_pending_.store(false, std::memory_order_release);
    refresh_latest_slot();
  }

  template <typename CancelFunction>
  void cancel_lifecycle_candidates(const CancelFunction& cancel) noexcept {
    if (slots_ == nullptr) {
      return;
    }
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      GenerationSlot& slot = slots_[index];
      if (slot.role != SlotRole::kLifecycleCandidate) {
        continue;
      }
      cancel(slot.update_sequence, slot.header.generation, slot.lifecycle_transaction);
      slot.role = SlotRole::kRetired;
      slot.lifecycle_transaction = 0U;
      slot.lifecycle_receipt = 0U;
      slot.claims.fetch_or(kRetiredClaim, std::memory_order_acq_rel);
    }
    refresh_latest_slot();
  }

  [[nodiscard]] const GenerationSlot* slot_for_view(ParamReadView view) const noexcept {
    if (slots_ == nullptr) {
      return nullptr;
    }
    if (view == ParamReadView::kPending) {
      return slots_[latest_slot_].role == SlotRole::kPending ||
                     slots_[latest_slot_].role == SlotRole::kLifecycleCandidate
                 ? &slots_[latest_slot_]
                 : nullptr;
    }
    if (view == ParamReadView::kLatestAccepted) {
      if (no_change_lineage_slot_ != kInvalidSlot) {
        return &slots_[no_change_lineage_slot_];
      }
      return &slots_[latest_slot_];
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    return &slots_[current->slot - 1U];
  }

  [[nodiscard]] bool latest_accepted_is_no_change() const noexcept {
    return no_change_lineage_slot_ != kInvalidSlot;
  }

  [[nodiscard]] ParamReadResult published_lineage() const noexcept {
    if (slots_ == nullptr) {
      return ParamReadResult{
          .generation = 0U,
          .update_sequence = 0U,
          .state = ParamUpdateState::kPublished,
      };
    }
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    const GenerationSlot& slot = slots_[current->slot - 1U];
    return ParamReadResult{
        .generation = current->generation,
        .update_sequence = slot.update_sequence,
        .state = slot.observed.load(std::memory_order_acquire) ? ParamUpdateState::kObserved
                                                               : ParamUpdateState::kPublished,
    };
  }

  [[nodiscard]] expected<ParameterGenerationPin, Error> pin() noexcept {
    if (slots_ == nullptr) {
      return ParameterGenerationPin{
          &static_header_, this, kInvalidSlot, derived_terms_for_slot(0U), derived_term_count_};
    }
    for (;;) {
      const PinnedParameterHeader* const header = published_.load(std::memory_order_acquire);
      if (header == nullptr || header->slot == 0U || header->slot > slot_count_) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      GenerationSlot& slot = slots_[header->slot - 1U];
      const std::uint64_t version = slot.version.load(std::memory_order_acquire);
      std::uint64_t claims = slot.claims.load(std::memory_order_acquire);
      while ((claims & kRetiredClaim) == 0U) {
        if ((claims & kPinCountMask) == kPinCountMask) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        if (slot.claims.compare_exchange_weak(
                claims, claims + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
          if (published_.load(std::memory_order_acquire) == header &&
              slot.version.load(std::memory_order_acquire) == version) {
            return ParameterGenerationPin{header,
                                          this,
                                          header->slot - 1U,
                                          derived_terms_for_slot(header->slot - 1U),
                                          derived_term_count_};
          }
          slot.claims.fetch_sub(1U, std::memory_order_release);
          break;
        }
      }
    }
  }

  // Pinning is one versioned claim loop: each retry must revalidate role, generation, and claims
  // against the same slot before exposing a pointer to callback code.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  [[nodiscard]] expected<ParameterGenerationPin, Error> pin_generation(
      ParamGeneration generation) noexcept {
    if (slots_ == nullptr) {
      return generation == 0U ? expected<ParameterGenerationPin, Error>{ParameterGenerationPin{
                                    &static_header_,
                                    this,
                                    kInvalidSlot,
                                    derived_terms_for_slot(0U),
                                    derived_term_count_}}
                              : expected<ParameterGenerationPin, Error>{
                                    make_unexpected(Error{ErrorCode::kNotFound})};
    }
    for (;;) {
      GenerationSlot* selected = nullptr;
      for (std::uint32_t index = 0U; index < slot_count_; ++index) {
        if (slots_[index].live && slots_[index].header.generation == generation &&
            (slots_[index].role == SlotRole::kPublished ||
             slots_[index].role == SlotRole::kLifecycleCandidate)) {
          selected = &slots_[index];
          break;
        }
      }
      if (selected == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotFound});
      }
      const std::uint64_t version = selected->version.load(std::memory_order_acquire);
      std::uint64_t claims = selected->claims.load(std::memory_order_acquire);
      while ((claims & kRetiredClaim) == 0U) {
        if ((claims & kPinCountMask) == kPinCountMask) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        if (selected->claims.compare_exchange_weak(
                claims, claims + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
          if (selected->live && selected->header.generation == generation &&
              selected->version.load(std::memory_order_acquire) == version &&
              (selected->role == SlotRole::kPublished ||
               selected->role == SlotRole::kLifecycleCandidate)) {
            const auto slot_index = static_cast<std::uint32_t>(selected - slots_.get());
            return ParameterGenerationPin{&selected->header,
                                          this,
                                          slot_index,
                                          derived_terms_for_slot(slot_index),
                                          derived_term_count_};
          }
          selected->claims.fetch_sub(1U, std::memory_order_release);
          break;
        }
      }
    }
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  void release(std::uint32_t slot_index) noexcept {
    if (slot_index != kInvalidSlot && slot_index < slot_count_) {
      slots_[slot_index].claims.fetch_sub(1U, std::memory_order_release);
    }
  }

  [[nodiscard]] expected<std::optional<ParamUpdateSequence>, Error> pending_observation(
      const ParameterGenerationPin& pin) noexcept {
    if (pin.store_cookie_ != this || pin.header_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (pin.slot_ == kInvalidSlot) {
      return std::optional<ParamUpdateSequence>{};
    }
    if (pin.slot_ >= slot_count_) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    GenerationSlot& slot = slots_[pin.slot_];
    if (!slot.live || slot.role != SlotRole::kPublished || &slot.header != pin.header_) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (slot.observed.load(std::memory_order_acquire)) {
      return std::optional<ParamUpdateSequence>{};
    }
    if (slot.update_sequence == 0U) {
      slot.observed.store(true, std::memory_order_release);
      return std::optional<ParamUpdateSequence>{};
    }
    return std::optional<ParamUpdateSequence>{slot.update_sequence};
  }

  [[nodiscard]] ObservationStatus observation_status(
      const ParameterGenerationPin& pin) const noexcept {
    if (pin.store_cookie_ != this || pin.header_ == nullptr) {
      return ObservationStatus::kInvalid;
    }
    if (pin.slot_ == kInvalidSlot) {
      return pin.header_ == &static_header_ ? ObservationStatus::kSatisfied
                                            : ObservationStatus::kInvalid;
    }
    if (pin.slot_ >= slot_count_ || &slots_[pin.slot_].header != pin.header_ ||
        published_.load(std::memory_order_acquire) != pin.header_) {
      return ObservationStatus::kInvalid;
    }
    return slots_[pin.slot_].observed.load(std::memory_order_acquire)
               ? ObservationStatus::kSatisfied
               : ObservationStatus::kPending;
  }

  void commit_observed(const ParameterGenerationPin& pin) noexcept {
    slots_[pin.slot_].observed.store(true, std::memory_order_release);
  }

 private:
  [[nodiscard]] expected<void, Error> initialize_derived_term_storage(
      std::uint32_t generation_slots) {
    if (derived_terms_.size() > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    derived_term_count_ = static_cast<std::uint32_t>(derived_terms_.size());
    if (derived_term_count_ == 0U) {
      return {};
    }
    std::size_t value_count{};
    if (!checked_multiply(static_cast<std::size_t>(generation_slots),
                          static_cast<std::size_t>(derived_term_count_),
                          value_count)) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    derived_term_values_ = DerivedTermStorage{new ParameterRuntimeDerivedTermValue[value_count]{}};
    ParameterRuntimeDerivedTermValue* const initial = derived_terms_for_slot(0U);
    // Slot 0 owns the first `derived_term_count_` entries of the `value_count`-element buffer.
    // Bound the write to the allocation so the store stays in range for every `generation_slots`
    // (a zero count would otherwise leave `initial` pointing at an empty buffer).
    const std::size_t initial_count = std::min<std::size_t>(derived_term_count_, value_count);
    for (std::size_t index = 0U; index < initial_count; ++index) {
      const ParameterDerivedTermPlanRecord& term = derived_terms_[index];
      initial[index] = ParameterRuntimeDerivedTermValue{
          .kind = term.kind,
          .input = term.input,
          .field = term.field,
          .value_ns = term.initial_value_ns,
      };
    }
    return {};
  }

  [[nodiscard]] ParameterRuntimeDerivedTermValue* derived_terms_for_slot(
      std::uint32_t slot) noexcept {
    return derived_term_count_ == 0U
               ? nullptr
               : derived_term_values_.get() + static_cast<std::size_t>(slot) * derived_term_count_;
  }

  [[nodiscard]] const ParameterRuntimeDerivedTermValue* derived_terms_for_slot(
      std::uint32_t slot) const noexcept {
    return derived_term_count_ == 0U
               ? nullptr
               : derived_term_values_.get() + static_cast<std::size_t>(slot) * derived_term_count_;
  }

  [[nodiscard]] expected<void, Error> verify_fields() const noexcept {
    if (schema_ == nullptr) {
      return fields_.empty() ? expected<void, Error>{}
                             : make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (fields_.size() != schema_->fields.size()) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    for (const ParameterFieldPlanRecord& planned : fields_) {
      if (planned.owner.id != plan_->owner.id ||
          planned.owner.plan_generation != plan_->owner.plan_generation ||
          !planned.owner.plan_bound()) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      const ParamFieldDescriptor* const local = find_field(planned.field);
      if (local == nullptr || local->canonical_name != planned.canonical_name ||
          local->logical_type != planned.logical_type || local->mutability != planned.mutability ||
          local->earliest_invalidated_stage != planned.earliest_invalidated_stage ||
          local->persistence != planned.persistence || local->visibility != planned.visibility ||
          local->external_access != planned.external_access ||
          local->sensitivity != planned.sensitivity ||
          ((planned.value_route == ParamValueRouteDisposition::kPortableScalar) !=
           has_portable_scalar_route(*local)) ||
          local->storage.owns_dynamic_storage != planned.storage.owns_dynamic_storage ||
          local->storage.fixed_object_extent != planned.storage.fixed_object_extent ||
          local->storage.has_portable_codec != planned.storage.has_portable_codec ||
          local->storage.supports_bounded_materialization !=
              planned.storage.supports_bounded_materialization ||
          local->storage.inline_bytes != planned.storage.inline_bytes) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      if (planned.update_disposition == ParameterFieldUpdateDisposition::kAdmitted) {
        const bool optional_supported =
            !local->type_shape.optional || local->local_operations.clear_optional_member != nullptr;
        const bool value_supported = [&]() {
          switch (local->logical_type) {
            case ParamLogicalType::kBool:
            case ParamLogicalType::kSignedInteger:
            case ParamLogicalType::kUnsignedInteger:
            case ParamLogicalType::kFloatingPoint:
            case ParamLogicalType::kDuration:
              return local->local_operations.replace_member_from_scalar != nullptr;
            case ParamLogicalType::kString:
              return local->local_operations.replace_member_from_string != nullptr;
            case ParamLogicalType::kSequence:
              return local->local_operations.replace_member_from_scalar_sequence != nullptr ||
                     local->local_operations.replace_member_from_string_sequence != nullptr;
            case ParamLogicalType::kObject:
              return local->nested && local->local_operations.mutable_nested_member != nullptr &&
                     local->local_operations.nested_schema != nullptr;
            case ParamLogicalType::kFilesystemPath:
              return false;
          }
          return false;
        }();
        if (!optional_supported || !value_supported) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
      }
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> verify_derived_terms() const noexcept {
    for (std::size_t index = 0U; index < derived_terms_.size(); ++index) {
      const ParameterDerivedTermPlanRecord& term = derived_terms_[index];
      const ParameterFieldPlanRecord* const field = find_plan_field(term.field);
      const ParamFieldDescriptor* const local = find_field(term.field);
      bool input_matches = false;
      switch (term.kind) {
        case ParameterDerivedTermKind::kLatestWithinMaxAge:
          input_matches = static_cast<bool>(term.input);
          break;
        case ParameterDerivedTermKind::kContractMaxStartDelay:
          input_matches = !term.input;
          break;
      }
      ParamScalarValue initial_scalar{};
      std::chrono::nanoseconds initial_duration{};
      if (field == nullptr || term.term_id.empty() || !input_matches ||
          term.owner.id != owner_.id || term.owner.plan_generation != owner_.plan_generation ||
          term.owner.run_generation != 0U || term.field_full_id != field->full_id ||
          term.field_path != field->canonical_name || term.schema_id != plan_->schema_id ||
          term.schema_version != plan_->schema_version || term.mutability != field->mutability ||
          field->logical_type != ParamLogicalType::kDuration || local == nullptr ||
          local->local_operations.read_member_as_scalar == nullptr ||
          !local->local_operations.read_member_as_scalar(static_header_.value, initial_scalar) ||
          !initial_scalar.try_as(initial_duration) ||
          initial_duration.count() != term.initial_value_ns || term.minimum_value_ns <= 0 ||
          term.maximum_value_ns < term.minimum_value_ns ||
          term.initial_value_ns < term.minimum_value_ns ||
          term.initial_value_ns > term.maximum_value_ns) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      if (term.update_disposition == ParameterDerivedTermUpdateDisposition::kRuntimeAtomic &&
          (field->update_disposition != ParameterFieldUpdateDisposition::kAdmitted ||
           field->mutability != ParamMutability::kRuntime || !runtime_updates_enabled())) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      switch (term.update_disposition) {
        case ParameterDerivedTermUpdateDisposition::kRuntimeAtomic:
        case ParameterDerivedTermUpdateDisposition::kFrozenCompileTime:
        case ParameterDerivedTermUpdateDisposition::kRecompileRequired:
          break;
        default:
          return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      for (std::size_t prior = 0U; prior < index; ++prior) {
        if (derived_terms_[prior].term_id == term.term_id ||
            (derived_terms_[prior].kind == term.kind &&
             derived_terms_[prior].input == term.input)) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
      }
    }
    return {};
  }

  void reclaim_retired() noexcept {
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      GenerationSlot& slot = slots_[index];
      if (slot.role == SlotRole::kRetired &&
          slot.claims.load(std::memory_order_acquire) == kRetiredClaim) {
        type_->destroy(const_cast<void*>(slot.header.value));
        slot.live = false;
        slot.role = SlotRole::kFree;
        slot.update_sequence = 0U;
        slot.lifecycle_transaction = 0U;
        slot.lifecycle_receipt = 0U;
      }
    }
  }

  void refresh_latest_slot() noexcept {
    const PinnedParameterHeader* const current = published_.load(std::memory_order_acquire);
    std::uint32_t selected = current->slot - 1U;
    ParamUpdateSequence selected_sequence = slots_[selected].update_sequence;
    for (std::uint32_t index = 0U; index < slot_count_; ++index) {
      if ((slots_[index].role == SlotRole::kPending ||
           slots_[index].role == SlotRole::kLifecycleCandidate) &&
          slots_[index].update_sequence >= selected_sequence) {
        selected = index;
        selected_sequence = slots_[index].update_sequence;
      }
    }
    latest_slot_ = selected;
  }

  void release_no_change_lineage() noexcept {
    if (no_change_lineage_slot_ != kInvalidSlot) {
      slots_[no_change_lineage_slot_].claims.fetch_sub(1U, std::memory_order_release);
      no_change_lineage_slot_ = kInvalidSlot;
    }
  }

  OperatorRef owner_{};
  const OperatorParameterPlanRecord* plan_{};
  std::span<const ParameterFieldPlanRecord> fields_;
  std::span<const ParameterDerivedTermPlanRecord> derived_terms_;
  const ParamTypeDescriptor* type_{};
  const ParamSchemaDescriptor* schema_{};
  PinnedParameterHeader static_header_{};
  void* owned_static_value_{};
  std::size_t owned_static_alignment_{};
  // All generation storage is compiler-bounded and allocated once during reconstruction.
  using SlotStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<GenerationSlot[]>;
  using PendingStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<PendingGeneration[]>;
  using DerivedTermStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ParameterRuntimeDerivedTermValue[]>;
  SlotStorage slots_;
  PendingStorage pending_;
  DerivedTermStorage derived_term_values_;
  std::byte* arena_{};
  std::size_t arena_alignment_{};
  std::size_t arena_bytes_{};
  std::size_t stride_{};
  std::uint32_t slot_count_{};
  std::uint32_t derived_term_count_{};
  std::uint32_t pending_capacity_{};
  std::uint32_t pending_head_{};
  std::uint32_t pending_count_{};
  std::uint32_t latest_slot_{};
  std::uint32_t no_change_lineage_slot_{kInvalidSlot};
  ParamUpdateSequence latest_accepted_sequence_{};
  std::atomic<const PinnedParameterHeader*> published_{};
  std::atomic_bool publication_pending_{};
};

struct ParameterUpdateCoordinator::State {
  struct PatchView {
    ParamAddress target{};
    ParamPatchValue value{};
    ParamGeneration expected_generation{};
    bool has_expected_generation{};
  };

  struct ComputeIngressRequest {
    std::uint32_t patch_count{};
  };

  void materialize_close_request() noexcept {
    if (!close_requested.load(std::memory_order_acquire) || closed) {
      return;
    }
    closed = true;
    terminal_frontier.store(accepted_frontier.load(std::memory_order_acquire),
                            std::memory_order_release);
    close_materialized.store(true, std::memory_order_release);
    state_changed.notify_all();
  }

  void clear_compute_ingress() noexcept {
    const std::lock_guard lock{compute_ingress_mutex};
    compute_ingress_head = 0U;
    compute_ingress_count = 0U;
    compute_ingress_pending.store(false, std::memory_order_release);
  }

  [[nodiscard]] ParameterGenerationStore* find_store(OperatorRef owner) noexcept {
    if (owner.id == 0U || owner.id > operator_store_count) {
      return nullptr;
    }
    ParameterGenerationStore* const store = operator_stores[owner.id - 1U];
    return store != nullptr && store->owner() == owner ? store : nullptr;
  }

  [[nodiscard]] const ParameterGenerationStore* find_store(OperatorRef owner) const noexcept {
    if (owner.id == 0U || owner.id > operator_store_count) {
      return nullptr;
    }
    const ParameterGenerationStore* const store = operator_stores[owner.id - 1U];
    return store != nullptr && store->owner() == owner ? store : nullptr;
  }

  [[nodiscard]] ResultLedgerEntry* find_result(ParamUpdateSequence sequence) noexcept {
    for (std::uint32_t index = 0U; index < result_capacity; ++index) {
      if (results[index].valid && results[index].result.update_sequence == sequence) {
        return &results[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] const ResultLedgerEntry* find_result(ParamUpdateSequence sequence) const noexcept {
    for (std::uint32_t index = 0U; index < result_capacity; ++index) {
      if (results[index].valid && results[index].result.update_sequence == sequence) {
        return &results[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] bool can_record_result() const noexcept {
    return select_reusable_result_slot(static_cast<const ResultLedgerEntry*>(results.get()),
                                       result_capacity) != nullptr;
  }

  [[nodiscard]] ResultLedgerEntry* reserve_result() noexcept {
    ResultLedgerEntry* const selected = select_reusable_result_slot(results.get(), result_capacity);
    if (selected == nullptr ||
        !events.replace_accepted_transition_reservation(selected->event_reservations)) {
      return nullptr;
    }
    *selected = ResultLedgerEntry{
        .event_reservations = FixedEventLedger::kAcceptedTransitionReservation,
        .reserved = true,
    };
    return selected;
  }

  void abort_result(ResultLedgerEntry& entry) noexcept {
    events.release_accepted_transitions(entry.event_reservations);
    entry = ResultLedgerEntry{};
  }

  void commit_result(ResultLedgerEntry& selected, ParamUpdateResult result, OperatorRef owner,
                     ParamFieldId sole_field, bool single_field, bool requires_internal,
                     bool requires_sensitive, bool external_tuning_visible,
                     const ParamUpdateOptions& options, std::uint32_t patch_count) noexcept {
    const std::uint32_t reservations = selected.event_reservations;
    selected = ResultLedgerEntry{
        .result = result,
        .owner = owner,
        .sole_field = sole_field,
        .audit_reason = options.reason.id(),
        .audit_actor = options.actor_id,
        .audit_patch_count = patch_count,
        .event_reservations = reservations,
        .single_field = single_field,
        .requires_internal = requires_internal,
        .requires_sensitive = requires_sensitive,
        .external_tuning_visible = external_tuning_visible,
        .valid = true,
        .reserved = false,
        .acknowledged = false,
    };
  }

  [[nodiscard]] bool result_visible(ParamUpdateSequence sequence,
                                    const ParamAccessPolicy& policy) const noexcept {
    const ResultLedgerEntry* const entry = find_result(sequence);
    if (entry == nullptr || (entry->requires_internal && !policy.include_internal) ||
        (entry->requires_sensitive && !policy.include_sensitive) ||
        (policy.origin == ParamControlOrigin::kExternalAdapter &&
         !entry->external_tuning_visible)) {
      return false;
    }
    const ParamScopeFilter scope = policy.allowed_scope;
    if (scope == ParamScopeFilter::run()) {
      return true;
    }
    if (scope.operator_instance != entry->owner) {
      return false;
    }
    return scope.field == 0U || (entry->single_field && scope.field == entry->sole_field);
  }

  [[nodiscard]] ParamUpdateResult project_result(ResultLedgerEntry& entry) noexcept {
    const ParamUpdateResult result = entry.result;
    if (terminal_state(result.state)) {
      entry.acknowledged = true;
    }
    return result;
  }

  [[nodiscard]] ParamUpdateResult project_wait_result(ResultLedgerEntry& entry) noexcept {
    const ParamUpdateResult result = entry.result;
    // A completed wait consumes its observation even when detach, rather than a terminal state,
    // satisfied the wait predicate. Keep this distinct from non-blocking result() projection.
    entry.acknowledged = true;
    return result;
  }

  [[nodiscard]] expected<ParamUpdateResult, Error> await_terminal_result(
      std::unique_lock<std::mutex>& lock, ParamUpdateSequence sequence,
      std::chrono::nanoseconds timeout, ErrorCode missing_error) {
    const auto complete = [&] {
      const ResultLedgerEntry* const entry = find_result(sequence);
      return detached || (entry != nullptr && terminal_state(entry->result.state));
    };
    if (!complete() && (timeout == std::chrono::nanoseconds::zero() ||
                        !state_changed.wait_for(lock, timeout, complete))) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    ResultLedgerEntry* const entry = find_result(sequence);
    if (entry == nullptr) {
      return make_unexpected(Error{detached ? ErrorCode::kClosed : missing_error});
    }
    return project_wait_result(*entry);
  }

  void update_result(ParamUpdateResult result) noexcept {
    if (ResultLedgerEntry* const entry = find_result(result.update_sequence); entry != nullptr) {
      entry->result = result;
    }
  }

  [[nodiscard]] bool event_visible(const ParamAccessPolicy& policy,
                                   const ParamEvent& event) const noexcept {
    const ParamScopeFilter scope = policy.allowed_scope;
    if (scope != ParamScopeFilter::run() && scope.operator_instance != event.operator_instance) {
      return false;
    }
    if (scope.field != 0U && scope.field != event.field) {
      return false;
    }
    if (event.field == 0U) {
      return scope.field == 0U;
    }
    const ParameterGenerationStore* const store = find_store(event.operator_instance);
    const ParamFieldDescriptor* const field =
        store == nullptr ? nullptr : store->find_field(event.field);
    if (field == nullptr ||
        (field->visibility == ParamVisibility::kInternal && !policy.include_internal) ||
        (field->sensitivity == ParamSensitivity::kSensitive && !policy.include_sensitive)) {
      return false;
    }
    return policy.origin != ParamControlOrigin::kExternalAdapter ||
           field->external_access != ParamExternalAccess::kNone;
  }

  void record_transition(OperatorRef owner, const ParamUpdateResult& result,
                         const ParamUpdateOptions& options, std::uint32_t patch_count,
                         ParamFieldId field = 0U, ParamGeneration prior_generation = 0U) noexcept {
    static_cast<void>(patch_count);
    switch (result.state) {
      case ParamUpdateState::kSuperseded:
      case ParamUpdateState::kCanceled:
      case ParamUpdateState::kRolledBack:
      case ParamUpdateState::kFailedStopped:
        if (result.update_sequence != 0U &&
            (first_unpublished_terminal_sequence == 0U ||
             result.update_sequence < first_unpublished_terminal_sequence)) {
          first_unpublished_terminal_sequence = result.update_sequence;
        }
        break;
      case ParamUpdateState::kSubmitted:
      case ParamUpdateState::kRejected:
      case ParamUpdateState::kAccepted:
      case ParamUpdateState::kAcceptedNoChange:
      case ParamUpdateState::kQuiescing:
      case ParamUpdateState::kRestarting:
      case ParamUpdateState::kPublished:
      case ParamUpdateState::kObserved:
        break;
    }
    ResultLedgerEntry* const evidence = find_result(result.update_sequence);
    if (evidence == nullptr || evidence->event_reservations == 0U) {
      runtime_log_and_terminate_noexcept([] {
        HOLOSCAN_LOG_ERROR("Parameter transition lost its preaccepted result/event reservation");
      });
    }
    const bool recorded = events.append(
        ParamEvent{
            .operator_instance = owner,
            .field = field,
            .update_sequence = result.update_sequence,
            .prior_generation = prior_generation,
            .target_generation = result.target_generation,
            .state = result.state,
            .status = result.status,
            .actor_id = options.actor_id,
            .reason_id = options.reason.id(),
        },
        true,
        [this](const auto& policy, const auto& event) { return event_visible(policy, event); });
    if (!recorded) {
      runtime_log_and_terminate_noexcept([] {
        HOLOSCAN_LOG_ERROR("Parameter transition exhausted a preaccepted event reservation");
      });
    }
    --evidence->event_reservations;
    if (final_evidence_state(result.state)) {
      events.release_accepted_transitions(evidence->event_reservations);
      evidence->event_reservations = 0U;
    }
  }

  [[nodiscard]] bool record_unaccepted_event(ParamAddress address, ParamUpdateState event_state,
                                             ParamUpdateStatus status,
                                             const ParamUpdateOptions& options) noexcept {
    return events.append(
        ParamEvent{
            .operator_instance = address.operator_instance,
            .field = address.field,
            .state = event_state,
            .status = status,
            .actor_id = options.actor_id,
            .reason_id = options.reason.id(),
        },
        false,
        [this](const auto& policy, const auto& event) { return event_visible(policy, event); });
  }

  void supersede_lifecycle_candidate(ParameterGenerationStore& store,
                                     LifecycleTransactionId transaction,
                                     LifecycleTransactionId related_transaction) noexcept {
    GenerationSlot* const candidate = store.lifecycle_candidate(transaction);
    if (candidate == nullptr) {
      return;
    }
    ResultLedgerEntry* const entry = find_result(candidate->update_sequence);
    if (entry != nullptr) {
      ParamUpdateResult superseded = entry->result;
      superseded.state = ParamUpdateState::kSuperseded;
      superseded.lifecycle_transaction_id = transaction;
      superseded.coalesced_into_transaction = related_transaction;
      update_result(superseded);
      record_transition(store.owner(),
                        superseded,
                        {},
                        0U,
                        entry->single_field ? entry->sole_field : 0U,
                        store.published_generation());
    }
    static_cast<void>(store.retire_lifecycle_candidate(transaction));
  }

  void terminalize_unpublished() noexcept {
    close_requested.store(true, std::memory_order_release);
    materialize_close_request();
    clear_compute_ingress();
    for (std::uint32_t index = 0U; index < store_count; ++index) {
      ParameterGenerationStore& store = stores[index];
      store.cancel_pending([&](ParamUpdateSequence sequence, ParamGeneration generation) {
        const ParamUpdateResult canceled{
            .status = ParamUpdateStatus::kClosed,
            .state = ParamUpdateState::kCanceled,
            .update_sequence = sequence,
            .target_generation = generation,
        };
        ResultLedgerEntry* const entry = find_result(sequence);
        update_result(canceled);
        record_transition(store.owner(),
                          canceled,
                          {},
                          0U,
                          entry != nullptr && entry->single_field ? entry->sole_field : 0U);
      });
      store.cancel_lifecycle_candidates([&](ParamUpdateSequence sequence,
                                            ParamGeneration generation,
                                            LifecycleTransactionId transaction) {
        const ParamUpdateResult canceled{
            .status = ParamUpdateStatus::kClosed,
            .state = ParamUpdateState::kCanceled,
            .update_sequence = sequence,
            .target_generation = generation,
            .lifecycle_transaction_id = transaction,
        };
        ResultLedgerEntry* const entry = find_result(sequence);
        update_result(canceled);
        record_transition(store.owner(),
                          canceled,
                          {},
                          0U,
                          entry != nullptr && entry->single_field ? entry->sole_field : 0U,
                          store.published_generation());
      });
    }
    state_changed.notify_all();
  }

  template <typename PatchAccessor>
  // Submission validates and reserves all finite storage before committing one generation and its
  // audit evidence. Keeping the mutex-held rollback paths together prevents partial publication.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  [[nodiscard]] ParamBatchUpdateResult submit_locked(std::size_t patch_count,
                                                     PatchAccessor patch_at,
                                                     ParamUpdateOptions options) {
    ParamBatchUpdateResult outcome;
    if (patch_count <= ParamBatchUpdateResult::kNoFailedPatch) {
      outcome.patch_count = static_cast<std::uint32_t>(patch_count);
    }
    if (!valid_audit(options.audit)) {
      return outcome;
    }
    materialize_close_request();
    if (detached || closed) {
      outcome.status = ParamUpdateStatus::kClosed;
      return outcome;
    }
    if (patch_count == 0U || patch_count > ParamBatchUpdateResult::kNoFailedPatch) {
      return outcome;
    }

    const PatchView first = patch_at(0U);
    ParameterGenerationStore* const store = find_store(first.target.operator_instance);
    if (store == nullptr) {
      outcome.status = ParamUpdateStatus::kStaleOperatorRef;
      outcome.failed_patch = 0U;
      return outcome;
    }
    if (!store->updates_enabled()) {
      outcome.status = ParamUpdateStatus::kUpdatesDisabled;
      outcome.failed_patch = 0U;
      return outcome;
    }
    if (options.audit == ParamAudit::kDetailed &&
        !profile_allows_detailed_parameter_audit(store->plan().effective_profile)) {
      outcome.status = ParamUpdateStatus::kUnsupportedInProfile;
      outcome.failed_patch = 0U;
      return outcome;
    }
    if (!can_record_result()) {
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }

    const ParamGeneration latest_generation = store->latest_generation();
    bool no_change = true;
    bool requires_candidate_comparison = false;
    bool lifecycle_update = false;
    bool runtime_update = false;
    LifecycleStage earliest_lifecycle_stage = LifecycleStage::kStart;
    bool requires_internal = false;
    bool requires_sensitive = false;
    bool external_tuning_visible = true;
    for (std::size_t index = 0U; index < patch_count; ++index) {
      const PatchView patch = patch_at(index);
      if (patch.target.operator_instance != first.target.operator_instance) {
        outcome.status = ParamUpdateStatus::kUnsupportedAtomicScope;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
      for (std::size_t prior = 0U; prior < index; ++prior) {
        if (patch_at(prior).target.field == patch.target.field) {
          outcome.status = ParamUpdateStatus::kRejected;
          outcome.failed_patch = static_cast<std::uint32_t>(index);
          return outcome;
        }
      }
      if (patch.has_expected_generation && patch.expected_generation != latest_generation) {
        outcome.status = ParamUpdateStatus::kExpectedGenerationMismatch;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
      const ParameterFieldPlanRecord* const planned = store->find_plan_field(patch.target.field);
      const ParamFieldDescriptor* const local = store->find_field(patch.target.field);
      if (planned == nullptr || local == nullptr) {
        outcome.status = ParamUpdateStatus::kUnknownParameter;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
      outcome.status = disposition_status(planned->update_disposition);
      if (outcome.status != ParamUpdateStatus::kAccepted) {
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
      if (planned->mutability == ParamMutability::kLifecycle) {
        lifecycle_update = true;
        if (static_cast<std::uint8_t>(planned->earliest_invalidated_stage) <
            static_cast<std::uint8_t>(earliest_lifecycle_stage)) {
          earliest_lifecycle_stage = planned->earliest_invalidated_stage;
        }
      } else if (planned->mutability == ParamMutability::kRuntime) {
        runtime_update = true;
      }
      requires_internal = requires_internal || local->visibility == ParamVisibility::kInternal;
      requires_sensitive = requires_sensitive || local->sensitivity == ParamSensitivity::kSensitive;
      external_tuning_visible =
          external_tuning_visible && local->external_access == ParamExternalAccess::kTuning;
      const auto scalar = patch_scalar_value(*local, patch.value);
      if (scalar) {
        ParamScalarValue current{};
        if (!local->local_operations.read_member_as_scalar(store->latest_value(), current)) {
          if (!local->type_shape.optional) {
            outcome.status = ParamUpdateStatus::kTypeMismatch;
            outcome.failed_patch = static_cast<std::uint32_t>(index);
            return outcome;
          }
        } else if (scalar_semantically_equal(current, *scalar)) {
          continue;
        }
        const ParamUpdateStatus derived_status =
            store->validate_derived_term_patch(patch.target.field, *scalar);
        if (derived_status != ParamUpdateStatus::kAccepted) {
          outcome.status = derived_status;
          outcome.failed_patch = static_cast<std::uint32_t>(index);
          return outcome;
        }
      } else if (patch.value.kind() == ParamPatchValueKind::kUnset) {
        outcome.status = ParamUpdateStatus::kTypeMismatch;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      } else {
        requires_candidate_comparison = true;
        continue;
      }
      no_change = false;
    }

    if (lifecycle_update) {
      if (!store->lifecycle_updates_enabled() || options.effective.has_value() ||
          lifecycle_submitter.state == nullptr || lifecycle_submitter.submit == nullptr ||
          lifecycle_submitter.cancel_accepted == nullptr) {
        outcome.status = ParamUpdateStatus::kRequiresLifecycleBarrier;
        return outcome;
      }
    } else if (!runtime_update || !store->runtime_updates_enabled()) {
      outcome.status = ParamUpdateStatus::kUpdatesDisabled;
      return outcome;
    } else if (store->lifecycle_candidate_active() || store->pending_full()) {
      // Once a lifecycle candidate exists, cloning it into a NextCompute slot would expose a
      // lifecycle-only field before successful kStart. Keep ordinary ingress closed for this owner
      // until the transaction terminalizes; pre-existing runtime candidates may still drain.
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }

    if (next_sequence == std::numeric_limits<ParamUpdateSequence>::max()) {
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }
    const ParamUpdateSequence sequence = next_sequence + 1U;
    const auto accept_runtime_no_change = [&]() {
      ResultLedgerEntry* const evidence = reserve_result();
      if (evidence == nullptr) {
        outcome.status = ParamUpdateStatus::kQueueFull;
        return;
      }
      if (auto retained = store->accept_latest_without_candidate(sequence); !retained) {
        abort_result(*evidence);
        outcome.status = retained.error().code == ErrorCode::kResourceExhausted
                             ? ParamUpdateStatus::kQueueFull
                             : ParamUpdateStatus::kRejected;
        return;
      }
      const ParamUpdateResult result{
          .status = ParamUpdateStatus::kAcceptedNoChange,
          .state = ParamUpdateState::kAcceptedNoChange,
          .update_sequence = sequence,
          .target_generation = latest_generation,
      };
      commit_result(*evidence,
                    result,
                    store->owner(),
                    first.target.field,
                    patch_count == 1U,
                    requires_internal,
                    requires_sensitive,
                    external_tuning_visible,
                    options,
                    outcome.patch_count);
      next_sequence = sequence;
      accepted_frontier.store(sequence, std::memory_order_release);
      record_transition(store->owner(),
                        result,
                        options,
                        outcome.patch_count,
                        patch_count == 1U ? first.target.field : 0U,
                        latest_generation);
      outcome.status = result.status;
      outcome.committed = true;
      outcome.update_sequence = sequence;
      outcome.target_generation = latest_generation;
      state_changed.notify_all();
    };
    if (no_change && !lifecycle_update && !requires_candidate_comparison) {
      accept_runtime_no_change();
      return outcome;
    }
    if (latest_generation == std::numeric_limits<ParamGeneration>::max()) {
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }

    auto candidate = store->prepare_candidate();
    if (!candidate) {
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }
    const std::uint32_t candidate_slot = *candidate;
    for (std::size_t index = 0U; index < patch_count; ++index) {
      const PatchView patch = patch_at(index);
      const ParamFieldDescriptor* const local = store->find_field(patch.target.field);
      ParamUpdateStatus replacement_status{ParamUpdateStatus::kRejected};
      try {
        replacement_status = store->replace_candidate_value(candidate_slot, *local, patch.value);
      } catch (const std::bad_alloc&) {
        replacement_status = ParamUpdateStatus::kQueueFull;
      } catch (...) {
        replacement_status = ParamUpdateStatus::kRejected;
      }
      if (replacement_status != ParamUpdateStatus::kAccepted) {
        store->discard_candidate(candidate_slot);
        outcome.status = replacement_status;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
      const auto scalar = patch_scalar_value(*local, patch.value);
      if (scalar &&
          !store->replace_candidate_derived_terms(candidate_slot, patch.target.field, *scalar)) {
        store->discard_candidate(candidate_slot);
        outcome.status = ParamUpdateStatus::kTypeMismatch;
        outcome.failed_patch = static_cast<std::uint32_t>(index);
        return outcome;
      }
    }
    const ParamUpdateStatus validated = store->validate_candidate(candidate_slot);
    if (validated != ParamUpdateStatus::kAccepted) {
      store->discard_candidate(candidate_slot);
      outcome.status = validated;
      return outcome;
    }
    if (no_change && !lifecycle_update) {
      for (std::size_t index = 0U; index < patch_count; ++index) {
        if (!store->candidate_field_equivalent_to_latest(candidate_slot,
                                                         patch_at(index).target.field)) {
          no_change = false;
          break;
        }
      }
      if (no_change) {
        store->discard_candidate(candidate_slot);
        accept_runtime_no_change();
        return outcome;
      }
    }

    ResultLedgerEntry* const evidence = reserve_result();
    if (evidence == nullptr) {
      store->discard_candidate(candidate_slot);
      outcome.status = ParamUpdateStatus::kQueueFull;
      return outcome;
    }

    const ParamGeneration generation = latest_generation + 1U;
    ParameterLifecycleSubmitOutcome lifecycle;
    if (lifecycle_update) {
      auto submitted = lifecycle_submitter.submit(
          lifecycle_submitter.state,
          ParameterLifecycleSubmitRequest{
              .owner = store->owner(),
              .generation = generation,
              .earliest_stage = earliest_lifecycle_stage,
              .equivalent_to_published = store->candidate_equivalent_to_published(candidate_slot),
          });
      if (!submitted) {
        abort_result(*evidence);
        store->discard_candidate(candidate_slot);
        outcome.status = lifecycle_submission_status(submitted.error().code);
        return outcome;
      }
      lifecycle = *submitted;
      if (lifecycle.receipt.receipt_id == 0U || lifecycle.receipt.target != store->owner() ||
          (lifecycle.state != LifecycleTransactionState::kAcceptedNoChange &&
           lifecycle.state != LifecycleTransactionState::kQueued) ||
          (lifecycle.state == LifecycleTransactionState::kAcceptedNoChange &&
           lifecycle.transaction != 0U) ||
          (lifecycle.state == LifecycleTransactionState::kQueued && lifecycle.transaction == 0U)) {
        // The shared executor must never return partial or structurally inconsistent acceptance.
        // Its submit callback promises atomic rejection, so a malformed success is a run-internal
        // contract failure and the Parameter candidate remains unpublished.
        lifecycle_submitter.cancel_accepted(lifecycle_submitter.state, lifecycle);
        abort_result(*evidence);
        store->discard_candidate(candidate_slot);
        outcome.status = ParamUpdateStatus::kRejected;
        return outcome;
      }
    }

    const bool accepted_no_change =
        lifecycle_update && lifecycle.state == LifecycleTransactionState::kAcceptedNoChange;
    if (accepted_no_change) {
      if (auto retained = store->accept_published_without_candidate(sequence); !retained) {
        lifecycle_submitter.cancel_accepted(lifecycle_submitter.state, lifecycle);
        abort_result(*evidence);
        store->discard_candidate(candidate_slot);
        outcome.status = retained.error().code == ErrorCode::kResourceExhausted
                             ? ParamUpdateStatus::kQueueFull
                             : ParamUpdateStatus::kRejected;
        return outcome;
      }
    }
    const ParamUpdateResult result{
        .status = accepted_no_change ? ParamUpdateStatus::kAcceptedNoChange
                                     : ParamUpdateStatus::kAccepted,
        .state =
            accepted_no_change ? ParamUpdateState::kAcceptedNoChange : ParamUpdateState::kAccepted,
        .update_sequence = sequence,
        .target_generation = accepted_no_change ? store->published_generation() : generation,
        .lifecycle_transaction_id = lifecycle.transaction,
        .coalesced_into_transaction = lifecycle.coalesced_into,
    };
    commit_result(*evidence,
                  result,
                  store->owner(),
                  first.target.field,
                  patch_count == 1U,
                  requires_internal,
                  requires_sensitive,
                  external_tuning_visible,
                  options,
                  outcome.patch_count);
    if (lifecycle_update && lifecycle.superseded_transaction != 0U) {
      supersede_lifecycle_candidate(*store,
                                    lifecycle.superseded_transaction,
                                    lifecycle.coalesced_into != 0U
                                        ? lifecycle.coalesced_into
                                        : lifecycle.superseded_transaction);
    }
    if (lifecycle_update && lifecycle.state == LifecycleTransactionState::kQueued) {
      supersede_lifecycle_candidate(*store, lifecycle.transaction, lifecycle.transaction);
    }
    if (accepted_no_change) {
      store->discard_candidate(candidate_slot);
    } else if (lifecycle_update) {
      store->commit_lifecycle_candidate(candidate_slot, sequence, generation, lifecycle);
    } else {
      store->commit_candidate(candidate_slot, sequence, generation);
    }
    next_sequence = sequence;
    accepted_frontier.store(sequence, std::memory_order_release);
    record_transition(store->owner(),
                      result,
                      options,
                      outcome.patch_count,
                      patch_count == 1U ? first.target.field : 0U,
                      latest_generation);
    outcome.status = result.status;
    outcome.committed = true;
    outcome.update_sequence = sequence;
    outcome.target_generation = result.target_generation;
    return outcome;
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  std::mutex mutex;
  std::condition_variable state_changed;
  std::mutex compute_ingress_mutex;
  // These queues are exact compiler-admitted capacities and never grow after reconstruction.
  using ComputeIngressStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ComputeIngressRequest[]>;
  using ComputePatchStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ParamInlinePatch[]>;
  using StoreStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ParameterGenerationStore[]>;
  using OperatorStorePointer = ParameterGenerationStore*;
  using OperatorStoreStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<OperatorStorePointer[]>;
  using ResultStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<ResultLedgerEntry[]>;
  ComputeIngressStorage compute_ingress_requests;
  ComputePatchStorage compute_ingress_patches;
  std::uint32_t compute_ingress_capacity{};
  std::uint32_t max_compute_patches_per_request{};
  std::uint32_t compute_ingress_head{};
  std::uint32_t compute_ingress_count{};
  std::atomic_bool compute_ingress_pending{};
  StoreStorage stores;
  std::uint32_t store_count{};
  OperatorStoreStorage operator_stores;
  std::uint32_t operator_store_count{};
  ResultStorage results;
  std::uint32_t result_capacity{};
  std::uint32_t max_event_cursors{};
  ParameterLifecycleSubmitBinding lifecycle_submitter{};
  FixedEventLedger events;
  PlanDigest plan_digest{};
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  ParameterCatalogView catalog;
  // This advances at acceptance. close_ingress freezes its current value as terminal N; detach
  // then terminalizes every still-pending sequence at or below that accepted frontier.
  std::atomic<ParamUpdateSequence> accepted_frontier{};
  std::atomic<ParamUpdateSequence> terminal_frontier{};
  ParamUpdateSequence first_unpublished_terminal_sequence{};
  std::atomic_bool close_requested{};
  std::atomic_bool close_materialized{};
  ParamUpdateSequence next_sequence{};
  bool closed{};
  std::atomic_bool detached{};
  bool run_terminalized{};
};

ParameterGenerationPin::ParameterGenerationPin(ParameterGenerationPin&& other) noexcept
    : header_(std::exchange(other.header_, nullptr)),
      store_cookie_(std::exchange(other.store_cookie_, nullptr)),
      slot_(std::exchange(other.slot_, 0U)),
      derived_terms_(std::exchange(other.derived_terms_, nullptr)),
      derived_term_count_(std::exchange(other.derived_term_count_, 0U)) {}

ParameterGenerationPin& ParameterGenerationPin::operator=(ParameterGenerationPin&& other) noexcept {
  if (this != &other) {
    if (header_ != nullptr && store_cookie_ != nullptr) {
      static_cast<ParameterGenerationStore*>(store_cookie_)->release(slot_);
    }
    header_ = std::exchange(other.header_, nullptr);
    store_cookie_ = std::exchange(other.store_cookie_, nullptr);
    slot_ = std::exchange(other.slot_, 0U);
    derived_terms_ = std::exchange(other.derived_terms_, nullptr);
    derived_term_count_ = std::exchange(other.derived_term_count_, 0U);
  }
  return *this;
}

std::optional<std::int64_t> ParameterGenerationPin::duration_term(ParameterDerivedTermKind kind,
                                                                  PortId input) const noexcept {
  if (header_ == nullptr) {
    return std::nullopt;
  }
  for (std::uint32_t index = 0U; index < derived_term_count_; ++index) {
    if (derived_terms_[index].kind == kind && derived_terms_[index].input == input) {
      return derived_terms_[index].value_ns;
    }
  }
  return std::nullopt;
}

std::optional<std::int64_t> ParameterGenerationPin::duration_term_at(std::uint32_t index,
                                                                     ParameterDerivedTermKind kind,
                                                                     PortId input) const noexcept {
  if (header_ == nullptr || index >= derived_term_count_ || derived_terms_[index].kind != kind ||
      derived_terms_[index].input != input) {
    return std::nullopt;
  }
  return derived_terms_[index].value_ns;
}

ParameterUpdateCoordinator::~ParameterUpdateCoordinator() = default;

// Reconstruction validates every compiler-derived capacity and allocates the complete bounded
// store before the coordinator becomes observable.
// NOLINTBEGIN(readability-function-cognitive-complexity)
expected<std::unique_ptr<ParameterUpdateCoordinator>, Error> ParameterUpdateCoordinator::create(
    std::span<const ParameterRuntimeOperatorBinding> bindings,
    ParameterRuntimeCapacityPlanRecord capacity, std::uint32_t operator_slot_count) {
  constexpr std::uint32_t kLifecycleCandidateSlots = 2U;
  const bool updates_enabled =
      capacity.runtime_updates_enabled || capacity.lifecycle_updates_enabled;
  const std::uint32_t lifecycle_slots =
      capacity.lifecycle_updates_enabled ? kLifecycleCandidateSlots : 0U;
  if (capacity.max_pending_updates_per_operator >
      std::numeric_limits<std::uint32_t>::max() - 2U - lifecycle_slots) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  const auto& lifecycle = capacity.required_lifecycle_capacity;
  const bool lifecycle_capacity_empty =
      lifecycle.max_pending_requests == 0U && lifecycle.max_pending_transactions == 0U &&
      lifecycle.max_receipts == 0U && lifecycle.max_fenced_operators == 0U &&
      lifecycle.max_candidate_generations == 0U && lifecycle.max_events == 0U &&
      lifecycle.max_audit_records == 0U && lifecycle.max_quiesce_records == 0U &&
      lifecycle.max_rollback_records == 0U;
  const bool active_capacity_invalid =
      updates_enabled && (capacity.enabled_operator_count == 0U || capacity.audit_capacity == 0U ||
                          capacity.event_capacity == 0U || capacity.max_event_cursors == 0U ||
                          capacity.max_pending_updates_per_operator == 0U ||
                          capacity.generation_slots_per_operator <
                              capacity.max_pending_updates_per_operator + 2U + lifecycle_slots);
  const bool inactive_capacity_invalid =
      !updates_enabled &&
      (capacity.enabled_operator_count != 0U ||
       capacity.resolved_update_policy != ParamRuntimeUpdatePolicy::kDisabled ||
       capacity.max_pending_updates_per_operator != 0U ||
       capacity.generation_slots_per_operator != 0U || capacity.compute_ingress_capacity != 0U ||
       capacity.max_compute_patches_per_request != 0U || capacity.audit_capacity != 0U ||
       capacity.event_capacity != 0U || capacity.max_event_cursors != 0U ||
       capacity.parameter_arena_bytes != 0U || capacity.max_arena_alignment != 0U ||
       !lifecycle_capacity_empty);
  if (active_capacity_invalid || inactive_capacity_invalid ||
      (capacity.runtime_updates_enabled && (capacity.compute_ingress_capacity == 0U ||
                                            capacity.max_compute_patches_per_request == 0U)) ||
      (!capacity.runtime_updates_enabled && (capacity.compute_ingress_capacity != 0U ||
                                             capacity.max_compute_patches_per_request != 0U)) ||
      bindings.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  if (operator_slot_count == 0U) {
    operator_slot_count = static_cast<std::uint32_t>(bindings.size());
  }
  if (bindings.size() > operator_slot_count) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  const bool projected_inventory = bindings.size() != operator_slot_count;
  try {
    auto coordinator = std::unique_ptr<ParameterUpdateCoordinator>{new ParameterUpdateCoordinator};
    auto state = std::make_unique<State>();
    state->store_count = static_cast<std::uint32_t>(bindings.size());
    state->stores = State::StoreStorage{new ParameterGenerationStore[state->store_count]{}};
    state->operator_store_count = operator_slot_count;
    state->operator_stores =
        State::OperatorStoreStorage{new State::OperatorStorePointer[state->operator_store_count]{}};
    state->result_capacity = capacity.audit_capacity;
    state->max_event_cursors = capacity.max_event_cursors;
    if (state->result_capacity != 0U) {
      state->results = State::ResultStorage{new ResultLedgerEntry[state->result_capacity]{}};
      state->events.initialize(capacity.event_capacity, capacity.max_event_cursors);
    }
    state->catalog.operators_.reserve(bindings.size());
    std::size_t catalog_field_count{};
    for (const auto& binding : bindings) {
      if (binding.fields.size() > std::numeric_limits<std::uint32_t>::max()) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      if (binding.fields.size() > std::numeric_limits<std::size_t>::max() - catalog_field_count) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      catalog_field_count += binding.fields.size();
    }
    state->catalog.fields_.reserve(catalog_field_count);
    if (!bindings.empty()) {
      state->plan_digest = bindings.front().plan_digest;
      state->plan_generation = bindings.front().run_bound_owner.plan_generation;
      state->run_generation = bindings.front().run_bound_owner.run_generation;
    }
    std::size_t actual_arena_bytes{};
    std::uint32_t enabled_operator_count{};
    std::uint32_t runtime_update_operator_count{};
    std::uint32_t max_compute_patches_per_request{};
    for (std::uint32_t index = 0U; index < state->store_count; ++index) {
      const OperatorInstanceId owner_id = bindings[index].run_bound_owner.id;
      if (owner_id == 0U || owner_id > state->operator_store_count ||
          state->operator_stores[owner_id - 1U] != nullptr) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      if (bindings[index].plan_digest != state->plan_digest ||
          bindings[index].run_bound_owner.plan_generation != state->plan_generation ||
          bindings[index].run_bound_owner.run_generation != state->run_generation ||
          state->plan_generation == 0U || state->run_generation == 0U) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      auto initialized = state->stores[index].initialize(bindings[index], capacity);
      if (!initialized) {
        return make_unexpected(std::move(initialized.error()));
      }
      state->operator_stores[owner_id - 1U] = &state->stores[index];
      if (state->stores[index].arena_bytes() >
          std::numeric_limits<std::size_t>::max() - actual_arena_bytes) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      actual_arena_bytes += state->stores[index].arena_bytes();
      const auto& binding = bindings[index];
      std::vector<std::string> operator_segments = binding.operator_path;
      if (operator_segments.empty()) {
        operator_segments.push_back("operator-" + std::to_string(binding.run_bound_owner.id));
      }
      auto operator_path = OperatorPath::from_segments(operator_segments);
      if (!operator_path) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      state->catalog.operators_.push_back(OperatorParamRecord{
          .operator_instance = binding.run_bound_owner,
          .path = std::move(operator_path).value(),
          .effective_profile = binding.plan->effective_profile,
          .schema_availability = binding.plan->schema_availability,
          .schema_disposition = binding.plan->schema_disposition,
          .materialization = binding.plan->materialization,
          .schema_id = binding.plan->schema_id,
          .schema_version = binding.plan->schema_version,
          .structural_hash = binding.plan->structural_hash,
          .contract_hash = binding.plan->contract_hash,
          .metadata_hash = binding.plan->metadata_hash,
          .schema_coverage = binding.plan->schema_coverage,
          .schema_proof = binding.plan->schema_proof,
      });
      for (const ParameterFieldPlanRecord& planned : binding.fields) {
        auto field_path = ParamPath::parse(planned.canonical_name);
        if (!field_path) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        state->catalog.fields_.push_back(ParamFieldCatalogRecord{
            .instance = planned.instance,
            .instance_identity_digest = planned.instance_identity_digest,
            .address =
                ParamAddress{
                    .operator_instance = binding.run_bound_owner,
                    .field = planned.field,
                },
            .full_id = planned.full_id,
            .path = std::move(field_path).value(),
            .logical_type = planned.logical_type,
            .type_shape = planned.type_shape,
            .mutability = planned.mutability,
            .earliest_invalidated_stage = planned.earliest_invalidated_stage,
            .persistence = planned.persistence,
            .visibility = planned.visibility,
            .external_access = planned.external_access,
            .sensitivity = planned.sensitivity,
            .required_on_load = planned.required_on_load,
            .alias = planned.alias,
            .validator_id = planned.validator_id,
            .constraints = planned.constraints,
            .enum_choices = planned.enum_choices,
            .string_encoding = planned.string_encoding,
            .element_string_encoding = planned.element_string_encoding,
            .value_route = planned.value_route,
            .storage = planned.storage,
            .materialization = planned.materialization,
            .projection = planned.projection,
            .update_disposition = planned.update_disposition,
            .adapter_eligible = planned.adapter_eligible,
        });
      }
      if (state->stores[index].updates_enabled()) {
        ++enabled_operator_count;
      }
      if (state->stores[index].runtime_updates_enabled()) {
        ++runtime_update_operator_count;
        const auto admitted_field_count = std::count_if(
            bindings[index].fields.begin(), bindings[index].fields.end(), [](const auto& field) {
              return field.update_disposition == ParameterFieldUpdateDisposition::kAdmitted &&
                     field.mutability == ParamMutability::kRuntime;
            });
        if (std::cmp_greater(admitted_field_count, std::numeric_limits<std::uint32_t>::max())) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        const auto admitted_fields = static_cast<std::uint32_t>(admitted_field_count);
        max_compute_patches_per_request =
            std::max(max_compute_patches_per_request, admitted_fields);
      }
    }
    state->catalog.digest_ = runtime_parameter_catalog_digest(state->plan_generation,
                                                              state->run_generation,
                                                              state->catalog.operators_,
                                                              state->catalog.fields_);
    if ((!projected_inventory && actual_arena_bytes != capacity.parameter_arena_bytes) ||
        (projected_inventory && actual_arena_bytes > capacity.parameter_arena_bytes)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if ((!projected_inventory && enabled_operator_count != capacity.enabled_operator_count) ||
        (projected_inventory && enabled_operator_count > capacity.enabled_operator_count)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (capacity.runtime_updates_enabled) {
      const bool local_runtime_capacity_invalid =
          runtime_update_operator_count != 0U &&
          (max_compute_patches_per_request == 0U ||
           runtime_update_operator_count > std::numeric_limits<std::uint32_t>::max() /
                                               capacity.max_pending_updates_per_operator ||
           capacity.compute_ingress_capacity <
               runtime_update_operator_count * capacity.max_pending_updates_per_operator ||
           capacity.max_compute_patches_per_request < max_compute_patches_per_request);
      const bool exact_runtime_capacity_invalid =
          !projected_inventory &&
          (runtime_update_operator_count == 0U ||
           capacity.compute_ingress_capacity !=
               runtime_update_operator_count * capacity.max_pending_updates_per_operator ||
           capacity.max_compute_patches_per_request != max_compute_patches_per_request);
      if (local_runtime_capacity_invalid || exact_runtime_capacity_invalid) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      std::size_t patch_storage_count{};
      if (!checked_multiply(capacity.compute_ingress_capacity,
                            capacity.max_compute_patches_per_request,
                            patch_storage_count)) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      state->compute_ingress_requests = State::ComputeIngressStorage{
          new State::ComputeIngressRequest[capacity.compute_ingress_capacity]{}};
      state->compute_ingress_patches =
          State::ComputePatchStorage{new ParamInlinePatch[patch_storage_count]{}};
      state->compute_ingress_capacity = capacity.compute_ingress_capacity;
      state->max_compute_patches_per_request = capacity.max_compute_patches_per_request;
    } else if (runtime_update_operator_count != 0U || max_compute_patches_per_request != 0U) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    coordinator->state_ = std::move(state);
    return coordinator;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}
// NOLINTEND(readability-function-cognitive-complexity)

expected<void, Error> ParameterUpdateCoordinator::bind_lifecycle_submitter(
    ParameterLifecycleSubmitBinding binding) noexcept {
  if (binding.state == nullptr || binding.submit == nullptr || binding.cancel_accepted == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::lock_guard lock{state_->mutex};
  if (state_->lifecycle_submitter.state != nullptr ||
      state_->lifecycle_submitter.submit != nullptr ||
      state_->lifecycle_submitter.cancel_accepted != nullptr || state_->detached ||
      state_->closed) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  state_->lifecycle_submitter = binding;
  return {};
}

ParamBatchUpdateResult ParameterUpdateCoordinator::submit(std::span<const ParamPatch> patches,
                                                          ParamUpdateOptions options) {
  const std::lock_guard lock{state_->mutex};
  if (!patches.empty() && !state_->events.can_append_unreserved(2U)) {
    return ParamBatchUpdateResult{
        .status = ParamUpdateStatus::kQueueFull,
        .committed = false,
        .patch_count = patches.size() <= ParamBatchUpdateResult::kNoFailedPatch
                           ? static_cast<std::uint32_t>(patches.size())
                           : 0U,
    };
  }
  const ParamAddress event_target =
      patches.empty() ? ParamAddress{}
      : patches.size() == 1U
          ? patches.front().target
          : ParamAddress{.operator_instance = patches.front().target.operator_instance};
  if (!patches.empty() &&
      !state_->record_unaccepted_event(
          event_target, ParamUpdateState::kSubmitted, ParamUpdateStatus::kSubmitted, options)) {
    return ParamBatchUpdateResult{.status = ParamUpdateStatus::kQueueFull, .committed = false};
  }
  ParamBatchUpdateResult result = state_->submit_locked(
      patches.size(),
      [&](std::size_t index) {
        const ParamPatch& patch = patches[index];
        return State::PatchView{
            .target = patch.target,
            .value = patch.value,
            .expected_generation = patch.expected_generation.value_or(0U),
            .has_expected_generation = patch.expected_generation.has_value(),
        };
      },
      options);
  if (!result.committed && !patches.empty()) {
    static_cast<void>(state_->record_unaccepted_event(
        event_target, ParamUpdateState::kRejected, result.status, options));
  }
  return result;
}

ParamSubmitResult ParameterUpdateCoordinator::try_submit_from_compute(
    const ParamInlinePatch* patches, std::size_t patch_count) noexcept {
  if (patches == nullptr || patch_count == 0U) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kRejected,
        .state = ParamUpdateState::kRejected,
    };
  }
  if (state_->close_requested.load(std::memory_order_acquire)) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kClosed,
        .state = ParamUpdateState::kRejected,
    };
  }
  if (patch_count > state_->max_compute_patches_per_request) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kOutOfBounds,
        .state = ParamUpdateState::kRejected,
    };
  }
  const std::unique_lock lock{state_->compute_ingress_mutex, std::try_to_lock};
  if (!lock.owns_lock()) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kQueueFull,
        .state = ParamUpdateState::kRejected,
    };
  }
  if (state_->close_requested.load(std::memory_order_acquire)) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kClosed,
        .state = ParamUpdateState::kRejected,
    };
  }
  if (state_->compute_ingress_count == state_->compute_ingress_capacity) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kQueueFull,
        .state = ParamUpdateState::kRejected,
    };
  }
  const std::uint32_t request_index =
      (state_->compute_ingress_head + state_->compute_ingress_count) %
      state_->compute_ingress_capacity;
  ParamInlinePatch* const destination =
      state_->compute_ingress_patches.get() +
      static_cast<std::size_t>(request_index) * state_->max_compute_patches_per_request;
  std::copy_n(patches, patch_count, destination);
  state_->compute_ingress_requests[request_index].patch_count =
      static_cast<std::uint32_t>(patch_count);
  ++state_->compute_ingress_count;
  state_->compute_ingress_pending.store(true, std::memory_order_release);
  return ParamSubmitResult{
      .status = ParamUpdateStatus::kSubmitted,
      .state = ParamUpdateState::kSubmitted,
  };
}

std::size_t ParameterUpdateCoordinator::drain_compute_ingress() noexcept {
  if (!state_->compute_ingress_pending.load(std::memory_order_acquire)) {
    return 0U;
  }
  const std::lock_guard mutation_lock{state_->mutex};
  const std::lock_guard ingress_lock{state_->compute_ingress_mutex};
  const std::size_t drained = state_->compute_ingress_count;
  while (state_->compute_ingress_count != 0U) {
    const std::uint32_t request_index = state_->compute_ingress_head;
    const State::ComputeIngressRequest request = state_->compute_ingress_requests[request_index];
    const ParamInlinePatch* const patches =
        state_->compute_ingress_patches.get() +
        static_cast<std::size_t>(request_index) * state_->max_compute_patches_per_request;
    const ParamAddress event_target =
        request.patch_count == 0U ? ParamAddress{}
        : request.patch_count == 1U
            ? patches[0].target
            : ParamAddress{.operator_instance = patches[0].target.operator_instance};
    const bool evidence_available =
        request.patch_count == 0U || state_->events.can_append_unreserved(2U);
    if (request.patch_count != 0U && evidence_available) {
      static_cast<void>(state_->record_unaccepted_event(
          event_target, ParamUpdateState::kSubmitted, ParamUpdateStatus::kSubmitted, {}));
    }
    const ParamBatchUpdateResult result =
        evidence_available
            ? state_->submit_locked(request.patch_count,
                                    [&](std::size_t index) {
                                      const ParamInlinePatch& patch = patches[index];
                                      return State::PatchView{
                                          .target = patch.target,
                                          .value = patch.value,
                                          .expected_generation = patch.expected_generation,
                                          .has_expected_generation = patch.has_expected_generation,
                                      };
                                    },
                                    {})
            : ParamBatchUpdateResult{
                  .status = ParamUpdateStatus::kQueueFull,
                  .committed = false,
                  .patch_count = request.patch_count,
              };
    if (!result.committed && request.patch_count != 0U) {
      static_cast<void>(state_->record_unaccepted_event(
          event_target, ParamUpdateState::kRejected, result.status, {}));
    }
    state_->compute_ingress_head =
        (state_->compute_ingress_head + 1U) % state_->compute_ingress_capacity;
    --state_->compute_ingress_count;
  }
  state_->compute_ingress_pending.store(false, std::memory_order_release);
  return drained;
}

expected<ParamReadResult, Error> ParameterUpdateCoordinator::read(ParamAddress address,
                                                                  ParamReadView view) {
  const std::lock_guard lock{state_->mutex};
  if (view != ParamReadView::kPublished && view != ParamReadView::kLatestAccepted &&
      view != ParamReadView::kPending) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ParameterGenerationStore* const store = find_store(address.operator_instance);
  if (store == nullptr) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  const ParamFieldDescriptor* const field = store->find_field(address.field);
  if (field == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (field->local_operations.read_member_as_scalar == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  const GenerationSlot* const slot = store->slot_for_view(view);
  if (view == ParamReadView::kPending && slot == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  const void* const value = slot == nullptr ? store->latest_value() : slot->header.value;
  ParamScalarValue scalar{};
  if (!field->local_operations.read_member_as_scalar(value, scalar)) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  const auto owned = value_from_scalar(scalar);
  if (!owned.has_value()) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  if (slot == nullptr) {
    return ParamReadResult{
        .value = *owned,
        .generation = 0U,
        .update_sequence = 0U,
        .state = ParamUpdateState::kPublished,
    };
  }
  return ParamReadResult{
      .value = *owned,
      .generation = slot->header.generation,
      .update_sequence =
          view == ParamReadView::kLatestAccepted && store->latest_accepted_is_no_change()
              ? store->latest_accepted_sequence()
              : slot->update_sequence,
      .state = view == ParamReadView::kLatestAccepted && store->latest_accepted_is_no_change()
                   ? ParamUpdateState::kAcceptedNoChange
               : (slot->role == SlotRole::kPending || slot->role == SlotRole::kLifecycleCandidate)
                   ? ParamUpdateState::kAccepted
               : slot->observed.load(std::memory_order_acquire) ? ParamUpdateState::kObserved
                                                                : ParamUpdateState::kPublished,
  };
}

const ParamFieldDescriptor* ParameterUpdateCoordinator::field_descriptor(
    ParamAddress address) const noexcept {
  const ParameterGenerationStore* const store = find_store(address.operator_instance);
  return store == nullptr ? nullptr : store->find_field(address.field);
}

const ParameterCatalogView& ParameterUpdateCoordinator::catalog() const noexcept {
  return state_->catalog;
}

std::uint32_t ParameterUpdateCoordinator::max_event_cursors() const noexcept {
  return state_->max_event_cursors;
}

expected<std::uint32_t, Error> ParameterUpdateCoordinator::register_event_cursor(
    const ParamAccessPolicy& policy) noexcept {
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->events.register_cursor(policy);
}

void ParameterUpdateCoordinator::unregister_event_cursor(std::uint32_t cursor) noexcept {
  const std::lock_guard lock{state_->mutex};
  state_->events.unregister_cursor(cursor);
}

expected<ParamSnapshot, Error> ParameterUpdateCoordinator::capture(const ParamAccessPolicy& policy,
                                                                   ParamCaptureOptions options) {
  // Capture provenance describes the records authorized by the fixed scope and
  // visibility policy, even when the capability intentionally omits discovery.
  // Enable read only for this private digest projection; value capture and the
  // public handle retain the original permission set.
  ParamAccessPolicy projection_policy = policy;
  projection_policy.permissions = projection_policy.permissions | ParamPermission::kRead;
  auto visible_catalog = filter_runtime_parameter_catalog(
      state_->catalog, projection_policy, state_->plan_generation, state_->run_generation);
  if (!visible_catalog) {
    return make_unexpected(visible_catalog.error());
  }
  return capture(policy, visible_catalog->compatibility_digest(), options);
}

// Capture is a single locked projection transaction: validation, authorization, lineage, and
// field-value classification must remain ordered so no value is read before its policy decision.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<ParamSnapshot, Error> ParameterUpdateCoordinator::capture(
    const ParamAccessPolicy& policy, ParamCatalogDigest source_catalog_digest,
    ParamCaptureOptions options) {
  if (options.view != ParamReadView::kPublished ||
      (options.consistency != ParamCaptureConsistency::kRequireStopped &&
       options.consistency != ParamCaptureConsistency::kLivePerOperator) ||
      options.filter != ParamSnapshotFilter::kPolicyDefault ||
      (options.unavailable_values != ParamUnavailableValuePolicy::kReject &&
       options.unavailable_values != ParamUnavailableValuePolicy::kRecordUnavailable)) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  ParamAccessPolicy projection_policy = policy;
  projection_policy.permissions = projection_policy.permissions | ParamPermission::kRead;
  auto visible_catalog = filter_runtime_parameter_catalog(
      state_->catalog, projection_policy, state_->plan_generation, state_->run_generation);
  if (!visible_catalog) {
    return make_unexpected(visible_catalog.error());
  }
  if (source_catalog_digest != visible_catalog->compatibility_digest()) {
    // A capture-only capability deliberately exposes an empty discovery
    // catalog. Accept that exact prepared identity, but stamp the snapshot with
    // the read-enabled projection used for capture provenance.
    if (has_param_permission(policy.permissions, ParamPermission::kRead)) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    auto capability_catalog = filter_runtime_parameter_catalog(
        state_->catalog, policy, state_->plan_generation, state_->run_generation);
    if (!capability_catalog ||
        source_catalog_digest != capability_catalog->compatibility_digest()) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
  }

  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (options.consistency == ParamCaptureConsistency::kRequireStopped &&
      !state_->run_terminalized) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  const ParamUpdateSequence accepted = state_->accepted_frontier.load(std::memory_order_acquire);
  if (options.after_update_sequence.has_value()) {
    if (options.consistency != ParamCaptureConsistency::kRequireStopped) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (state_->terminal_frontier.load(std::memory_order_acquire) <
        *options.after_update_sequence) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
  }

  ParamSnapshot snapshot;
  snapshot.plan_digest_ = state_->plan_digest;
  snapshot.plan_generation_ = state_->plan_generation;
  snapshot.run_generation_ = state_->run_generation;
  snapshot.catalog_digest_ = visible_catalog->compatibility_digest();
  snapshot.view_ = options.view;
  snapshot.consistency_ = options.consistency;
  if (options.consistency == ParamCaptureConsistency::kRequireStopped) {
    snapshot.accepted_through_ = accepted;
    snapshot.published_through_ = state_->first_unpublished_terminal_sequence == 0U
                                      ? accepted
                                      : state_->first_unpublished_terminal_sequence - 1U;
  }
  try {
    const ParamScopeFilter scope = policy.allowed_scope;
    if (scope != ParamScopeFilter::run()) {
      const auto selected = std::find_if(
          state_->catalog.operators_.begin(),
          state_->catalog.operators_.end(),
          [&](const auto& record) { return record.operator_instance == scope.operator_instance; });
      if (selected == state_->catalog.operators_.end()) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      // The scope labels the snapshot's structural Operator-record envelope;
      // a field restriction remains represented by the policy-filtered field
      // vector and catalog digest. Offline slicing can remove records but can
      // never reconstruct an omitted field or grant control authority.
      snapshot.scope_ = ParamSnapshotScope::operator_(selected->path);
    }

    // This coordinator lock is also the publication/retirement linearization
    // lock. Each Operator record is therefore copied from one immutable
    // published slot. Live capture deliberately does not claim that the
    // per-Operator records share a graph-wide update frontier.
    snapshot.operators_.reserve(visible_catalog->operators().size());
    for (const OperatorParamRecord& catalog_operator : visible_catalog->operators()) {
      ParameterGenerationStore* const selected_store =
          find_store(catalog_operator.operator_instance);
      if (selected_store == nullptr) {
        return make_unexpected(Error{ErrorCode::kIncompatible});
      }
      const ParameterGenerationStore& store = *selected_store;

      const ParamReadResult lineage = store.published_lineage();
      ParamSnapshotOperatorRecord record{
          .operator_instance = catalog_operator.operator_instance,
          .path = catalog_operator.path,
          .effective_profile = catalog_operator.effective_profile,
          .schema_availability = catalog_operator.schema_availability,
          .schema_id = catalog_operator.schema_id,
          .schema_version = catalog_operator.schema_version,
          .structural_hash = catalog_operator.structural_hash,
          .contract_hash = catalog_operator.contract_hash,
          .metadata_hash = catalog_operator.metadata_hash,
          .schema_coverage = catalog_operator.schema_coverage,
          .schema_proof = catalog_operator.schema_proof,
          .generation = lineage.generation,
          .accepted_update_sequence = store.latest_accepted_sequence(),
          .published_update_sequence = lineage.update_sequence,
          .state = lineage.state,
          .filter = options.filter,
      };
      if (store.plan().schema_availability == ParamSchemaAvailability::kOpaqueLocal &&
          !store.plan().no_params) {
        if (options.unavailable_values == ParamUnavailableValuePolicy::kReject) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
        record.value_coverage = ParamSnapshotValueCoverage::kUnavailable;
        snapshot.operators_.push_back(std::move(record));
        continue;
      }

      std::size_t included{};
      bool filtered{};
      const auto visible_field_count = std::count_if(visible_catalog->fields().begin(),
                                                     visible_catalog->fields().end(),
                                                     [&](const auto& field) {
                                                       return field.address.operator_instance ==
                                                              catalog_operator.operator_instance;
                                                     });
      record.fields.reserve(static_cast<std::size_t>(visible_field_count));
      for (const ParamFieldCatalogRecord& catalog_field : visible_catalog->fields()) {
        if (catalog_field.address.operator_instance != catalog_operator.operator_instance) {
          continue;
        }
        const auto planned = std::find_if(store.fields().begin(),
                                          store.fields().end(),
                                          [&](const ParameterFieldPlanRecord& field) {
                                            return field.field == catalog_field.address.field &&
                                                   field.full_id == catalog_field.full_id;
                                          });
        const ParamFieldDescriptor* const local = store.find_field(catalog_field.address.field);
        if (planned == store.fields().end() || local == nullptr) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        ParamSnapshotFieldRecord field{
            .address = catalog_field.address,
            .full_id = catalog_field.full_id,
            .path = catalog_field.path,
            .logical_type = catalog_field.logical_type,
            .type = snapshot_type_identity(*planned),
            .codec = snapshot_codec_identity(*planned),
        };
        // Reuse the compile-frozen metadata-only classifier. This ordering is
        // security relevant: no value operation is invoked until the exact
        // transient/sensitive/internal/profile/codec disposition is known.
        field.projection = catalog_field.projection;
        if (field.projection == params::ParamProjectionStatus::kIncluded &&
            catalog_field.sensitivity == ParamSensitivity::kSensitive &&
            !policy.include_sensitive) {
          field.projection = params::ParamProjectionStatus::kRedactedSensitive;
        }

        if (field.projection == params::ParamProjectionStatus::kExcludedUnsupportedCodec &&
            options.unavailable_values == ParamUnavailableValuePolicy::kReject) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
        if (field.projection != params::ParamProjectionStatus::kIncluded) {
          filtered = true;
        } else {
          if (planned->value_route != ParamValueRouteDisposition::kPortableScalar ||
              local->local_operations.read_member_as_scalar == nullptr) {
            return make_unexpected(Error{ErrorCode::kIncompatible});
          }
          ParamScalarValue scalar{};
          if (!local->local_operations.read_member_as_scalar(store.published_value(), scalar)) {
            return make_unexpected(Error{ErrorCode::kIncompatible});
          }
          field.value = value_from_scalar(scalar);
          if (!field.value.has_value()) {
            return make_unexpected(Error{ErrorCode::kNotSupported});
          }
          ++included;
        }
        record.fields.push_back(std::move(field));
      }
      std::sort(
          record.fields.begin(), record.fields.end(), [](const auto& left, const auto& right) {
            return left.address.field < right.address.field;
          });
      const bool all_declared_fields = (record.fields.empty() && store.plan().no_params) ||
                                       (!filtered && included == record.fields.size());
      if (all_declared_fields) {
        record.value_coverage = ParamSnapshotValueCoverage::kAllDeclaredFields;
      } else if (included == 0U && !record.fields.empty() && !filtered) {
        record.value_coverage = ParamSnapshotValueCoverage::kUnavailable;
      } else {
        record.value_coverage = ParamSnapshotValueCoverage::kPolicyFiltered;
      }
      snapshot.operators_.push_back(std::move(record));
    }
    std::sort(snapshot.operators_.begin(),
              snapshot.operators_.end(),
              [](const auto& left, const auto& right) {
                return operator_path_less(left.path, right.path);
              });
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return snapshot;
}

expected<std::size_t, Error> ParameterUpdateCoordinator::drain_events(
    std::uint32_t cursor, std::span<ParamEvent> output) noexcept {
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->events.drain(cursor, output, [this](const auto& policy, const auto& event) {
    return state_->event_visible(policy, event);
  });
}

expected<ParamUpdateResult, Error> ParameterUpdateCoordinator::result(
    ParamUpdateSequence sequence) {
  if (sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::lock_guard lock{state_->mutex};
  ResultLedgerEntry* const entry = state_->find_result(sequence);
  if (entry == nullptr) {
    return make_unexpected(Error{state_->detached ? ErrorCode::kClosed : ErrorCode::kNotFound});
  }
  return state_->project_result(*entry);
}

expected<ParamUpdateResult, Error> ParameterUpdateCoordinator::await(
    ParamUpdateSequence sequence, std::chrono::nanoseconds timeout) {
  if (sequence == 0U || timeout.count() < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::unique_lock lock{state_->mutex};
  if (state_->find_result(sequence) == nullptr) {
    return make_unexpected(Error{state_->detached ? ErrorCode::kClosed : ErrorCode::kNotFound});
  }
  return state_->await_terminal_result(lock, sequence, timeout, ErrorCode::kNotFound);
}

expected<ParamUpdateResult, Error> ParameterUpdateCoordinator::await_authorized(
    ParamUpdateSequence sequence, std::chrono::nanoseconds timeout,
    const ParamAccessPolicy& policy) {
  if (sequence == 0U || timeout.count() < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::unique_lock lock{state_->mutex};
  if (!state_->result_visible(sequence, policy)) {
    // A restricted capability gets the same response for an absent sequence and for one outside
    // its field/visibility boundary, so guessed run-local sequence numbers are not an oracle.
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  return state_->await_terminal_result(lock, sequence, timeout, ErrorCode::kPermissionDenied);
}

std::optional<ParamUpdateResult> ParameterUpdateCoordinator::publish_next_at_callback_boundary(
    OperatorRef owner, ParamUpdateSequence accepted_cutoff) {
  ParameterGenerationStore* const store = find_store(owner);
  if (store == nullptr || !store->publication_pending()) {
    return std::nullopt;
  }
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return std::nullopt;
  }
  auto published = store->publish_next(accepted_cutoff);
  if (published.has_value()) {
    const ResultLedgerEntry* const entry = state_->find_result(published->update_sequence);
    state_->update_result(*published);
    state_->record_transition(
        owner,
        *published,
        {},
        0U,
        entry != nullptr && entry->single_field ? entry->sole_field : 0U,
        published->target_generation == 0U ? 0U : published->target_generation - 1U);
    state_->state_changed.notify_all();
  }
  return published;
}

ParamUpdateSequence ParameterUpdateCoordinator::publication_frontier() const noexcept {
  return state_->accepted_frontier.load(std::memory_order_acquire);
}

expected<ParameterGenerationPin, Error> ParameterUpdateCoordinator::pin_published(
    OperatorRef owner) noexcept {
  ParameterGenerationStore* const store = find_store(owner);
  return store == nullptr ? expected<ParameterGenerationPin, Error>{make_unexpected(
                                Error{ErrorCode::kStaleOperatorRef})}
                          : store->pin();
}

expected<void, Error> ParameterUpdateCoordinator::commit_activation_observation(
    OperatorRef owner, const ParameterGenerationPin& pin) noexcept {
  ParameterGenerationStore* const store = find_store(owner);
  if (store == nullptr) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (!state_->detached.load(std::memory_order_acquire) &&
      store->observation_status(pin) == ObservationStatus::kSatisfied) {
    return {};
  }
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto observed = store->pending_observation(pin);
  if (!observed) {
    return make_unexpected(std::move(observed.error()));
  }
  const std::optional<ParamUpdateSequence> pending_observation = std::move(observed).value();
  const bool has_pending_observation = pending_observation.has_value();
  const ParamUpdateSequence observed_sequence = pending_observation.value_or(0U);
  if (!has_pending_observation) {
    return {};
  }
  ResultLedgerEntry* const entry = state_->find_result(observed_sequence);
  if (entry == nullptr || entry->result.state != ParamUpdateState::kPublished) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }
  store->commit_observed(pin);
  ParamUpdateResult result = entry->result;
  result.state = ParamUpdateState::kObserved;
  state_->update_result(result);
  state_->record_transition(owner,
                            result,
                            {},
                            0U,
                            entry->single_field ? entry->sole_field : 0U,
                            result.target_generation);
  state_->state_changed.notify_all();
  return {};
}

expected<ParameterGenerationPin, Error> ParameterUpdateCoordinator::pin_lifecycle_generation(
    OperatorRef owner, ParamGeneration generation) noexcept {
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ParameterGenerationStore* const store = find_store(owner);
  return store == nullptr ? expected<ParameterGenerationPin, Error>{make_unexpected(
                                Error{ErrorCode::kStaleOperatorRef})}
                          : store->pin_generation(generation);
}

expected<void, Error> ParameterUpdateCoordinator::transition_lifecycle_candidate(
    OperatorRef owner, ParamGeneration generation, LifecycleTransactionId transaction,
    LifecycleTransactionState lifecycle_state) noexcept {
  if (transaction == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto mapped = parameter_state(lifecycle_state);
  if (!mapped.has_value() || lifecycle_state == LifecycleTransactionState::kAcceptedNoChange ||
      lifecycle_state == LifecycleTransactionState::kQueued) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ParameterGenerationStore* const store = find_store(owner);
  if (store == nullptr) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  GenerationSlot* const candidate = store->lifecycle_candidate(generation, transaction);
  if (candidate == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const ParamUpdateSequence sequence = candidate->update_sequence;
  ResultLedgerEntry* const entry = state_->find_result(sequence);
  if (entry == nullptr) {
    return make_unexpected(Error{ErrorCode::kIncompatible});
  }

  const ParamGeneration prior_generation = store->published_generation();
  ParamUpdateResult result = entry->result;
  result.state = *mapped;
  result.lifecycle_transaction_id = transaction;
  if (lifecycle_state == LifecycleTransactionState::kPublished) {
    const LifecyclePublication publication =
        store->publish_lifecycle_candidate(generation, transaction);
    if (publication.published_sequence != sequence) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    if (publication.retired_sequence != 0U && !publication.retired_was_observed) {
      ResultLedgerEntry* const retired = state_->find_result(publication.retired_sequence);
      if (retired == nullptr || retired->result.state != ParamUpdateState::kPublished) {
        runtime_log_and_terminate_noexcept([] {
          HOLOSCAN_LOG_ERROR("Unobserved parameter publication lost its retained result evidence");
        });
      }
      ParamUpdateResult superseded = retired->result;
      superseded.state = ParamUpdateState::kSuperseded;
      state_->update_result(superseded);
      state_->record_transition(owner,
                                superseded,
                                {},
                                0U,
                                retired->single_field ? retired->sole_field : 0U,
                                prior_generation);
    }
  } else if (lifecycle_state == LifecycleTransactionState::kSuperseded ||
             lifecycle_state == LifecycleTransactionState::kCanceled ||
             lifecycle_state == LifecycleTransactionState::kRolledBack ||
             lifecycle_state == LifecycleTransactionState::kFailedStopped) {
    if (store->retire_lifecycle_candidate(transaction) != sequence) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
  }
  state_->update_result(result);
  state_->record_transition(
      owner, result, {}, 0U, entry->single_field ? entry->sole_field : 0U, prior_generation);
  state_->state_changed.notify_all();
  return {};
}

void ParameterUpdateCoordinator::release(ParameterGenerationPin& pin) noexcept {
  if (pin.header_ == nullptr || pin.store_cookie_ == nullptr) {
    return;
  }
  static_cast<ParameterGenerationStore*>(pin.store_cookie_)->release(pin.slot_);
  pin.header_ = nullptr;
  pin.store_cookie_ = nullptr;
  pin.slot_ = 0U;
  pin.derived_terms_ = nullptr;
  pin.derived_term_count_ = 0U;
}

ParamStopFenceResult ParameterUpdateCoordinator::close_ingress() {
  request_close_ingress();
  const std::lock_guard lock{state_->mutex};
  const bool already_closed = state_->closed;
  state_->materialize_close_request();
  return ParamStopFenceResult{
      .terminal_update_sequence = state_->terminal_frontier.load(std::memory_order_acquire),
      .already_closed = already_closed,
  };
}

void ParameterUpdateCoordinator::request_close_ingress() noexcept {
  state_->close_requested.store(true, std::memory_order_release);
}

void ParameterUpdateCoordinator::terminalize_unpublished_at_run_end() noexcept {
  const std::lock_guard lock{state_->mutex};
  state_->terminalize_unpublished();
  state_->run_terminalized = true;
}

void ParameterUpdateCoordinator::begin_detach() noexcept {
  const std::lock_guard lock{state_->mutex};
  if (state_->detached) {
    return;
  }
  state_->terminalize_unpublished();
  state_->run_terminalized = true;
  state_->detached = true;
  state_->state_changed.notify_all();
}

ParamUpdateSequence ParameterUpdateCoordinator::terminal_sequence() const noexcept {
  return state_->close_materialized.load(std::memory_order_acquire)
             ? state_->terminal_frontier.load(std::memory_order_acquire)
             : state_->accepted_frontier.load(std::memory_order_acquire);
}

void ParameterUpdateCoordinatorTestAccess::hold_mutation_lock(
    ParameterUpdateCoordinator& coordinator, std::atomic_bool& entered,
    const std::atomic_bool& release) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  entered.store(true, std::memory_order_release);
  while (!release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

bool ParameterUpdateCoordinatorTestAccess::scalar_semantically_equal(
    ParamScalarValue left, ParamScalarValue right) noexcept {
  return ::holoscan::detail::scalar_semantically_equal(left, right);
}

std::uint64_t ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(
    ParameterUpdateCoordinator& coordinator) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  return coordinator.state_->events.reserved_sequences();
}

void ParameterUpdateCoordinatorTestAccess::set_event_sequence(
    ParameterUpdateCoordinator& coordinator, ParamEventSequence sequence) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  coordinator.state_->events.set_next_sequence_for_test(sequence);
}

void ParameterUpdateCoordinatorTestAccess::set_update_sequence(
    ParameterUpdateCoordinator& coordinator, ParamUpdateSequence sequence) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  coordinator.state_->next_sequence = sequence;
}

void ParameterUpdateCoordinatorTestAccess::set_published_generation(
    ParameterUpdateCoordinator& coordinator, OperatorRef owner,
    ParamGeneration generation) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  if (ParameterGenerationStore* const store = coordinator.find_store(owner); store != nullptr) {
    store->set_published_generation_for_test(generation);
  }
}

void ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
    ParameterUpdateCoordinator& coordinator, OperatorRef owner, std::uint64_t pin_count) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  if (ParameterGenerationStore* const store = coordinator.find_store(owner); store != nullptr) {
    store->set_published_pin_count_for_test(pin_count);
  }
}

std::uint64_t ParameterUpdateCoordinatorTestAccess::published_claim_word(
    ParameterUpdateCoordinator& coordinator, OperatorRef owner) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  const ParameterGenerationStore* const store = coordinator.find_store(owner);
  return store == nullptr ? 0U : store->published_claim_word_for_test();
}

ParameterGenerationStore* ParameterUpdateCoordinator::find_store(OperatorRef owner) noexcept {
  return state_ == nullptr ? nullptr : state_->find_store(owner);
}

const ParameterGenerationStore* ParameterUpdateCoordinator::find_store(
    OperatorRef owner) const noexcept {
  return state_ == nullptr ? nullptr : state_->find_store(owner);
}

void ParameterUpdateCoordinatorTestAccess::replace_event_reservation(
    ParameterUpdateCoordinator& coordinator, std::uint32_t released_count) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  static_cast<void>(
      coordinator.state_->events.replace_accepted_transition_reservation(released_count));
}

void ParameterUpdateCoordinatorTestAccess::release_event_reservations(
    ParameterUpdateCoordinator& coordinator, std::uint32_t count) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  coordinator.state_->events.release_accepted_transitions(count);
}

void ParameterUpdateCoordinatorTestAccess::set_result_event_reservations(
    ParameterUpdateCoordinator& coordinator, ParamUpdateSequence sequence,
    std::uint32_t count) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  if (auto* result = coordinator.state_->find_result(sequence)) {
    result->event_reservations = count;
  }
}

void ParameterUpdateCoordinatorTestAccess::invalidate_result(
    ParameterUpdateCoordinator& coordinator, ParamUpdateSequence sequence) noexcept {
  const std::lock_guard lock{coordinator.state_->mutex};
  if (auto* result = coordinator.state_->find_result(sequence)) {
    result->valid = false;
  }
}

}  // namespace holoscan::detail
