/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/activation.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/payload_kind.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

namespace detail {
template <typename T>
inline constexpr bool kPublishedScalar = std::is_arithmetic_v<T> || std::is_enum_v<T>;

}  // namespace detail

/// @brief Constrain a public typed port to an explicitly admitted payload family.
///
/// The baseline families are unqualified arithmetic or enum scalars, generated self-contained
/// copy-constructible schema roots with an explicit \ref schema_identity specialization,
/// descriptor-backed schema owning types with explicit identity and Tensor-schema traits, raw POD
/// types with both explicit `ChannelTypeIdentity<T>` and `ChannelPodLayout<T>` specializations, and
/// exactly \ref Tensor. Descriptor-backed schema metadata remains separate from its retained Tensor
/// storage. Schema/POD metadata does not override the owning-copy requirement used by scalar
/// receive and finite read claims. Copyability or triviality alone is never admission. References,
/// cv-qualified types, pointers, arbitrary classes, strings and views, backend handles,
/// \c MemoryBuffer, move-only schema roots, and unpublished loan or write-guard types therefore do
/// not satisfy this concept.
///
/// @tparam T Candidate public payload type.
template <typename T>
concept PublishedPayload = std::same_as<T, std::remove_cvref_t<T>> && std::is_object_v<T> &&
                           (detail::kPublishedScalar<T> || std::same_as<T, Tensor> ||
                            (detail::HasSchemaIdentity<T> && std::copy_constructible<T>) ||
                            (detail::HasChannelPodContract<T> && std::copy_constructible<T>));

template <PublishedPayload T>
class Input;

namespace detail {

// Private input bindings return this callback record only after reserving one finite read-credit
// slot. It is intentionally not sufficient to manufacture a public claim: only Input<T> can adopt
// the record into a Sample<T>.
struct PublishedReadClaimBinding {
  void* state{};
  void (*retain)(void*) noexcept {};
  void (*release)(void*) noexcept {};
};

class PublishedReadClaim {
 public:
  PublishedReadClaim() noexcept = default;

  PublishedReadClaim(const PublishedReadClaim& other) noexcept : binding_(other.binding_) {
    if (binding_.state != nullptr) {
      binding_.retain(binding_.state);
    }
  }

  PublishedReadClaim& operator=(const PublishedReadClaim& other) noexcept {
    if (this == &other) {
      return *this;
    }
    PublishedReadClaim replacement{other};
    swap(replacement);
    return *this;
  }

  PublishedReadClaim(PublishedReadClaim&& other) noexcept
      : binding_(std::exchange(other.binding_, PublishedReadClaimBinding{})) {}

  PublishedReadClaim& operator=(PublishedReadClaim&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    binding_ = std::exchange(other.binding_, PublishedReadClaimBinding{});
    return *this;
  }

  ~PublishedReadClaim() noexcept { reset(); }

 private:
  explicit PublishedReadClaim(PublishedReadClaimBinding binding) noexcept : binding_(binding) {}

  void reset() noexcept {
    const PublishedReadClaimBinding old = std::exchange(binding_, PublishedReadClaimBinding{});
    if (old.state != nullptr) {
      old.release(old.state);
    }
  }

  void swap(PublishedReadClaim& other) noexcept { std::swap(binding_, other.binding_); }

  PublishedReadClaimBinding binding_{};

  template <PublishedPayload T>
  friend class ::holoscan::Input;
};

}  // namespace detail

/// @brief Identify the version of the transport-neutral Core sample metadata record.
struct SampleMetadataVersion {
  std::uint16_t major{1};  ///< Major semantic record version.
  std::uint16_t minor{0};  ///< Backward-compatible semantic record revision.
};

/// @brief Carry semantic validity and degradation flags for one sample.
enum class SampleFlags : std::uint32_t {
  kNone = 0,            ///< Sample carries no invalid or degraded indication.
  kInvalid = 1U << 0,   ///< Sample is not valid for ordinary processing.
  kDegraded = 1U << 1,  ///< Sample remains valid but carries degraded evidence.
};

/// @brief Combine independent sample validity and degradation flags.
/// @param lhs Left flag set.
/// @param rhs Right flag set.
/// @return Bitwise union of both flag sets.
/// @note A bitwise combination of declared flags is a valid value even when it has no separately
/// named enumerator.
[[nodiscard]] constexpr SampleFlags operator|(SampleFlags lhs, SampleFlags rhs) noexcept {
  // SampleFlags is a bitmask; the analyzer otherwise treats an intentional combination as invalid.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<SampleFlags>(static_cast<std::uint32_t>(lhs) |
                                  static_cast<std::uint32_t>(rhs));
}

/// @brief Intersect sample validity and degradation flags.
/// @param lhs Left flag set.
/// @param rhs Right flag set.
/// @return Flags present in both sets.
[[nodiscard]] constexpr SampleFlags operator&(SampleFlags lhs, SampleFlags rhs) noexcept {
  // SampleFlags is a bitmask; the analyzer otherwise treats an intentional combination as invalid.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<SampleFlags>(static_cast<std::uint32_t>(lhs) &
                                  static_cast<std::uint32_t>(rhs));
}

/// @brief Add flags to an existing sample flag set.
/// @param lhs Flag set to update.
/// @param rhs Flags to add.
/// @return Reference to the updated flag set.
constexpr SampleFlags& operator|=(SampleFlags& lhs, SampleFlags rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

/// @brief Carry transport-neutral identity and timing metadata for one logical sample.
///
/// The aggregate contains no backend descriptor, slot identity, lease token,
/// payload reference, or activation \c Tag. Capture-time presence is independent
/// of payload validity. The default record is valid and has canonical absent
/// capture and publish times because both clock identities are zero.
///
/// The compact clock identifiers carry no originating plan, clock-catalog, or
/// registry-generation identity. A sample inherits that scope from its validated
/// RunSession or endpoint binding. Persistence and transport must carry the scope
/// directly or inherit it unambiguously from an immutable validated envelope or
/// attachment before exposing or locally rebinding these identifiers.
struct SampleMetadata {
  SampleMetadataVersion version{};  ///< Semantic metadata record version.
  /// Producer or Runtime publication sequence; zero when unstamped.
  /// Runtime-owned outputs start at one in a new session and retain their publication counter
  /// across lifecycle reset and restart. The counter belongs to each output, independently of
  /// producer epochs and logical flow identity; re-emission uses the emitting output's counter.
  /// External endpoints retain their own sequence authority and transport projection.
  std::uint64_t sequence{};
  std::int64_t capture_timestamp_ns{};  ///< Genuine acquisition coordinate when its clock is valid.
  ClockId source_clock_id{};  ///< Capture-time clock domain, or zero when capture time is absent.
  std::int64_t publish_timestamp_ns{};  ///< Publication coordinate when its clock is valid.
  ClockId publish_clock_id{};           ///< Publish-time clock domain, or zero when unavailable.
  std::uint64_t trace_id{};             ///< Trace correlation identity, or zero when absent.
  std::uint64_t frame_id{};             ///< Application frame/sample identity, or zero when absent.
  std::uint64_t byte_size{};            ///< Logical payload size in bytes when known.
  PayloadKind payload_kind{PayloadKind::kInlineTyped};  ///< Whole-sample semantic representation.
  SampleFlags flags{SampleFlags::kNone};                ///< Validity and degradation flags.

  /// @brief Return whether the sample is valid for ordinary processing.
  /// @return \c false only when the \c kInvalid flag is present.
  [[nodiscard]] constexpr bool valid() const noexcept {
    return (static_cast<std::uint32_t>(flags) &
            static_cast<std::uint32_t>(SampleFlags::kInvalid)) == 0;
  }

  /// @brief Return whether the sample carries degraded evidence.
  /// @return \c true when the \c kDegraded flag is present, independent of validity.
  [[nodiscard]] constexpr bool degraded() const noexcept {
    return (static_cast<std::uint32_t>(flags) &
            static_cast<std::uint32_t>(SampleFlags::kDegraded)) != 0;
  }

  /// @brief Return the genuine capture time carried by this sample.
  /// @return Capture coordinate and source clock, or canonical
  /// \c TimePoint{0,ClockId{0}} when capture time was omitted.
  /// @note Core never substitutes activation time, current time, publication
  /// time, arrival time, or a valid zero coordinate for omitted capture time.
  /// The result remains scoped to this sample's validated clock catalog.
  [[nodiscard]] constexpr TimePoint capture_time() const noexcept {
    return TimePoint{.timestamp_ns = capture_timestamp_ns, .clock_id = source_clock_id};
  }

  /// @brief Return the publication time carried by this sample.
  ///
  /// Publication time is the coordinate at which Runtime submitted this sample for publication.
  /// Runtime reads its publish clock once per publication, immediately after it assigns
  /// \ref sequence and before any recipient or transport accepts the sample. Every fanout copy of
  /// one publication therefore reports a single coordinate, which is what lets two recipients
  /// measure delivery latency against the same event. The coordinate precedes transport queueing,
  /// serialization, and delivery, so it does not describe when any recipient observed the sample.
  ///
  /// In-graph publications carry the Graph default clock. An exported sample carries the same
  /// instant on the same clock, but named in the receiving endpoint catalog's clock namespace,
  /// because that catalog numbers its clocks independently of the publishing plan. An external
  /// client resolves \ref publish_clock_id against the clock records its catalog declares.
  ///
  /// A sample imported into this graph across an external boundary carries no publication time. Its
  /// publication belongs to the peer that produced it, whose clock this graph cannot place.
  ///
  /// @return Publication coordinate and publish clock, or canonical absent time
  /// when the publish clock identity is zero.
  /// @note Publication is never capture. Core does not substitute this coordinate for an omitted
  /// \ref capture_time, and an author-supplied publication coordinate does not survive
  /// publication. The result remains scoped to this sample's validated clock catalog.
  [[nodiscard]] constexpr TimePoint publish_time() const noexcept {
    return TimePoint{.timestamp_ns = publish_timestamp_ns, .clock_id = publish_clock_id};
  }
};

/// @brief Own one admitted typed payload together with its semantic metadata.
///
/// A scalar receive may attach a finite Holoscan-owned read claim so the owning
/// sample can outlive the callback. That claim remains private and never exposes
/// a backend handle, pool slot, transport descriptor, or activation \c Tag.
/// Copying a sample copies \p T and retains that already-admitted bounded claim;
/// moving transfers both. Copy and move exception behavior follows \p T.
///
/// @tparam T Explicitly admitted published payload type.
template <PublishedPayload T>
class Sample {
 public:
  /// @brief Construct an owning sample from a payload and metadata value.
  /// @param data Payload value consumed into this sample.
  /// @param metadata Transport-neutral semantic metadata to copy.
  Sample(T data, SampleMetadata metadata) noexcept(std::is_nothrow_move_constructible_v<T>)
      : data(std::move(data)), metadata(metadata) {}

  /// @brief Copy a sample and retain its finite read claim when present.
  /// @param other Owning sample to copy.
  Sample(const Sample& other) = default;

  /// @brief Move a sample and transfer its finite read claim.
  /// @param other Owning sample to consume.
  Sample(Sample&& other) noexcept(std::is_nothrow_move_constructible_v<T>) = default;

  /// @brief Copy-assign a sample and its finite read claim.
  /// @param other Owning sample to copy.
  /// @return This sample after assignment.
  Sample& operator=(const Sample& other) = default;

  /// @brief Move-assign a sample and transfer its finite read claim.
  /// @param other Owning sample to consume.
  /// @return This sample after assignment.
  Sample& operator=(Sample&& other) noexcept(std::is_nothrow_move_assignable_v<T>) = default;

  /// @brief Release the payload and any private finite read claim without throwing.
  ~Sample() noexcept = default;

  T data;                     ///< Owning public payload value.
  SampleMetadata metadata{};  ///< Transport-neutral semantic sample metadata.

 private:
  detail::PublishedReadClaim read_claim_{};

  friend class Input<T>;
};

/// @brief Reuse sample flag semantics for one output publication request.
using EmitFlags = SampleFlags;

/// @brief Supply optional semantic metadata when publishing a payload value.
///
/// Capture time is optional unless the compiled binding requires it. Omission
/// remains canonical absence and never authorizes an implicit clock read or
/// timestamp substitution. Runtime stamps sequence, payload kind, and logical
/// byte size according to the admitted output binding, and stamps publication
/// time from the Graph default clock at the submission boundary. There is no
/// option here for publication time because it is not the author's to state: a
/// value naming when Runtime submitted the sample can only be measured at that
/// boundary. A supplied capture time must belong to the clock-catalog scope
/// admitted for that binding; numeric \ref ClockId equality alone does not
/// establish this.
struct EmitOptions {
  std::optional<TimePoint> capture_time{};  ///< Genuine capture/acquisition time when supplied.
  std::optional<std::uint64_t> frame_id{};  ///< Application frame/sample identity when supplied.
  std::optional<TraceContext> trace{};      ///< Explicit trace context when supplied.
  EmitFlags flags{EmitFlags::kNone};        ///< Caller-supplied validity/degradation flags.
};

static_assert(std::is_standard_layout_v<SampleMetadataVersion>);
static_assert(std::is_trivially_copyable_v<SampleMetadataVersion>);
static_assert(sizeof(SampleMetadataVersion) == 4U);
static_assert(alignof(SampleMetadataVersion) == 2U);
static_assert(offsetof(SampleMetadataVersion, major) == 0U);
static_assert(offsetof(SampleMetadataVersion, minor) == 2U);
static_assert(std::is_standard_layout_v<SampleMetadata>);
static_assert(std::is_trivially_copyable_v<SampleMetadata>);
static_assert(sizeof(SampleMetadata) == 80U);
static_assert(alignof(SampleMetadata) == 8U);
static_assert(offsetof(SampleMetadata, version) == 0U);
static_assert(offsetof(SampleMetadata, sequence) == 8U);
static_assert(offsetof(SampleMetadata, capture_timestamp_ns) == 16U);
static_assert(offsetof(SampleMetadata, source_clock_id) == 24U);
static_assert(offsetof(SampleMetadata, publish_timestamp_ns) == 32U);
static_assert(offsetof(SampleMetadata, publish_clock_id) == 40U);
static_assert(offsetof(SampleMetadata, trace_id) == 48U);
static_assert(offsetof(SampleMetadata, frame_id) == 56U);
static_assert(offsetof(SampleMetadata, byte_size) == 64U);
static_assert(offsetof(SampleMetadata, payload_kind) == 72U);
static_assert(offsetof(SampleMetadata, flags) == 76U);

}  // namespace holoscan
