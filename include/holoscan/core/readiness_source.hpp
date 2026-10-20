/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/value_lifecycle.hpp>

namespace holoscan {

namespace detail {
struct ReadinessSourceAccess;
}  // namespace detail

/// @brief Identify the immutable logical kind of a readiness source.
enum class ReadinessSourceKind : std::uint8_t {
  kReadable = 0,      ///< Level-readable waitable source with capacity exactly one.
  kNotification = 1,  ///< Bounded counted notification source posted from foreign callbacks.
  kDoorbell = 2,      ///< Bounded monotonic device-completion source.
  kSpin = 3,          ///< Capacity-one false-to-true polling source.
};

/// @brief Refer to one logical source without carrying adapter ownership.
///
/// The reference contains only a strong plan-local \ref SourceId and private
/// immutable plan provenance. It never contains an authoring-object address,
/// file descriptor, CUDA event, HoloMQ handle, adapter pointer, or Runtime
/// object. A default value is invalid. Copying does not retain a binding or
/// source generation.
///
/// @tparam Kind Immutable logical source kind represented by this type.
template <ReadinessSourceKind Kind>
class BasicSourceRef {
 public:
  /// @brief Construct an invalid source reference.
  constexpr BasicSourceRef() noexcept = default;

  /// @brief Return the plan-local logical source identity.
  /// @return Assigned source identity, or invalid zero for an unbound value.
  [[nodiscard]] constexpr SourceId id() const noexcept { return id_; }

  /// @brief Return the immutable logical kind carried by this reference type.
  /// @return Template source kind.
  [[nodiscard]] static constexpr ReadinessSourceKind kind() noexcept { return Kind; }

  /// @brief Return whether this reference contains an assigned logical identity.
  /// @return \c true when \ref id is nonzero.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return static_cast<bool>(id_); }

  /// @brief Compare source references by logical identity and plan provenance.
  /// @param lhs Left source reference.
  /// @param rhs Right source reference.
  /// @return \c true when both identity and private plan provenance match.
  friend constexpr bool operator==(BasicSourceRef lhs, BasicSourceRef rhs) noexcept = default;

 private:
  constexpr BasicSourceRef(SourceId id, std::uint64_t plan_provenance) noexcept
      : id_(id), plan_provenance_(plan_provenance) {}

  SourceId id_{};
  std::uint64_t plan_provenance_{};

  friend struct detail::ReadinessSourceAccess;
};

/// @brief Refer to one logical level-readable source.
using ReadableSourceRef = BasicSourceRef<ReadinessSourceKind::kReadable>;

/// @brief Refer to one logical counted-notification source.
using NotificationSourceRef = BasicSourceRef<ReadinessSourceKind::kNotification>;

/// @brief Refer to one logical device-completion source.
using DoorbellSourceRef = BasicSourceRef<ReadinessSourceKind::kDoorbell>;

/// @brief Refer to one logical false-to-true polling source.
using SpinSourceRef = BasicSourceRef<ReadinessSourceKind::kSpin>;

/// @brief Declare one level-readable readiness source owned by an operator.
///
/// The declaration is a stable setup object with effective capacity exactly
/// one. It contains no physical adapter or backend handle. Operator setup binds
/// its logical name, clock, wake requirements, and profile constraints before
/// compilation assigns a source reference.
class ReadableSource {
 public:
  /// @brief Construct an unbound logical source declaration.
  ReadableSource() noexcept = default;

  /// @brief Reject copying a stable source declaration.
  /// @param other Declaration that would otherwise be copied.
  ReadableSource(const ReadableSource& other) = delete;

  /// @brief Reject copy assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  ReadableSource& operator=(const ReadableSource& other) = delete;

  /// @brief Reject moving a stable source declaration.
  /// @param other Declaration that would otherwise be moved.
  ReadableSource(ReadableSource&& other) = delete;

  /// @brief Reject move assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  ReadableSource& operator=(ReadableSource&& other) = delete;

  /// @brief Destroy an unbound or setup-bound logical declaration.
  ~ReadableSource() = default;

  /// @brief Convert this declaration to its current typed logical source reference.
  /// @return Bound logical reference, or a default-invalid reference before compilation binds it.
  /// @note This implicit conversion enables canonical syntax such as
  /// `OnReadable{.event = readable_source}` without exposing declaration addresses or bindings.
  [[nodiscard]] operator ReadableSourceRef() const noexcept { return ref_; }

 private:
  ReadableSourceRef ref_{};
  friend struct detail::ReadinessSourceAccess;
};

/// @brief Declare one bounded counted notification source owned by an operator.
///
/// Every successful post accepts one distinct witness; notification witnesses
/// are neither coalesced nor overwritten. Capacity is supplied through the
/// setup builder and covers each accepted witness through complete retirement.
/// Capacity rejection accepts nothing, so a lossless producer retries the same
/// logical notification until it is accepted or the sender becomes terminal.
/// The declaration contains no callback, adapter, or run-session ownership.
class NotificationSource {
 public:
  /// @brief Construct an unbound logical source declaration.
  NotificationSource() noexcept = default;

  /// @brief Reject copying a stable source declaration.
  /// @param other Declaration that would otherwise be copied.
  NotificationSource(const NotificationSource& other) = delete;

  /// @brief Reject copy assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  NotificationSource& operator=(const NotificationSource& other) = delete;

  /// @brief Reject moving a stable source declaration.
  /// @param other Declaration that would otherwise be moved.
  NotificationSource(NotificationSource&& other) = delete;

  /// @brief Reject move assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  NotificationSource& operator=(NotificationSource&& other) = delete;

  /// @brief Destroy an unbound or setup-bound logical declaration.
  ~NotificationSource() = default;

  /// @brief Convert this declaration to its current typed logical source reference.
  /// @return Bound logical reference, or a default-invalid reference before compilation binds it.
  /// @note This implicit conversion enables canonical syntax such as
  /// `OnNotified{.event = notification_source}` without acquiring a sender capability.
  [[nodiscard]] operator NotificationSourceRef() const noexcept { return ref_; }

 private:
  NotificationSourceRef ref_{};
  friend struct detail::ReadinessSourceAccess;
};

/// @brief Declare one bounded monotonic device-completion source.
///
/// Runtime adapters interpret nonzero completion tokens according to the
/// compiled source contract. The declaration itself contains no device event,
/// adapter, or backend handle.
class DoorbellSource {
 public:
  /// @brief Construct an unbound logical source declaration.
  DoorbellSource() noexcept = default;

  /// @brief Reject copying a stable source declaration.
  /// @param other Declaration that would otherwise be copied.
  DoorbellSource(const DoorbellSource& other) = delete;

  /// @brief Reject copy assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  DoorbellSource& operator=(const DoorbellSource& other) = delete;

  /// @brief Reject moving a stable source declaration.
  /// @param other Declaration that would otherwise be moved.
  DoorbellSource(DoorbellSource&& other) = delete;

  /// @brief Reject move assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  DoorbellSource& operator=(DoorbellSource&& other) = delete;

  /// @brief Destroy an unbound or setup-bound logical declaration.
  ~DoorbellSource() = default;

  /// @brief Convert this declaration to its current typed logical source reference.
  /// @return Bound logical reference, or a default-invalid reference before compilation binds it.
  /// @note This implicit conversion enables canonical syntax such as
  /// `OnDoorbell{.event = doorbell_source}` without exposing device-completion state.
  [[nodiscard]] operator DoorbellSourceRef() const noexcept { return ref_; }

 private:
  DoorbellSourceRef ref_{};
  friend struct detail::ReadinessSourceAccess;
};

/// @brief Declare one capacity-one false-to-true polling source.
///
/// The source begins false and may accept one witness only after observing a
/// false-to-true edge. The public declaration owns no poller or callback.
class SpinSource {
 public:
  /// @brief Construct an unbound logical source declaration.
  SpinSource() noexcept = default;

  /// @brief Reject copying a stable source declaration.
  /// @param other Declaration that would otherwise be copied.
  SpinSource(const SpinSource& other) = delete;

  /// @brief Reject copy assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  SpinSource& operator=(const SpinSource& other) = delete;

  /// @brief Reject moving a stable source declaration.
  /// @param other Declaration that would otherwise be moved.
  SpinSource(SpinSource&& other) = delete;

  /// @brief Reject move assignment of a stable source declaration.
  /// @param other Declaration that would otherwise replace this one.
  SpinSource& operator=(SpinSource&& other) = delete;

  /// @brief Destroy an unbound or setup-bound logical declaration.
  ~SpinSource() = default;

  /// @brief Convert this declaration to its current typed logical source reference.
  /// @return Bound logical reference, or a default-invalid reference before compilation binds it.
  /// @note This implicit conversion enables canonical syntax such as
  /// `OnSpin{.event = spin_source}` without exposing a poller or callback.
  [[nodiscard]] operator SpinSourceRef() const noexcept { return ref_; }

 private:
  SpinSourceRef ref_{};
  friend struct detail::ReadinessSourceAccess;
};

/// @brief Post bounded notification readiness from a foreign thread.
///
/// A valid value is copyable, movable, source-generation-bound, and thread-safe
/// for concurrent \ref post calls. Copies share only an admitted finite Core
/// control/tombstone record; they never retain a Runtime session or adapter and
/// never widen source capacity. The public default value is invalid.
class NotificationSender {
 public:
  /// @brief Construct a default-invalid sender.
  NotificationSender() noexcept = default;

  /// @brief Copy the same narrow generation/source authority.
  /// @param other Sender whose already-admitted reference authority is copied.
  NotificationSender(const NotificationSender&) noexcept = default;

  /// @brief Replace this sender with the same authority as another sender.
  /// @param other Sender whose already-admitted reference authority is copied.
  /// @return This sender after replacement.
  NotificationSender& operator=(const NotificationSender&) noexcept = default;

  /// @brief Move generation/source authority from another sender.
  /// @param other Sender whose authority is transferred; it becomes invalid.
  NotificationSender(NotificationSender&&) noexcept = default;

  /// @brief Replace this sender by moving authority from another sender.
  /// @param other Sender whose authority is transferred; it becomes invalid.
  /// @return This sender after replacement.
  NotificationSender& operator=(NotificationSender&&) noexcept = default;

  /// @brief Release this sender reference without blocking or throwing.
  ~NotificationSender() noexcept = default;

  /// @brief Attempt to accept one notification witness without blocking.
  /// @return Success only after current-generation validation, finite credit
  /// reservation, source-sequence assignment, and witness publication.
  /// @retval ErrorCode::kNotReady Binding exists but its operator has not completed \c kStart.
  /// @retval ErrorCode::kBackpressured Accepted-through-retirement capacity is full.
  /// @retval ErrorCode::kStaleEndpoint A successful restart replaced this sender generation.
  /// @retval ErrorCode::kClosed Normal stop or session destruction closed the source.
  /// @retval ErrorCode::kInvalidArgument This value is default-invalid or moved from.
  /// @note Acceptance assigns no activation \c Tag. The successful MPSC
  /// linearization point belongs to the private capacity reservation callback.
  /// Every success accepts one distinct counted witness. \c kBackpressured
  /// accepts nothing, so a lossless producer may retry the same logical
  /// notification until success or a terminal sender error.
  [[nodiscard]] expected<void, Error> post() const noexcept {
    if (!lifecycle_ || post_ == nullptr) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = {}});
    }
    return post_(lifecycle_.state_);
  }

 private:
  using PostFunction = expected<void, Error> (*)(void*) noexcept;

  NotificationSender(detail::ValueLifecycle lifecycle, PostFunction post) noexcept
      : lifecycle_(std::move(lifecycle)), post_(post) {}

  detail::ValueLifecycle lifecycle_{};
  PostFunction post_{};

  friend struct detail::ReadinessSourceAccess;
};

/// @brief Trigger an operator when a typed readable source accepts readiness.
struct OnReadable {
  ReadableSourceRef event;  ///< Logical readable source; no timeout or backend handle.
};

/// @brief Trigger an operator for each accepted counted notification witness.
struct OnNotified {
  NotificationSourceRef event;  ///< Logical notification source reference.
};

/// @brief Trigger an operator for an accepted device-completion witness.
struct OnDoorbell {
  DoorbellSourceRef event;  ///< Logical doorbell source reference.
};

/// @brief Trigger an operator on an accepted false-to-true polling edge.
struct OnSpin {
  SpinSourceRef event;  ///< Logical spin source reference.
};

static_assert(std::is_trivially_copyable_v<ReadableSourceRef>);
static_assert(std::is_trivially_copyable_v<NotificationSourceRef>);
static_assert(std::is_trivially_copyable_v<DoorbellSourceRef>);
static_assert(std::is_trivially_copyable_v<SpinSourceRef>);

}  // namespace holoscan
