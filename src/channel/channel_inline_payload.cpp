/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_inline_payload.hpp"

#include <holomq/delivery_attribution.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/subscriber.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>
#include "channel_readiness_handle.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::channel_internal {
namespace {

template <bool Enabled>
struct ProtectedTerminalPublisherBase {};

template <>
struct ProtectedTerminalPublisherBase<true> : ProtectedTerminalInlinePublisher {};

template <bool Enabled>
struct ProtectedTerminalSubscriberBase {};

template <>
struct ProtectedTerminalSubscriberBase<true> : ProtectedTerminalWaitSetSubscriber {};

struct NoTerminalPublisher {};
struct NoTerminalSubscriber {};
struct NoTerminalAttachment {};
struct NoTerminalWaitHandle {};

template <std::size_t Size, bool Enabled>
class PendingTerminalStorage {
 public:
  [[nodiscard]] constexpr bool has_value() const noexcept { return false; }
  void reset() noexcept {}
};

template <std::size_t Size>
class PendingTerminalStorage<Size, true> {
 public:
  [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }
  template <typename Value>
  void emplace(Value&& value) {
    value_.emplace(std::forward<Value>(value));
  }
  void reset() noexcept { value_.reset(); }
  // Callers use these storage-like operators only after has_value(); preserve optional's unchecked
  // access semantics without adding exception machinery to the receive path.
  [[nodiscard]] InlinePayloadBucket<Size>* operator->() noexcept {
    return &*value_;  // NOLINT(bugprone-unchecked-optional-access)
  }
  [[nodiscard]] const InlinePayloadBucket<Size>* operator->() const noexcept {
    return &*value_;  // NOLINT(bugprone-unchecked-optional-access)
  }
  [[nodiscard]] InlinePayloadBucket<Size>& operator*() noexcept {
    return *value_;  // NOLINT(bugprone-unchecked-optional-access)
  }
  [[nodiscard]] const InlinePayloadBucket<Size>& operator*() const noexcept {
    return *value_;  // NOLINT(bugprone-unchecked-optional-access)
  }

 private:
  std::optional<InlinePayloadBucket<Size>> value_;
};

static_assert(sizeof(PendingTerminalStorage<kMaximumInlinePayloadBytes, false>) == 1U);
static_assert(sizeof(PendingTerminalStorage<kMaximumInlinePayloadBytes, true>) >=
              kMaximumInlinePayloadBytes);

template <std::size_t Size, bool Enabled>
using TerminalPublisher =
    std::conditional_t<Enabled, holomq::Publisher<InlinePayloadBucket<Size>>, NoTerminalPublisher>;

template <std::size_t Size, bool Enabled>
using TerminalSubscriber =
    std::conditional_t<Enabled, holomq::Subscriber<InlinePayloadBucket<Size>>,
                       NoTerminalSubscriber>;

template <bool Enabled>
using TerminalAttachment =
    std::conditional_t<Enabled, std::optional<holomq::WaitSetAttachment>, NoTerminalAttachment>;

template <bool Enabled>
using TerminalWaitHandle = std::conditional_t<Enabled, std::uint64_t, NoTerminalWaitHandle>;

template <std::size_t Size, bool GraphRetryBoundary, bool ProtectedTerminal>
class InlinePayloadPublisherImpl final : public InlinePayloadPublisher,
                                         public ProtectedTerminalPublisherBase<ProtectedTerminal> {
 public:
  explicit InlinePayloadPublisherImpl(
      holomq::Publisher<InlinePayloadBucket<Size>>&& publisher,
      TerminalPublisher<Size, ProtectedTerminal>&& terminal_publisher,
      const std::uint32_t expected_recipient_count,
      const bool zero_recipients_proven_undelivered) noexcept
      : terminal_publisher_(std::move(terminal_publisher)),
        publisher_(std::move(publisher)),
        expected_recipient_count_(expected_recipient_count),
        zero_recipients_proven_undelivered_(zero_recipients_proven_undelivered) {}

  [[nodiscard]] InlinePublicationResult emit_phase(const std::span<const std::byte> payload,
                                                   const SampleMetadata& metadata) override {
    if (closed_) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kClosed, "inline channel publisher is closed"),
      };
    }
    if (payload.size() > Size) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kInvalidArgument,
                                      "inline channel payload is larger than its bucket"),
      };
    }
    auto message = publisher_.loan();
    if (!message) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = from_holomq_error(std::move(message).error()),
      };
    }

    auto& bucket = message->payload();
    bucket.operation = InlinePayloadOperation::kData;
    bucket.metadata = metadata;
    bucket.payload_size = static_cast<std::uint32_t>(payload.size());
    if (!payload.empty()) {
      std::memcpy(bucket.bytes.data(), payload.data(), payload.size());
    }
    auto sent = publisher_.send(std::move(message).value());
    if (sent && *sent == 0 && zero_recipients_proven_undelivered_) {
      // HoloMQ's iceoryx2 backend returns the exact number of receiver connections that accepted
      // this sample. A successful submit with a zero count therefore proves no subscriber can ever
      // observe it: with publisher history disabled the sample is reclaimed, never replayed to a
      // late-connecting peer. The registry may already report the subscriber (has_subscribers
      // true) while its receiver side of the zero-copy connection is still unattached; the send is
      // then provably invisible, so retain producer authority and let the caller retry.
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kNotReady,
                                      "inline channel publication reached zero subscribers"),
      };
    }
    auto published = publication_result(std::move(sent));
    if (!published) {
      const ErrorCode code = published.error().code;
      const bool pre_submit = code == ErrorCode::kBackpressured || code == ErrorCode::kNotReady;
      return InlinePublicationResult{
          .phase = pre_submit ? InlinePublicationPhase::kPreSubmitRejected
                              : InlinePublicationPhase::kPostSubmitUncertain,
          .error = std::move(published).error(),
      };
    }
    return {.phase = InlinePublicationPhase::kAccepted, .recipient_count = published.value()};
  }

  [[nodiscard]] InlinePublicationResult emit_encoded_phase(
      const InlinePayloadEncoder& encoder, const SampleMetadata& metadata) override {
    if (closed_) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kClosed, "inline channel publisher is closed"),
      };
    }
    if (encoder.state == nullptr || encoder.encode == nullptr || encoder.capacity == 0U ||
        encoder.capacity > Size) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kInvalidArgument,
                                      "inline channel encoder exceeds its bounded bucket"),
      };
    }
    auto message = publisher_.loan();
    if (!message) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = from_holomq_error(std::move(message).error()),
      };
    }
    auto& bucket = message->payload();
    auto encoded =
        encoder.encode(encoder.state, std::span<std::byte>{bucket.bytes}.first(encoder.capacity));
    if (!encoded || *encoded == 0U || *encoded > encoder.capacity ||
        *encoded > std::numeric_limits<std::uint32_t>::max()) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = encoded ? make_channel_error(ErrorCode::kProtocolError,
                                                "inline channel encoder returned invalid size")
                           : std::move(encoded).error(),
      };
    }
    bucket.operation = InlinePayloadOperation::kData;
    bucket.metadata = metadata;
    // Runtime's direct encoder produces the semantic generated-schema bytes themselves. Catalog
    // schema envelopes use the ordinary emit path so their metadata can retain the inner size.
    bucket.metadata.byte_size = *encoded;
    bucket.payload_size = static_cast<std::uint32_t>(*encoded);
    auto sent = publisher_.send(std::move(message).value());
    if (sent && *sent == 0 && zero_recipients_proven_undelivered_) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = make_channel_error(ErrorCode::kNotReady,
                                      "inline channel publication reached zero subscribers"),
      };
    }
    auto published = publication_result(std::move(sent));
    if (!published) {
      const ErrorCode code = published.error().code;
      const bool pre_submit = code == ErrorCode::kBackpressured || code == ErrorCode::kNotReady;
      return InlinePublicationResult{
          .phase = pre_submit ? InlinePublicationPhase::kPreSubmitRejected
                              : InlinePublicationPhase::kPostSubmitUncertain,
          .error = std::move(published).error(),
      };
    }
    return {.phase = InlinePublicationPhase::kAccepted, .recipient_count = published.value()};
  }

  [[nodiscard]] expected<void, Error> close_stream() override {
    const SampleMetadata metadata{
        .byte_size = 0U,
        .payload_kind = PayloadKind::kInlineTyped,
    };
    InlinePublicationResult result = emit_protected_terminal({}, metadata);
    if (result.phase == InlinePublicationPhase::kAccepted) {
      return {};
    }
    if (result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      if (terminal_publication_error_consumes_authority(result.error.code)) {
        return make_unexpected(std::move(result.error));
      }
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
    return make_unexpected(std::move(result.error));
  }

  // Only the ProtectedTerminal=true specialization has the corresponding virtual base.
  // NOLINTNEXTLINE(clang-diagnostic-inconsistent-missing-override)
  [[nodiscard]] InlinePublicationResult emit_protected_terminal(
      const std::span<const std::byte> payload, const SampleMetadata& metadata) {
    if (closed_) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kClosed},
      };
    }
    if (payload.size() > Size) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kInvalidArgument},
      };
    }
    if (!publisher_.has_subscribers()) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kNotReady},
      };
    }
    if constexpr (ProtectedTerminal) {
      if (!terminal_publisher_.has_subscribers()) {
        return InlinePublicationResult{
            .phase = InlinePublicationPhase::kPreSubmitRejected,
            .error = Error{ErrorCode::kNotReady},
        };
      }
    }
    auto& terminal_lane = [&]() -> auto& {
      if constexpr (ProtectedTerminal) {
        return terminal_publisher_;
      } else {
        return publisher_;
      }
    }();
    auto message = terminal_lane.loan();
    if (!message) {
      return InlinePublicationResult{
          .phase = InlinePublicationPhase::kPreSubmitRejected,
          .error = from_holomq_error(std::move(message).error()),
      };
    }
    auto& bucket = message->payload();
    bucket.operation =
        payload.empty() ? InlinePayloadOperation::kEndOfStream : InlinePayloadOperation::kData;
    bucket.metadata = metadata;
    bucket.payload_size = static_cast<std::uint32_t>(payload.size());
    if (!payload.empty()) {
      std::memcpy(bucket.bytes.data(), payload.data(), payload.size());
    }
    // HoloMQ send is synchronous through recipient enqueue: every data send accepted for an arm
    // returns before close can submit this terminal record. The receiver always probes/takes its
    // own data queue before the terminal queue, so an empty data queue at terminal observation
    // means that arm legitimately missed a prior partial publication; it is not an ordering race.
    auto sent = terminal_lane.send(std::move(message).value());
    auto published = [&]() {
      if constexpr (ProtectedTerminal) {
        return from_holomq_publication_result(std::move(sent), expected_recipient_count_);
      } else {
        return publication_result(std::move(sent));
      }
    }();
    if (!published) {
      Error error = std::move(published).error();
      const bool consumes_authority = terminal_publication_error_consumes_authority(error.code);
      if (consumes_authority) {
        closed_ = true;
      }
      return InlinePublicationResult{
          .phase = consumes_authority ? InlinePublicationPhase::kPostSubmitUncertain
                                      : InlinePublicationPhase::kPreSubmitRejected,
          .error = std::move(error),
      };
    }
    closed_ = true;
    return {
        .phase = InlinePublicationPhase::kAccepted,
        .recipient_count = published.value(),
    };
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override {
    auto data = from_holomq_result(publisher_.update_connections());
    if (!data) {
      return data;
    }
    if constexpr (!ProtectedTerminal) {
      return {};
    } else {
      return from_holomq_result(terminal_publisher_.update_connections());
    }
  }

  [[nodiscard]] bool has_subscribers() const noexcept override {
    if constexpr (ProtectedTerminal) {
      return publisher_.has_subscribers() && terminal_publisher_.has_subscribers();
    }
    return publisher_.has_subscribers();
  }

 private:
  [[nodiscard]] expected<std::uint32_t, Error> publication_result(
      holomq::expected<int, holomq::Error> result) const noexcept {
    if constexpr (GraphRetryBoundary) {
      return from_holomq_publication_result(std::move(result), expected_recipient_count_, true);
    }
    return from_holomq_publication_result(std::move(result), expected_recipient_count_);
  }

  [[no_unique_address]] TerminalPublisher<Size, ProtectedTerminal> terminal_publisher_;
  holomq::Publisher<InlinePayloadBucket<Size>> publisher_;
  std::uint32_t expected_recipient_count_{};
  bool closed_{};
  bool zero_recipients_proven_undelivered_{};
};

template <std::size_t Size, bool ProtectedTerminal>
class InlinePayloadSubscriberImpl final : public InlinePayloadSubscriber,
                                          public ProtectedTerminalSubscriberBase<ProtectedTerminal>,
                                          public StagedDataWaitSetSubscriber {
 public:
  explicit InlinePayloadSubscriberImpl(
      holomq::Context& context, holomq::Subscriber<InlinePayloadBucket<Size>>&& subscriber,
      TerminalSubscriber<Size, ProtectedTerminal>&& terminal_subscriber) noexcept
      : terminal_subscriber_(std::move(terminal_subscriber)),
        subscriber_(std::move(subscriber)),
        context_(&context) {}

  // Read from the port rather than a cached flag so this can never disagree with the transport.
  [[nodiscard]] bool event_wake_enabled() const noexcept override {
    return subscriber_.is_event_wake_enabled();
  }

  // The parent Channel runtime owns this transport context and outlives every endpoint created
  // from it, so this borrow stays valid for the subscriber's lifetime.
  [[nodiscard]] void* wait_context_runtime() noexcept override { return context_; }

  [[nodiscard]] bool wait_set_attached() const noexcept override {
    if constexpr (ProtectedTerminal) {
      return subscriber_.has_wait_owner() || terminal_subscriber_.has_wait_owner();
    }
    return subscriber_.has_wait_owner();
  }

  [[nodiscard]] bool readiness_handle_attached() const noexcept override {
    if constexpr (ProtectedTerminal) {
      return subscriber_.has_readiness_owner() || terminal_subscriber_.has_readiness_owner();
    }
    return subscriber_.has_readiness_owner();
  }

  [[nodiscard]] expected<ChannelReadinessHandle, Error> readiness() override {
    auto claim = claim_readiness_owner();
    if (!claim) {
      return make_unexpected(std::move(claim).error());
    }
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto ready = has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }

    std::vector<holomq::ReadinessHandle> handles;
    handles.reserve(ProtectedTerminal ? 2U : 1U);
    auto data = subscriber_.readiness();
    if (!data) {
      return make_unexpected(from_holomq_error(std::move(data).error()));
    }
    handles.push_back(std::move(*data));
    if constexpr (ProtectedTerminal) {
      auto terminal = terminal_subscriber_.readiness();
      if (!terminal) {
        return make_unexpected(from_holomq_error(std::move(terminal).error()));
      }
      handles.push_back(std::move(*terminal));
    }
    return ChannelReadinessHandleAccess::create(std::move(handles), {}, *ready);
  }

  [[nodiscard]] expected<bool, Error> has_samples() const override {
    if (closed_) {
      return false;
    }
    auto data = from_holomq_result(subscriber_.has_samples());
    if (!data) {
      return data;
    }
    if constexpr (ProtectedTerminal) {
      // Refresh both receiver mappings before returning the OR result. A continuously ready data
      // lane must not starve the private terminal lane's backend refresh; otherwise the publisher
      // can observe a registered terminal peer whose transport mapping has already gone stale.
      auto terminal = from_holomq_result(terminal_subscriber_.has_samples());
      if (!terminal) {
        return terminal;
      }
      return inbound_.has_value() || *data || pending_terminal_.has_value() || *terminal;
    } else {
      return inbound_.has_value() || pending_terminal_.has_value() || *data;
    }
  }

  [[nodiscard]] bool has_publishers() const noexcept override {
    if constexpr (ProtectedTerminal) {
      return !closed_ && subscriber_.has_publishers() && terminal_subscriber_.has_publishers();
    }
    return !closed_ && subscriber_.has_publishers();
  }

  [[nodiscard]] expected<InlinePayloadReceiveResult, Error> receive(
      std::span<std::byte> payload) override {
    auto received = receive_borrowed();
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    if (received->payload().size() > payload.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "inline channel destination is smaller than its payload"));
    }
    // Report the borrowed length, not metadata.byte_size. The bucket carries the two independently,
    // so a short frame claiming the full width must remain distinguishable to the caller.
    std::memcpy(payload.data(), received->payload().data(), received->payload().size());
    return InlinePayloadReceiveResult{.metadata = received->metadata(),
                                      .payload_size = received->payload().size()};
  }

  [[nodiscard]] expected<InlinePayloadReceiveGuard, Error> receive_borrowed() override {
    return receive_borrowed_if_data_accepted(true);
  }

  // Data, protected-terminal, and incarnation arbitration must remain one ordered hot-path state
  // machine; extracting branches would add indirection without simplifying its invariants.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<InlinePayloadReceiveGuard, Error> receive_borrowed_if_data_accepted(
      const bool accept_data) override {
    if (payload_loan_active_ || borrowed_terminal_) {
      return make_unexpected(
          make_channel_error(ErrorCode::kFailure, "inline channel already has an active loan"));
    }
    // Keep the terminal mapping refreshed even while the data lane remains continuously ready.
    std::optional<bool> terminal_ready;  // NOLINT(misc-const-correctness)
    if constexpr (ProtectedTerminal) {
      auto ready = from_holomq_result(terminal_subscriber_.has_samples());
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
      terminal_ready = *ready;
    }
    if (!inbound_) {
      auto inbound = subscriber_.try_take();
      if (inbound) {
        inbound_.emplace(std::move(inbound).value());
      } else {
        Error data_error = from_holomq_error(std::move(inbound).error());
        if (data_error.code != ErrorCode::kNotReady) {
          return make_unexpected(std::move(data_error));
        }
      }
    }
    if constexpr (ProtectedTerminal) {
      if (!pending_terminal_.has_value() && terminal_ready.value_or(false)) {
        auto terminal_inbound = terminal_subscriber_.try_take();
        if (!terminal_inbound) {
          return make_unexpected(from_holomq_error(std::move(terminal_inbound).error()));
        }
        pending_terminal_.emplace(terminal_inbound->payload());
      }
    }
    if (inbound_) {
      if (!accept_data && inbound_->payload().operation == InlinePayloadOperation::kData) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      return borrow_bucket(inbound_->payload());
    }
    if constexpr (!ProtectedTerminal) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    } else {
      if (!pending_terminal_.has_value()) {
        if (!terminal_ready.value_or(false)) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        auto terminal_inbound = terminal_subscriber_.try_take();
        if (!terminal_inbound) {
          return make_unexpected(from_holomq_error(std::move(terminal_inbound).error()));
        }
        pending_terminal_.emplace(terminal_inbound->payload());
      }
      if (pending_terminal_->operation == InlinePayloadOperation::kEndOfStream) {
        closed_ = true;
        pending_terminal_.reset();
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (!accept_data) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      borrowed_terminal_ = true;
      return borrow_bucket(*pending_terminal_);
    }
  }

  void release_payload_loan() noexcept override {
    if (!payload_loan_active_) {
      return;
    }
    discard_staged_operation();
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void* wait_set) override {
    if (wait_set == nullptr || attachment_.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if constexpr (ProtectedTerminal) {
      if (terminal_attachment_.has_value()) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    auto* const typed_wait_set = static_cast<holomq::WaitSet*>(wait_set);
    auto attached = typed_wait_set->attach(subscriber_);
    if (!attached) {
      return make_unexpected(from_holomq_error(std::move(attached).error()));
    }
    if constexpr (ProtectedTerminal) {
      auto terminal_attached = typed_wait_set->attach(terminal_subscriber_);
      if (!terminal_attached) {
        return make_unexpected(from_holomq_error(std::move(terminal_attached).error()));
      }
      terminal_wait_handle_ = terminal_attached->handle().id();
      terminal_attachment_.emplace(std::move(*terminal_attached));
    }
    const std::uint64_t handle = attached->handle().id();
    attachment_.emplace(std::move(*attached));
    return handle;
  }

  // Only the ProtectedTerminal=true specialization has the corresponding protected-lane base.
  // NOLINTNEXTLINE(clang-diagnostic-inconsistent-missing-override)
  [[nodiscard]] std::optional<std::uint64_t> protected_terminal_wait_handle() const noexcept {
    if constexpr (ProtectedTerminal) {
      return terminal_wait_handle_ == 0U ? std::nullopt
                                         : std::optional<std::uint64_t>{terminal_wait_handle_};
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<StagedDataState> staged_data_state() const noexcept override {
    if (!inbound_ || inbound_->payload().operation != InlinePayloadOperation::kData ||
        inbound_->payload().payload_size == 0U || inbound_->payload().payload_size > Size ||
        inbound_->payload().metadata.sequence == 0U) {
      return std::nullopt;
    }
    return StagedDataState{.sequence = inbound_->payload().metadata.sequence,
                           .capture_time = inbound_->payload().metadata.capture_time()};
  }

  [[nodiscard]] std::optional<std::uint64_t> staged_terminal_sequence() const noexcept override {
    if constexpr (!ProtectedTerminal) {
      return std::nullopt;
    } else {
      return pending_terminal_.has_value() &&
                     pending_terminal_->operation == InlinePayloadOperation::kEndOfStream &&
                     pending_terminal_->metadata.sequence != 0U
                 ? std::optional<std::uint64_t>{pending_terminal_->metadata.sequence}
                 : std::nullopt;
    }
  }

  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override {
    if constexpr (ProtectedTerminal) {
      terminal_attachment_.reset();
      terminal_wait_handle_ = 0U;
    }
    attachment_.reset();
    return {};
  }

 private:
  void discard_staged_operation() noexcept {
    inbound_.reset();
    if (borrowed_terminal_) {
      pending_terminal_.reset();
      borrowed_terminal_ = false;
    }
    payload_loan_active_ = false;
  }

  [[nodiscard]] expected<InlinePayloadReceiveGuard, Error> borrow_bucket(
      const InlinePayloadBucket<Size>& bucket) {
    if (bucket.operation == InlinePayloadOperation::kEndOfStream) {
      closed_ = true;
      discard_staged_operation();
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (bucket.operation != InlinePayloadOperation::kData || bucket.payload_size == 0U ||
        bucket.payload_size > Size) {
      discard_staged_operation();
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto payload = std::span<const std::byte>{bucket.bytes}.first(bucket.payload_size);
    payload_loan_active_ = true;
    return InlinePayloadReceiveGuard{*this, payload, bucket.metadata};
  }

  [[no_unique_address]] TerminalSubscriber<Size, ProtectedTerminal> terminal_subscriber_;
  [[no_unique_address]] TerminalAttachment<ProtectedTerminal> terminal_attachment_;
  [[no_unique_address]] PendingTerminalStorage<Size, ProtectedTerminal> pending_terminal_;
  [[no_unique_address]] TerminalWaitHandle<ProtectedTerminal> terminal_wait_handle_{};
  holomq::Subscriber<InlinePayloadBucket<Size>> subscriber_;
  std::optional<holomq::WaitSetAttachment> attachment_;
  std::optional<holomq::InboundMessage<InlinePayloadBucket<Size>>> inbound_;
  holomq::Context* context_{};
  bool payload_loan_active_{};
  bool borrowed_terminal_{};
  bool closed_{};
};

class InlinePayloadWaitSetImpl final : public InlinePayloadWaitSet {
 public:
  explicit InlinePayloadWaitSetImpl(holomq::WaitSet wait_set) noexcept
      : wait_set_(std::move(wait_set)) {}

  [[nodiscard]] expected<std::uint64_t, Error> attach(WaitSetSubscriber& subscriber) override {
    return subscriber.attach_wait_set(&wait_set_);
  }

  [[nodiscard]] expected<bool, Error> wait_for(
      const std::chrono::nanoseconds timeout) noexcept override {
    const holomq::WaitStrategy strategy{
        .mode = holomq::WaitMode::KernelWait,
        .spin_duration = std::chrono::nanoseconds{0},
        .timeout = timeout,
    };
    // Use the checked wait. The unchecked entry point assumed the pump's preceding drain left
    // nothing ready, so the pre-block readiness scan would be redundant. That does not hold: an
    // endpoint can become ready between the drain and the block, and the pump then commits to
    // blocking on a sample it already owns, waiting for the next publisher notification to release
    // it. Measured on graph_intraprocess_docs_pod_1to2_per_sink_latency, that cost 41 blocks per
    // 60k samples in a 0.9-1.2 ms band -- every one of them ended in a notification, not a timeout
    // -- and moved end-to-end p999 from ~35 us to ~1.07 ms.
    //
    // The scan is not a cost on this path: it retires roughly a quarter of the blocks outright
    // (84.4k -> 63.3k per run) because it finds the sample already queued, so this trades a
    // blocking syscall for an O(N) shared-memory probe. The median is unchanged or slightly better.
    //
    // Known cost: while a subscriber's typed slot is still occupied and its raw queue holds a
    // further sample, the scan reports ready on every iteration, so the pump re-probes that
    // subscriber (a failed reservation under its input cache mutex) instead of blocking until the
    // slot frees. That window is bounded by the consumer's per-sample occupancy, not by the whole
    // backpressure interval; measured at roughly 15-20 % more pump CPU on the paced, saturated, and
    // slow-sink rows of the same benchmark, never a full core. Excluding a subscriber with a
    // retained head from the pre-block scan is the follow-up; skipping the scan for the pump again
    // is not, that was this defect.
    auto waited = wait_set_.wait(strategy);
    if (!waited) {
      return make_unexpected(from_holomq_error(std::move(waited).error()));
    }
    return *waited == holomq::WaitResult::ready;
  }

  [[nodiscard]] expected<std::size_t, Error> ready_handles(
      std::span<std::uint64_t> destination) const noexcept override {
    const auto ready = wait_set_.ready_handles();
    if (ready.size() > destination.size()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    for (std::size_t index = 0U; index < ready.size(); ++index) {
      destination[index] = ready[index].id();
    }
    return ready.size();
  }

 private:
  holomq::WaitSet wait_set_;
};

template <std::size_t Size, bool GraphRetryBoundary>
[[nodiscard]] expected<std::unique_ptr<InlinePayloadPublisher>, Error>
create_inline_payload_publisher_impl(holomq::Context& context, const std::string_view channel_name,
                                     const ChannelOptions& options,
                                     const PublisherPressureOptions pressure) {
  if (is_ordered_terminal_service_id(channel_name)) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "inline channel name occupies the reserved terminal namespace"));
  }
  std::string terminal_service_id;
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    terminal_service_id = ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "inline channel name cannot derive a protected terminal service identity"));
    }
  }
  auto publisher_options = to_holomq_publisher_options(options, pressure);
  publisher_options.enable_sample_release_callbacks = false;
  auto builder = context.publisher_builder(std::string(channel_name))
                     .service_options(to_holomq_service_options(options))
                     .publisher_options(publisher_options)
                     .template payload_type<InlinePayloadBucket<Size>>();
  if (options.replacement_attribution_required) {
    // Inline payloads own no external claims. Runtime needs HoloMQ's setup-time replacement
    // attribution contract for admitted lossy queues, but an overwritten pre-witness POD sample
    // has no additional resource to retire here.
    static_cast<void>(builder.on_sample_replaced(
        [](const InlinePayloadBucket<Size>&, const holomq::SampleReplacedInfo&) noexcept {}));
  }
  auto publisher = std::move(builder).create();
  if (!publisher) {
    return make_unexpected(from_holomq_error(std::move(publisher).error(),
                                             "failed to create inline payload channel publisher"));
  }
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    auto created = context.publisher_builder(terminal_service_id)
                       .service_options(ordered_terminal_service_options(options))
                       .publisher_options(ordered_terminal_publisher_options(options))
                       .template payload_type<InlinePayloadBucket<Size>>()
                       .create();
    if (!created) {
      return make_unexpected(from_holomq_error(
          std::move(created).error(), "failed to create protected inline terminal publisher"));
    }
    return std::make_unique<InlinePayloadPublisherImpl<Size, GraphRetryBoundary, true>>(
        std::move(publisher).value(),
        std::move(*created),
        options.expected_recipient_count,
        options.publisher_history_depth == 0U);
  }
  return std::make_unique<InlinePayloadPublisherImpl<Size, GraphRetryBoundary, false>>(
      std::move(publisher).value(),
      NoTerminalPublisher{},
      options.expected_recipient_count,
      options.publisher_history_depth == 0U);
}

template <std::size_t Size>
[[nodiscard]] expected<std::unique_ptr<InlinePayloadSubscriber>, Error>
create_inline_payload_subscriber_impl(holomq::Context& context, const std::string_view channel_name,
                                      const ChannelOptions& options) {
  if (is_ordered_terminal_service_id(channel_name)) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "inline channel name occupies the reserved terminal namespace"));
  }
  std::string terminal_service_id;
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    terminal_service_id = ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "inline channel name cannot derive a protected terminal service identity"));
    }
  }
  auto subscriber = context.subscriber_builder(std::string(channel_name))
                        .service_options(to_holomq_service_options(options))
                        .subscriber_options(to_holomq_subscriber_options(options))
                        .template payload_type<InlinePayloadBucket<Size>>()
                        .create();
  if (!subscriber) {
    return make_unexpected(from_holomq_error(std::move(subscriber).error(),
                                             "failed to create inline payload channel subscriber"));
  }
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    auto created = context.subscriber_builder(terminal_service_id)
                       .service_options(ordered_terminal_service_options(options))
                       .subscriber_options(ordered_terminal_subscriber_options())
                       .template payload_type<InlinePayloadBucket<Size>>()
                       .create();
    if (!created) {
      return make_unexpected(from_holomq_error(
          std::move(created).error(), "failed to create protected inline terminal subscriber"));
    }
    return std::make_unique<InlinePayloadSubscriberImpl<Size, true>>(
        context, std::move(subscriber).value(), std::move(*created));
  }
  return std::make_unique<InlinePayloadSubscriberImpl<Size, false>>(
      context, std::move(subscriber).value(), NoTerminalSubscriber{});
}

template <bool GraphRetryBoundary>
struct InlinePayloadPublisherFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;
  PublisherPressureOptions pressure;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<InlinePayloadPublisher>, Error> operator()() const {
    return create_inline_payload_publisher_impl<Size, GraphRetryBoundary>(
        context, channel_name, options, pressure);
  }
};

struct InlinePayloadSubscriberFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<InlinePayloadSubscriber>, Error> operator()() const {
    return create_inline_payload_subscriber_impl<Size>(context, channel_name, options);
  }
};

template <typename Factory>
[[nodiscard]] auto dispatch_inline_payload_bucket(const std::size_t payload_size, Factory& factory)
    -> decltype(factory.template operator()<8U>()) {
  const auto bucket_size = inline_payload_bucket_size(payload_size);
  if (!bucket_size.has_value()) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "inline channel payload size must be in the range [1, " +
                               std::to_string(kMaximumInlinePayloadBytes) + "] bytes"));
  }

  switch (bucket_size.value()) {
    case 8U:
      return factory.template operator()<8U>();
    case 16U:
      return factory.template operator()<16U>();
    case 32U:
      return factory.template operator()<32U>();
    case 64U:
      return factory.template operator()<64U>();
    case 128U:
      return factory.template operator()<128U>();
    case 256U:
      return factory.template operator()<256U>();
    case 512U:
      return factory.template operator()<512U>();
    case 1024U:
      return factory.template operator()<1024U>();
    case 2048U:
      return factory.template operator()<2048U>();
    case 4096U:
      return factory.template operator()<4096U>();
    case 8192U:
      return factory.template operator()<8192U>();
    case 16384U:
      return factory.template operator()<16384U>();
    case kMaximumInlinePayloadBytes:
      return factory.template operator()<kMaximumInlinePayloadBytes>();
    default:
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "unsupported inline channel payload bucket size"));
  }
}

}  // namespace

expected<std::unique_ptr<InlinePayloadPublisher>, Error> create_inline_payload_publisher(
    holomq::Context& context, const std::string_view channel_name, const std::size_t payload_size,
    const ChannelOptions& options, const TransportCapabilities& capabilities,
    const PublisherPressureOptions pressure) {
  if (auto valid = validate_inline_payload_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (auto valid = validate_inline_payload_capabilities(capabilities, options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (pressure.enables_graph_retry_boundary(options.publisher_history_depth)) {
    InlinePayloadPublisherFactory<true> factory{context, channel_name, options, pressure};
    return dispatch_inline_payload_bucket(payload_size, factory);
  }
  InlinePayloadPublisherFactory<false> factory{context, channel_name, options, pressure};
  return dispatch_inline_payload_bucket(payload_size, factory);
}

expected<std::unique_ptr<InlinePayloadSubscriber>, Error> create_inline_payload_subscriber(
    holomq::Context& context, const std::string_view channel_name, const std::size_t payload_size,
    const ChannelOptions& options, const TransportCapabilities& capabilities) {
  if (auto valid = validate_inline_payload_options(options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (auto valid = validate_inline_payload_capabilities(capabilities, options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  InlinePayloadSubscriberFactory factory{context, channel_name, options};
  return dispatch_inline_payload_bucket(payload_size, factory);
}

expected<std::unique_ptr<InlinePayloadWaitSet>, Error> create_inline_payload_wait_set(
    holomq::Context& context) {
  auto wait_set = context.create_waitset();
  if (!wait_set) {
    return make_unexpected(
        from_holomq_error(std::move(wait_set).error(), "failed to create inline channel wait set"));
  }
  return std::make_unique<InlinePayloadWaitSetImpl>(std::move(*wait_set));
}

}  // namespace holoscan::channel_internal
