/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/connection_error.hpp>
#include <holomq/detail/transport/iox2/loan_error.hpp>
#include <holomq/detail/transport/iox2/outbound_loan.hpp>
#include <holomq/detail/transport/iox2/pubsub_wake_event.hpp>
#include <holomq/detail/transport/iox2/send_error.hpp>
#include <holomq/detail/transport/iox2/unable_to_deliver_evidence.hpp>

#include <holomq/expected.hpp>

#include "iox2/backpressure_handler.hpp"
#include "iox2/notifier.hpp"
#include "iox2/port_factory_publish_subscribe.hpp"
#include "iox2/publisher.hpp"
#include "iox2/service_type.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace holomq::detail::transport {

template <typename Payload>
class PublisherPort {
  using IoxPublisher = iox2::Publisher<iox2::ServiceType::Ipc, Payload, void>;
  using IoxPortFactory = iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, Payload, void>;

 public:
  PublisherPort() = delete;
  PublisherPort(PublisherPort&&) noexcept = default;
  PublisherPort& operator=(PublisherPort&&) noexcept = delete;
  ~PublisherPort() {
    // Member destruction alone would release the borrowed handler before the backend publisher.
    inner_.reset();
    backpressure_handler_.reset();
  }

  PublisherPort(const PublisherPort&) = delete;
  PublisherPort& operator=(const PublisherPort&) = delete;

  [[nodiscard]] auto loan() -> holomq::expected<OutboundLoan<Payload>, holomq::Error> {
    auto sample = inner_->loan_uninit();
    if (!sample.has_value()) {
      return holomq::make_unexpected(map_loan_error(sample.error()));
    }
    return OutboundLoan<Payload>(std::move(sample).value());
  }

  [[nodiscard]] auto send(OutboundLoan<Payload>&& sample) noexcept
      -> holomq::expected<int, holomq::Error> {
    auto result = iox2::send(iox2::assume_init(std::move(sample).release()));
    if (!result.has_value()) {
      holomq::Error error = map_send_error(result.error());
      if (result.error() == iox2::SendError::UnableToDeliver) {
        const auto recipient = take_unable_to_deliver_recipient_if(backpressure_handler_installed_);
        if (recipient) {
          try {
            error.message = unable_to_deliver_recipient_diagnostic(*recipient);
          } catch (...) {
            // Recipient attribution is best-effort. Preserve the mapped transport error if
            // failure-path diagnostic allocation is unavailable.
          }
        }
      }
      return holomq::make_unexpected(std::move(error));
    }
    if (wake_notifier_.has_value()) {
      static_cast<void>(wake_notifier_->notify_with_custom_event_id(make_sent_sample_event_id()));
    }
    return static_cast<int>(result.value());
  }

  [[nodiscard]] auto update_connections() noexcept -> holomq::expected<void, holomq::Error> {
    auto updated = inner_->update_connections();
    if (!updated.has_value()) {
      return holomq::make_unexpected(map_connection_error(updated.error()));
    }
    return {};
  }

  [[nodiscard]] std::uint64_t number_of_subscribers() const noexcept {
    return port_factory_.dynamic_config().number_of_subscribers();
  }

 private:
  friend class Runtime;

  explicit PublisherPort(IoxPublisher inner, IoxPortFactory port_factory,
                         std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> wake_notifier,
                         std::unique_ptr<iox2::BackpressureHandler> backpressure_handler,
                         bool backpressure_handler_installed) noexcept
      : backpressure_handler_(std::move(backpressure_handler)),
        inner_(std::move(inner)),
        port_factory_(std::move(port_factory)),
        wake_notifier_(std::move(wake_notifier)),
        backpressure_handler_installed_(backpressure_handler_installed) {}

  // Declared first so default move construction establishes handler ownership before moving the
  // publisher that borrows it. Destruction explicitly retires inner_ first.
  std::unique_ptr<iox2::BackpressureHandler> backpressure_handler_;
  std::optional<IoxPublisher> inner_;
  IoxPortFactory port_factory_;
  std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> wake_notifier_;
  bool backpressure_handler_installed_{false};
};

}  // namespace holomq::detail::transport
