/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/bb/semantic_string.hpp"
#include "iox2/listener_error.hpp"
#include "iox2/node_failure_enums.hpp"
#include "iox2/notifier_error.hpp"
#include "iox2/publisher_error.hpp"
#include "iox2/service_builder_event_error.hpp"
#include "iox2/service_builder_publish_subscribe_error.hpp"
#include "iox2/subscriber_error.hpp"
#include "iox2/waitset_enums.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace holomq::detail::transport {

[[nodiscard]] inline auto make_setup_error(const holomq::ErrorCode code,
                                           const std::string_view message) -> holomq::Error {
  return holomq::Error{code, std::string(message)};
}

[[nodiscard]] inline auto prepend_error_context(holomq::Error error, std::string context)
    -> holomq::Error {
  if (!error.message.empty()) {
    context.append(": ").append(error.message);
  }
  error.message = std::move(context);
  return error;
}

[[nodiscard]] inline auto map_service_name_error(const iox2::bb::SemanticStringError error)
    -> holomq::Error {
  switch (error) {
    case iox2::bb::SemanticStringError::InvalidContent:
      return make_setup_error(holomq::ErrorCode::kInvalidArgument,
                              "service name contains invalid content");
    case iox2::bb::SemanticStringError::ExceedsMaximumLength:
      return make_setup_error(holomq::ErrorCode::kInvalidArgument,
                              "service name exceeds the maximum length");
  }
  return make_setup_error(holomq::ErrorCode::kInvalidArgument, "invalid service name");
}

[[nodiscard]] inline auto map_node_creation_error(const iox2::NodeCreationFailure error)
    -> holomq::Error {
  switch (error) {
    case iox2::NodeCreationFailure::InsufficientPermissions:
      return make_setup_error(holomq::ErrorCode::kPermissionDenied,
                              "insufficient permission to create transport node");
    case iox2::NodeCreationFailure::SystemCorrupted:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "transport node resources are corrupted");
    case iox2::NodeCreationFailure::InternalError:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "transport node creation failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "transport node creation failed");
}

[[nodiscard]] inline auto is_missing_publish_subscribe_service(
    const iox2::PublishSubscribeOpenError error) noexcept -> bool {
  return error == iox2::PublishSubscribeOpenError::DoesNotExist;
}

[[nodiscard]] inline auto map_publish_subscribe_open_error(
    const iox2::PublishSubscribeOpenError error) -> holomq::Error {
  using IoxError = iox2::PublishSubscribeOpenError;
  switch (error) {
    case IoxError::DoesNotExist:
      return make_setup_error(holomq::ErrorCode::kNotFound,
                              "publish-subscribe service does not exist");
    case IoxError::IncompatibleTypes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "publish-subscribe payload or user-header type is incompatible");
    case IoxError::IncompatibleMessagingPattern:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service uses an incompatible messaging pattern");
    case IoxError::IncompatibleAttributes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "publish-subscribe service attributes are incompatible");
    case IoxError::DoesNotSupportRequestedMinBufferSize:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service buffer capacity is below the requested minimum");
    case IoxError::DoesNotSupportRequestedMinHistorySize:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service history capacity is below the requested minimum");
    case IoxError::DoesNotSupportRequestedMinSubscriberBorrowedSamples:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service borrowed-sample capacity is below the requested minimum");
    case IoxError::DoesNotSupportRequestedAmountOfPublishers:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service publisher capacity is below the requested minimum");
    case IoxError::DoesNotSupportRequestedAmountOfSubscribers:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service subscriber capacity is below the requested minimum");
    case IoxError::DoesNotSupportRequestedAmountOfNodes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service node capacity is below the requested minimum");
    case IoxError::IncompatibleOverflowBehavior:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service queue overflow policy is incompatible");
    case IoxError::InsufficientPermissions:
      return make_setup_error(holomq::ErrorCode::kPermissionDenied,
                              "insufficient permission to open service");
    case IoxError::ServiceInCorruptedState:
      return make_setup_error(holomq::ErrorCode::kStaleEndpoint,
                              "publish-subscribe service resources are stale or corrupted");
    case IoxError::ExceedsMaxNumberOfNodes:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "publish-subscribe service node capacity is exhausted");
    case IoxError::HangsInCreation:
    case IoxError::IsMarkedForDestruction:
      return make_setup_error(holomq::ErrorCode::kNotReady,
                              "publish-subscribe service is transitioning; retry later");
    case IoxError::InternalFailure:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "publish-subscribe service open failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "publish-subscribe service open failed");
}

[[nodiscard]] inline auto map_publish_subscribe_open_or_create_error(
    const iox2::PublishSubscribeOpenOrCreateError error) -> holomq::Error {
  using IoxError = iox2::PublishSubscribeOpenOrCreateError;
  switch (error) {
    case IoxError::OpenDoesNotExist:
      return make_setup_error(holomq::ErrorCode::kNotFound,
                              "publish-subscribe service does not exist");
    case IoxError::OpenIncompatibleTypes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "publish-subscribe payload or user-header type is incompatible");
    case IoxError::OpenIncompatibleMessagingPattern:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service uses an incompatible messaging pattern");
    case IoxError::OpenIncompatibleAttributes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "publish-subscribe service attributes are incompatible");
    case IoxError::OpenDoesNotSupportRequestedMinBufferSize:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service buffer capacity is below the requested minimum");
    case IoxError::OpenDoesNotSupportRequestedMinHistorySize:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service history capacity is below the requested minimum");
    case IoxError::OpenDoesNotSupportRequestedMinSubscriberBorrowedSamples:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service borrowed-sample capacity is below the requested minimum");
    case IoxError::OpenDoesNotSupportRequestedAmountOfPublishers:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service publisher capacity is below the requested minimum");
    case IoxError::OpenDoesNotSupportRequestedAmountOfSubscribers:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service subscriber capacity is below the requested minimum");
    case IoxError::OpenDoesNotSupportRequestedAmountOfNodes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service node capacity is below the requested minimum");
    case IoxError::OpenIncompatibleOverflowBehavior:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "service queue overflow policy is incompatible");
    case IoxError::OpenInsufficientPermissions:
    case IoxError::CreateInsufficientPermissions:
      return make_setup_error(holomq::ErrorCode::kPermissionDenied,
                              "insufficient permission to open or create service");
    case IoxError::OpenServiceInCorruptedState:
    case IoxError::CreateServiceInCorruptedState:
    case IoxError::CreateOldConnectionsStillActive:
      return make_setup_error(holomq::ErrorCode::kStaleEndpoint,
                              "publish-subscribe service resources are stale or corrupted");
    case IoxError::OpenExceedsMaxNumberOfNodes:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "publish-subscribe service node capacity is exhausted");
    case IoxError::CreateSubscriberBufferMustBeLargerThanHistorySize:
      return make_setup_error(holomq::ErrorCode::kInvalidArgument,
                              "subscriber buffer must be at least as large as service history");
    case IoxError::OpenHangsInCreation:
    case IoxError::OpenIsMarkedForDestruction:
    case IoxError::CreateAlreadyExists:
    case IoxError::CreateIsBeingCreatedByAnotherInstance:
    case IoxError::CreateHangsInCreation:
    case IoxError::SystemInFlux:
      return make_setup_error(holomq::ErrorCode::kNotReady,
                              "publish-subscribe service is transitioning; retry later");
    case IoxError::OpenInternalFailure:
    case IoxError::CreateInternalFailure:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "publish-subscribe service setup failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "publish-subscribe service setup failed");
}

[[nodiscard]] inline auto map_event_open_or_create_error(const iox2::EventOpenOrCreateError error)
    -> holomq::Error {
  using IoxError = iox2::EventOpenOrCreateError;
  switch (error) {
    case IoxError::OpenDoesNotExist:
      return make_setup_error(holomq::ErrorCode::kNotFound, "event service does not exist");
    case IoxError::OpenIncompatibleDeadline:
    case IoxError::OpenIncompatibleMessagingPattern:
    case IoxError::OpenIncompatibleAttributes:
    case IoxError::OpenIncompatibleNotifierCreatedEvent:
    case IoxError::OpenIncompatibleNotifierDroppedEvent:
    case IoxError::OpenIncompatibleNotifierDeadEvent:
    case IoxError::OpenDoesNotSupportRequestedAmountOfNotifiers:
    case IoxError::OpenDoesNotSupportRequestedAmountOfListeners:
    case IoxError::OpenDoesNotSupportRequestedMaxEventId:
    case IoxError::OpenDoesNotSupportRequestedAmountOfNodes:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "event service configuration is incompatible");
    case IoxError::OpenInsufficientPermissions:
    case IoxError::CreateInsufficientPermissions:
      return make_setup_error(holomq::ErrorCode::kPermissionDenied,
                              "insufficient permission to open or create event service");
    case IoxError::OpenServiceInCorruptedState:
    case IoxError::CreateServiceInCorruptedState:
    case IoxError::CreateOldConnectionsStillActive:
      return make_setup_error(holomq::ErrorCode::kStaleEndpoint,
                              "event service resources are stale or corrupted");
    case IoxError::OpenExceedsMaxNumberOfNodes:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "event service node capacity is exhausted");
    case IoxError::OpenHangsInCreation:
    case IoxError::OpenIsMarkedForDestruction:
    case IoxError::CreateIsBeingCreatedByAnotherInstance:
    case IoxError::CreateAlreadyExists:
    case IoxError::CreateHangsInCreation:
    case IoxError::SystemInFlux:
      return make_setup_error(holomq::ErrorCode::kNotReady,
                              "event service is transitioning; retry later");
    case IoxError::OpenInternalFailure:
    case IoxError::CreateInternalFailure:
      return make_setup_error(holomq::ErrorCode::kFailure, "event service setup failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "event service setup failed");
}

[[nodiscard]] inline auto map_publisher_create_error(const iox2::PublisherCreateError error)
    -> holomq::Error {
  switch (error) {
    case iox2::PublisherCreateError::ExceedsMaxSupportedPublishers:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "publisher capacity is exhausted");
    case iox2::PublisherCreateError::UnableToCreateDataSegment:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "publisher data segment could not be created");
    case iox2::PublisherCreateError::FailedToDeployThreadsafetyPolicy:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "publisher thread-safety policy setup failed");
    case iox2::PublisherCreateError::UnableToCreatePortTag:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "publisher cleanup tag could not be created");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "publisher creation failed");
}

[[nodiscard]] inline auto map_subscriber_create_error(const iox2::SubscriberCreateError error)
    -> holomq::Error {
  switch (error) {
    case iox2::SubscriberCreateError::ExceedsMaxSupportedSubscribers:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "subscriber capacity is exhausted");
    case iox2::SubscriberCreateError::BufferSizeExceedsMaxSupportedBufferSizeOfService:
      return make_setup_error(holomq::ErrorCode::kIncompatible,
                              "subscriber queue exceeds the service buffer capacity");
    case iox2::SubscriberCreateError::FailedToDeployThreadsafetyPolicy:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "subscriber thread-safety policy setup failed");
    case iox2::SubscriberCreateError::UnableToCreatePortTag:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "subscriber cleanup tag could not be created");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "subscriber creation failed");
}

[[nodiscard]] inline auto map_notifier_create_error(const iox2::NotifierCreateError error)
    -> holomq::Error {
  switch (error) {
    case iox2::NotifierCreateError::ExceedsMaxSupportedNotifiers:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "event notifier capacity is exhausted");
    case iox2::NotifierCreateError::FailedToDeployThreadsafetyPolicy:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "event notifier thread-safety policy setup failed");
    case iox2::NotifierCreateError::UnableToCreatePortTag:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "event notifier cleanup tag could not be created");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "event notifier creation failed");
}

[[nodiscard]] inline auto map_listener_create_error(const iox2::ListenerCreateError error)
    -> holomq::Error {
  switch (error) {
    case iox2::ListenerCreateError::ExceedsMaxSupportedListeners:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "event listener capacity is exhausted");
    case iox2::ListenerCreateError::ResourceCreationFailed:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "event listener resource creation failed");
    case iox2::ListenerCreateError::FailedToDeployThreadsafetyPolicy:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "event listener thread-safety policy setup failed");
    case iox2::ListenerCreateError::UnableToCreatePortTag:
      return make_setup_error(holomq::ErrorCode::kFailure,
                              "event listener cleanup tag could not be created");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "event listener creation failed");
}

[[nodiscard]] inline auto map_wait_set_create_error(const iox2::WaitSetCreateError error)
    -> holomq::Error {
  switch (error) {
    case iox2::WaitSetCreateError::InsufficientResources:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "insufficient resources to create wait set");
    case iox2::WaitSetCreateError::InternalError:
      return make_setup_error(holomq::ErrorCode::kFailure, "wait-set creation failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "wait-set creation failed");
}

[[nodiscard]] inline auto map_wait_set_attachment_error(const iox2::WaitSetAttachmentError error)
    -> holomq::Error {
  switch (error) {
    case iox2::WaitSetAttachmentError::InsufficientCapacity:
    case iox2::WaitSetAttachmentError::InsufficientResources:
      return make_setup_error(holomq::ErrorCode::kResourceExhausted,
                              "wait-set attachment capacity or resources are exhausted");
    case iox2::WaitSetAttachmentError::AlreadyAttached:
      return make_setup_error(holomq::ErrorCode::kInvalidArgument,
                              "notification is already attached to a wait set");
    case iox2::WaitSetAttachmentError::InternalError:
      return make_setup_error(holomq::ErrorCode::kFailure, "wait-set attachment failed internally");
  }
  return make_setup_error(holomq::ErrorCode::kFailure, "wait-set attachment failed");
}

[[nodiscard]] inline auto map_wait_set_run_error_code(const iox2::WaitSetRunError error)
    -> holomq::ErrorCode {
  switch (error) {
    case iox2::WaitSetRunError::InsufficientPermissions:
      return holomq::ErrorCode::kPermissionDenied;
    case iox2::WaitSetRunError::NoAttachments:
      return holomq::ErrorCode::kInvalidArgument;
    case iox2::WaitSetRunError::TerminationRequest:
      return holomq::ErrorCode::kClosed;
    case iox2::WaitSetRunError::Interrupt:
      return holomq::ErrorCode::kInterrupted;
    case iox2::WaitSetRunError::InternalError:
      return holomq::ErrorCode::kFailure;
  }
  return holomq::ErrorCode::kFailure;
}

}  // namespace holomq::detail::transport
