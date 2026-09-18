/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <initializer_list>
#include <utility>

#include <gtest/gtest.h>

#include <holomq/detail/transport/iox2/setup_error.hpp>
#include <holomq/errors.hpp>

namespace {

template <typename IoxError>
struct MappingCase {
  IoxError input;
  holomq::ErrorCode expected;
};

template <typename IoxError, typename Mapper>
void expect_mappings(const std::initializer_list<MappingCase<IoxError>> cases, Mapper mapper) {
  for (const auto& [input, expected] : cases) {
    const holomq::Error mapped = mapper(input);
    EXPECT_EQ(mapped.code, expected);
    EXPECT_FALSE(mapped.message.empty());
  }
}

TEST(Iox2SetupErrorTest, MapsServiceNameAndNodeErrors) {
  using holomq::ErrorCode;
  using holomq::detail::transport::map_node_creation_error;
  using holomq::detail::transport::map_service_name_error;

  expect_mappings<iox2::bb::SemanticStringError>(
      {{iox2::bb::SemanticStringError::InvalidContent, ErrorCode::kInvalidArgument},
       {iox2::bb::SemanticStringError::ExceedsMaximumLength, ErrorCode::kInvalidArgument}},
      map_service_name_error);
  expect_mappings<iox2::NodeCreationFailure>(
      {{iox2::NodeCreationFailure::InsufficientPermissions, ErrorCode::kPermissionDenied},
       {iox2::NodeCreationFailure::InternalError, ErrorCode::kFailure},
       {iox2::NodeCreationFailure::SystemCorrupted, ErrorCode::kFailure}},
      map_node_creation_error);
}

TEST(Iox2SetupErrorTest, MapsOpenErrorsAndOnlyCreatesWhenServiceIsMissing) {
  using IoxError = iox2::PublishSubscribeOpenError;
  using holomq::ErrorCode;
  using holomq::detail::transport::is_missing_publish_subscribe_service;
  using holomq::detail::transport::map_publish_subscribe_open_error;

  const std::initializer_list<MappingCase<IoxError>> cases = {
      {IoxError::DoesNotExist, ErrorCode::kNotFound},
      {IoxError::InternalFailure, ErrorCode::kFailure},
      {IoxError::IncompatibleTypes, ErrorCode::kIncompatible},
      {IoxError::IncompatibleMessagingPattern, ErrorCode::kIncompatible},
      {IoxError::IncompatibleAttributes, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedMinBufferSize, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedMinHistorySize, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedMinSubscriberBorrowedSamples, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedAmountOfPublishers, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedAmountOfSubscribers, ErrorCode::kIncompatible},
      {IoxError::DoesNotSupportRequestedAmountOfNodes, ErrorCode::kIncompatible},
      {IoxError::IncompatibleOverflowBehavior, ErrorCode::kIncompatible},
      {IoxError::InsufficientPermissions, ErrorCode::kPermissionDenied},
      {IoxError::ServiceInCorruptedState, ErrorCode::kStaleEndpoint},
      {IoxError::HangsInCreation, ErrorCode::kNotReady},
      {IoxError::ExceedsMaxNumberOfNodes, ErrorCode::kResourceExhausted},
      {IoxError::IsMarkedForDestruction, ErrorCode::kNotReady},
  };

  expect_mappings<IoxError>(cases, map_publish_subscribe_open_error);
  for (const auto& mapping : cases) {
    EXPECT_EQ(is_missing_publish_subscribe_service(mapping.input),
              mapping.input == IoxError::DoesNotExist);
  }
}

TEST(Iox2SetupErrorTest, MapsPublishSubscribeServiceErrors) {
  using IoxError = iox2::PublishSubscribeOpenOrCreateError;
  using holomq::ErrorCode;
  using holomq::detail::transport::map_publish_subscribe_open_or_create_error;

  expect_mappings<IoxError>(
      {{IoxError::OpenDoesNotExist, ErrorCode::kNotFound},
       {IoxError::OpenInternalFailure, ErrorCode::kFailure},
       {IoxError::OpenIncompatibleTypes, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleMessagingPattern, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleAttributes, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedMinBufferSize, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedMinHistorySize, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedMinSubscriberBorrowedSamples,
        ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedAmountOfPublishers, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedAmountOfSubscribers, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedAmountOfNodes, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleOverflowBehavior, ErrorCode::kIncompatible},
       {IoxError::OpenInsufficientPermissions, ErrorCode::kPermissionDenied},
       {IoxError::OpenServiceInCorruptedState, ErrorCode::kStaleEndpoint},
       {IoxError::OpenHangsInCreation, ErrorCode::kNotReady},
       {IoxError::OpenExceedsMaxNumberOfNodes, ErrorCode::kResourceExhausted},
       {IoxError::OpenIsMarkedForDestruction, ErrorCode::kNotReady},
       {IoxError::CreateServiceInCorruptedState, ErrorCode::kStaleEndpoint},
       {IoxError::CreateSubscriberBufferMustBeLargerThanHistorySize, ErrorCode::kInvalidArgument},
       {IoxError::CreateAlreadyExists, ErrorCode::kNotReady},
       {IoxError::CreateInsufficientPermissions, ErrorCode::kPermissionDenied},
       {IoxError::CreateInternalFailure, ErrorCode::kFailure},
       {IoxError::CreateIsBeingCreatedByAnotherInstance, ErrorCode::kNotReady},
       {IoxError::CreateOldConnectionsStillActive, ErrorCode::kStaleEndpoint},
       {IoxError::CreateHangsInCreation, ErrorCode::kNotReady},
       {IoxError::SystemInFlux, ErrorCode::kNotReady}},
      map_publish_subscribe_open_or_create_error);
}

TEST(Iox2SetupErrorTest, MapsEventServiceErrors) {
  using IoxError = iox2::EventOpenOrCreateError;
  using holomq::ErrorCode;
  using holomq::detail::transport::map_event_open_or_create_error;

  expect_mappings<IoxError>(
      {{IoxError::OpenDoesNotExist, ErrorCode::kNotFound},
       {IoxError::OpenInsufficientPermissions, ErrorCode::kPermissionDenied},
       {IoxError::OpenServiceInCorruptedState, ErrorCode::kStaleEndpoint},
       {IoxError::OpenIncompatibleDeadline, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleMessagingPattern, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleAttributes, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleNotifierCreatedEvent, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleNotifierDroppedEvent, ErrorCode::kIncompatible},
       {IoxError::OpenIncompatibleNotifierDeadEvent, ErrorCode::kIncompatible},
       {IoxError::OpenInternalFailure, ErrorCode::kFailure},
       {IoxError::OpenHangsInCreation, ErrorCode::kNotReady},
       {IoxError::OpenDoesNotSupportRequestedAmountOfNotifiers, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedAmountOfListeners, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedMaxEventId, ErrorCode::kIncompatible},
       {IoxError::OpenDoesNotSupportRequestedAmountOfNodes, ErrorCode::kIncompatible},
       {IoxError::OpenExceedsMaxNumberOfNodes, ErrorCode::kResourceExhausted},
       {IoxError::OpenIsMarkedForDestruction, ErrorCode::kNotReady},
       {IoxError::CreateServiceInCorruptedState, ErrorCode::kStaleEndpoint},
       {IoxError::CreateInternalFailure, ErrorCode::kFailure},
       {IoxError::CreateIsBeingCreatedByAnotherInstance, ErrorCode::kNotReady},
       {IoxError::CreateAlreadyExists, ErrorCode::kNotReady},
       {IoxError::CreateHangsInCreation, ErrorCode::kNotReady},
       {IoxError::CreateInsufficientPermissions, ErrorCode::kPermissionDenied},
       {IoxError::CreateOldConnectionsStillActive, ErrorCode::kStaleEndpoint},
       {IoxError::SystemInFlux, ErrorCode::kNotReady}},
      map_event_open_or_create_error);
}

TEST(Iox2SetupErrorTest, RetriesOnlyTransientEventServiceCleanupErrors) {
  using IoxError = iox2::EventOpenOrCreateError;
  using holomq::detail::transport::is_retryable_event_open_or_create_error;

  EXPECT_TRUE(is_retryable_event_open_or_create_error(IoxError::OpenDoesNotExist));
  EXPECT_TRUE(is_retryable_event_open_or_create_error(IoxError::OpenServiceInCorruptedState));
  EXPECT_TRUE(is_retryable_event_open_or_create_error(IoxError::OpenIsMarkedForDestruction));
  EXPECT_TRUE(is_retryable_event_open_or_create_error(IoxError::CreateServiceInCorruptedState));
  EXPECT_TRUE(is_retryable_event_open_or_create_error(IoxError::CreateOldConnectionsStillActive));

  EXPECT_FALSE(is_retryable_event_open_or_create_error(IoxError::OpenIncompatibleAttributes));
  EXPECT_FALSE(is_retryable_event_open_or_create_error(IoxError::OpenInsufficientPermissions));
  EXPECT_FALSE(is_retryable_event_open_or_create_error(IoxError::CreateInternalFailure));
  EXPECT_FALSE(is_retryable_event_open_or_create_error(IoxError::SystemInFlux));
}

TEST(Iox2SetupErrorTest, MapsEndpointCreationErrors) {
  using holomq::ErrorCode;
  using namespace holomq::detail::transport;

  expect_mappings<iox2::PublisherCreateError>(
      {{iox2::PublisherCreateError::ExceedsMaxSupportedPublishers, ErrorCode::kResourceExhausted},
       {iox2::PublisherCreateError::UnableToCreateDataSegment, ErrorCode::kResourceExhausted},
       {iox2::PublisherCreateError::FailedToDeployThreadsafetyPolicy, ErrorCode::kFailure},
       {iox2::PublisherCreateError::UnableToCreatePortTag, ErrorCode::kFailure}},
      map_publisher_create_error);
  expect_mappings<iox2::SubscriberCreateError>(
      {{iox2::SubscriberCreateError::ExceedsMaxSupportedSubscribers, ErrorCode::kResourceExhausted},
       {iox2::SubscriberCreateError::BufferSizeExceedsMaxSupportedBufferSizeOfService,
        ErrorCode::kIncompatible},
       {iox2::SubscriberCreateError::FailedToDeployThreadsafetyPolicy, ErrorCode::kFailure},
       {iox2::SubscriberCreateError::UnableToCreatePortTag, ErrorCode::kFailure}},
      map_subscriber_create_error);
  expect_mappings<iox2::NotifierCreateError>(
      {{iox2::NotifierCreateError::ExceedsMaxSupportedNotifiers, ErrorCode::kResourceExhausted},
       {iox2::NotifierCreateError::FailedToDeployThreadsafetyPolicy, ErrorCode::kFailure},
       {iox2::NotifierCreateError::UnableToCreatePortTag, ErrorCode::kFailure}},
      map_notifier_create_error);
  expect_mappings<iox2::ListenerCreateError>(
      {{iox2::ListenerCreateError::ExceedsMaxSupportedListeners, ErrorCode::kResourceExhausted},
       {iox2::ListenerCreateError::ResourceCreationFailed, ErrorCode::kResourceExhausted},
       {iox2::ListenerCreateError::FailedToDeployThreadsafetyPolicy, ErrorCode::kFailure},
       {iox2::ListenerCreateError::UnableToCreatePortTag, ErrorCode::kFailure}},
      map_listener_create_error);
}

TEST(Iox2SetupErrorTest, MapsWaitSetErrors) {
  using holomq::ErrorCode;
  using holomq::detail::transport::map_wait_set_attachment_error;
  using holomq::detail::transport::map_wait_set_create_error;
  using holomq::detail::transport::map_wait_set_run_error_code;

  expect_mappings<iox2::WaitSetCreateError>(
      {{iox2::WaitSetCreateError::InsufficientResources, ErrorCode::kResourceExhausted},
       {iox2::WaitSetCreateError::InternalError, ErrorCode::kFailure}},
      map_wait_set_create_error);
  expect_mappings<iox2::WaitSetAttachmentError>(
      {{iox2::WaitSetAttachmentError::InsufficientCapacity, ErrorCode::kResourceExhausted},
       {iox2::WaitSetAttachmentError::AlreadyAttached, ErrorCode::kInvalidArgument},
       {iox2::WaitSetAttachmentError::InternalError, ErrorCode::kFailure},
       {iox2::WaitSetAttachmentError::InsufficientResources, ErrorCode::kResourceExhausted}},
      map_wait_set_attachment_error);

  EXPECT_EQ(map_wait_set_run_error_code(iox2::WaitSetRunError::InsufficientPermissions),
            ErrorCode::kPermissionDenied);
  EXPECT_EQ(map_wait_set_run_error_code(iox2::WaitSetRunError::InternalError), ErrorCode::kFailure);
  EXPECT_EQ(map_wait_set_run_error_code(iox2::WaitSetRunError::NoAttachments),
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(map_wait_set_run_error_code(iox2::WaitSetRunError::TerminationRequest),
            ErrorCode::kClosed);
  EXPECT_EQ(map_wait_set_run_error_code(iox2::WaitSetRunError::Interrupt), ErrorCode::kInterrupted);
}

TEST(Iox2SetupErrorTest, PrependsSetupContext) {
  holomq::Error error{holomq::ErrorCode::kIncompatible, "incompatible service"};
  error = holomq::detail::transport::prepend_error_context(std::move(error), "publisher 'video'");

  EXPECT_EQ(error.code, holomq::ErrorCode::kIncompatible);
  EXPECT_EQ(error.message, "publisher 'video': incompatible service");
}

}  // namespace
