/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/authoring/channel_registry.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::authoring::detail {

/// @file
/// @brief Prototype build-time payload registration and typed Graph thunks.

/// @brief Select the public Channel endpoint templates for one registered type.
/// @tparam T Exact published payload type.
template <PublishedPayload T,
          bool DescriptorTensor = holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>>
struct RegisteredChannelEndpointTypes {
  using Publisher = ChannelPublisher<T>;
  using Subscriber = ChannelSubscriber<T>;
};

template <PublishedPayload T>
struct RegisteredChannelEndpointTypes<T, true> {
  using Root = typename DescriptorTensorSchemaNativeTraits<T>::root_type;
  using Payload =
      schema::DescriptorFlatBufferPayload<Root,
                                          DescriptorTensorSchemaNativeTraits<T>::metadata_capacity,
                                          DescriptorTensorSchemaNativeTraits<T>::tensor_count>;
  using Publisher = ChannelTensorSchemaPublisher<Payload>;
  using Subscriber = ChannelTensorSchemaSubscriber<Payload>;
};

template <PublishedPayload T,
          bool DescriptorTensor = holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>>
struct RegisteredDescriptorChannelAdapter {};

template <PublishedPayload T>
struct RegisteredDescriptorChannelAdapter<T, true> {
  using ChannelTypes = RegisteredChannelEndpointTypes<T, true>;
  using Root = typename ChannelTypes::Root;
  using Payload = typename ChannelTypes::Payload;
  using Publisher = typename ChannelTypes::Publisher;
  using Subscriber = typename ChannelTypes::Subscriber;
  using WireEnvelope = channel_internal::TensorSchemaRuntimeEnvelope<Payload>;

  [[nodiscard]] static expected<std::array<Tensor, Payload::max_tensors>, Error> tensors(
      const T& value) {
    std::array<Tensor, Payload::max_tensors> result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
      const Tensor* tensor = DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, index);
      if (tensor == nullptr) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "PYTHON_CHANNEL_SCHEMA_TENSOR_FIELD_MISSING"});
      }
      result[index] = *tensor;
    }
    return result;
  }

  [[nodiscard]] static expected<std::span<const std::byte>, Error> encode_metadata(
      const T& value, std::span<const Tensor> tensors, WireEnvelope& envelope) {
    const holoscan::detail::RuntimePortTypeOps* operations =
        holoscan::detail::runtime_port_type_ops<T>();
    if (operations == nullptr || operations->encode_descriptor_tensor_loans == nullptr ||
        operations->wire_capacity != sizeof(WireEnvelope)) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_SCHEMA_TENSOR_CODEC_UNAVAILABLE"});
    }
    auto encoded = operations->encode_descriptor_tensor_loans(
        &value,
        tensors,
        SampleMetadata{},
        channel_internal::kCompatibilitySchemaEndpointGeneration,
        std::as_writable_bytes(std::span<WireEnvelope, 1U>{&envelope, 1U}));
    if (!encoded) {
      return make_unexpected(std::move(encoded).error());
    }
    return envelope.payload.bytes();
  }

  [[nodiscard]] static expected<std::uint32_t, Error> emit(Publisher& publisher,
                                                           ForeignSchemaValue value,
                                                           EmitOptions options) {
    auto native = PythonSchemaNativeTraits<T>::from_foreign(value);
    if (!native) {
      return make_unexpected(std::move(native).error());
    }
    auto field_tensors = tensors(*native);
    if (!field_tensors) {
      return make_unexpected(std::move(field_tensors).error());
    }
    WireEnvelope envelope{};
    auto metadata = encode_metadata(*native, *field_tensors, envelope);
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    return publisher.emit(*metadata, *field_tensors, std::move(options));
  }

  [[nodiscard]] static expected<std::uint32_t, Error> emit_captured(
      Publisher& publisher, ForeignSchemaValue value, std::int64_t capture_timestamp_ns,
      EmitOptions options) {
    auto native = PythonSchemaNativeTraits<T>::from_foreign(value);
    if (!native) {
      return make_unexpected(std::move(native).error());
    }
    auto field_tensors = tensors(*native);
    if (!field_tensors) {
      return make_unexpected(std::move(field_tensors).error());
    }
    WireEnvelope envelope{};
    auto metadata = encode_metadata(*native, *field_tensors, envelope);
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    return publisher.emit_captured(
        *metadata, *field_tensors, capture_timestamp_ns, std::move(options));
  }

  [[nodiscard]] static expected<ForeignSchemaSample, Error> materialize(
      ChannelResolvedTensorBundle<Payload> bundle) {
    auto payload = Payload::copy_from_trusted_metadata(
        bundle.metadata.bytes(),
        std::span<const PayloadDescriptor>{bundle.payloads},
        channel_internal::kCompatibilitySchemaEndpointGeneration);
    if (!payload) {
      return make_unexpected(std::move(payload).error());
    }
    WireEnvelope envelope{
        .metadata = bundle.sample_metadata,
        .payload = std::move(*payload),
    };
    const holoscan::detail::RuntimePortTypeOps* operations =
        holoscan::detail::runtime_port_type_ops<T>();
    if (operations == nullptr || operations->prepare_sample_storage == nullptr ||
        operations->destroy_sample == nullptr ||
        operations->assign_descriptor_tensors_sample == nullptr ||
        operations->sample_payload == nullptr || operations->sample_size != sizeof(Sample<T>) ||
        operations->sample_alignment != alignof(Sample<T>)) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_SCHEMA_TENSOR_CODEC_UNAVAILABLE"});
    }

    alignas(Sample<T>) std::array<std::byte, sizeof(Sample<T>)> sample_storage{};
    auto prepared = operations->prepare_sample_storage(sample_storage.data());
    if (!prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    struct SampleGuard {
      const holoscan::detail::RuntimePortTypeOps* operations;
      void* storage;
      ~SampleGuard() { operations->destroy_sample(storage); }
    };
    const SampleGuard guard{operations, sample_storage.data()};

    auto assigned = operations->assign_descriptor_tensors_sample(
        sample_storage.data(),
        std::as_bytes(std::span<const WireEnvelope, 1U>{&envelope, 1U}),
        std::span<Tensor>{bundle.tensors},
        holoscan::detail::DescriptorTensorWireMetadata{bundle.sample_metadata},
        holoscan::detail::DescriptorTensorSampleMetadata{bundle.sample_metadata});
    if (!assigned) {
      return make_unexpected(std::move(assigned).error());
    }
    auto foreign = PythonSchemaNativeTraits<T>::to_foreign(
        *static_cast<const T*>(operations->sample_payload(sample_storage.data())));
    if (!foreign) {
      return make_unexpected(std::move(foreign).error());
    }
    return ForeignSchemaSample{
        .has_data = true,
        .data = std::move(*foreign),
        .metadata = bundle.sample_metadata,
    };
  }

  [[nodiscard]] static expected<ForeignSchemaSample, Error> try_receive(Subscriber& subscriber) {
    auto bundle = subscriber.receive_tensors();
    return bundle
               ? materialize(std::move(*bundle))
               : expected<ForeignSchemaSample, Error>{make_unexpected(std::move(bundle).error())};
  }

  [[nodiscard]] static expected<ForeignSchemaSample, Error> wait_and_receive(
      Subscriber& subscriber, std::chrono::steady_clock::time_point deadline,
      const std::stop_token& token) {
    while (true) {
      auto bundle = subscriber.receive_tensors();
      if (bundle) {
        return materialize(std::move(*bundle));
      }
      if (bundle.error().code != ErrorCode::kNotReady) {
        return make_unexpected(std::move(bundle).error());
      }
      auto ready = subscriber.wait_for_sample(deadline, token);
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
    }
  }
};

/// @brief Instantiate the public Graph payload templates for one registered type.
/// @tparam T Exact payload type supported by Graph external topics.
template <holoscan::detail::ExternalTopicPayload T>
struct PayloadRegistrationThunks {
  using ChannelTypes = RegisteredChannelEndpointTypes<T>;
  using RegisteredChannelPublisher = typename ChannelTypes::Publisher;
  using RegisteredChannelSubscriber = typename ChannelTypes::Subscriber;

  inline static const SchemaTransportProfile schema_transport_profile = [] {
    if constexpr (holoscan::detail::kRuntimeGeneratedSchemaPayload<T> ||
                  holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      return schema_package<T>().transport_profile();
    } else {
      return SchemaTransportProfile{};
    }
  }();

  static constexpr auto descriptor_tensor_fields = [] {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      constexpr auto names = DescriptorTensorSchemaNativeTraits<T>::tensor_field_names;
      constexpr auto ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
      static_assert(names.size() == ids.size(),
                    "descriptor Tensor field names and IDs must have equal lengths");
      static_assert(names.size() == holoscan::detail::descriptor_tensor_schema_tensor_count_v<T>,
                    "descriptor Tensor field names must match the generated field inventory");
      static_assert(names.size() <= kMaximumTensorSchemaLoanFields,
                    "descriptor Tensor field inventory exceeds the multi-Tensor loan limit");
      std::array<RegisteredDescriptorTensorField, names.size()> fields{};
      for (std::size_t index = 0U; index < names.size(); ++index) {
        fields[index] = RegisteredDescriptorTensorField{
            .name = names[index],
            .field_id = ids[index],
            .ordinal = static_cast<std::uint16_t>(index),
        };
      }
      return fields;
    } else {
      return std::array<RegisteredDescriptorTensorField, 0U>{};
    }
  }();

  static constexpr bool channel_inline_supported =
      CatalogChannelScalar<T> || (CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T> &&
                                  !holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>);
  static constexpr bool channel_catalog_publisher_supported =
      channel_inline_supported || std::same_as<T, Tensor> ||
      (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
       HasPythonSchemaNativeTraits<T>);
  static constexpr bool channel_subscriber_supported = channel_catalog_publisher_supported;

  template <typename Result, typename Callable>
  [[nodiscard]] static Result channel_call(Callable&& callable) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::forward<Callable>(callable)();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  [[nodiscard]] static expected<OpaqueChannelPublisher, Error> hold_channel_publisher(
      expected<RegisteredChannelPublisher, Error> publisher) noexcept {
    if (!publisher) {
      return make_unexpected(std::move(publisher).error());
    }
    return channel_call<expected<OpaqueChannelPublisher, Error>>([&] {
      OpaqueChannelPublisher result;
      result.holder_ = std::make_shared<RegisteredChannelPublisher>(std::move(*publisher));
      result.local_type_token_ = &holoscan::detail::kPublishedPayloadLocalTypeToken<T>;
      return result;
    });
  }

  [[nodiscard]] static expected<OpaqueChannelSubscriber, Error> hold_channel_subscriber(
      expected<RegisteredChannelSubscriber, Error> subscriber) noexcept {
    if (!subscriber) {
      return make_unexpected(std::move(subscriber).error());
    }
    return channel_call<expected<OpaqueChannelSubscriber, Error>>([&] {
      OpaqueChannelSubscriber result;
      result.holder_ = std::make_shared<RegisteredChannelSubscriber>(std::move(*subscriber));
      result.local_type_token_ = &holoscan::detail::kPublishedPayloadLocalTypeToken<T>;
      return result;
    });
  }

  [[nodiscard]] static expected<RegisteredChannelPublisher*, Error> channel_publisher(
      OpaqueChannelPublisher& publisher) noexcept {
    if (publisher.local_type_token_ != &holoscan::detail::kPublishedPayloadLocalTypeToken<T> ||
        publisher.holder_ == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "AUTHORING_CHANNEL_PUBLISHER_TYPE_MISMATCH"});
    }
    return static_cast<RegisteredChannelPublisher*>(publisher.holder_.get());
  }

  [[nodiscard]] static expected<const RegisteredChannelPublisher*, Error> channel_publisher(
      const OpaqueChannelPublisher& publisher) noexcept {
    if (publisher.local_type_token_ != &holoscan::detail::kPublishedPayloadLocalTypeToken<T> ||
        publisher.holder_ == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "AUTHORING_CHANNEL_PUBLISHER_TYPE_MISMATCH"});
    }
    return static_cast<const RegisteredChannelPublisher*>(publisher.holder_.get());
  }

  [[nodiscard]] static expected<RegisteredChannelSubscriber*, Error> channel_subscriber(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if (subscriber.local_type_token_ != &holoscan::detail::kPublishedPayloadLocalTypeToken<T> ||
        subscriber.holder_ == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "AUTHORING_CHANNEL_SUBSCRIBER_TYPE_MISMATCH"});
    }
    return static_cast<RegisteredChannelSubscriber*>(subscriber.holder_.get());
  }

  [[nodiscard]] static expected<const RegisteredChannelSubscriber*, Error> channel_subscriber(
      const OpaqueChannelSubscriber& subscriber) noexcept {
    if (subscriber.local_type_token_ != &holoscan::detail::kPublishedPayloadLocalTypeToken<T> ||
        subscriber.holder_ == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "AUTHORING_CHANNEL_SUBSCRIBER_TYPE_MISMATCH"});
    }
    return static_cast<const RegisteredChannelSubscriber*>(subscriber.holder_.get());
  }

  static expected<OpaqueChannelPublisher, Error> channel_create_publisher(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      ChannelAttachOptions options) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
    } else {
      return channel_call<expected<OpaqueChannelPublisher, Error>>([&] {
        if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
          return hold_channel_publisher(
              runtime.create_tensor_schema_publisher<typename ChannelTypes::Payload>(
                  endpoint, std::move(options)));
        } else {
          return hold_channel_publisher(runtime.create_publisher<T>(endpoint, std::move(options)));
        }
      });
    }
  }

  static expected<OpaqueChannelPublisher, Error> channel_create_name_publisher(
      ChannelRuntime& runtime, std::string_view channel_name,
      const ChannelOptions& options) noexcept {
    if constexpr (std::same_as<T, Tensor>) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "CHANNEL_NAME_TENSOR_PUBLISHER_REQUIRES_CATALOG"});
    } else if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_NAME_PAYLOAD_TYPE_UNSUPPORTED"});
    } else {
      return channel_call<expected<OpaqueChannelPublisher, Error>>([&] {
        if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
          return hold_channel_publisher(
              runtime.create_tensor_schema_publisher<typename ChannelTypes::Payload>(channel_name,
                                                                                     options));
        } else {
          return hold_channel_publisher(runtime.create_publisher<T>(channel_name, options));
        }
      });
    }
  }

  static expected<OpaqueChannelSubscriber, Error> channel_create_subscriber(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      ChannelAttachOptions options) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
    } else {
      return channel_call<expected<OpaqueChannelSubscriber, Error>>([&] {
        if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
          return hold_channel_subscriber(
              runtime.create_tensor_schema_subscriber<typename ChannelTypes::Payload>(
                  endpoint, std::move(options)));
        } else {
          return hold_channel_subscriber(
              runtime.create_subscriber<T>(endpoint, std::move(options)));
        }
      });
    }
  }

  static expected<OpaqueChannelSubscriber, Error> channel_create_name_subscriber(
      ChannelRuntime& runtime, std::string_view channel_name,
      const ChannelOptions& options) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_NAME_PAYLOAD_TYPE_UNSUPPORTED"});
    } else {
      return channel_call<expected<OpaqueChannelSubscriber, Error>>([&] {
        if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
          return hold_channel_subscriber(
              runtime.create_tensor_schema_subscriber<typename ChannelTypes::Payload>(channel_name,
                                                                                      options));
        } else {
          return hold_channel_subscriber(runtime.create_subscriber<T>(channel_name, options));
        }
      });
    }
  }

  static expected<std::uint32_t, Error> channel_emit_scalar(OpaqueChannelPublisher& publisher,
                                                            ParamScalarValue value,
                                                            EmitOptions options) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        if (!typed) {
          return make_unexpected(std::move(typed).error());
        }
        T decoded{};
        if (!value.try_as(decoded)) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE"});
        }
        return (*typed)->emit(std::move(decoded), std::move(options));
      });
    }
  }

  static expected<std::uint32_t, Error> channel_emit_captured_scalar(
      OpaqueChannelPublisher& publisher, ParamScalarValue value, std::int64_t capture_timestamp_ns,
      EmitOptions options) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        if (!typed) {
          return make_unexpected(std::move(typed).error());
        }
        T decoded{};
        if (!value.try_as(decoded)) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE"});
        }
        return (*typed)->emit_captured(
            std::move(decoded), capture_timestamp_ns, std::move(options));
      });
    }
  }

  // The erased callback ABI owns payload arguments consistently across supported and unsupported
  // rows; these fallback thunks intentionally accept and discard that owning value.
  static expected<std::uint32_t, Error> channel_emit_object(
      OpaqueChannelPublisher&,
      ForeignPayloadValue,  // NOLINT(performance-unnecessary-value-param)
      EmitOptions) noexcept {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_OBJECT_PAYLOAD_UNSUPPORTED"});
  }

  static expected<std::uint32_t, Error> channel_emit_captured_object(
      OpaqueChannelPublisher&,
      ForeignPayloadValue,  // NOLINT(performance-unnecessary-value-param)
      std::int64_t, EmitOptions) noexcept {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_OBJECT_PAYLOAD_UNSUPPORTED"});
  }

  static expected<std::uint32_t, Error> channel_emit_schema(OpaqueChannelPublisher& publisher,
                                                            ForeignSchemaValue value,
                                                            EmitOptions options) noexcept {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
                  HasPythonSchemaNativeTraits<T>) {
      return channel_call<expected<std::uint32_t, Error>>([&]() {
        auto typed = channel_publisher(publisher);
        return typed ? RegisteredDescriptorChannelAdapter<T>::emit(
                           **typed, std::move(value), std::move(options))
                     : expected<std::uint32_t, Error>{make_unexpected(std::move(typed).error())};
      });
    } else if constexpr (!(CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T>)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        if (!typed) {
          return make_unexpected(std::move(typed).error());
        }
        auto decoded = PythonSchemaNativeTraits<T>::from_foreign(value);
        if (!decoded) {
          return make_unexpected(std::move(decoded).error());
        }
        return (*typed)->emit(std::move(*decoded), std::move(options));
      });
    }
  }

  static expected<std::uint32_t, Error> channel_emit_captured_schema(
      OpaqueChannelPublisher& publisher, ForeignSchemaValue value,
      std::int64_t capture_timestamp_ns, EmitOptions options) noexcept {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
                  HasPythonSchemaNativeTraits<T>) {
      return channel_call<expected<std::uint32_t, Error>>([&]() {
        auto typed = channel_publisher(publisher);
        return typed ? RegisteredDescriptorChannelAdapter<T>::emit_captured(
                           **typed, std::move(value), capture_timestamp_ns, std::move(options))
                     : expected<std::uint32_t, Error>{make_unexpected(std::move(typed).error())};
      });
    } else if constexpr (!(CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T>)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        if (!typed) {
          return make_unexpected(std::move(typed).error());
        }
        auto decoded = PythonSchemaNativeTraits<T>::from_foreign(value);
        if (!decoded) {
          return make_unexpected(std::move(decoded).error());
        }
        return (*typed)->emit_captured(
            std::move(*decoded), capture_timestamp_ns, std::move(options));
      });
    }
  }

  static expected<TensorPublishLoan, Error> channel_loan_tensor(
      OpaqueChannelPublisher& publisher, TensorLoanRequest request) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<TensorPublishLoan, Error>>(
          [&]() -> expected<TensorPublishLoan, Error> {
            auto typed = channel_publisher(publisher);
            return typed ? (*typed)->loan_tensor(request)
                         : make_unexpected(std::move(typed).error());
          });
    }
  }

  static expected<std::uint32_t, Error> channel_emit_tensor_loan(OpaqueChannelPublisher& publisher,
                                                                 TensorPublishLoan loan,
                                                                 EmitOptions options) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        return typed ? (*typed)->emit(std::move(loan), std::move(options))
                     : make_unexpected(std::move(typed).error());
      });
    }
  }

  static expected<std::uint32_t, Error> channel_emit_tensor_copy(OpaqueChannelPublisher& publisher,
                                                                 const Tensor& source,
                                                                 ::cuda::stream_ref producer_stream,
                                                                 EmitOptions options) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<std::uint32_t, Error>>([&]() -> expected<std::uint32_t, Error> {
        auto typed = channel_publisher(publisher);
        return typed ? (*typed)->emit_copy(source, producer_stream, std::move(options))
                     : make_unexpected(std::move(typed).error());
      });
    }
  }

  static expected<void, Error> channel_publisher_close(OpaqueChannelPublisher& publisher) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<void, Error>>([&]() -> expected<void, Error> {
        auto typed = channel_publisher(publisher);
        return typed ? (*typed)->close() : make_unexpected(std::move(typed).error());
      });
    }
  }

  static expected<void, Error> channel_update_connections(
      OpaqueChannelPublisher& publisher) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      auto typed = channel_publisher(publisher);
      return typed ? (*typed)->update_connections() : make_unexpected(std::move(typed).error());
    }
  }

  static bool channel_has_subscribers(OpaqueChannelPublisher& publisher) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return false;
    } else {
      auto typed = channel_publisher(publisher);
      return typed && (*typed)->has_subscribers();
    }
  }

  static expected<TimePoint, Error> channel_make_capture_time(
      const OpaqueChannelPublisher& publisher, std::int64_t timestamp_ns) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      auto typed = channel_publisher(publisher);
      return typed ? (*typed)->make_capture_time(timestamp_ns)
                   : make_unexpected(std::move(typed).error());
    }
  }

  static expected<void, Error> channel_report_clock_health(OpaqueChannelPublisher& publisher,
                                                           ClockHealthReport report) noexcept {
    if constexpr (!channel_catalog_publisher_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<void, Error>>([&]() -> expected<void, Error> {
        auto typed = channel_publisher(publisher);
        return typed ? (*typed)->report_clock_health(std::move(report))
                     : make_unexpected(std::move(typed).error());
      });
    }
  }

  template <typename Receive>
  static expected<ForeignScalarSample, Error> channel_receive_scalar(
      OpaqueChannelSubscriber& subscriber, Receive&& receive) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<ForeignScalarSample, Error>>(
          [&]() -> expected<ForeignScalarSample, Error> {
            auto typed = channel_subscriber(subscriber);
            if (!typed) {
              return make_unexpected(std::move(typed).error());
            }
            auto sample = std::forward<Receive>(receive)(**typed);
            if (!sample) {
              return make_unexpected(std::move(sample).error());
            }
            return ForeignScalarSample{.has_data = true,
                                       .data = ParamScalarValue::from(sample->data),
                                       .metadata = sample->metadata};
          });
    }
  }

  static expected<ForeignScalarSample, Error> channel_try_receive_scalar(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_scalar(subscriber, [](ChannelSubscriber<T>& typed) {
        return typed.template receive_sample<T>();
      });
    }
  }

  static expected<ForeignScalarSample, Error> channel_wait_and_receive_scalar(
      OpaqueChannelSubscriber& subscriber, std::chrono::steady_clock::time_point deadline,
      const std::stop_token& token) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_scalar(subscriber, [&](ChannelSubscriber<T>& typed) {
        return typed.template wait_and_receive_sample<T>(deadline, token);
      });
    }
  }

  static expected<ForeignObjectSample, Error> channel_try_receive_object(
      OpaqueChannelSubscriber&) noexcept {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_OBJECT_PAYLOAD_UNSUPPORTED"});
  }

  static expected<ForeignObjectSample, Error> channel_wait_and_receive_object(
      OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
      const std::stop_token&) noexcept {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "AUTHORING_CHANNEL_OBJECT_PAYLOAD_UNSUPPORTED"});
  }

  template <typename Receive>
  static expected<ForeignSchemaSample, Error> channel_receive_schema(
      OpaqueChannelSubscriber& subscriber, Receive&& receive) noexcept {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
                  HasPythonSchemaNativeTraits<T>) {
      return channel_call<expected<ForeignSchemaSample, Error>>([&]() {
        auto typed = channel_subscriber(subscriber);
        return typed ? std::forward<Receive>(receive)(**typed)
                     : expected<ForeignSchemaSample, Error>{
                           make_unexpected(std::move(typed).error())};
      });
    } else if constexpr (!(CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T>)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<ForeignSchemaSample, Error>>(
          [&]() -> expected<ForeignSchemaSample, Error> {
            auto typed = channel_subscriber(subscriber);
            if (!typed) {
              return make_unexpected(std::move(typed).error());
            }
            auto sample = std::forward<Receive>(receive)(**typed);
            if (!sample) {
              return make_unexpected(std::move(sample).error());
            }
            auto converted = PythonSchemaNativeTraits<T>::to_foreign(sample->data);
            if (!converted) {
              return make_unexpected(std::move(converted).error());
            }
            return ForeignSchemaSample{
                .has_data = true, .data = std::move(*converted), .metadata = sample->metadata};
          });
    }
  }

  static expected<ForeignSchemaSample, Error> channel_try_receive_schema(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
                  HasPythonSchemaNativeTraits<T>) {
      return channel_receive_schema(subscriber, [](RegisteredChannelSubscriber& typed) {
        return RegisteredDescriptorChannelAdapter<T>::try_receive(typed);
      });
    } else if constexpr (!(CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T>)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_schema(subscriber, [](ChannelSubscriber<T>& typed) {
        return typed.template receive_sample<T>();
      });
    }
  }

  static expected<ForeignSchemaSample, Error> channel_wait_and_receive_schema(
      OpaqueChannelSubscriber& subscriber, std::chrono::steady_clock::time_point deadline,
      const std::stop_token& token) noexcept {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> &&
                  HasPythonSchemaNativeTraits<T>) {
      return channel_receive_schema(subscriber, [&](RegisteredChannelSubscriber& typed) {
        return RegisteredDescriptorChannelAdapter<T>::wait_and_receive(typed, deadline, token);
      });
    } else if constexpr (!(CatalogChannelSchema<T> && HasPythonSchemaNativeTraits<T>)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_schema(subscriber, [&](ChannelSubscriber<T>& typed) {
        return typed.template wait_and_receive_sample<T>(deadline, token);
      });
    }
  }

  template <typename Receive>
  static expected<ForeignTensorSample, Error> channel_receive_tensor(
      OpaqueChannelSubscriber& subscriber, Receive&& receive) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<ForeignTensorSample, Error>>(
          [&]() -> expected<ForeignTensorSample, Error> {
            auto typed = channel_subscriber(subscriber);
            if (!typed) {
              return make_unexpected(std::move(typed).error());
            }
            auto sample = std::forward<Receive>(receive)(**typed);
            if (!sample) {
              return make_unexpected(std::move(sample).error());
            }
            return ForeignTensorSample{
                .has_data = true, .data = std::move(sample->data), .metadata = sample->metadata};
          });
    }
  }

  static expected<ForeignTensorSample, Error> channel_try_receive_tensor(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_tensor(
          subscriber, [](ChannelSubscriber<Tensor>& typed) { return typed.receive_sample(); });
    }
  }

  static expected<ForeignTensorSample, Error> channel_wait_and_receive_tensor(
      OpaqueChannelSubscriber& subscriber, std::chrono::steady_clock::time_point deadline,
      const std::stop_token& token) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_receive_tensor(subscriber, [&](ChannelSubscriber<Tensor>& typed) {
        return typed.wait_and_receive_sample(deadline, token);
      });
    }
  }

  static expected<void, Error> channel_wait(OpaqueChannelSubscriber& subscriber,
                                            std::chrono::steady_clock::time_point deadline,
                                            const std::stop_token& token) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<void, Error>>([&]() -> expected<void, Error> {
        auto typed = channel_subscriber(subscriber);
        return typed ? (*typed)->wait_for_sample(deadline, token)
                     : make_unexpected(std::move(typed).error());
      });
    }
  }

  static expected<ChannelReadinessHandle, Error> channel_subscriber_readiness(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<ChannelReadinessHandle, Error>>(
          [&]() -> expected<ChannelReadinessHandle, Error> {
            auto typed = channel_subscriber(subscriber);
            return typed ? (*typed)->readiness() : make_unexpected(std::move(typed).error());
          });
    }
  }

  static expected<void, Error> channel_subscriber_close(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<void, Error>>([&]() -> expected<void, Error> {
        auto typed = channel_subscriber(subscriber);
        return typed ? (*typed)->close() : make_unexpected(std::move(typed).error());
      });
    }
  }

  static expected<void, Error> channel_refresh_connections(
      OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<void, Error>>([&]() -> expected<void, Error> {
        auto typed = channel_subscriber(subscriber);
        if (!typed) {
          return make_unexpected(std::move(typed).error());
        }
        if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
          // Descriptor-schema subscriber discovery is updated by its backend endpoint directly.
          return {};
        } else {
          return (*typed)->refresh_connections();
        }
      });
    }
  }

  static expected<bool, Error> channel_has_samples(
      const OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
      return channel_call<expected<bool, Error>>([&]() -> expected<bool, Error> {
        auto typed = channel_subscriber(subscriber);
        return typed ? (*typed)->has_samples() : make_unexpected(std::move(typed).error());
      });
    }
  }

  static bool channel_has_publishers(const OpaqueChannelSubscriber& subscriber) noexcept {
    if constexpr (!channel_subscriber_supported) {
      return false;
    } else {
      auto typed = channel_subscriber(subscriber);
      return typed && (*typed)->has_publishers();
    }
  }

  inline static const RegisteredChannelThunks channel_thunks{
      .create_publisher = &PayloadRegistrationThunks::channel_create_publisher,
      .create_name_publisher = &PayloadRegistrationThunks::channel_create_name_publisher,
      .create_subscriber = &PayloadRegistrationThunks::channel_create_subscriber,
      .create_name_subscriber = &PayloadRegistrationThunks::channel_create_name_subscriber,
      .emit_scalar = &PayloadRegistrationThunks::channel_emit_scalar,
      .emit_captured_scalar = &PayloadRegistrationThunks::channel_emit_captured_scalar,
      .emit_object = &PayloadRegistrationThunks::channel_emit_object,
      .emit_captured_object = &PayloadRegistrationThunks::channel_emit_captured_object,
      .emit_schema = &PayloadRegistrationThunks::channel_emit_schema,
      .emit_captured_schema = &PayloadRegistrationThunks::channel_emit_captured_schema,
      .loan_tensor = &PayloadRegistrationThunks::channel_loan_tensor,
      .emit_tensor_loan = &PayloadRegistrationThunks::channel_emit_tensor_loan,
      .emit_tensor_copy = &PayloadRegistrationThunks::channel_emit_tensor_copy,
      .publisher_close = &PayloadRegistrationThunks::channel_publisher_close,
      .update_connections = &PayloadRegistrationThunks::channel_update_connections,
      .has_subscribers = &PayloadRegistrationThunks::channel_has_subscribers,
      .make_capture_time = &PayloadRegistrationThunks::channel_make_capture_time,
      .report_clock_health = &PayloadRegistrationThunks::channel_report_clock_health,
      .try_receive_scalar = &PayloadRegistrationThunks::channel_try_receive_scalar,
      .wait_and_receive_scalar = &PayloadRegistrationThunks::channel_wait_and_receive_scalar,
      .try_receive_object = &PayloadRegistrationThunks::channel_try_receive_object,
      .wait_and_receive_object = &PayloadRegistrationThunks::channel_wait_and_receive_object,
      .try_receive_schema = &PayloadRegistrationThunks::channel_try_receive_schema,
      .wait_and_receive_schema = &PayloadRegistrationThunks::channel_wait_and_receive_schema,
      .try_receive_tensor = &PayloadRegistrationThunks::channel_try_receive_tensor,
      .wait_and_receive_tensor = &PayloadRegistrationThunks::channel_wait_and_receive_tensor,
      .wait = &PayloadRegistrationThunks::channel_wait,
      .subscriber_readiness = &PayloadRegistrationThunks::channel_subscriber_readiness,
      .subscriber_close = &PayloadRegistrationThunks::channel_subscriber_close,
      .refresh_connections = &PayloadRegistrationThunks::channel_refresh_connections,
      .has_samples = &PayloadRegistrationThunks::channel_has_samples,
      .has_publishers = &PayloadRegistrationThunks::channel_has_publishers,
  };

  static OpaqueFlowRef add_flow(Graph& graph, const OpaquePortRef& output,
                                const OpaquePortRef& input, ConnectionOptions options) {
    const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* output_declaration = output.resolve(expected, PortDirection::kOutput);
    const void* input_declaration = input.resolve(expected, PortDirection::kInput);
    const auto flow = holoscan::detail::GraphPayloadRegistrationAccess::add_flow(
        graph,
        *static_cast<const OutputBase*>(output_declaration),
        *static_cast<const InputBase*>(input_declaration),
        expected,
        expected,
        std::move(options));
    return OpaqueFlowRef{holoscan::detail::GraphPayloadRegistrationAccess::state(graph), flow};
  }

  static expected<OpaquePortRef, Error> bind_spec_input(OperatorSpec& spec, std::string_view name,
                                                        std::optional<std::uint32_t> queue_depth,
                                                        bool optional) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto holder = std::make_shared<Input<T>>(name);
      auto& builder = spec.input(*holder, name);
      if (queue_depth) {
        builder.queue_depth(*queue_depth);
      }
      if (optional) {
        builder.optional();
      }
      return OpaquePortRef::held<T>(
          holder,
          [](const void* value) -> const void* { return static_cast<const Input<T>*>(value); },
          PortDirection::kInput,
          nullptr,
          &builder);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(Error{error.code()});
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<OpaquePortRef, Error> bind_spec_output(
      OperatorSpec& spec, std::string_view name, std::optional<std::uint32_t> max_emits) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto holder = std::make_shared<Output<T>>(name);
      auto& builder = spec.output(*holder, name);
      if (max_emits) {
        builder.max_emits_per_compute(*max_emits);
      }
      return OpaquePortRef::held<T>(
          holder,
          [](const void* value) -> const void* { return static_cast<const Output<T>*>(value); },
          PortDirection::kOutput,
          nullptr,
          &builder,
          std::same_as<T, Tensor> || holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(Error{error.code()});
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<void, Error> input_expected_rate(const OpaquePortRef& input, Rate rate) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
      (void)input.resolve(expected, PortDirection::kInput);
      auto* builder = static_cast<InputSpecBuilder*>(const_cast<void*>(input.authoring_builder_));
      if (builder == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      builder->expected_rate(rate);
      return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<void, Error> tensor_input_expects(const OpaquePortRef& input,
                                                    const TensorInputSpec& spec) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        (void)input.resolve(expected, PortDirection::kInput);
        auto* builder = static_cast<InputSpecBuilder*>(const_cast<void*>(input.authoring_builder_));
        if (builder == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        builder->expects_tensor(spec);
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> tensor_input_accepts_projection(
      const OpaquePortRef& input) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        (void)input.resolve(expected, PortDirection::kInput);
        auto* builder = static_cast<InputSpecBuilder*>(const_cast<void*>(input.authoring_builder_));
        if (builder == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        builder->accepts_projected_tensor();
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static OpaqueFlowRef add_tensor_projection(Graph& graph, const OpaquePortRef& output,
                                             const OpaquePortRef& input,
                                             std::string_view field_name,
                                             ConnectionOptions options) {
    if constexpr (!holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      throw RuntimeError{
          Error{ErrorCode::kNotSupported, "GRAPH_TENSOR_PROJECTION_SOURCE_UNSUPPORTED"}};
    } else {
      const PayloadIdentityRecord source_identity =
          holoscan::detail::operator_spec_payload_identity<T>();
      const auto* source =
          static_cast<const Output<T>*>(output.resolve(source_identity, PortDirection::kOutput));
      const auto field = std::ranges::find_if(
          descriptor_tensor_fields, [field_name](const RegisteredDescriptorTensorField& candidate) {
            return candidate.name == field_name;
          });
      if (field == descriptor_tensor_fields.end()) {
        std::string message{"GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN port="};
        const auto state = output.state_.lock();
        const auto owner = state && output.operator_type_token_ != nullptr
                               ? holoscan::detail::graph_operator_name(
                                     state, output.index_, output.handle_identity_.get())
                               : std::string_view{};
        if (!owner.empty()) {
          message.append(owner);
          message.push_back('.');
        }
        message.append(source->name());
        message.append(" field_name=");
        message.append(field_name);
        message.append(" known_fields=");
        for (std::size_t index = 0U; index < descriptor_tensor_fields.size(); ++index) {
          if (index != 0U) {
            message.push_back(',');
          }
          message.append(descriptor_tensor_fields[index].name);
        }
        throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
      }
      const PayloadIdentityRecord target_identity =
          holoscan::detail::operator_spec_payload_identity<Tensor>();
      const auto* target =
          static_cast<const Input<Tensor>*>(input.resolve(target_identity, PortDirection::kInput));
      const auto flow = holoscan::detail::GraphPayloadRegistrationAccess::add_tensor_projection(
          graph,
          *source,
          *target,
          source_identity,
          TensorProjectionRecord{.field_id = field->field_id, .tensor_index = field->ordinal},
          std::move(options));
      return OpaqueFlowRef{holoscan::detail::GraphPayloadRegistrationAccess::state(graph), flow};
    }
  }

  static expected<void, Error> output_nominal_rate(const OpaquePortRef& output,
                                                   Rate rate) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
      (void)output.resolve(expected, PortDirection::kOutput);
      if (output.authoring_builder_ == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      if constexpr (std::same_as<T, Tensor> ||
                    holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
        auto* builder =
            static_cast<TensorOutputSpecBuilder*>(const_cast<void*>(output.authoring_builder_));
        builder->nominal_rate(rate);
      } else {
        auto* builder =
            static_cast<OutputSpecBuilder*>(const_cast<void*>(output.authoring_builder_));
        builder->nominal_rate(rate);
      }
      return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<void, Error> output_follows_input(const OpaquePortRef& output,
                                                    const OpaquePortRef& input,
                                                    RateRatio ratio) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
      const auto* declaration =
          static_cast<const Input<T>*>(input.resolve(expected, PortDirection::kInput));
      // The output may intentionally have a different payload type. Resolve its own identity only
      // to enforce output direction and declaration liveness before selecting the erased builder.
      (void)output.resolve(output.payload_identity_, PortDirection::kOutput);
      if (output.authoring_builder_ == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      if (output.tensor_output_builder_) {
        auto* builder =
            static_cast<TensorOutputSpecBuilder*>(const_cast<void*>(output.authoring_builder_));
        builder->follows(*declaration, ratio);
      } else {
        auto* builder =
            static_cast<OutputSpecBuilder*>(const_cast<void*>(output.authoring_builder_));
        builder->follows(*declaration, ratio);
      }
      return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static PortRef contract_input_ref(const OpaquePortRef& input) {
    return PortRef{resolve_contract_input(input)};
  }

  static Contract& contract_input_exact(Contract& contract, const OpaquePortRef& input,
                                        Exact selector) {
    return contract.input(resolve_contract_input(input), selector);
  }

  static Contract& contract_input_latest_within(Contract& contract, const OpaquePortRef& input,
                                                LatestWithin selector) {
    return contract.input(resolve_contract_input(input), selector);
  }

  static Contract& contract_input_drain_all(Contract& contract, const OpaquePortRef& input,
                                            DrainAll selector) {
    return contract.input(resolve_contract_input(input), selector);
  }

  static Contract& contract_input_bracketed_window(Contract& contract, const OpaquePortRef& input,
                                                   BracketedWindow selector) {
    return contract.input(resolve_contract_input(input), selector);
  }

 private:
  static const Input<T>& resolve_contract_input(const OpaquePortRef& input) {
    const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const auto* declaration =
        static_cast<const Input<T>*>(input.resolve(expected, PortDirection::kInput));
    return *declaration;
  }

 public:
  static expected<InputSelection, Error> input_selection(const OpaquePortRef& input) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
      const auto* declaration =
          static_cast<const Input<T>*>(input.resolve(expected, PortDirection::kInput));
      return declaration->selection();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<ForeignScalarSample, Error> receive_scalar(const OpaquePortRef& input) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        const bool missing = declaration->is_missing();
        const bool stale = declaration->is_stale();
        if (missing) {
          return ForeignScalarSample{.is_stale = stale, .is_missing = true};
        }
        auto received = declaration->receive();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return ForeignScalarSample{
            .has_data = true,
            .data = ParamScalarValue::from(received->data),
            .metadata = received->metadata,
            .is_stale = stale,
            .is_missing = false,
        };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<std::size_t, Error> selected_batch_size(const OpaquePortRef& input) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
      auto* declaration =
          static_cast<Input<T>*>(const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
      auto selected = declaration->selected_batch();
      if (!selected) {
        return make_unexpected(std::move(selected).error());
      }
      return selected->size();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<ForeignScalarSample, Error> selected_scalar_batch_item(
      const OpaquePortRef& input, std::size_t index) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto selected = declaration->selected_batch();
        if (!selected) {
          return make_unexpected(std::move(selected).error());
        }
        if (index >= selected->size()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const Sample<T>& sample = (*selected)[index];
        return ForeignScalarSample{.has_data = true,
                                   .data = ParamScalarValue::from(sample.data),
                                   .metadata = sample.metadata};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ParamScalarValue, Error> receive_scalar_data(
      const OpaquePortRef& input) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto received = declaration->receive_data();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return ParamScalarValue::from(*received);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_scalar(const OpaquePortRef& output, ParamScalarValue value,
                                           EmitOptions options) noexcept {
    if constexpr (!holoscan::detail::kFixedParameterScalar<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        T decoded{};
        if (!value.try_as(decoded)) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE"});
        }
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Output<T>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->emit(std::move(decoded), options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignObjectSample, Error> receive_object(const OpaquePortRef& input) noexcept {
    if constexpr (!holoscan::detail::kPyObjectPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        const bool missing = declaration->is_missing();
        const bool stale = declaration->is_stale();
        if (missing) {
          return ForeignObjectSample{.is_stale = stale, .is_missing = true};
        }
        auto received = declaration->receive();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return ForeignObjectSample{
            .has_data = true,
            .data = ForeignPayloadValue::from(std::move(received->data)),
            .metadata = received->metadata,
            .is_stale = stale,
            .is_missing = false,
        };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignObjectSample, Error> selected_object_batch_item(
      const OpaquePortRef& input, std::size_t index) noexcept {
    if constexpr (!holoscan::detail::kPyObjectPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto selected = declaration->selected_batch();
        if (!selected) {
          return make_unexpected(std::move(selected).error());
        }
        if (index >= selected->size()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const Sample<T>& sample = (*selected)[index];
        return ForeignObjectSample{
            .has_data = true,
            .data = ForeignPayloadValue::from(sample.data),
            .metadata = sample.metadata,
        };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignPayloadValue, Error> receive_object_data(
      const OpaquePortRef& input) noexcept {
    if constexpr (!holoscan::detail::kPyObjectPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto received = declaration->receive_data();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return ForeignPayloadValue::from(std::move(*received));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_object(const OpaquePortRef& output, ForeignPayloadValue value,
                                           EmitOptions options) noexcept {
    if constexpr (!holoscan::detail::kPyObjectPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const T* decoded = value.template get<T>();
        if (decoded == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"});
        }
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Output<T>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->emit(*decoded, options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignSchemaSample, Error> receive_schema(const OpaquePortRef& input) noexcept {
    if constexpr (!HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        const bool missing = declaration->is_missing();
        const bool stale = declaration->is_stale();
        if (missing) {
          return ForeignSchemaSample{.is_stale = stale, .is_missing = true};
        }
        auto received = declaration->receive();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        auto converted = PythonSchemaNativeTraits<T>::to_foreign(received->data);
        if (!converted) {
          return make_unexpected(std::move(converted).error());
        }
        return ForeignSchemaSample{.has_data = true,
                                   .data = std::move(*converted),
                                   .metadata = received->metadata,
                                   .is_stale = stale,
                                   .is_missing = false};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignSchemaSample, Error> selected_schema_batch_item(
      const OpaquePortRef& input, std::size_t index) noexcept {
    if constexpr (!HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto selected = declaration->selected_batch();
        if (!selected) {
          return make_unexpected(std::move(selected).error());
        }
        if (index >= selected->size()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const Sample<T>& sample = (*selected)[index];
        auto converted = PythonSchemaNativeTraits<T>::to_foreign(sample.data);
        if (!converted) {
          return make_unexpected(std::move(converted).error());
        }
        return ForeignSchemaSample{
            .has_data = true,
            .data = std::move(*converted),
            .metadata = sample.metadata,
        };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignSchemaValue, Error> receive_schema_data(
      const OpaquePortRef& input) noexcept {
    if constexpr (!HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Input<T>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto received = declaration->receive_data();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return PythonSchemaNativeTraits<T>::to_foreign(*received);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_schema(const OpaquePortRef& output, ForeignSchemaValue value,
                                           EmitOptions options) noexcept {
    if constexpr (!HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        auto decoded = PythonSchemaNativeTraits<T>::from_foreign(value);
        if (!decoded) {
          return make_unexpected(std::move(decoded).error());
        }
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Output<T>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->emit(std::move(*decoded), options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> validate_schema(const ForeignSchemaValue& value) noexcept {
    if constexpr (!HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        auto decoded = PythonSchemaNativeTraits<T>::from_foreign(value);
        if (!decoded) {
          return make_unexpected(std::move(decoded).error());
        }
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> schema_tensor_output_produces(
      const OpaquePortRef& output, std::string_view field_name, TensorOutputSpec spec,
      TensorByteSpanExpression byte_span) noexcept {
    if constexpr (!holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        (void)output.resolve(expected, PortDirection::kOutput);
        auto* builder =
            static_cast<TensorOutputSpecBuilder*>(const_cast<void*>(output.authoring_builder_));
        if (builder == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        const auto field =
            std::ranges::find_if(descriptor_tensor_fields,
                                 [field_name](const RegisteredDescriptorTensorField& candidate) {
                                   return candidate.name == field_name;
                                 });
        if (field == descriptor_tensor_fields.end()) {
          return make_unexpected(
              Error{ErrorCode::kNotFound, "AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN"});
        }
        builder
            ->tensor_field(
                TensorFieldSelector<T>{.field_id = field->field_id, .tensor_index = field->ordinal})
            .produces_tensor(spec, std::move(byte_span));
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignTensorSchemaLoan, Error> allocate_schema_tensors(
      const OpaquePortRef& output,
      std::span<const ForeignTensorFieldLoanRequest> requests) noexcept {
    if constexpr (!holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        if (requests.size() != descriptor_tensor_fields.size()) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_COUNT_MISMATCH"});
        }
        std::array<TensorFieldLoanRequest<T>, kMaximumTensorSchemaLoanFields> typed_requests;
        std::array<bool, kMaximumTensorSchemaLoanFields> selected_fields{};
        for (std::size_t index = 0U; index < requests.size(); ++index) {
          const auto field = std::ranges::find_if(
              descriptor_tensor_fields,
              [&request = requests[index]](const RegisteredDescriptorTensorField& candidate) {
                return candidate.name == request.field_name;
              });
          if (field == descriptor_tensor_fields.end()) {
            return make_unexpected(
                Error{ErrorCode::kNotFound, "AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN"});
          }
          if (field->ordinal >= selected_fields.size() || selected_fields[field->ordinal]) {
            return make_unexpected(
                Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_FIELD_DUPLICATE"});
          }
          selected_fields[field->ordinal] = true;
          typed_requests[index] = TensorFieldLoanRequest<T>{
              .field = TensorFieldSelector<T>{.field_id = field->field_id,
                                              .tensor_index = field->ordinal},
              .tensor = requests[index].tensor,
          };
        }
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Output<T>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        if constexpr (holoscan::detail::kRuntimeSingularDescriptorTensorSchemaPayload<T>) {
          auto allocated = declaration->allocate_tensor(typed_requests.front().tensor);
          if (!allocated) {
            return make_unexpected(std::move(allocated).error());
          }
          holoscan::detail::TensorSchemaLoanBundleStorage storage;
          storage.field_count = 1U;
          storage.field_ids.front() = typed_requests.front().field.field_id;
          storage.fields.front().emplace(std::move(*allocated));
          return ForeignTensorSchemaLoan{std::move(storage)};
        } else {
          auto allocated = declaration->allocate_tensors(
              std::span<const TensorFieldLoanRequest<T>>{typed_requests}.first(requests.size()));
          if (!allocated) {
            return make_unexpected(std::move(allocated).error());
          }
          return ForeignTensorSchemaLoan{
              holoscan::detail::TensorSchemaOutputLoanAccess::release(std::move(*allocated))};
        }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_schema_tensors(const OpaquePortRef& output,
                                                   ForeignTensorSchemaLoan loan,
                                                   ForeignSchemaValue value,
                                                   EmitOptions options) noexcept {
    if constexpr (!holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T> ||
                  !HasPythonSchemaNativeTraits<T>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        auto decoded = PythonSchemaNativeTraits<T>::from_foreign(value);
        if (!decoded) {
          return make_unexpected(std::move(decoded).error());
        }
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<T>();
        auto* declaration = static_cast<Output<T>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        if constexpr (holoscan::detail::kRuntimeSingularDescriptorTensorSchemaPayload<T>) {
          auto storage = std::move(loan).release();
          if (storage.field_count != 1U ||
              storage.field_ids.front() != descriptor_tensor_fields.front().field_id ||
              !storage.fields.front()) {
            return make_unexpected(
                Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_INVALID"});
          }
          TensorOutputLoan tensor = std::move(*storage.fields.front());
          storage.fields.front().reset();
          storage.field_count = 0U;
          return declaration->emit_tensor(std::move(tensor), *decoded, options);
        } else {
          auto typed =
              holoscan::detail::TensorSchemaOutputLoanAccess::make<T>(std::move(loan).release());
          return declaration->emit_tensors(std::move(typed), *decoded, options);
        }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignTensorSample, Error> receive_tensor(const OpaquePortRef& input) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Input<Tensor>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        const bool missing = declaration->is_missing();
        const bool stale = declaration->is_stale();
        if (missing) {
          return ForeignTensorSample{.is_stale = stale, .is_missing = true};
        }
        auto received = declaration->receive();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        return ForeignTensorSample{.has_data = true,
                                   .data = std::move(received->data),
                                   .metadata = received->metadata,
                                   .is_stale = stale,
                                   .is_missing = false};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<ForeignTensorSample, Error> selected_tensor_batch_item(
      const OpaquePortRef& input, std::size_t index) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Input<Tensor>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        auto selected = declaration->selected_batch();
        if (!selected) {
          return make_unexpected(std::move(selected).error());
        }
        if (index >= selected->size()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const Sample<Tensor>& sample = (*selected)[index];
        return ForeignTensorSample{
            .has_data = true,
            .data = sample.data,
            .metadata = sample.metadata,
        };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<Tensor, Error> receive_tensor_data(const OpaquePortRef& input) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Input<Tensor>*>(
            const_cast<void*>(input.resolve(expected, PortDirection::kInput)));
        return declaration->receive_data();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_tensor(const OpaquePortRef& output, Tensor value,
                                           EmitOptions options) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Output<Tensor>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->emit(std::move(value), options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<TensorOutputLoan, Error> allocate_tensor(const OpaquePortRef& output,
                                                           TensorLoanRequest request) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Output<Tensor>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->allocate_tensor(request);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> emit_tensor_loan(const OpaquePortRef& output, TensorOutputLoan loan,
                                                EmitOptions options) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        auto* declaration = static_cast<Output<Tensor>*>(
            const_cast<void*>(output.resolve(expected, PortDirection::kOutput)));
        return declaration->emit(std::move(loan), options);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<void, Error> tensor_output_produces(
      const OpaquePortRef& output, const TensorOutputSpec& spec,
      TensorOutputDerivations derivations) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        (void)output.resolve(expected, PortDirection::kOutput);
        if (output.authoring_builder_ == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kNotReady, "FOREIGN_OPERATOR_TENSOR_BUILDER_NOT_BOUND"});
        }
        auto* builder = const_cast<TensorOutputSpecBuilder*>(
            static_cast<const TensorOutputSpecBuilder*>(output.authoring_builder_));
        if (derivations.max_byte_span || derivations.memory_kind) {
          builder->produces_tensor(spec, std::move(derivations));
        } else {
          builder->produces_tensor(spec);
        }
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static expected<TensorByteSpanExpression, Error> tensor_input_max_byte_span(
      const OpaquePortRef& input) noexcept {
    if constexpr (!std::same_as<T, Tensor>) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    } else {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const PayloadIdentityRecord expected =
            holoscan::detail::operator_spec_payload_identity<Tensor>();
        const auto* declaration =
            static_cast<const Input<Tensor>*>(input.resolve(expected, PortDirection::kInput));
        return detail_make_input_max_byte_span(declaration);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const RuntimeError& error) {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }
  }

  static OpaquePortConfigurator configure_input(Graph& graph, const OpaquePortRef& input) {
    const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* declaration = input.resolve(expected, PortDirection::kInput);
    const auto port = holoscan::detail::GraphPayloadRegistrationAccess::port(
        graph, static_cast<const InputBase*>(declaration), PortDirection::kInput, expected);
    const auto identity = holoscan::detail::graph_port_registration_identity(
        holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock(), port);
    if (!identity) {
      throw RuntimeError{identity.error()};
    }
    return OpaquePortConfigurator{holoscan::detail::GraphPayloadRegistrationAccess::state(graph),
                                  port,
                                  identity.value(),
                                  PortDirection::kInput,
                                  input.index_,
                                  input.graph_handle_identity_};
  }

  static OpaquePortConfigurator configure_output(Graph& graph, const OpaquePortRef& output) {
    const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* declaration = output.resolve(expected, PortDirection::kOutput);
    const auto port = holoscan::detail::GraphPayloadRegistrationAccess::port(
        graph, static_cast<const OutputBase*>(declaration), PortDirection::kOutput, expected);
    const auto identity = holoscan::detail::graph_port_registration_identity(
        holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock(), port);
    if (!identity) {
      throw RuntimeError{identity.error()};
    }
    return OpaquePortConfigurator{holoscan::detail::GraphPayloadRegistrationAccess::state(graph),
                                  port,
                                  identity.value(),
                                  PortDirection::kOutput,
                                  output.index_,
                                  output.graph_handle_identity_};
  }

  static void export_input(Graph& graph, const OpaquePortRef& public_port,
                           const OpaquePortRef& internal_port) {
    PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* public_declaration = public_port.resolve(expected, PortDirection::kInput);
    const void* internal_declaration = internal_port.resolve(expected, PortDirection::kInput);
    holoscan::detail::GraphPayloadRegistrationAccess::export_input(
        graph,
        *static_cast<const InputBase*>(public_declaration),
        *static_cast<const InputBase*>(internal_declaration),
        std::move(expected));
  }

  static void export_output(Graph& graph, const OpaquePortRef& public_port,
                            const OpaquePortRef& internal_port) {
    PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* public_declaration = public_port.resolve(expected, PortDirection::kOutput);
    const void* internal_declaration = internal_port.resolve(expected, PortDirection::kOutput);
    holoscan::detail::GraphPayloadRegistrationAccess::export_output(
        graph,
        *static_cast<const OutputBase*>(public_declaration),
        *static_cast<const OutputBase*>(internal_declaration),
        std::move(expected));
  }

  static OpaqueImportedEndpoint import_topic(Graph& graph, std::string_view uri,
                                             TopicOptions options) {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED"}};
    } else {
      static_assert(std::is_nothrow_move_constructible_v<ImportedTopic<T>>);
      auto holder = std::make_shared<std::optional<ImportedTopic<T>>>();
      OpaqueImportedEndpoint result{holder};
      holder->emplace(graph.import_topic<T>(uri, std::move(options)));
      return result;
    }
  }

  static OpaqueExportedEndpoint export_topic(Graph& graph, std::string_view uri,
                                             TopicOptions options) {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED"}};
    } else {
      static_assert(std::is_nothrow_move_constructible_v<ExportedTopic<T>>);
      auto holder = std::make_shared<std::optional<ExportedTopic<T>>>();
      OpaqueExportedEndpoint result{holder};
      holder->emplace(graph.export_topic<T>(uri, std::move(options)));
      return result;
    }
  }

  static OpaqueImportedEndpoint import_descriptor_topic(
      Graph& graph, std::string_view uri, TopicOptions options,
      std::span<const std::pair<std::string_view, TensorBoundaryContract>> fields) {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      static_assert(std::is_nothrow_move_constructible_v<ImportedTopic<T>>);
      auto holder = std::make_shared<std::optional<ImportedTopic<T>>>();
      OpaqueImportedEndpoint result{holder};
      holder->emplace(
          graph.import_topic<T>(uri, std::move(options), make_descriptor_topic_contract(fields)));
      return result;
    } else {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED"}};
    }
  }

  static OpaqueExportedEndpoint export_descriptor_topic(
      Graph& graph, std::string_view uri, TopicOptions options,
      std::span<const std::pair<std::string_view, TensorBoundaryContract>> fields) {
    if constexpr (holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<T>) {
      static_assert(std::is_nothrow_move_constructible_v<ExportedTopic<T>>);
      auto holder = std::make_shared<std::optional<ExportedTopic<T>>>();
      OpaqueExportedEndpoint result{holder};
      holder->emplace(
          graph.export_topic<T>(uri, std::move(options), make_descriptor_topic_contract(fields)));
      return result;
    } else {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED"}};
    }
  }

  static InterfacePortPair make_interface_ports(std::string_view name) {
    return InterfacePortPair::make<T>(name);
  }

 private:
  template <typename U = T>
    requires holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<U>
  static DescriptorTensorTopicContract<U> make_descriptor_topic_contract(
      std::span<const std::pair<std::string_view, TensorBoundaryContract>> fields) {
    DescriptorTensorTopicContract<U> result;
    for (const auto& [name, contract] : fields) {
      std::size_t index = 0U;
      while (index < descriptor_tensor_fields.size() &&
             descriptor_tensor_fields[index].name != name) {
        ++index;
      }
      if (index == descriptor_tensor_fields.size()) {
        std::string message{"AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN name="};
        message.append(name);
        message.append(" known_fields=");
        for (std::size_t known = 0U; known < descriptor_tensor_fields.size(); ++known) {
          if (known != 0U) {
            message.push_back(',');
          }
          message.append(descriptor_tensor_fields[known].name);
        }
        throw RuntimeError{Error{ErrorCode::kNotFound, std::move(message)}};
      }
      const auto& field = descriptor_tensor_fields[index];
      result.field(TensorFieldSelector<U>{field.field_id, field.ordinal}, contract);
    }
    return result;
  }
};

/// @brief Build one complete payload registry row.
/// @note Process-local enums require typed local C++ Graph authoring and cannot be registered.
/// @tparam T Exact payload type supported by Graph external topics.
/// @param name Stable language-facing registry name.
/// @return Complete identity and thunk row.
template <holoscan::detail::ExternalTopicPayload T>
[[nodiscard]] RegisteredPayloadType make_registered_payload(std::string_view name) {
  using Thunks = PayloadRegistrationThunks<T>;
  return RegisteredPayloadType{
      .name = name,
      .identity = holoscan::detail::operator_spec_payload_identity<T>(),
      .schema_transport_profile = Thunks::schema_transport_profile,
      .generated_schema_facade = HasPythonSchemaNativeTraits<T>,
      .channel = &Thunks::channel_thunks,
      .bind_spec_input = &Thunks::bind_spec_input,
      .bind_spec_output = &Thunks::bind_spec_output,
      .input_expected_rate = &Thunks::input_expected_rate,
      .tensor_input_expects = &Thunks::tensor_input_expects,
      .tensor_input_accepts_projection = &Thunks::tensor_input_accepts_projection,
      .output_nominal_rate = &Thunks::output_nominal_rate,
      .output_follows_input = &Thunks::output_follows_input,
      .contract_input_ref = &Thunks::contract_input_ref,
      .contract_input_exact = &Thunks::contract_input_exact,
      .contract_input_latest_within = &Thunks::contract_input_latest_within,
      .contract_input_drain_all = &Thunks::contract_input_drain_all,
      .contract_input_bracketed_window = &Thunks::contract_input_bracketed_window,
      .input_selection = &Thunks::input_selection,
      .receive_scalar = &Thunks::receive_scalar,
      .selected_batch_size = &Thunks::selected_batch_size,
      .selected_scalar_batch_item = &Thunks::selected_scalar_batch_item,
      .receive_scalar_data = &Thunks::receive_scalar_data,
      .emit_scalar = &Thunks::emit_scalar,
      .receive_object = &Thunks::receive_object,
      .selected_object_batch_item = &Thunks::selected_object_batch_item,
      .receive_object_data = &Thunks::receive_object_data,
      .emit_object = &Thunks::emit_object,
      .receive_schema = &Thunks::receive_schema,
      .selected_schema_batch_item = &Thunks::selected_schema_batch_item,
      .receive_schema_data = &Thunks::receive_schema_data,
      .emit_schema = &Thunks::emit_schema,
      .validate_schema = &Thunks::validate_schema,
      .schema_tensor_output_produces = &Thunks::schema_tensor_output_produces,
      .allocate_schema_tensors = &Thunks::allocate_schema_tensors,
      .emit_schema_tensors = &Thunks::emit_schema_tensors,
      .receive_tensor = &Thunks::receive_tensor,
      .selected_tensor_batch_item = &Thunks::selected_tensor_batch_item,
      .receive_tensor_data = &Thunks::receive_tensor_data,
      .emit_tensor = &Thunks::emit_tensor,
      .allocate_tensor = &Thunks::allocate_tensor,
      .emit_tensor_loan = &Thunks::emit_tensor_loan,
      .tensor_output_produces = &Thunks::tensor_output_produces,
      .tensor_input_max_byte_span = &Thunks::tensor_input_max_byte_span,
      .add_flow = &Thunks::add_flow,
      .add_tensor_projection = &Thunks::add_tensor_projection,
      .configure_input = &Thunks::configure_input,
      .configure_output = &Thunks::configure_output,
      .export_input = &Thunks::export_input,
      .export_output = &Thunks::export_output,
      .import_topic = &Thunks::import_topic,
      .export_topic = &Thunks::export_topic,
      .import_descriptor_topic = &Thunks::import_descriptor_topic,
      .export_descriptor_topic = &Thunks::export_descriptor_topic,
      .descriptor_tensor_fields = Thunks::descriptor_tensor_fields,
      .make_interface_ports = &Thunks::make_interface_ports,
  };
}

/// @brief File one payload row during module initialization without throwing.
/// @tparam T Exact payload type supported by Graph external topics.
/// @param name Process-lifetime stable registry name.
/// @return `true` after successful registration.
template <holoscan::detail::ExternalTopicPayload T>
[[nodiscard]] bool register_payload_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_payload_automatically(make_registered_payload<T>(name));
  } catch (...) {
    AutomaticRegistrationFailureAccess::record(registry);
    return false;
  }
}

}  // namespace holoscan::authoring::detail

#ifndef HOLOSCAN_DETAIL_AUTHORING_CONCAT
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y) X##Y
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT(X, Y) HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y)
#endif

/// @brief Register one build-time payload type in the process-local authoring registry.
/// @param TYPE Exact payload type supported by Graph external topics.
/// @param NAME Stable process-lifetime registry name string literal.
/// @note Invoke once in a translation unit that has seen all payload traits. A duplicate identity
/// or name is retained by `AuthoringRegistry::registration_status()` without throwing across module
/// initialization. Process-local enums remain available only through typed local C++ Graph
/// authoring; this macro rejects them at its template constraint.
#define HOLOSCAN_REGISTER_PAYLOAD(TYPE, NAME)                                               \
  namespace {                                                                               \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(kHoloscanRegisteredPayload_, \
                                                               __COUNTER__) =               \
      ::holoscan::authoring::detail::register_payload_automatically<TYPE>(NAME);            \
  }
