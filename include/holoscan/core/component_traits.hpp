/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_COMPONENT_TRAITS_HPP
#define HOLOSCAN_CORE_COMPONENT_TRAITS_HPP

namespace holoscan {

// Forward declarations for condition types
class AsynchronousCondition;
class BooleanCondition;
class CountCondition;
class CudaBufferAvailableCondition;
class CudaEventCondition;
class CudaStreamCondition;
class DownstreamMessageAffordableCondition;
class ExpiringMessageAvailableCondition;
class MemoryAvailableCondition;
class MessageAvailableCondition;
class MultiMessageAvailableCondition;
class MultiMessageAvailableTimeoutCondition;
#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
class PendingExportCondition;
#endif
class PeriodicCondition;
class PublisherAvailableCondition;
class SubscriberAvailableCondition;

// Forward declarations for resource types
class AsyncBufferReceiver;
class AsyncBufferTransmitter;
class BlockMemoryPool;
class CPUThread;
class CudaGreenContext;
class CudaGreenContextPool;
class CudaStreamPool;
class DoubleBufferReceiver;
class DoubleBufferTransmitter;
class GPUDevice;
class ManualClock;
class OrConditionCombiner;
class PubSubReceiver;
class PubSubTransmitter;
class RealtimeClock;
class RMMAllocator;
class SerializationBuffer;
class StdComponentSerializer;
class StdEntitySerializer;
class StreamOrderedAllocator;
class SyntheticClock;
class ThreadPool;
class UcxComponentSerializer;
class UcxHoloscanComponentSerializer;
class UcxEntitySerializer;
class UcxReceiver;
class UcxSerializationBuffer;
class UcxTransmitter;
class UnboundedAllocator;

// Forward declarations for data logger types (in holoscan::data_loggers namespace)
namespace data_loggers {
class AsyncConsoleLogger;
class BasicConsoleLogger;
class GXFConsoleLogger;
class SimpleTextSerializer;
}  // namespace data_loggers

// Forward declarations for scheduler types
class EventBasedScheduler;
class GreedyScheduler;
class MultiThreadScheduler;

// Forward declarations for network context types
class FastDdsPubSubNetworkContext;
class InMemoryPubSubNetworkContext;
class PubSubContext;
class UcxContext;

// Forward declarations for operator types (in holoscan::ops namespace)
namespace ops {
class AsyncPingRxOp;
class AsyncPingTxOp;
class BayerDemosaicOp;
class DataTypeRxTestOp;
class DataTypeTxTestOp;
class FormatConverterOp;
class HolovizOp;
class RawImageProcessorOp;
class InferenceOp;
class InferenceProcessorOp;
class PingRxOp;
class PingTxOp;
class PingTensorRxOp;
class PingTensorTxOp;
class SegmentationPostprocessorOp;
class V4L2VideoCaptureOp;
class VideoStreamReplayerOp;
class VideoStreamRecorderOp;
class GXFCodeletOp;

}  // namespace ops

/**
 * @brief Type trait to provide unique default names for condition types.
 *
 * This trait ensures that different condition types have unique default names when
 * created without an explicit name parameter. This prevents naming conflicts when
 * multiple unnamed conditions of different types are added to the same operator.
 *
 * Primary template provides fallback to generic name for any condition types not
 * explicitly specialized below.
 *
 * @tparam ConditionT The condition type.
 */
template <typename ConditionT>
struct condition_default_name {
  static constexpr const char* value = "noname_condition";
};

// Specializations for each condition type
template <>
struct condition_default_name<AsynchronousCondition> {
  static constexpr const char* value = "async_condition";
};

template <>
struct condition_default_name<BooleanCondition> {
  static constexpr const char* value = "boolean_condition";
};

template <>
struct condition_default_name<CountCondition> {
  static constexpr const char* value = "count_condition";
};

template <>
struct condition_default_name<CudaBufferAvailableCondition> {
  static constexpr const char* value = "cuda_buffer_available_condition";
};

template <>
struct condition_default_name<CudaEventCondition> {
  static constexpr const char* value = "cuda_event_condition";
};

template <>
struct condition_default_name<CudaStreamCondition> {
  static constexpr const char* value = "cuda_stream_condition";
};

template <>
struct condition_default_name<DownstreamMessageAffordableCondition> {
  static constexpr const char* value = "downstream_affordable_condition";
};

template <>
struct condition_default_name<ExpiringMessageAvailableCondition> {
  static constexpr const char* value = "expiring_message_available_condition";
};

template <>
struct condition_default_name<MemoryAvailableCondition> {
  static constexpr const char* value = "memory_available_condition";
};

template <>
struct condition_default_name<MessageAvailableCondition> {
  static constexpr const char* value = "message_available_condition";
};

template <>
struct condition_default_name<MultiMessageAvailableCondition> {
  static constexpr const char* value = "multi_message_condition";
};

template <>
struct condition_default_name<MultiMessageAvailableTimeoutCondition> {
  static constexpr const char* value = "multi_message_timeout_condition";
};

#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
template <>
struct condition_default_name<PendingExportCondition> {
  static constexpr const char* value = "pending_export_condition";
};
#endif

template <>
struct condition_default_name<PeriodicCondition> {
  static constexpr const char* value = "periodic_condition";
};

template <>
struct condition_default_name<PublisherAvailableCondition> {
  static constexpr const char* value = "publisher_available_condition";
};

template <>
struct condition_default_name<SubscriberAvailableCondition> {
  static constexpr const char* value = "subscriber_available_condition";
};

/**
 * @brief Helper template variable for easier access to condition default names (C++17).
 *
 * Usage: condition_default_name_v<CountCondition> returns "count_condition"
 *
 * @tparam ConditionT The condition type.
 */
template <typename ConditionT>
inline constexpr const char* condition_default_name_v = condition_default_name<ConditionT>::value;

/**
 * @brief Type trait to provide unique default names for resource types.
 *
 * This trait ensures that different resource types have unique default names when
 * created without an explicit name parameter. This prevents naming conflicts when
 * multiple unnamed resources of different types are added to the same operator or
 * fragment.
 *
 * Primary template provides fallback to generic name for any resource types not
 * explicitly specialized below.
 *
 * @tparam ResourceT The resource type.
 */
template <typename ResourceT>
struct resource_default_name {
  static constexpr const char* value = "noname_resource";
};

template <>
struct resource_default_name<AsyncBufferReceiver> {
  static constexpr const char* value = "async_buffer_receiver";
};

template <>
struct resource_default_name<AsyncBufferTransmitter> {
  static constexpr const char* value = "async_buffer_transmitter";
};

template <>
struct resource_default_name<BlockMemoryPool> {
  static constexpr const char* value = "block_memory_pool";
};

template <>
struct resource_default_name<CPUThread> {
  static constexpr const char* value = "cpu_thread";
};

template <>
struct resource_default_name<CudaGreenContext> {
  static constexpr const char* value = "cuda_green_context";
};

template <>
struct resource_default_name<CudaGreenContextPool> {
  static constexpr const char* value = "cuda_green_context_pool";
};

template <>
struct resource_default_name<CudaStreamPool> {
  static constexpr const char* value = "cuda_stream_pool";
};

template <>
struct resource_default_name<DoubleBufferReceiver> {
  static constexpr const char* value = "double_buffer_receiver";
};

template <>
struct resource_default_name<DoubleBufferTransmitter> {
  static constexpr const char* value = "double_buffer_transmitter";
};

template <>
struct resource_default_name<GPUDevice> {
  static constexpr const char* value = "gpu_device";
};

template <>
struct resource_default_name<ManualClock> {
  static constexpr const char* value = "manual_clock";
};

template <>
struct resource_default_name<OrConditionCombiner> {
  static constexpr const char* value = "or_condition_combiner";
};

template <>
struct resource_default_name<PubSubReceiver> {
  static constexpr const char* value = "pubsub_receiver";
};

template <>
struct resource_default_name<PubSubTransmitter> {
  static constexpr const char* value = "pubsub_transmitter";
};

template <>
struct resource_default_name<RealtimeClock> {
  static constexpr const char* value = "realtime_clock";
};

template <>
struct resource_default_name<RMMAllocator> {
  static constexpr const char* value = "rmm_pool";
};

template <>
struct resource_default_name<SerializationBuffer> {
  static constexpr const char* value = "serialization_buffer";
};

template <>
struct resource_default_name<StdComponentSerializer> {
  static constexpr const char* value = "standard_component_serializer";
};

template <>
struct resource_default_name<StdEntitySerializer> {
  static constexpr const char* value = "standard_entity_serializer";
};

template <>
struct resource_default_name<StreamOrderedAllocator> {
  static constexpr const char* value = "stream_ordered_allocator";
};

template <>
struct resource_default_name<SyntheticClock> {
  static constexpr const char* value = "synthetic_clock";
};

template <>
struct resource_default_name<ThreadPool> {
  static constexpr const char* value = "thread_pool";
};

template <>
struct resource_default_name<UcxComponentSerializer> {
  static constexpr const char* value = "ucx_component_serializer";
};

template <>
struct resource_default_name<UcxHoloscanComponentSerializer> {
  static constexpr const char* value = "ucx_holoscan_component_serializer";
};

template <>
struct resource_default_name<UcxEntitySerializer> {
  static constexpr const char* value = "ucx_entity_serializer";
};

template <>
struct resource_default_name<UcxReceiver> {
  static constexpr const char* value = "ucx_receiver";
};

template <>
struct resource_default_name<UcxSerializationBuffer> {
  static constexpr const char* value = "ucx_serialization_buffer";
};

template <>
struct resource_default_name<UcxTransmitter> {
  static constexpr const char* value = "ucx_transmitter";
};

template <>
struct resource_default_name<UnboundedAllocator> {
  static constexpr const char* value = "unbounded_allocator";
};

// Specializations for data logger types

template <>
struct resource_default_name<data_loggers::AsyncConsoleLogger> {
  static constexpr const char* value = "async_console_logger";
};

template <>
struct resource_default_name<data_loggers::BasicConsoleLogger> {
  static constexpr const char* value = "basic_console_logger";
};

template <>
struct resource_default_name<data_loggers::GXFConsoleLogger> {
  static constexpr const char* value = "gxf_basic_console_logger";
};

template <>
struct resource_default_name<data_loggers::SimpleTextSerializer> {
  static constexpr const char* value = "simple_text_serializer";
};

/**
 * @brief Helper template variable for easier access to resource default names (C++17).
 *
 * Usage: resource_default_name_v<UnboundedAllocator> returns "unbounded_allocator"
 *
 * @tparam ResourceT The resource type.
 */
template <typename ResourceT>
inline constexpr const char* resource_default_name_v = resource_default_name<ResourceT>::value;

/**
 * @brief Type trait to provide unique default names for scheduler types.
 *
 * This trait ensures that different scheduler types have unique default names when
 * created without an explicit name parameter.
 *
 * Primary template provides fallback to generic name for any scheduler types not
 * explicitly specialized below.
 *
 * @tparam SchedulerT The scheduler type.
 */
template <typename SchedulerT>
struct scheduler_default_name {
  static constexpr const char* value = "noname_scheduler";
};

// Specializations for each scheduler type
template <>
struct scheduler_default_name<EventBasedScheduler> {
  static constexpr const char* value = "event_based_scheduler";
};

template <>
struct scheduler_default_name<GreedyScheduler> {
  static constexpr const char* value = "greedy_scheduler";
};

template <>
struct scheduler_default_name<MultiThreadScheduler> {
  static constexpr const char* value = "multithread_scheduler";
};

/**
 * @brief Helper template variable for easier access to scheduler default names (C++17).
 *
 * Usage: scheduler_default_name_v<GreedyScheduler> returns "greedy_scheduler"
 *
 * @tparam SchedulerT The scheduler type.
 */
template <typename SchedulerT>
inline constexpr const char* scheduler_default_name_v = scheduler_default_name<SchedulerT>::value;

/**
 * @brief Type trait to provide unique default names for network context types.
 *
 * This trait ensures that different network context types have unique default names when
 * created without an explicit name parameter.
 *
 * Primary template provides fallback to generic name for any network context types not
 * explicitly specialized below.
 *
 * @tparam NetworkContextT The network context type.
 */
template <typename NetworkContextT>
struct network_context_default_name {
  static constexpr const char* value = "noname_network_context";
};

// Specializations for each network context type
template <>
struct network_context_default_name<PubSubContext> {
  static constexpr const char* value = "pubsub_context";
};

template <>
struct network_context_default_name<UcxContext> {
  static constexpr const char* value = "ucx_context";
};

template <>
struct network_context_default_name<InMemoryPubSubNetworkContext> {
  static constexpr const char* value = "in_memory_pubsub_network_context";
};

template <>
struct network_context_default_name<FastDdsPubSubNetworkContext> {
  static constexpr const char* value = "dds_pubsub_network_context";
};

/**
 * @brief Helper template variable for easier access to network context default names (C++17).
 *
 * Usage: network_context_default_name_v<UcxContext> returns "ucx_context"
 *
 * @tparam NetworkContextT The network context type.
 */
template <typename NetworkContextT>
inline constexpr const char* network_context_default_name_v =
    network_context_default_name<NetworkContextT>::value;

/**
 * @brief Type trait to provide unique default names for operator types.
 *
 * This trait ensures that different operator types have unique default names when
 * created without an explicit name parameter.
 *
 * Primary template provides fallback to generic name for any operator types not
 * explicitly specialized below.
 *
 * @tparam OperatorT The operator type.
 */
template <typename OperatorT>
struct operator_default_name {
  static constexpr const char* value = "noname_operator";
};

// Specializations for each operator type
template <>
struct operator_default_name<ops::AsyncPingRxOp> {
  static constexpr const char* value = "async_ping_rx";
};
template <>
struct operator_default_name<ops::AsyncPingTxOp> {
  static constexpr const char* value = "async_ping_tx";
};
template <>
struct operator_default_name<ops::BayerDemosaicOp> {
  static constexpr const char* value = "bayer_demosaic";
};

template <>
struct operator_default_name<ops::DataTypeRxTestOp> {
  static constexpr const char* value = "data_type_rx_test_op";
};

template <>
struct operator_default_name<ops::DataTypeTxTestOp> {
  static constexpr const char* value = "data_type_tx_test_op";
};

template <>
struct operator_default_name<ops::FormatConverterOp> {
  static constexpr const char* value = "format_converter";
};

template <>
struct operator_default_name<ops::GXFCodeletOp> {
  static constexpr const char* value = "gxf_codelet";
};

template <>
struct operator_default_name<ops::HolovizOp> {
  static constexpr const char* value = "holoviz_op";
};

template <>
struct operator_default_name<ops::RawImageProcessorOp> {
  static constexpr const char* value = "raw_image_processor";
};

template <>
struct operator_default_name<ops::InferenceOp> {
  static constexpr const char* value = "inference";
};

template <>
struct operator_default_name<ops::InferenceProcessorOp> {
  static constexpr const char* value = "postprocessor";
};

template <>
struct operator_default_name<ops::PingRxOp> {
  static constexpr const char* value = "ping_rx";
};

template <>
struct operator_default_name<ops::PingTensorRxOp> {
  static constexpr const char* value = "ping_tensor_rx";
};

template <>
struct operator_default_name<ops::PingTensorTxOp> {
  static constexpr const char* value = "ping_tensor_tx";
};

template <>
struct operator_default_name<ops::PingTxOp> {
  static constexpr const char* value = "ping_tx";
};

template <>
struct operator_default_name<ops::SegmentationPostprocessorOp> {
  static constexpr const char* value = "segmentation_postprocessor";
};

template <>
struct operator_default_name<ops::V4L2VideoCaptureOp> {
  static constexpr const char* value = "v4l2_video_capture";
};

template <>
struct operator_default_name<ops::VideoStreamRecorderOp> {
  static constexpr const char* value = "video_stream_recorder";
};

template <>
struct operator_default_name<ops::VideoStreamReplayerOp> {
  static constexpr const char* value = "video_stream_replayer";
};

/**
 * @brief Helper template variable for easier access to operator default names (C++17).
 *
 * Usage: operator_default_name_v<InferenceOp> returns "inference"
 *
 * @tparam OperatorT The operator type.
 */
template <typename OperatorT>
inline constexpr const char* operator_default_name_v = operator_default_name<OperatorT>::value;

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_COMPONENT_TRAITS_HPP */
