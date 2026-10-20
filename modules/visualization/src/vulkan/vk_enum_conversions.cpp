/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_enum_conversions.hpp"

namespace holoviz::vulkan {

expected<VkFormat> from_holoviz_format(Format format) {
  switch (format) {
    case Format::kR8G8Unorm:
      return VK_FORMAT_R8G8_UNORM;
    case Format::kR8Unorm:
      return VK_FORMAT_R8_UNORM;
    case Format::kR8G8B8A8Unorm:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::kB8G8R8A8Unorm:
      return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::kR8G8B8A8Srgb:
      return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::kB8G8R8A8Srgb:
      return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::kA2B10G10R10Unorm:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::kR16G16B16A16Sfloat:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::kR32Uint:
      return VK_FORMAT_R32_UINT;
    case Format::kR32Sfloat:
      return VK_FORMAT_R32_SFLOAT;
    case Format::kR32G32Sfloat:
      return VK_FORMAT_R32G32_SFLOAT;
    case Format::kR32G32B32Sfloat:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::kR32G32B32A32Sfloat:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case Format::kD32Sfloat:
      return VK_FORMAT_D32_SFLOAT;
    case Format::kR16Unorm:
      return VK_FORMAT_R16_UNORM;
    case Format::kR16G16B16A16Unorm:
      return VK_FORMAT_R16G16B16A16_UNORM;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported format conversion requested."});
  }
}

expected<uint32_t> vk_format_texel_size_bytes(VkFormat vk_format) {
  switch (vk_format) {
    case VK_FORMAT_R8G8_UNORM:
      return 2;
    case VK_FORMAT_R8_UNORM:
      return 1;
    case VK_FORMAT_R16_UNORM:
      return 2;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R32G32_SFLOAT:
      return 8;
    case VK_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported format for texel size query."});
  }
}

expected<Format> to_holoviz_format(VkFormat vk_format) {
  switch (vk_format) {
    case VK_FORMAT_R8G8_UNORM:
      return Format::kR8G8Unorm;
    case VK_FORMAT_R8_UNORM:
      return Format::kR8Unorm;
    case VK_FORMAT_R8G8B8A8_UNORM:
      return Format::kR8G8B8A8Unorm;
    case VK_FORMAT_B8G8R8A8_UNORM:
      return Format::kB8G8R8A8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
      return Format::kR8G8B8A8Srgb;
    case VK_FORMAT_B8G8R8A8_SRGB:
      return Format::kB8G8R8A8Srgb;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return Format::kA2B10G10R10Unorm;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return Format::kR16G16B16A16Sfloat;
    case VK_FORMAT_R32_UINT:
      return Format::kR32Uint;
    case VK_FORMAT_R32_SFLOAT:
      return Format::kR32Sfloat;
    case VK_FORMAT_R32G32_SFLOAT:
      return Format::kR32G32Sfloat;
    case VK_FORMAT_R32G32B32_SFLOAT:
      return Format::kR32G32B32Sfloat;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return Format::kR32G32B32A32Sfloat;
    case VK_FORMAT_D32_SFLOAT:
      return Format::kD32Sfloat;
    case VK_FORMAT_R16_UNORM:
      return Format::kR16Unorm;
    case VK_FORMAT_R16G16B16A16_UNORM:
      return Format::kR16G16B16A16Unorm;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported format conversion requested."});
  }
}

VkPipelineStageFlags from_holoviz_gate_wait_stage(GateWaitStage wait_stage) {
  switch (wait_stage) {
    case GateWaitStage::kColorAttachmentOutput:
      return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    case GateWaitStage::kAllCommands:
      return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  }
  return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

expected<VkPresentModeKHR> from_holoviz_present_mode(PresentMode present_mode) {
  switch (present_mode) {
    case PresentMode::kAuto:
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidArgument,
          .message = "[holoviz] PresentMode::kAuto must be resolved to a concrete present mode "
                     "before swapchain creation."});
    case PresentMode::kFIFO:
      return VK_PRESENT_MODE_FIFO_KHR;
    case PresentMode::kFIFORelaxed:
      return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    case PresentMode::kFIFOLatestReady:
#if defined(VK_EXT_present_mode_fifo_latest_ready)
      return VK_PRESENT_MODE_FIFO_LATEST_READY_EXT;
#else
      return make_unexpected(
          Error{.code = ErrorCode::kNotImplemented,
                .message = "[holoviz] PresentMode::kFifoLatestReady requires Vulkan headers with "
                           "VK_EXT_present_mode_fifo_latest_ready."});
#endif
    case PresentMode::kMailbox:
      return VK_PRESENT_MODE_MAILBOX_KHR;
    case PresentMode::kImmediate:
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    case PresentMode::kSharedDemandRefresh:
      return VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR;
    case PresentMode::kSharedContinuousRefresh:
      return VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported present mode conversion requested."});
  }
}

expected<PresentMode> to_holoviz_present_mode(VkPresentModeKHR vk_present_mode) {
  switch (vk_present_mode) {
    case VK_PRESENT_MODE_FIFO_KHR:
      return PresentMode::kFIFO;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
      return PresentMode::kFIFORelaxed;
#if defined(VK_EXT_present_mode_fifo_latest_ready)
    case VK_PRESENT_MODE_FIFO_LATEST_READY_EXT:
      return PresentMode::kFIFOLatestReady;
#endif
    case VK_PRESENT_MODE_MAILBOX_KHR:
      return PresentMode::kMailbox;
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
      return PresentMode::kImmediate;
    case VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR:
      return PresentMode::kSharedDemandRefresh;
    case VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR:
      return PresentMode::kSharedContinuousRefresh;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported present mode conversion requested."});
  }
}

expected<VkColorSpaceKHR> from_holoviz_color_space(ColorSpace color_space) {
  switch (color_space) {
    case ColorSpace::kSrgbNonlinear:
      return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    case ColorSpace::kHdr10St2084:
      return VK_COLOR_SPACE_HDR10_ST2084_EXT;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported color space conversion requested."});
  }
}

expected<ColorSpace> to_holoviz_color_space(VkColorSpaceKHR vk_color_space) {
  switch (vk_color_space) {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
      return ColorSpace::kSrgbNonlinear;
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
      return ColorSpace::kHdr10St2084;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported color space conversion requested."});
  }
}

expected<VkDescriptorType> from_holoviz_binding_kind(BindingKind kind) {
  switch (kind) {
    case BindingKind::kStorageBuffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case BindingKind::kSampler:
      return VK_DESCRIPTOR_TYPE_SAMPLER;
    case BindingKind::kSampledImage:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported binding kind conversion requested."});
  }
}

expected<BindingKind> to_holoviz_binding_kind(VkDescriptorType vk_descriptor_type) {
  switch (vk_descriptor_type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return BindingKind::kStorageBuffer;
    case VK_DESCRIPTOR_TYPE_SAMPLER:
      return BindingKind::kSampler;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return BindingKind::kSampledImage;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported descriptor type conversion requested."});
  }
}

expected<VkPrimitiveTopology> from_holoviz_primitive_topology(PrimitiveTopology topology) {
  switch (topology) {
    case PrimitiveTopology::kPointList:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case PrimitiveTopology::kLineList:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PrimitiveTopology::kLineStrip:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PrimitiveTopology::kTriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PrimitiveTopology::kTriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported primitive topology conversion requested."});
  }
}

expected<PrimitiveTopology> to_holoviz_primitive_topology(VkPrimitiveTopology vk_topology) {
  switch (vk_topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return PrimitiveTopology::kPointList;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return PrimitiveTopology::kLineList;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return PrimitiveTopology::kLineStrip;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return PrimitiveTopology::kTriangleList;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return PrimitiveTopology::kTriangleStrip;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported primitive topology conversion requested."});
  }
}

expected<VkCullModeFlags> from_holoviz_cull_mode(CullMode cull_mode) {
  switch (cull_mode) {
    case CullMode::kNone:
      return VK_CULL_MODE_NONE;
    case CullMode::kFront:
      return VK_CULL_MODE_FRONT_BIT;
    case CullMode::kBack:
      return VK_CULL_MODE_BACK_BIT;
    case CullMode::kFrontAndBack:
      return VK_CULL_MODE_FRONT_AND_BACK;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported cull mode conversion requested."});
  }
}

expected<VkPolygonMode> from_holoviz_polygon_mode(PolygonMode polygon_mode) {
  switch (polygon_mode) {
    case PolygonMode::kFill:
      return VK_POLYGON_MODE_FILL;
    case PolygonMode::kFillRectangle:
      return VK_POLYGON_MODE_FILL_RECTANGLE_NV;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported polygon mode conversion requested."});
  }
}

expected<CullMode> to_holoviz_cull_mode(VkCullModeFlags vk_cull_mode) {
  switch (vk_cull_mode) {
    case VK_CULL_MODE_NONE:
      return CullMode::kNone;
    case VK_CULL_MODE_FRONT_BIT:
      return CullMode::kFront;
    case VK_CULL_MODE_BACK_BIT:
      return CullMode::kBack;
    case VK_CULL_MODE_FRONT_AND_BACK:
      return CullMode::kFrontAndBack;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported cull mode conversion requested."});
  }
}

expected<VkCompareOp> from_holoviz_compare_op(CompareOp op) {
  switch (op) {
    case CompareOp::kAlways:
      return VK_COMPARE_OP_ALWAYS;
    case CompareOp::kLess:
      return VK_COMPARE_OP_LESS;
    case CompareOp::kLessEqual:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::kGreater:
      return VK_COMPARE_OP_GREATER;
    case CompareOp::kGreaterEqual:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::kEqual:
      return VK_COMPARE_OP_EQUAL;
    case CompareOp::kNever:
      return VK_COMPARE_OP_NEVER;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported compare op conversion requested."});
  }
}

expected<CompareOp> to_holoviz_compare_op(VkCompareOp vk_compare_op) {
  switch (vk_compare_op) {
    case VK_COMPARE_OP_ALWAYS:
      return CompareOp::kAlways;
    case VK_COMPARE_OP_LESS:
      return CompareOp::kLess;
    case VK_COMPARE_OP_LESS_OR_EQUAL:
      return CompareOp::kLessEqual;
    case VK_COMPARE_OP_GREATER:
      return CompareOp::kGreater;
    case VK_COMPARE_OP_GREATER_OR_EQUAL:
      return CompareOp::kGreaterEqual;
    case VK_COMPARE_OP_EQUAL:
      return CompareOp::kEqual;
    case VK_COMPARE_OP_NEVER:
      return CompareOp::kNever;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported compare op conversion requested."});
  }
}

expected<VkBlendFactor> from_holoviz_blend_factor(BlendFactor factor) {
  switch (factor) {
    case BlendFactor::kZero:
      return VK_BLEND_FACTOR_ZERO;
    case BlendFactor::kOne:
      return VK_BLEND_FACTOR_ONE;
    case BlendFactor::kSrcAlpha:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case BlendFactor::kOneMinusSrcAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::kDstAlpha:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case BlendFactor::kOneMinusDstAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported blend factor conversion requested."});
  }
}

expected<BlendFactor> to_holoviz_blend_factor(VkBlendFactor vk_blend_factor) {
  switch (vk_blend_factor) {
    case VK_BLEND_FACTOR_ZERO:
      return BlendFactor::kZero;
    case VK_BLEND_FACTOR_ONE:
      return BlendFactor::kOne;
    case VK_BLEND_FACTOR_SRC_ALPHA:
      return BlendFactor::kSrcAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return BlendFactor::kOneMinusSrcAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA:
      return BlendFactor::kDstAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return BlendFactor::kOneMinusDstAlpha;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported blend factor conversion requested."});
  }
}

expected<VkBlendOp> from_holoviz_blend_op(BlendOp op) {
  switch (op) {
    case BlendOp::kAdd:
      return VK_BLEND_OP_ADD;
    case BlendOp::kMin:
      return VK_BLEND_OP_MIN;
    case BlendOp::kMax:
      return VK_BLEND_OP_MAX;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported blend op conversion requested."});
  }
}

expected<BlendOp> to_holoviz_blend_op(VkBlendOp vk_blend_op) {
  switch (vk_blend_op) {
    case VK_BLEND_OP_ADD:
      return BlendOp::kAdd;
    case VK_BLEND_OP_MIN:
      return BlendOp::kMin;
    case VK_BLEND_OP_MAX:
      return BlendOp::kMax;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported blend op conversion requested."});
  }
}

expected<VkIndexType> from_holoviz_index_type(IndexType type) {
  switch (type) {
    case IndexType::kUint16:
      return VK_INDEX_TYPE_UINT16;
    case IndexType::kUint32:
      return VK_INDEX_TYPE_UINT32;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported index type conversion requested."});
  }
}

expected<IndexType> to_holoviz_index_type(VkIndexType vk_index_type) {
  switch (vk_index_type) {
    case VK_INDEX_TYPE_UINT16:
      return IndexType::kUint16;
    case VK_INDEX_TYPE_UINT32:
      return IndexType::kUint32;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported index type conversion requested."});
  }
}

expected<VkVertexInputRate> from_holoviz_vertex_step_mode(VertexStepMode step_mode) {
  switch (step_mode) {
    case VertexStepMode::kVertex:
      return VK_VERTEX_INPUT_RATE_VERTEX;
    case VertexStepMode::kInstance:
      return VK_VERTEX_INPUT_RATE_INSTANCE;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported vertex step mode conversion requested."});
  }
}

expected<VertexStepMode> to_holoviz_vertex_step_mode(VkVertexInputRate vk_vertex_input_rate) {
  switch (vk_vertex_input_rate) {
    case VK_VERTEX_INPUT_RATE_VERTEX:
      return VertexStepMode::kVertex;
    case VK_VERTEX_INPUT_RATE_INSTANCE:
      return VertexStepMode::kInstance;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported vertex step mode conversion requested."});
  }
}

expected<VkFilter> from_holoviz_filter(FilterMode filter_mode) {
  switch (filter_mode) {
    case FilterMode::kNearest:
      return VK_FILTER_NEAREST;
    case FilterMode::kLinear:
      return VK_FILTER_LINEAR;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported filter mode conversion requested."});
  }
}

expected<FilterMode> to_holoviz_filter(VkFilter vk_filter) {
  switch (vk_filter) {
    case VK_FILTER_NEAREST:
      return FilterMode::kNearest;
    case VK_FILTER_LINEAR:
      return FilterMode::kLinear;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported filter mode conversion requested."});
  }
}

expected<VkSamplerAddressMode> from_holoviz_address_mode(AddressMode address_mode) {
  switch (address_mode) {
    case AddressMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case AddressMode::kMirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case AddressMode::kClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case AddressMode::kClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported address mode conversion requested."});
  }
}

expected<AddressMode> to_holoviz_address_mode(VkSamplerAddressMode vk_address_mode) {
  switch (vk_address_mode) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return AddressMode::kRepeat;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return AddressMode::kMirroredRepeat;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
      return AddressMode::kClampToEdge;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return AddressMode::kClampToBorder;
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported address mode conversion requested."});
  }
}

VkTimeDomainKHR to_vk_time_domain(TimeDomain domain) {
  switch (domain) {
    case TimeDomain::kHostMonotonic:
#ifdef _WIN32
      return VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
      return VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
#if defined(VK_EXT_present_timing)
    case TimeDomain::kSwapchainLocal:
      return VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT;
    case TimeDomain::kPresentStageLocal:
      return VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT;
#endif
    case TimeDomain::kDevice:
    default:
      return VK_TIME_DOMAIN_DEVICE_KHR;
  }
}

std::optional<TimeDomain> from_vk_time_domain(VkTimeDomainKHR vk_domain) {
  switch (vk_domain) {
    case VK_TIME_DOMAIN_DEVICE_KHR:
      return TimeDomain::kDevice;
#ifdef _WIN32
    case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR:
      return TimeDomain::kHostMonotonic;
#else
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR:
      return TimeDomain::kHostMonotonic;
#endif
#if defined(VK_EXT_present_timing)
    case VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT:
      return TimeDomain::kSwapchainLocal;
    case VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT:
      return TimeDomain::kPresentStageLocal;
#endif
    default:
      return std::nullopt;
  }
}

PresentStageSet to_present_stage_set(uint32_t vk_present_stages) {
  PresentStageSet stages;
#if defined(VK_EXT_present_timing)
  if ((vk_present_stages & VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT) != 0) {
    stages.add(PresentStage::kQueueOperationsEnd);
  }
  if ((vk_present_stages & VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT) != 0) {
    stages.add(PresentStage::kRequestDequeued);
  }
  if ((vk_present_stages & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) != 0) {
    stages.add(PresentStage::kFirstPixelOut);
  }
  if ((vk_present_stages & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT) != 0) {
    stages.add(PresentStage::kFirstPixelVisible);
  }
#else
  (void)vk_present_stages;
#endif
  return stages;
}

}  // namespace holoviz::vulkan
