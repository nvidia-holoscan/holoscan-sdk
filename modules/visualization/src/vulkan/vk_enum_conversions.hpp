/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/pipeline.hpp>
#include <holoviz/swapchain.hpp>

#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::vulkan {

expected<VkFormat> from_holoviz_format(Format format);
expected<Format> to_holoviz_format(VkFormat vk_format);

/// Size in bytes of one texel of `vk_format` (only the formats from_holoviz_format()
/// produces — all single-plane, non-compressed). Used to convert a caller's
/// row_pitch_bytes into VkBufferImageCopy::bufferRowLength (which counts texels).
expected<uint32_t> vk_format_texel_size_bytes(VkFormat vk_format);

/// Map a swapchain gate's wait stage to the Vulkan stage the consuming submit blocks at.
/// Total and infallible: an unrecognized value falls back to the conservative all-commands
/// wait, which is always correct and never silently weakens the barrier.
VkPipelineStageFlags from_holoviz_gate_wait_stage(GateWaitStage wait_stage);

expected<VkPresentModeKHR> from_holoviz_present_mode(PresentMode present_mode);
expected<PresentMode> to_holoviz_present_mode(VkPresentModeKHR vk_present_mode);

expected<VkColorSpaceKHR> from_holoviz_color_space(ColorSpace color_space);
expected<ColorSpace> to_holoviz_color_space(VkColorSpaceKHR vk_color_space);

expected<VkDescriptorType> from_holoviz_binding_kind(BindingKind kind);
expected<BindingKind> to_holoviz_binding_kind(VkDescriptorType vk_descriptor_type);

expected<VkPrimitiveTopology> from_holoviz_primitive_topology(PrimitiveTopology topology);
expected<PrimitiveTopology> to_holoviz_primitive_topology(VkPrimitiveTopology vk_topology);

expected<VkCullModeFlags> from_holoviz_cull_mode(CullMode cull_mode);
expected<VkPolygonMode> from_holoviz_polygon_mode(PolygonMode polygon_mode);
expected<CullMode> to_holoviz_cull_mode(VkCullModeFlags vk_cull_mode);

expected<VkCompareOp> from_holoviz_compare_op(CompareOp op);
expected<CompareOp> to_holoviz_compare_op(VkCompareOp vk_compare_op);

expected<VkBlendFactor> from_holoviz_blend_factor(BlendFactor factor);
expected<BlendFactor> to_holoviz_blend_factor(VkBlendFactor vk_blend_factor);

expected<VkBlendOp> from_holoviz_blend_op(BlendOp op);
expected<BlendOp> to_holoviz_blend_op(VkBlendOp vk_blend_op);

expected<VkIndexType> from_holoviz_index_type(IndexType type);
expected<IndexType> to_holoviz_index_type(VkIndexType vk_index_type);

expected<VkVertexInputRate> from_holoviz_vertex_step_mode(VertexStepMode step_mode);
expected<VertexStepMode> to_holoviz_vertex_step_mode(VkVertexInputRate vk_vertex_input_rate);

expected<VkFilter> from_holoviz_filter(FilterMode filter_mode);
expected<FilterMode> to_holoviz_filter(VkFilter vk_filter);

expected<VkSamplerAddressMode> from_holoviz_address_mode(AddressMode address_mode);
expected<AddressMode> to_holoviz_address_mode(VkSamplerAddressMode vk_address_mode);

}  // namespace holoviz::vulkan
