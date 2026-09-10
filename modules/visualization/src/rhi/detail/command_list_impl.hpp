/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <span>

#include <holoviz/rhi/command_list.hpp>

namespace holoviz::rhi::detail {

class DescriptorImpl;
class PipelineImpl;
class QueryPoolImpl;
class RenderPassImpl;

/// Backend interface for CommandList; see the CommandList handle for API docs.
class CommandListImpl {
 public:
  CommandListImpl(const CommandListImpl&) = delete;
  CommandListImpl& operator=(const CommandListImpl&) = delete;
  CommandListImpl(CommandListImpl&&) = delete;
  CommandListImpl& operator=(CommandListImpl&&) = delete;
  virtual ~CommandListImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  [[nodiscard]] virtual expected<void> begin() = 0;
  [[nodiscard]] virtual expected<void> end() = 0;
  [[nodiscard]] virtual expected<void> begin_render(RenderPassImpl& pass, Rect2D render_area) = 0;
  [[nodiscard]] virtual expected<void> end_render() = 0;
  [[nodiscard]] virtual expected<void> bind_vertex_buffer(uint32_t slot, BufferSlice slice) = 0;
  [[nodiscard]] virtual expected<void> bind_index_buffer(BufferSlice slice, IndexType type) = 0;
  [[nodiscard]] virtual expected<void> bind_pipeline(PipelineImpl& pipeline) = 0;
  [[nodiscard]] virtual expected<void> bind_descriptor(const DescriptorImpl& descriptor) = 0;
  [[nodiscard]] virtual expected<void> draw(uint32_t vertex_count, uint32_t instance_count,
                                            uint32_t first_vertex, uint32_t first_instance) = 0;
  [[nodiscard]] virtual expected<void> draw_indexed(uint32_t index_count, uint32_t instance_count,
                                                    uint32_t first_index, int32_t vertex_offset,
                                                    uint32_t first_instance) = 0;
  [[nodiscard]] virtual expected<void> set_push_data(const PipelineImpl& pipeline,
                                                     std::span<const std::byte> bytes) = 0;
  [[nodiscard]] virtual expected<void> set_viewport(Rect2D viewport) = 0;
  [[nodiscard]] virtual expected<void> set_scissor(Rect2D scissor) = 0;
  [[nodiscard]] virtual expected<void> reset_queries(QueryPoolImpl& pool, uint32_t first,
                                                     uint32_t count) = 0;
  [[nodiscard]] virtual expected<void> write_timestamp(QueryPoolImpl& pool, uint32_t index,
                                                       TimestampLocation location) = 0;
  [[nodiscard]] virtual expected<void> resolve_queries(QueryPoolImpl& pool, uint32_t first,
                                                       uint32_t count) = 0;
  [[nodiscard]] virtual expected<void> reset(std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  CommandListImpl() = default;
};

}  // namespace holoviz::rhi::detail
