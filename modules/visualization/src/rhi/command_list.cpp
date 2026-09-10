/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/command_list.hpp"

#include <memory>
#include <utility>

#include "holoviz/rhi/descriptor.hpp"

#include "detail/command_list_impl.hpp"
#include "detail/descriptor_impl.hpp"
#include "detail/handle_errors.hpp"
#include "detail/pipeline_impl.hpp"
#include "detail/query_impl.hpp"
#include "detail/render_pass_impl.hpp"

namespace holoviz::rhi {

CommandList::CommandList(std::unique_ptr<detail::CommandListImpl> impl) : impl_(std::move(impl)) {}
CommandList::CommandList() = default;
CommandList::CommandList(CommandList&&) noexcept = default;
CommandList& CommandList::operator=(CommandList&&) noexcept = default;
CommandList::~CommandList() = default;

expected<void> CommandList::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> CommandList::begin() {
  if (!impl_) {
    return detail::null_impl_error("CommandList::begin");
  }
  return impl_->begin();
}

expected<void> CommandList::end() {
  if (!impl_) {
    return detail::null_impl_error("CommandList::end");
  }
  return impl_->end();
}

expected<void> CommandList::begin_render(RenderPass& pass, Rect2D render_area) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::begin_render");
  }
  if (!pass) {
    return detail::null_arg_error("CommandList::begin_render", "pass");
  }
  return impl_->begin_render(pass.impl(), render_area);
}

expected<void> CommandList::end_render() {
  if (!impl_) {
    return detail::null_impl_error("CommandList::end_render");
  }
  return impl_->end_render();
}

expected<void> CommandList::bind_vertex_buffer(uint32_t slot, BufferSlice slice) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::bind_vertex_buffer");
  }
  if (slice.buffer == nullptr) {
    return detail::null_arg_error("CommandList::bind_vertex_buffer", "slice");
  }
  return impl_->bind_vertex_buffer(slot, slice);
}

expected<void> CommandList::bind_index_buffer(BufferSlice slice, IndexType type) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::bind_index_buffer");
  }
  if (slice.buffer == nullptr) {
    return detail::null_arg_error("CommandList::bind_index_buffer", "slice");
  }
  return impl_->bind_index_buffer(slice, type);
}

expected<void> CommandList::bind_pipeline(Pipeline& pipeline) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::bind_pipeline");
  }
  if (!pipeline) {
    return detail::null_arg_error("CommandList::bind_pipeline", "pipeline");
  }
  return impl_->bind_pipeline(pipeline.impl());
}

expected<void> CommandList::bind_descriptor(const Descriptor& descriptor) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::bind_descriptor");
  }
  if (!descriptor) {
    return detail::null_arg_error("CommandList::bind_descriptor", "descriptor");
  }
  return impl_->bind_descriptor(descriptor.impl());
}

expected<void> CommandList::draw(uint32_t vertex_count, uint32_t instance_count,
                                 uint32_t first_vertex, uint32_t first_instance) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::draw");
  }
  return impl_->draw(vertex_count, instance_count, first_vertex, first_instance);
}

expected<void> CommandList::draw_indexed(uint32_t index_count, uint32_t instance_count,
                                         uint32_t first_index, int32_t vertex_offset,
                                         uint32_t first_instance) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::draw_indexed");
  }
  return impl_->draw_indexed(
      index_count, instance_count, first_index, vertex_offset, first_instance);
}

expected<void> CommandList::set_push_data(const Pipeline& pipeline,
                                          std::span<const std::byte> bytes) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::set_push_data");
  }
  if (!pipeline) {
    return detail::null_arg_error("CommandList::set_push_data", "pipeline");
  }
  return impl_->set_push_data(pipeline.impl(), bytes);
}

expected<void> CommandList::set_viewport(Rect2D viewport) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::set_viewport");
  }
  return impl_->set_viewport(viewport);
}

expected<void> CommandList::set_scissor(Rect2D scissor) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::set_scissor");
  }
  return impl_->set_scissor(scissor);
}

expected<void> CommandList::reset_queries(QueryPool& pool, uint32_t first, uint32_t count) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::reset_queries");
  }
  if (!pool) {
    return detail::null_arg_error("CommandList::reset_queries", "pool");
  }
  return impl_->reset_queries(pool.impl(), first, count);
}

expected<void> CommandList::write_timestamp(QueryPool& pool, uint32_t index,
                                            TimestampLocation location) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::write_timestamp");
  }
  if (!pool) {
    return detail::null_arg_error("CommandList::write_timestamp", "pool");
  }
  return impl_->write_timestamp(pool.impl(), index, location);
}

expected<void> CommandList::resolve_queries(QueryPool& pool, uint32_t first, uint32_t count) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::resolve_queries");
  }
  if (!pool) {
    return detail::null_arg_error("CommandList::resolve_queries", "pool");
  }
  return impl_->resolve_queries(pool.impl(), first, count);
}

expected<void> CommandList::reset(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("CommandList::reset");
  }
  return impl_->reset(timeout);
}

expected<NativeObject> CommandList::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("CommandList::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
