/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CommandList: recorded GPU work, submitted through Device::submit(). The two
// list types encode the module's two execution styles — re-record every frame
// (kTransient) or record once during setup and replay forever (kReusable, the
// steady-state contract this API is built around).

#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <span>

#include <holoviz/holoviz_export.h>

#include "buffer.hpp"
#include "common.hpp"
#include "pipeline.hpp"
#include "query.hpp"
#include "render_pass.hpp"

namespace holoviz::rhi {

enum class CommandListType : uint8_t {
  kTransient,  ///< Record, submit once, then reset() to recycle.
  kReusable    ///< Record once, then submit any number of times — including while previous
               ///< submissions of the same list are still executing on the GPU. "Concurrent"
               ///< means GPU-side overlap of submissions, NOT concurrent record()/submit()
               ///< calls from multiple threads (recording is single-threaded per list; see
               ///< the class doc). Never reset.
};

struct CommandListDesc {
  CommandListType type = CommandListType::kReusable;
  const char* debug_name = nullptr;
};

enum class IndexType : uint8_t { kUint16, kUint32 };

inline constexpr uint32_t kInvalidBindingIndex = ~0u;

class Descriptor;

namespace detail {
class CommandListImpl;
}

/// Recording surface for one command buffer. Recording is single-threaded per
/// list and bracketed by begin()/end(); all binds and draws must additionally
/// sit inside a begin_render()/end_render() scope. State (pipeline, bindings,
/// viewport/scissor) does not persist across recordings — set everything each
/// time.
class CommandList {
 public:
  HOLOVIZ_EXPORT explicit CommandList(std::unique_ptr<detail::CommandListImpl> impl);
  HOLOVIZ_EXPORT CommandList();
  HOLOVIZ_EXPORT CommandList(CommandList&&) noexcept;
  HOLOVIZ_EXPORT CommandList& operator=(CommandList&&) noexcept;
  CommandList(const CommandList&) = delete;
  CommandList& operator=(const CommandList&) = delete;
  HOLOVIZ_EXPORT ~CommandList();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> begin();
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> end();

  /// Begin rendering into the pass's currently set attachments (see
  /// RenderPass::set_render_image). Render scopes cannot nest.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> begin_render(RenderPass& pass, Rect2D render_area);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> end_render();

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_vertex_buffer(uint32_t slot, BufferSlice slice);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_index_buffer(BufferSlice slice, IndexType type);

  /// Bind a pipeline. First use with a given render pass compiles the
  /// pass-specific backend pipeline variant, which is then cached — warm every
  /// pipeline x pass combination during setup to keep the frame loop
  /// compile-free.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_pipeline(Pipeline& pipeline);

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_descriptor(const Descriptor& descriptor);

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> draw(uint32_t vertex_count,
                                                   uint32_t instance_count = 1,
                                                   uint32_t first_vertex = 0,
                                                   uint32_t first_instance = 0);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> draw_indexed(uint32_t index_count,
                                                           uint32_t instance_count = 1,
                                                           uint32_t first_index = 0,
                                                           int32_t vertex_offset = 0,
                                                           uint32_t first_instance = 0);

  /// Record push-constant data, visible to both shader stages. `bytes` must
  /// fit the pipeline's reflected push_constant_size. This is the intended
  /// path for small per-draw runtime data — there are no uniform buffers.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_push_data(const Pipeline& pipeline,
                                                            std::span<const std::byte> bytes);

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_viewport(Rect2D viewport);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_scissor(Rect2D scissor);

  /// Reset timestamp slots [first, first + count) so they can be written by
  /// later commands in this list. Slots allocate unreset, so every slot must
  /// be reset on the GPU timeline before its first write — record this at the
  /// top of the list covering every slot the list writes, and each replay of a
  /// reusable list recycles its own slots. Must be recorded outside a render
  /// scope. A recording no-op on backends whose queries need no reset (D3D12).
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> reset_queries(QueryPool& pool, uint32_t first,
                                                            uint32_t count);

  /// Record a GPU timestamp into one slot. Valid inside or outside a render
  /// scope. Bracket the work to measure with a kStart and a kEnd slot; the
  /// difference of the resolved values is the queue-execution span between the
  /// two write points, in nanoseconds.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> write_timestamp(QueryPool& pool, uint32_t index,
                                                              TimestampLocation location);

  /// Make slots [first, first + count) host-readable: once the submit carrying
  /// this command completes, QueryPool::results_ns() for the range returns
  /// their values. Record it after the range's last write_timestamp, outside a
  /// render scope. Explicit because some backends (D3D12) can only read
  /// queries through an in-command-list resolve into a readback buffer; on
  /// Vulkan it records nothing (results are read straight from the pool), so
  /// it costs nothing there — but call it anyway or the code breaks on those
  /// backends.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> resolve_queries(QueryPool& pool, uint32_t first,
                                                              uint32_t count);

  /// Recycle a kTransient list for re-recording. Blocks until the device
  /// timeline passes this list's last submit (timeout -> kBackpressured), so
  /// it doubles as the per-frame pacing point. Fails on kReusable lists.
  /// May also be called on a partially recorded list (begun, possibly inside
  /// a render scope) that was never submitted: the recorded commands are
  /// discarded and the recording state is cleared — this is the abort path
  /// for a frame that failed mid-record.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> reset(std::chrono::nanoseconds timeout);

  /// Vulkan: answers kVkDevice, kVkQueue, kVkCommandBuffer, kVkCommandPool.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::CommandListImpl& impl() { return *impl_; }
  const detail::CommandListImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::CommandListImpl> impl_;
};

}  // namespace holoviz::rhi
