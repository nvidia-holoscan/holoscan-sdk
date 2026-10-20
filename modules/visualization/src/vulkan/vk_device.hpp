/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/device_impl.hpp"

#include "vk_adapter.hpp"
#include "vk_impls.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace holoviz::vulkan {

class VulkanRenderPass;

class VulkanDevice final : public detail::DeviceImpl {
 public:
  VulkanDevice() = default;
  ~VulkanDevice() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  /// When \p present_surface is not VK_NULL_HANDLE the queue family is required to both
  /// render and present to it (the device's single queue serves both roles).
  expected<void> init(const VulkanInstance& instance, VkPhysicalDevice physical_device,
                      const DeviceDesc& desc, const Adapter& adapter,
                      VkSurfaceKHR present_surface = VK_NULL_HANDLE);

  Backend backend() const override { return Backend::kVulkan; }
  DevicePolicy policy() const override { return policy_; }

  const Adapter& adapter() const override { return adapter_; }
  const CapabilityReport& capabilities() const override { return capabilities_; }

  expected<void> seal() override;
  bool sealed() const override { return sealed_; }
  expected<void> wait_idle() override;
  expected<void> wait_on_submit(SubmitSyncToken sync_token,
                                std::chrono::nanoseconds timeout) override;

  expected<SubmitSyncToken> submit(const SubmitDesc& batch) override;

  expected<SurfaceCapabilities> surface_capabilities(
      const detail::SurfaceImpl& surface, std::optional<PresentMode> present_mode) const override;

  expected<std::vector<SurfaceFormat>> surface_formats(
      const detail::SurfaceImpl& surface) const override;

  expected<void> wait_for_display_event(const Display& display, DisplayEventType event,
                                        std::chrono::nanoseconds timeout) override;
  expected<CalibratedTimestamps> calibrated_timestamps() const override;

  expected<CUdevice> cuda_device() const override;
  expected<CUcontext> cuda_context() const override;
  expected<bool> cuda_is_cig() const override { return cuda_context_is_cig_; }

  expected<std::unique_ptr<detail::BufferImpl>> create_buffer(
      const BufferDesc& desc) const override;
  expected<std::unique_ptr<detail::ImageImpl>> create_image(const ImageDesc& desc) const override;
  expected<std::unique_ptr<detail::RenderPassImpl>> create_render_pass(
      const RenderPassDesc& desc) const override;
  expected<std::unique_ptr<detail::PipelineImpl>> create_pipeline(
      const PipelineDesc& desc) const override;
  expected<std::unique_ptr<detail::CommandListImpl>> create_command_list(
      const CommandListDesc& desc) const override;
  expected<std::unique_ptr<detail::SwapchainImpl>> create_swapchain(
      const SwapchainDesc& desc) const override;
  expected<std::unique_ptr<detail::QueryPoolImpl>> create_query_pool(
      const QueryPoolDesc& desc) const override;
  expected<std::unique_ptr<detail::ExternalSemaphoreImpl>> create_external_semaphore(
      CUstream stream) const override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

  expected<void> upload_to_buffer(const BufferSlice& dst,
                                  std::span<const std::byte> bytes) override;
  expected<void> download_from_buffer(const BufferSlice& src, std::span<std::byte> bytes) override;

  expected<void> upload_to_image(const detail::ImageImpl& dst, uint32_t mip,
                                 std::span<const std::byte> bytes, size_t row_pitch_bytes) override;

  expected<void> download_from_image(const detail::ImageImpl& src, uint32_t mip,
                                     std::span<std::byte> bytes, size_t row_pitch_bytes) override;

  expected<VkSampler> get_sampler(const SamplerDesc& desc);

  expected<void> staging_buffer_min_size(size_t size_bytes);

  const VkAllocationCallbacks* allocator() const { return allocator_; }
  const VmaAllocator& vma_allocator() const { return vma_allocator_; }
  uint32_t vulkan_queue_family_index() const { return queue_family_index_; }

  expected<VkCommandBuffer> start_cmd_buffer_record() const;
  expected<void> submit_cmd_buffer(VkCommandBuffer cmd_buffer) const;

  /// Submit an empty batch that signals \p semaphore and the device timeline at
  /// timeline_value_ + 1, and return that timeline value so the caller can
  /// record it (a present gate armed this way records it on its swapchain frame
  /// slot). Internal queue submits must never be unobservable: signaling the
  /// timeline keeps them covered by the bounded drain in
  /// destroy()/wait_on_submit(). const because gate holders only have a const
  /// device pointer (same rationale as register_render_pass); timeline_value_
  /// is mutable for this.
  expected<uint64_t> submit_semaphore_signal(VkSemaphore semaphore) const;

  /// Submit the recorded \p cmd_buffer waiting on the binary \p wait_semaphore at
  /// \p wait_stage, and signal \p resignal_semaphore plus the device timeline at
  /// timeline_value_ + 1; return that timeline value so the caller can record it on its
  /// frame slot. \p resignal_semaphore may be \p wait_semaphore itself: the wait retires the
  /// binary semaphore before the signal executes, which is how the swapchain re-arms an
  /// acquire semaphore after slipping a one-time layout transition in behind the acquire.
  /// Non-blocking, unlike submit_cmd_buffer(); the caller owns \p cmd_buffer's lifetime and
  /// must keep it alive until the returned timeline value has been waited out.
  expected<uint64_t> submit_cmd_buffer_gated(VkCommandBuffer cmd_buffer, VkSemaphore wait_semaphore,
                                             VkPipelineStageFlags wait_stage,
                                             VkSemaphore resignal_semaphore) const;

  VkInstance vk_instance() const { return instance_; }
  VkDevice vk_device() const { return device_; }

  /// True when the FIFO_LATEST_READY device extension + feature were enabled at creation.
  /// PresentMode::kFifoLatestReady is only valid to request when this holds; surface
  /// capability reporting masks the mode out otherwise.
  bool fifo_latest_ready_enabled() const { return fifo_latest_ready_enabled_; }

  /// True when VK_EXT_display_control was enabled at creation (with its
  /// VK_EXT_display_surface_counter instance dependency). Gates
  /// wait_for_display_event() and swapchain vblank counters.
  bool display_control_enabled() const { return display_control_enabled_; }

  /// True when VK_KHR_shared_presentable_image was enabled at creation. Gates the shared
  /// present modes; surface capability reporting masks them out otherwise.
  bool shared_present_enabled() const { return shared_present_enabled_; }

  /// True when VK_EXT_present_timing (+ its feature) was enabled at creation. Gates
  /// SwapchainDesc::enable_present_timing.
  bool present_timing_enabled() const { return present_timing_enabled_; }

  /// True when the graphics queue can write timestamps (timestampValidBits != 0).
  bool timestamps_supported() const { return timestamp_valid_bits_mask_ != 0; }

  /// Convert a raw GPU timestamp (query result or calibrated device sample) to nanoseconds
  /// on TimeDomain::kDevice. The single conversion shared by both producers so their values
  /// are directly comparable. Present-stage times are not raw ticks: VK_EXT_present_timing
  /// reports them in nanoseconds already, in every domain.
  uint64_t device_ticks_to_ns(uint64_t ticks) const {
    return static_cast<uint64_t>(static_cast<double>(ticks & timestamp_valid_bits_mask_) *
                                 static_cast<double>(timestamp_period_ns_));
  }

  VkSemaphore timeline_semaphore() const { return timeline_semaphore_; }
  /// Timeline value signalled by the most recent successful queue submission on this
  /// device (0 before the first). Waiting for it retires every submission issued so far,
  /// including gate-less ones no swapchain slot recorded (shared present modes).
  uint64_t last_submitted_timeline_value() const { return timeline_value_; }
  CUexternalSemaphore timeline_external_semaphore() const { return timeline_external_semaphore_; }

  const VolkDeviceTable* fn() const { return &fn_; }

  expected<VkDeviceMemory> allocate_device_memory(
      const VkMemoryRequirements& mem_reqs,
      VkMemoryPropertyFlags required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      void* pNext = nullptr) const;

  /// Live render passes register here so the device can evict their cached
  /// framebuffers when an image view is about to be destroyed (see
  /// notify_image_view_destroyed). const because every holder of a render pass
  /// or image only has a const device pointer; the registry is mutable and
  /// mutex-guarded. Unregistering an unregistered pass is a no-op.
  void register_render_pass(VulkanRenderPass* pass) const;
  void unregister_render_pass(VulkanRenderPass* pass) const;

  /// Notify that \p view is about to be destroyed: evicts and destroys every
  /// cached framebuffer referencing it in every live render pass on this
  /// device. Callers must have drained all submissions referencing the view
  /// first (the existing precondition for destroying the view itself), which
  /// is also vkDestroyFramebuffer's precondition. Without this eviction, a
  /// later view with a recycled handle value could match a stale cache entry
  /// and return a framebuffer built over the destroyed view.
  void notify_image_view_destroyed(VkImageView view) const;

  /// Debug bookkeeping: child objects (buffers, images, pipelines, command lists,
  /// descriptors, query pools, render passes, external semaphores, swapchains) register
  /// after init sets their device pointer and unregister exactly once on destroy()'s
  /// success path. destroy() on this device logs an error when children are still alive
  /// (checked only under HOLOVIZ_VALIDATION); destroying children after the device
  /// remains undefined behavior. const for the same reason as the render-pass registry.
  void register_child() const { live_child_count_.fetch_add(1, std::memory_order_relaxed); }
  void unregister_child() const { live_child_count_.fetch_sub(1, std::memory_order_relaxed); }
  int live_child_count() const { return live_child_count_.load(std::memory_order_relaxed); }

  /// Terminal poison: a child's teardown gave up on wedged pending work — a swapchain
  /// destructor on a wedged presentation engine, or a render backend destructor whose
  /// bounded drain timed out — and leaked/quarantined objects that still reference this
  /// VkDevice, with work possibly still pending on its queue or a producer stream.
  /// destroy() then leaks the native device and everything under it instead of destroying
  /// the parent of live children. Sticky for the device's lifetime; const because children
  /// hold a const device pointer (same rationale as register_child()).
  void report_leaked_resources() const override {
    resources_leaked_.store(true, std::memory_order_relaxed);
  }
  [[nodiscard]] bool resources_leaked() const override {
    return resources_leaked_.load(std::memory_order_relaxed);
  }

 private:
  /// The instance's cached record for physical_device_ — the complete set of supported
  /// extensions, extension properties, and extension features queried at instance
  /// creation. Resolved once at the top of init(); valid for the device's lifetime
  /// (the records live in the sticky instance singleton).
  const PhysicalDeviceInfo& physical_device_info() const { return *physical_device_info_; }
  const PhysicalDeviceInfo* physical_device_info_ = nullptr;

  Adapter adapter_;
  DevicePolicy policy_ = DevicePolicy::kDynamic;
  bool sealed_ = false;
  bool fifo_latest_ready_enabled_ = false;
  bool display_control_enabled_ = false;
  bool shared_present_enabled_ = false;
  bool present_timing_enabled_ = false;
  /// Non-null when a calibrated-timestamps extension was enabled (the spelling chosen).
  const char* calibrated_timestamps_ext_name_ = nullptr;
  /// Graphics-queue timestamp parameters, fixed per physical device (see device_ticks_to_ns).
  float timestamp_period_ns_ = 0.0F;
  uint64_t timestamp_valid_bits_mask_ = 0;
  CapabilityReport capabilities_{};

  VkInstance instance_ = VK_NULL_HANDLE;
  VkAllocationCallbacks* allocator_ = nullptr;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;

  CUdevice cuda_device_ = -1;

  bool owns_cuda_context_ = false;
  bool cuda_context_is_cig_ = false;
  CUcontext cuda_context_ = nullptr;
  VkExternalComputeQueueNV external_compute_queue_ = VK_NULL_HANDLE;

  VkSemaphore timeline_semaphore_ = VK_NULL_HANDLE;
  CUexternalSemaphore timeline_external_semaphore_ = nullptr;
  /// mutable so submit_semaphore_signal() can bump it through the const device
  /// pointer that child objects hold (same rationale as render_pass_registry_).
  mutable uint64_t timeline_value_ = 0;

  VolkDeviceTable fn_{};

  VkPhysicalDeviceMemoryProperties memory_properties_;

  VmaAllocator vma_allocator_ = VK_NULL_HANDLE;

  uint64_t staging_buffer_size_ = 0;
  VkBuffer staging_buffer_ = VK_NULL_HANDLE;
  VmaAllocation staging_allocation_ = VK_NULL_HANDLE;
  VmaAllocationInfo staging_allocation_info_{};

  VkFence cmd_fence_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;

  std::vector<SamplerDesc> sampler_descs_;
  std::vector<VkSampler> samplers_;

  uint32_t queue_family_index_ = static_cast<uint32_t>(-1);

  VolkDeviceTable device_table_{};

  mutable std::mutex render_pass_registry_mutex_;
  mutable std::vector<VulkanRenderPass*> render_pass_registry_;
  mutable std::atomic<int> live_child_count_{0};
  mutable std::atomic<bool> resources_leaked_{false};  // see report_leaked_resources()
};

}  // namespace holoviz::vulkan
