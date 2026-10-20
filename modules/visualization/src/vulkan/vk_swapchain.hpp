/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/swapchain_impl.hpp"
#include "vk_device.hpp"
#include "vk_image.hpp"
#include "vulkan/vk_swapchain_fault_injection.hpp"

#include "vk_impls.hpp"  // IWYU pragma: keep

#include <span>
#include <vector>

namespace holoviz::vulkan {

class VulkanAcquireSignalGate final : public detail::AcquireSignalGateImpl {
 public:
  VulkanAcquireSignalGate(const VulkanDevice* device, VkQueue queue)
      : device_(device), queue_(queue) {}
  VulkanAcquireSignalGate(VulkanAcquireSignalGate&&) noexcept;
  ~VulkanAcquireSignalGate() override;

  expected<void> set_semaphore(VkSemaphore semaphore, VkBool32* consumed_ptr,
                               SubmitSyncToken* slot_token_ptr) {
    semaphore_ = semaphore;
    consumed_ptr_ = consumed_ptr;
    slot_token_ptr_ = slot_token_ptr;
    return {};
  }

  /// Record the device-timeline value of the submit that consumed this gate on the owning
  /// swapchain frame slot, so the slot-drain waits in resize()/destroy() cover that submit
  /// even when the frame is never presented. Max-merge: the slot stays monotonic no matter
  /// the order the two gates' submits land in.
  void record_submit_timeline(uint64_t timeline_value) {
    if (slot_token_ptr_ != nullptr && timeline_value > slot_token_ptr_->timeline_value) {
      slot_token_ptr_->timeline_value = timeline_value;
    }
  }

  expected<void> wait(std::chrono::nanoseconds timeout) override;
  expected<bool> consumed() const override;
  expected<void> consume() override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

  /// Finish a drain submit an earlier timed-out wait() left in flight: wait its fence
  /// (kBackpressured on timeout, gate untouched and retryable). No-op when nothing is
  /// pending. The swapchain calls this before invalidating or destroying the gate — the
  /// pending submit references both the fence and the acquire semaphore.
  expected<void> drain_pending_wait(std::chrono::nanoseconds timeout);

  /// Destroy the persistent drain fence. Only valid once no drain is pending
  /// (drain_pending_wait() succeeded) or the device is lost.
  void destroy_drain_fence();

 private:
  const VulkanDevice* device_ = nullptr;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;

  VkBool32* consumed_ptr_ = nullptr;
  /// Into the owning swapchain's slot_submit_sync_tokens_; same lifetime story as
  /// consumed_ptr_ (recreate() invalidates it, generation-checked handles guard access).
  SubmitSyncToken* slot_token_ptr_ = nullptr;

  /// Persistent completion fence for wait()'s drain submit, reused across calls; never
  /// destroyed while a submission referencing it can still be pending (drain_pending_).
  VkFence drain_fence_ = VK_NULL_HANDLE;
  bool drain_pending_ = false;
};

class VulkanPresentWaitGate final : public detail::PresentWaitGateImpl {
 public:
  VulkanPresentWaitGate(const VulkanDevice* device, VkQueue queue)
      : device_(device), queue_(queue) {}
  VulkanPresentWaitGate(VulkanPresentWaitGate&&) noexcept;
  ~VulkanPresentWaitGate() override;

  expected<void> set_semaphore(VkSemaphore semaphore, VkBool32* consumed_ptr,
                               SubmitSyncToken* slot_token_ptr) {
    semaphore_ = semaphore;
    consumed_ptr_ = consumed_ptr;
    slot_token_ptr_ = slot_token_ptr;
    return {};
  }

  /// See VulkanAcquireSignalGate::record_submit_timeline; signal() records its own
  /// arm-submit through this too.
  void record_submit_timeline(uint64_t timeline_value) {
    if (slot_token_ptr_ != nullptr && timeline_value > slot_token_ptr_->timeline_value) {
      slot_token_ptr_->timeline_value = timeline_value;
    }
  }

  expected<void> signal() override;
  expected<bool> consumed() const override;
  expected<void> consume() override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  const VulkanDevice* device_ = nullptr;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;

  VkBool32* consumed_ptr_ = nullptr;
  /// Into the owning swapchain's slot_submit_sync_tokens_; same lifetime story as
  /// consumed_ptr_ (recreate() invalidates it, generation-checked handles guard access).
  SubmitSyncToken* slot_token_ptr_ = nullptr;
};

class VulkanSwapchain final : public detail::SwapchainImpl {
 public:
  VulkanSwapchain() = default;
  ~VulkanSwapchain() override;

  expected<void> init(const VulkanDevice& device, const SwapchainDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  [[nodiscard]] expected<void> resize(Extent2D new_extent,
                                      std::chrono::nanoseconds timeout) override;

  Extent2D extent() const override { return extent_; }

  Format format() const override;
  ColorSpace color_space() const override;
  PresentMode present_mode() const override;
  [[nodiscard]] expected<void> set_present_mode(PresentMode mode) override;
  [[nodiscard]] expected<void> set_format(Format format, ColorSpace color_space,
                                          std::chrono::nanoseconds timeout) override;
  [[nodiscard]] expected<uint64_t> vblank_counter() const override;
  expected<std::reference_wrapper<const Image>> image(uint32_t index) const override;

  uint32_t image_count() const override;
  uint64_t generation() const override { return generation_; }

  expected<AcquireSyncToken> acquire(std::chrono::nanoseconds timeout) override;
  expected<void> present(AcquireSyncToken&& acquired, SubmitSyncToken submit_sync_token,
                         uint64_t present_id, PresentConsumed* consumed) override;
  expected<void> wait_for_present(uint64_t present_id, std::chrono::nanoseconds timeout) override;
  expected<uint32_t> past_present_timings(std::span<PresentTimingSample> out,
                                          bool allow_partial) override;
  expected<uint64_t> refresh_duration_ns() const override;
  expected<RefreshMode> refresh_mode() const override;
  expected<TimeDomain> present_timing_domain() const override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

  /// Sanctioned fault-injection knobs (tests only; see vk_swapchain_fault_injection.hpp).
  // Mutable: const queries (query_timing_properties()) consume their one-shot knob too.
  mutable vulkan::SwapchainFaultInjection fault_injection_;

 private:
  friend SwapchainFenceCounts swapchain_fence_counts(const Swapchain& swapchain);

  /// (Re)create the VkSwapchainKHR and everything sized to its images, retiring any current
  /// swapchain via VkSwapchainCreateInfoKHR::oldSwapchain. Shared by init() and resize();
  /// requires the format/color space/present mode members to be selected already. A zero
  /// \p desired_extent means "use the surface's current extent". Mutates state only after
  /// the new swapchain is created, so a failure leaves the current one usable for destroy().
  expected<void> recreate(Extent2D desired_extent);
  /// The image-view format for a scanout format under srgb_attachment_: the sRGB sibling of
  /// RGBA8/BGRA8 (requires VK_KHR_swapchain_mutable_format when it differs), else the scanout
  /// format itself.
  expected<VkFormat> resolve_attachment_format(VkFormat scanout_format,
                                               VkColorSpaceKHR color_space) const;

  /// resize()'s body behind the destroyed-object guard: drain this swapchain's in-flight
  /// frame slots, release an unpresented acquire, reap retired generations, then
  /// recreate(). acquire()/present() call it (with a zero extent = surface's current size)
  /// to recover from out-of-date conditions automatically.
  expected<void> drain_and_recreate(Extent2D new_extent, std::chrono::nanoseconds timeout);

  /// Wait (bounded by \p timeout) for every submission that may still render into the
  /// current images: the frame-slot tokens, plus — in shared present modes — the device's
  /// last submitted timeline value, because gate-less shared-image submits leave no slot
  /// record. kBackpressured on expiry with nothing changed; \p action names the caller's
  /// operation in that message ("destroying" / "resizing").
  expected<void> drain_in_flight_rendering(std::chrono::nanoseconds timeout, const char* action);

  /// The internal recovery path: drain_and_recreate({0,0}) with failures mapped for the
  /// frame loop — kBackpressured passes through (retry), anything else is wrapped as
  /// kOutOfDate pointing the caller at an explicit resize().
  expected<void> auto_resize(std::chrono::nanoseconds timeout);

  /// Destroy retired swapchains whose present fences have all signaled, oldest first, and
  /// recycle their fences. \p force destroys every retired generation regardless of fence
  /// state (destroy()-after-wait only).
  void reap_retired(bool force);

  /// Move the current swapchain into retired_ and destroy its image views; the
  /// recreation paths call this once the current generation is no longer in use.
  void retire_current_swapchain();

  /// Retire the current swapchain and invalidate everything keyed to its generation: gate
  /// storage, trackers, semaphore lists, layout command buffers and the generation counter.
  /// Image views are destroyed; the Image wrappers in images_ are left for the caller to
  /// re-initialize or clear.
  void invalidate_current_generation();

  /// Wait (bounded by \p timeout) for the present fences of the current swapchain and of
  /// every retired generation. kBackpressured on expiry with nothing changed.
  expected<void> wait_all_present_fences(std::chrono::nanoseconds timeout, const char* action);

  /// Take a reset fence for the next present, recycling signaled ones first.
  expected<VkFence> take_present_fence();

  /// Return current-swapchain present fences that have signaled to the free pool.
  void recycle_signaled_present_fences();

  /// True when at least one of \p fences is unsignaled and every unsignaled one belongs to a
  /// present the engine rejected with VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT (current
  /// generation or retired). destroy() uses it to diagnose a driver that never signals them.
  [[nodiscard]] bool only_timing_rejected_fences_unsignaled(std::span<const VkFence> fences) const;

  /// Give an acquired-but-unpresented image back to the presentation engine
  /// (vkReleaseSwapchainImages), draining its acquire semaphore first if no submit
  /// consumed it. When a submit did consume the gates, the callers' preceding slot-drain
  /// wait already retired it (submit() records its timeline on the frame slot through the
  /// gates), which is what makes releasing the image — and later destroying/retiring the
  /// present semaphore it signaled — safe here. No-op when nothing is outstanding.
  expected<void> release_unpresented_acquire(std::chrono::nanoseconds timeout);

  /// Record and submit the one-time home-layout transition for the just-acquired
  /// \p image_index behind frame slot \p slot's acquire semaphore, re-arming that semaphore
  /// for the caller's gate and recording the submit on the slot. No-op once done for the
  /// image in this generation.
  expected<void> transition_acquired_image(uint32_t image_index, uint32_t slot);

  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  /// destroy() logged the timing-rejected-fence timeout diagnostic once already.
  bool warned_timing_rejected_fence_timeout_ = false;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;  ///< Borrowed from the caller's Surface; not owned.
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;

  VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
  VkFormat attachment_format_ = VK_FORMAT_UNDEFINED;
  bool srgb_attachment_ = false;  ///< SwapchainDesc::srgb_attachment, re-applied on set_format()
  VkColorSpaceKHR swapchain_color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  /// From SwapchainDesc::enable_vblank_counter: every (re)creation chains
  /// VkSwapchainCounterCreateInfoEXT and vblank_counter() reads the live counter.
  bool vblank_counter_enabled_ = false;
  /// From SwapchainDesc::enable_present_timing: every (re)creation sets the create flag,
  /// applies the timing queue size and re-chooses the time domain; tagged presents carry a
  /// timing request and past_present_timings() drains the results.
  bool present_timing_enabled_ = false;
  /// One-time warning latch for refresh_mode() on an engine whose refreshInterval is neither
  /// UINT64_MAX nor equal to refreshDuration.
  mutable bool refresh_interval_warned_ = false;
  /// One-time warning latch for past_present_timings() results in a VkTimeDomainKHR that
  /// TimeDomain cannot represent (those samples are dropped, not mislabelled).
  bool unmapped_timing_domain_warned_ = false;
#if defined(VK_EXT_present_timing)
  /// Shared gate + vkGetSwapchainTimingPropertiesEXT call behind refresh_duration_ns() and
  /// refresh_mode(); \p caller names the public method in error messages.
  [[nodiscard]] expected<VkSwapchainTimingPropertiesEXT> query_timing_properties(
      const char* caller) const;
#endif
  uint32_t present_timing_queue_size_ = 0;
  std::optional<TimeDomain> requested_timing_domain_;
  TimeDomain timing_domain_ = TimeDomain::kDevice;
  uint64_t timing_domain_id_ = 0;       ///< Driver id of timing_domain_ for this VkSwapchainKHR.
  uint32_t present_stage_queries_ = 0;  ///< VkPresentStageFlagsEXT the surface reports.
  /// Tagged presents whose samples have not been drained yet; present() rejects new timing
  /// requests pre-queue once this reaches the queue size.
  uint32_t outstanding_timing_requests_ = 0;
#if defined(VK_EXT_present_timing)
  /// Reused across past_present_timings() calls so the steady state allocates nothing.
  std::vector<VkPastPresentationTimingEXT> timing_scratch_;
  std::vector<VkPresentStageTimeEXT> stage_scratch_;
#endif
  /// Shared present mode (kSharedDemandRefresh / kSharedContinuousRefresh): one shared
  /// image, acquired once. After the first frame's normal handshake, acquire() mints
  /// pre-consumed gates (shared_synthetic_acquire_) and present() waits no semaphore.
  /// Gate-less render submits therefore leave no per-slot timeline record, so retirement
  /// (drain_in_flight_rendering) also waits the device's last submitted timeline value.
  bool shared_present_ = false;
  bool shared_image_acquired_ = false;
  bool shared_synthetic_acquire_ = false;
  VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  uint32_t requested_image_count_ = 0;  ///< The caller's original request, re-clamped per recreate.
  SwapchainScaling scaling_ = SwapchainScaling::kNone;
  /// From Surface::resizable() at creation. Gates every per-frame resize check in
  /// acquire(): a fixed-size window or a direct-to-display output never runs any of the
  /// automatic-resize machinery.
  bool resizable_ = false;
  /// False on a Tegra iGPU's driver-managed direct display: a successor created with oldSwapchain
  /// cannot present there, so recreation destroys the current swapchain first.
  bool hand_over_on_recreate_ = true;

  /// Present modes the surface reports as switchable-per-present for the current
  /// swapchain's base mode; the swapchain is created with this list
  /// (VkSwapchainPresentModesCreateInfoEXT) and set_present_mode() validates against it.
  std::vector<VkPresentModeKHR> compatible_present_modes_;

  /// A swapchain retired by resize(). The presentation engine may still be displaying its
  /// frames and consuming the semaphores of its queued presents; its present fences
  /// (VK_EXT_swapchain_maintenance1) tell exactly when that stops, so retired generations
  /// are destroyed as soon as all their fences have signaled — deterministically, not
  /// heuristically.
  struct RetiredSwapchain {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkSemaphore> semaphores;
    std::vector<VkFence> fences;  ///< Present fences still in flight when retired.
    /// The subset of \c fences belonging to timing-rejected presents (see
    /// timing_rejected_present_fences_).
    std::vector<VkFence> timing_rejected_fences;
  };
  std::vector<RetiredSwapchain> retired_;

  /// Present fence pool. Every present carries a fence (VkSwapchainPresentFenceInfoEXT)
  /// that signals when the presentation engine is done with the present's resources.
  std::vector<VkFence> free_present_fences_;     ///< Reset, ready for the next present.
  std::vector<VkFence> pending_present_fences_;  ///< In flight on this swapchain, oldest first.
  /// The subset of pending_present_fences_ whose present the engine rejected with
  /// VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT. Such a present is enqueued like any other (its
  /// semaphore wait executes, its image is released, its fence must signal), so its fence
  /// stays in the ordinary retirement proof — but the frame was never displayed, which is
  /// what lets destroy() fall back to a queue-idle proof for exactly these fences when a
  /// driver never signals them (NVIDIA 595.84 observed). Ordered like pending_present_fences_.
  std::vector<VkFence> timing_rejected_present_fences_;

  /// True between a successful acquire() and the following present(); resize() then owes
  /// the presentation engine a vkReleaseSwapchainImages for current_image_.
  bool has_unpresented_acquire_ = false;

  /// Persistent completion fence for release_unpresented_acquire()'s drain submit, reused
  /// across calls. While acquire_drain_pending_ the submit is still in flight: the fence is
  /// re-waited on retry instead of resubmitting, and never destroyed until it signals (or
  /// the device is lost).
  VkFence acquire_drain_fence_ = VK_NULL_HANDLE;
  bool acquire_drain_pending_ = false;

  /// One-time UNDEFINED -> home-layout transition per swapchain image, submitted behind the
  /// image's first acquire in each generation (the spec forbids touching a presentable image
  /// before vkAcquireNextImageKHR hands it out, so this cannot happen at creation). The pool
  /// is reset whole at recreate(), after the slot drains retired every prior submit; the
  /// command buffers are allocated per generation, one per image, and recorded on demand.
  VkCommandPool layout_cmd_pool_ = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> layout_cmds_;
  std::vector<VkBool32> image_layout_initialized_;

  std::vector<VulkanImage*> images_;  ///< Non-owning pointers for internal Vulkan-specific access.
  std::vector<Image> image_handles_;  ///< Public Image handles; owns the VulkanImage objects.

  std::vector<VulkanAcquireSignalGate> acquire_gates_;
  std::vector<VulkanPresentWaitGate> present_gates_;

  std::vector<uint32_t> frame_tracker_;

  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkBool32> image_gate_consumed_;
  std::vector<SubmitSyncToken> slot_submit_sync_tokens_;

  std::vector<VkSemaphore> present_finished_semaphores_;
  std::vector<VkBool32> present_gate_consumed_;

  uint32_t current_image_ = 0;
  uint32_t current_frame_ = 0;
  uint32_t image_count_ = 0;
  uint32_t frame_count_ = 0;
  Extent2D extent_{0, 0};
  /// Highest nonzero present_id tagged so far (VkPresentIdKHR ids must be strictly
  /// increasing per swapchain).
  uint64_t last_present_id_ = 0;
  /// Incremented by every successful recreate() (including the one inside init()). Lets
  /// layered code detect recreations that happened inside acquire()/present() and drop
  /// stale AcquireSyncTokens without touching their dead gates.
  uint64_t generation_ = 0;
  /// image_count() as last reported to the caller (at init, after an explicit resize(), or
  /// with kImageCountChanged); an internal recreation that lands on a different count is
  /// surfaced exactly once.
  uint32_t caller_visible_image_count_ = 0;
};

}  // namespace holoviz::vulkan
