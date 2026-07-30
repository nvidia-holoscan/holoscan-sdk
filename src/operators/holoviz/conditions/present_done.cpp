/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/holoviz/conditions/present_done.hpp>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <holoscan/core/fragment.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace holoscan {

struct PresentDoneCondition::Impl {
  std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op_;

  std::thread thread_;
  std::atomic<bool> stop_requested_ = false;

  std::atomic<SchedulingStatusType> current_state_{SchedulingStatusType::kReady};

  uint64_t present_id_ = 0;

  void thread_func(gxf_context_t gxf_context, gxf_uid_t gxf_eid) {
    while (!stop_requested_) {
      bool is_ready = false;
      try {
        // choose the timeout to be longer than the display period (assume 10Hz)
        is_ready = holoviz_op_->wait_for_present(present_id_, 100'000'000);
      } catch (const std::runtime_error& e) {
        // wait_for_present throws when the underlying VK_KHR_present_wait extension is not
        // available (e.g. Jetson AGX Thor iGPU, Jetson Orin iGPU). Without this catch the
        // exception would escape this std::thread callable and trigger std::terminate(),
        // crashing the application. Instead, transition the condition to kNever so the
        // owning operator stops ticking, then ask the fragment to wind down all operators
        // gracefully. The application exits cleanly with a clear ERROR log identifying the
        // missing extension and the FirstPixelOutCondition alternative.
        HOLOSCAN_LOG_ERROR(
            "PresentDoneCondition: {}. Stopping fragment execution. Use FirstPixelOutCondition "
            "instead, or run on a platform where the `VK_KHR_present_wait` Vulkan device "
            "extension is supported.",
            e.what());
        current_state_.store(SchedulingStatusType::kNever);
        const gxf_result_t notify_result = GxfEntityEventNotify(gxf_context, gxf_eid);
        if (notify_result != GXF_SUCCESS) {
          HOLOSCAN_LOG_ERROR("GxfEntityEventNotify failed during shutdown: {}",
                             GxfResultStr(notify_result));
        }
        // The owning HolovizOp lives in the same Fragment as this condition; reuse its
        // back-pointer.
        if (auto* fragment = holoviz_op_->fragment()) {
          fragment->stop_execution();
        }
        return;
      }

      if (is_ready) {
        present_id_++;
        SchedulingStatusType expected = SchedulingStatusType::kWaitEvent;
        current_state_.compare_exchange_strong(expected, SchedulingStatusType::kReady);

        if (expected == SchedulingStatusType::kWaitEvent) {
          const gxf_result_t result = GxfEntityEventNotify(gxf_context, gxf_eid);
          if (result != GXF_SUCCESS) {
            HOLOSCAN_LOG_ERROR("GxfEntityEventNotify failed: {}", GxfResultStr(result));
            throw std::runtime_error(
                fmt::format("Failed to notify event update, GXF error: {}", GxfResultStr(result)));
          }
        }
      }
    }
  }
};

PresentDoneCondition::PresentDoneCondition(std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op)
    : impl_(std::make_shared<Impl>()) {
  impl_->holoviz_op_ = std::move(holoviz_op);
}

PresentDoneCondition::~PresentDoneCondition() {
  impl_->stop_requested_ = true;
  if (impl_->thread_.joinable()) {
    impl_->thread_.join();
  }
}

void PresentDoneCondition::initialize() {
  Condition::initialize();

  auto gxf_context = fragment()->executor().context();
  auto gxf_eid = holoscan::gxf::get_component_eid(gxf_context, wrapper_cid());
  impl_->thread_ =
      std::thread(&PresentDoneCondition::Impl::thread_func, impl_.get(), gxf_context, gxf_eid);
}

void PresentDoneCondition::check(int64_t timestamp, SchedulingStatusType* type,
                                 int64_t* target_timestamp) const {
  *type = impl_->current_state_.load();
  *target_timestamp = timestamp;
}

void PresentDoneCondition::on_execute([[maybe_unused]] int64_t timestamp) {
  impl_->current_state_.store(SchedulingStatusType::kWaitEvent);
}

}  // namespace holoscan
