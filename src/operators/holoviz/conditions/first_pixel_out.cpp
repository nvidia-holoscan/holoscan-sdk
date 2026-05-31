/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <holoscan/operators/holoviz/conditions/first_pixel_out.hpp>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <holoscan/core/fragment.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace holoscan {

struct FirstPixelOutCondition::Impl {
  std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op_;

  std::thread thread_;
  std::atomic<bool> stop_requested_ = false;

  std::atomic<SchedulingStatusType> current_state_{SchedulingStatusType::kReady};

  void thread_func(gxf_context_t gxf_context, gxf_uid_t gxf_eid) {
    while (!stop_requested_) {
      bool is_ready = false;
      try {
        // choose the timeout to be longer than the display period (assume 10Hz)
        is_ready = holoviz_op_->wait_for_first_pixel_out(100'000'000);
      } catch (const std::runtime_error& e) {
        // wait_for_first_pixel_out throws when the underlying VK_EXT_display_control
        // extension is unavailable or broken at runtime (e.g. AGX Thor Jedha, where
        // the extension is advertised but registerDisplayEventEXT returns
        // VK_ERROR_UNKNOWN). Without this catch the exception would escape this
        // std::thread callable and trigger std::terminate(), crashing the
        // application. Instead, transition the condition to kNever so the owning
        // operator stops ticking, then ask the fragment to wind down all operators
        // gracefully. The application exits cleanly with a clear ERROR log
        // identifying the missing extension and the PresentDoneCondition alternative.
        HOLOSCAN_LOG_ERROR(
            "FirstPixelOutCondition: {}. Stopping fragment execution. Use PresentDoneCondition "
            "instead, or run on a platform where the `VK_EXT_display_control` Vulkan device "
            "extension is fully functional.",
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

FirstPixelOutCondition::FirstPixelOutCondition(std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op)
    : impl_(std::make_shared<Impl>()) {
  impl_->holoviz_op_ = std::move(holoviz_op);
}

FirstPixelOutCondition::~FirstPixelOutCondition() {
  impl_->stop_requested_ = true;
  if (impl_->thread_.joinable()) {
    impl_->thread_.join();
  }
}

void FirstPixelOutCondition::initialize() {
  Condition::initialize();

  auto gxf_context = fragment()->executor().context();
  auto gxf_eid = holoscan::gxf::get_component_eid(gxf_context, wrapper_cid());
  impl_->thread_ =
      std::thread(&FirstPixelOutCondition::Impl::thread_func, impl_.get(), gxf_context, gxf_eid);
}

void FirstPixelOutCondition::check(int64_t timestamp, SchedulingStatusType* type,
                                   int64_t* target_timestamp) const {
  *type = impl_->current_state_.load();
  *target_timestamp = timestamp;
}

void FirstPixelOutCondition::on_execute([[maybe_unused]] int64_t timestamp) {
  impl_->current_state_.store(SchedulingStatusType::kWaitEvent);
}

}  // namespace holoscan
