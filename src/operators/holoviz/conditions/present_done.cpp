/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/holoviz/conditions/present_done.hpp"

#include <atomic>
#include <memory>
#include <thread>

#include "holoscan/operators/holoviz/holoviz.hpp"

namespace holoscan {

struct PresentDoneCondition::Impl {
  std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op_;

  std::thread thread_;
  std::atomic<bool> stop_requested_ = false;

  std::atomic<SchedulingStatusType> current_state_{SchedulingStatusType::kReady};

  uint64_t present_id_ = 0;

  void thread_func(gxf_context_t gxf_context, gxf_uid_t gxf_eid) {
    while (!stop_requested_) {
      // choose the timeout to be longer than the display period (assume 10Hz)
      const bool is_ready = holoviz_op_->wait_for_present(present_id_, 100'000'000);

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
  impl_->holoviz_op_ = holoviz_op;
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
