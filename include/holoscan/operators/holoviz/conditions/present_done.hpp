/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_CONDITIONS_PRESENT_DONE_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_CONDITIONS_PRESENT_DONE_HPP

#include <memory>

#include <holoscan/core/condition.hpp>

namespace holoscan {

namespace ops {
class HolovizOp;
}  // namespace ops

class PresentDoneCondition : public Condition {
 public:
  explicit PresentDoneCondition(std::shared_ptr<holoscan::ops::HolovizOp> holoviz_op);
  PresentDoneCondition() = delete;

  ~PresentDoneCondition();

  void initialize() override;
  void check(int64_t timestamp, SchedulingStatusType* type,
             int64_t* target_timestamp) const override;
  void on_execute(int64_t timestamp) override;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_OPERATORS_HOLOVIZ_CONDITIONS_PRESENT_DONE_HPP
