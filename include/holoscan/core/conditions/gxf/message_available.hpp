/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_MESSAGE_AVAILABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_MESSAGE_AVAILABLE_HPP

#include <memory>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"
#include "../../resources/gxf/receiver.hpp"

namespace holoscan {

/**
 * @brief Condition class that allows an operator to execute only when one or more messages are
 * available on a given input port.
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * This condition can also be set via the `Operator::setup` method using `IOSpec::condition` with
 * `ConditionType::kMessageAvailable`. In that case, the receiver is already known from the port
 * corresponding to the `IOSpec` object, so the "receiver" argument is unnecessary.
 *
 * ==Parameters==
 *
 * - **min_size** (uint64_t): The minimum number of messages that must be available on the input
 * port before the operator will be considered READY.
 * - **front_stage_max_size** (size_t): If set, the condition will only allow execution if the
 * number of messages in the front stage of the receiver's double-buffer queue does not exceed this
 * count. In most cases, this parameter does not need to be set.
 * - **receiver** (std::string): The receiver whose message queue will be checked. This should be
 * specified by the name of the Operator's input port the condition will apply to. The Holoscan SDK
 * will then automatically replace the port name with the actual receiver object at application run
 * time.
 */
class MessageAvailableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(MessageAvailableCondition, GXFCondition)
  MessageAvailableCondition() = default;
  explicit MessageAvailableCondition(size_t min_size) : min_size_(min_size) {}
  MessageAvailableCondition(size_t min_size, size_t front_stage_max_size)
      : min_size_(min_size), front_stage_max_size_(front_stage_max_size) {}

  const char* gxf_typename() const override {
    return "nvidia::gxf::MessageAvailableSchedulingTerm";
  }

  void receiver(std::shared_ptr<Receiver> receiver) { receiver_ = receiver; }
  std::shared_ptr<Receiver> receiver() { return receiver_.get(); }

  void min_size(uint64_t min_size);
  uint64_t min_size() { return min_size_; }

  void front_stage_max_size(size_t front_stage_max_size);
  size_t front_stage_max_size() { return front_stage_max_size_; }

  void setup(ComponentSpec& spec) override;

  void initialize() override { GXFCondition::initialize(); }

  nvidia::gxf::MessageAvailableSchedulingTerm* get() const;

  // TODO(GXF4):   Expected<void> setReceiver(Handle<Receiver> value)

 private:
  Parameter<std::shared_ptr<Receiver>> receiver_;
  Parameter<uint64_t> min_size_;
  Parameter<size_t> front_stage_max_size_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_MESSAGE_AVAILABLE_HPP */
