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

#include <gtest/gtest.h>

#include <string>

#include <gxf/std/scheduling_terms.hpp>
#include <gxf/std/unbounded_allocator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Applications
///////////////////////////////////////////////////////////////////////////////

class CheckResourceConditionOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CheckResourceConditionOp)

  CheckResourceConditionOp() = default;

  void setup(OperatorSpec& spec) override {}

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto count_condition = condition<CountCondition>("count");
    auto pool = resource<UnboundedAllocator>("pool");

    if (count_condition == nullptr) {
      HOLOSCAN_LOG_ERROR("null count condition");
    } else {
      auto count = count_condition->count();
      // Note: this count method always returns the initial count
      HOLOSCAN_LOG_INFO("{}: count = {}", name(), count);

      // can get a typed pointer to the underlying GXF class
      auto gxf_count_condition_ptr = count_condition->get();
      if (gxf_count_condition_ptr == nullptr) {
        HOLOSCAN_LOG_ERROR("unexpected null nvidia::gxf::CountCondition*");
      } else {
        if (dynamic_cast<nvidia::gxf::CountSchedulingTerm*>(gxf_count_condition_ptr) == nullptr) {
          HOLOSCAN_LOG_ERROR("typid mismatch for gxf_count_condition_ptr (found {}, expected {})",
                             typeid(gxf_count_condition_ptr).name(),
                             typeid(nvidia::gxf::CountSchedulingTerm*).name());
        }
      }

      // can get a void* pointer to the underlying GXF class
      if (count_condition->gxf_cptr() == nullptr) {
        HOLOSCAN_LOG_ERROR("null get_cptr() for count_condition");
      }
    }

    if (pool == nullptr) {
      HOLOSCAN_LOG_ERROR("null pool resource");
    } else {
      bool memory_available = pool->is_available(1000);
      HOLOSCAN_LOG_INFO(
          "{}: system {} 1000 bytes available", name(), memory_available ? "has" : "does not have");

      // can get a typed pointer to the underlying GXF class
      auto gxf_unbounded_allocator_ptr = pool->get();
      if (gxf_unbounded_allocator_ptr == nullptr) {
        HOLOSCAN_LOG_ERROR("null nvidia::gxf::CountCondition*");
      } else {
        if (dynamic_cast<nvidia::gxf::UnboundedAllocator*>(gxf_unbounded_allocator_ptr) ==
            nullptr) {
          HOLOSCAN_LOG_ERROR(
              "typid mismatch for gxf_unbounded_allocator_ptr (found {}, expected {})",
              typeid(gxf_unbounded_allocator_ptr).name(),
              typeid(nvidia::gxf::UnboundedAllocator*).name());
        }
      }

      // can get a void* pointer to the underlying GXF class
      if (pool->gxf_cptr() == nullptr) { HOLOSCAN_LOG_ERROR("null get_cptr() for pool"); }
    }
  }
};

class ConditionResourceTestApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    // Create the resource and condition with names and types matching the ones assumed in
    // `CheckResourceConditionOp::compute`.
    auto pool = make_resource<UnboundedAllocator>("pool");
    auto count = make_condition<CountCondition>("count", 10);

    auto test_op = make_operator<CheckResourceConditionOp>("my_op", pool, count);
    add_operator(test_op);
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(OperatorComponentAccess, ConditionsAndResourcesFromComputeApp) {
  auto app = make_application<ConditionResourceTestApp>();

  // Capture stderr output to check for specific error messages
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  bool found = log_output.find("my_op: system has 1000 bytes available") != std::string::npos;
  ASSERT_TRUE(found) << "Expected to find message 'system has 1000 bytes available'"
                     << "\n=== LOG ===\n"
                     << log_output << "\n===========\n";
  found = log_output.find("my_op: count = 10") != std::string::npos;
  ASSERT_TRUE(found) << "Expected to find 'count = 10' in the output"
                     << "\n=== LOG ===\n"
                     << log_output << "\n===========\n";
  bool gxf_condition_found =
      log_output.find("null nvidia::gxf::CountCondition") == std::string::npos;
  ASSERT_TRUE(gxf_condition_found) << "Cast of condition to nvidia::gxf::CountCondition* failed"
                                   << "\n=== LOG ===\n"
                                   << log_output << "\n===========\n";
  bool gxf_resource_found =
      log_output.find("null nvidia::gxf::UnboundedAllocator") == std::string::npos;
  ASSERT_TRUE(gxf_resource_found) << "Cast of resource to nvidia::gxf::UnboundedAllocator* failed"
                                  << "\n=== LOG ===\n"
                                  << log_output << "\n===========\n";
  gxf_condition_found = log_output.find("null get_cptr() for count_condition") == std::string::npos;
  ASSERT_TRUE(gxf_condition_found) << "null gxf_cptr() returned by CountCondition"
                                   << "\n=== LOG ===\n"
                                   << log_output << "\n===========\n";
  gxf_resource_found = log_output.find("null get_cptr() for pool") == std::string::npos;
  ASSERT_TRUE(gxf_resource_found) << "null gxf_cptr() returned by UnboundedAllocator"
                                  << "\n=== LOG ===\n"
                                  << log_output << "\n===========\n";

  bool type_match =
      log_output.find("typid mismatch for gxf_count_condition_ptr") == std::string::npos;
  ASSERT_TRUE(type_match) << "unexpected typid from CountCondition::get()"
                          << "\n=== LOG ===\n"
                          << log_output << "\n===========\n";
  type_match =
      log_output.find("typid mismatch for gxf_unbounded_allocator_ptr") == std::string::npos;
  ASSERT_TRUE(type_match) << "unexpected typid from UnboundedAllocator::get()"
                          << "\n=== LOG ===\n"
                          << log_output << "\n===========\n";
}

}  // namespace holoscan
