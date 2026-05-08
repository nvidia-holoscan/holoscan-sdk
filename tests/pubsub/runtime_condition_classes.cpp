/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/fragment.hpp>
#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
#include <holoscan/pubsub/common/native_buffer_protocol_adapter.hpp>
#include <holoscan/pubsub/runtime/conditions/pending_export_condition.hpp>
#endif
#include <holoscan/pubsub/runtime/conditions/publisher_available.hpp>
#include <holoscan/pubsub/runtime/conditions/subscriber_available.hpp>

namespace holoscan {

#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
TEST(PubSubRuntimeConditionClasses, TestPendingExportCondition) {
  Fragment F;
  const std::string name{"pending-export-condition"};
  auto condition =
      F.make_condition<PendingExportCondition>(name, Arg{"max_pending", static_cast<uint64_t>(2)});
  EXPECT_EQ(condition->name(), name);
  EXPECT_NE(std::dynamic_pointer_cast<PendingExportCondition>(condition), nullptr);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(PubSubRuntimeConditionClasses, TestPendingExportConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<PendingExportCondition>();
  EXPECT_NE(condition, nullptr);
}

// Minimal fake adapter for testing the condition state machine.
class FakeNativeBufferAdapter : public NativeBufferProtocolAdapter {
 public:
  bool is_initialized() const override { return true; }
  const std::string& default_protocol_name() const override {
    static const std::string name = "fake";
    return name;
  }
  bool supports_protocol(const std::string&) const override { return false; }
  bool can_export_tensor(const nvidia::gxf::Tensor&) const override { return false; }
  uint8_t descriptor_format_version() const override { return 1; }
  nvidia::gxf::Expected<std::vector<uint8_t>> export_tensor(void*, const std::shared_ptr<void>&,
                                                            const NativeTensorMetadata&,
                                                            const std::string&, uint32_t) override {
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }
  nvidia::gxf::Expected<ImportedNativeTensor> import_tensor_generic(
      const std::vector<uint8_t>&, const std::string&, std::chrono::milliseconds) override {
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }
  size_t pending_export_count() const override { return pending_count_; }
  void set_on_pending_export_count_changed(PendingExportCountChangedCallback cb) override {
    callback_ = std::move(cb);
  }

  void set_pending_count(size_t count) { pending_count_ = count; }

  // Simulate an export release by updating the count and invoking the callback.
  void simulate_release(size_t new_count) {
    pending_count_ = new_count;
    if (callback_) {
      callback_(new_count);
    }
  }

 private:
  size_t pending_count_ = 0;
  PendingExportCountChangedCallback callback_;
};

TEST(PubSubRuntimeConditionClasses, TestPendingExportConditionStateMachine) {
  Fragment F;
  auto condition = F.make_condition<PendingExportCondition>(
      "pending_export_test", Arg{"max_pending", static_cast<uint64_t>(2)});
  ASSERT_NE(condition, nullptr);
  auto* cond = dynamic_cast<PendingExportCondition*>(condition.get());
  ASSERT_NE(cond, nullptr);

  auto adapter = std::make_shared<FakeNativeBufferAdapter>();
  cond->adapter(adapter);

  // Initialize the condition (sets up the callback on the adapter).
  cond->initialize();

  // Initially pending_count=0, max_pending=2 -> should be kReady.
  SchedulingStatusType status;
  int64_t target_ts = 0;
  cond->check(1000, &status, &target_ts);
  EXPECT_EQ(status, SchedulingStatusType::kReady);

  // Simulate pending count reaching max_pending (2) via update_state.
  adapter->set_pending_count(2);
  cond->update_state(2000);
  cond->check(2000, &status, &target_ts);
  EXPECT_EQ(status, SchedulingStatusType::kWaitEvent);

  // Simulate pending count exceeding max_pending -> still waiting.
  adapter->set_pending_count(3);
  cond->update_state(3000);
  cond->check(3000, &status, &target_ts);
  EXPECT_EQ(status, SchedulingStatusType::kWaitEvent);

  // Simulate a release that brings count below max_pending via the callback.
  adapter->simulate_release(1);
  cond->check(4000, &status, &target_ts);
  EXPECT_EQ(status, SchedulingStatusType::kReady);
}
#endif  // HOLOSCAN_HAS_PENDING_EXPORT_CONDITION

TEST(PubSubRuntimeConditionClasses, TestSubscriberAvailableCondition) {
  Fragment F;
  const std::string name{"subscriber-available-condition"};
  auto condition = F.make_condition<SubscriberAvailableCondition>(
      name, Arg{"min_subscriber_count", static_cast<uint64_t>(2)});
  EXPECT_EQ(condition->name(), name);
  EXPECT_NE(std::dynamic_pointer_cast<SubscriberAvailableCondition>(condition), nullptr);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(PubSubRuntimeConditionClasses, TestSubscriberAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<SubscriberAvailableCondition>();
  EXPECT_NE(condition, nullptr);
}

TEST(PubSubRuntimeConditionClasses, TestPublisherAvailableCondition) {
  Fragment F;
  const std::string name{"publisher-available-condition"};
  auto condition = F.make_condition<PublisherAvailableCondition>(
      name, Arg{"min_publisher_count", static_cast<uint64_t>(2)});
  EXPECT_EQ(condition->name(), name);
  EXPECT_NE(std::dynamic_pointer_cast<PublisherAvailableCondition>(condition), nullptr);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(PubSubRuntimeConditionClasses, TestPublisherAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<PublisherAvailableCondition>();
  EXPECT_NE(condition, nullptr);
}

TEST(PubSubRuntimeConditionClasses, TestRuntimeConditionUniqueDefaultNames) {
  Fragment F;

#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
  auto pending_export_cond = F.make_condition<PendingExportCondition>();
#endif
  auto publisher_available_cond = F.make_condition<PublisherAvailableCondition>();
  auto subscriber_available_cond = F.make_condition<SubscriberAvailableCondition>();

#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION
  EXPECT_EQ(pending_export_cond->name(), "pending_export_condition");
#endif
  EXPECT_EQ(publisher_available_cond->name(), "publisher_available_condition");
  EXPECT_EQ(subscriber_available_cond->name(), "subscriber_available_condition");
}

}  // namespace holoscan
