/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"

#include "../config.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

// create a GXF codelet-based operator
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(GXFForwardCodeletOp, "nvidia::gxf::Forward")

class PingTxMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxMetadataOp)

  PingTxMetadataOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    if (is_metadata_enabled()) {
      auto dynamic_metadata = metadata();
      if (num_keys_ == 1) {
        dynamic_metadata->set("PingTxMetadataOp.label", "my title"s);
      } else {
        for (auto ki = 0; ki < num_keys_; ki++) {
          dynamic_metadata->set(fmt::format("key{:09d}", ki), count_);
        }
      }
      HOLOSCAN_LOG_INFO("tx metadata()->size() = {}", metadata()->size());
    }
    HOLOSCAN_LOG_INFO("tx count = {}", count_);
    op_output.emit(count_, "out");
    count_++;
  }

  void set_num_keys(int num_keys) { num_keys_ = num_keys; }

 private:
  int count_ = 1;
  int num_keys_ = 1;
};

class ForwardOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardOp)

  ForwardOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int>("in").value();
    if (is_metadata_enabled()) {
      HOLOSCAN_LOG_INFO("fwd metadata()->size() = {}", metadata()->size());
    }
    op_output.emit(value, "out");
  }
};

class ForwardAddMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardAddMetadataOp)

  ForwardAddMetadataOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }
  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int>("in").value();
    if (is_metadata_enabled()) {
      auto meta = metadata();
      meta->set("ForwardAddMetadataOp.date", "2024-07-16"s);
      HOLOSCAN_LOG_INFO("fwd_with_meta metadata()->size() = {}", metadata()->size());
    }
    op_output.emit(value, "out");
  }
};

class ForwardAddMetadataOp2 : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardAddMetadataOp2)

  ForwardAddMetadataOp2() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }
  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int>("in").value();
    if (is_metadata_enabled()) {
      auto meta = metadata();
      meta->set("ForwardAddMetadataOp2.value", value);
      HOLOSCAN_LOG_INFO(
          "fwd_with_meta2 metadata()->size() = {}, value={}", metadata()->size(), value);
    }
    op_output.emit(value, "out");
  }
};

class PingThreeRxMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingThreeRxMetadataOp)

  PingThreeRxMetadataOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in1");
    spec.input<int>("in2");
    spec.input<int>("in3");
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value1 = op_input.receive<int>("in1").value();
    auto value2 = op_input.receive<int>("in2").value();
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto value3 = op_input.receive<int>("in3").value();
    auto meta = metadata();
    HOLOSCAN_LOG_INFO("rx metadata has {} keys", meta->size());
    HOLOSCAN_LOG_INFO("value1 {} value2", (value1 == value2) ? "==" : "!=");
    if (is_metadata_enabled()) {
      HOLOSCAN_LOG_INFO("tx label: {}", meta->get<std::string>("PingTxMetadataOp.label"));
      HOLOSCAN_LOG_INFO("fwd date: {}", meta->get<std::string>("ForwardAddMetadataOp.date"));
      HOLOSCAN_LOG_INFO("fwd2 value: {}", meta->get<int>("ForwardAddMetadataOp2.value"));
    }
  }

 private:
  int count_ = 1;
};

class PingSingleRxMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingSingleRxMetadataOp)

  PingSingleRxMetadataOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<int>("in1"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto value1 = op_input.receive<int>("in1").value();
    auto meta = metadata();
    HOLOSCAN_LOG_INFO("{} metadata has {} keys", name(), meta->size());
    if (is_metadata_enabled()) {
      HOLOSCAN_LOG_INFO("tx label: {}", meta->get<std::string>("PingTxMetadataOp.label"));
    }
  }

 private:
  int count_ = 1;
};

// This operator sends metadata and a TensorMap
class PingTxTensorMapMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxTensorMapMetadataOp)

  PingTxTensorMapMetadataOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out");
    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  }

  void initialize() override {
    // Set up prerequisite parameters before calling Operator::initialize()
    auto frag = fragment();

    // Find if there is an argument for 'allocator'
    auto has_allocator = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
    // Create the allocator if there is no argument provided.
    if (has_allocator == args().end()) {
      allocator_ = frag->make_resource<UnboundedAllocator>("allocator");
      add_arg(allocator_.get());
    }
    Operator::initialize();
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    if (is_metadata_enabled()) {
      auto dynamic_metadata = metadata();
      if (num_keys_ == 1) {
        dynamic_metadata->set("PingTxTensorMapMetadataOp.label", "my title"s);
      } else {
        for (auto ki = 0; ki < num_keys_; ki++) {
          dynamic_metadata->set(fmt::format("key{:09d}", ki), count_);
        }
      }
      HOLOSCAN_LOG_INFO("tx metadata()->size() = {}", metadata()->size());
    }
    HOLOSCAN_LOG_INFO("tensormap tx count = {}", count_);

    auto gxf_context = context.context();

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());

    // Create a pair of GXF Tensors
    auto gxf_tensor1 = std::make_shared<nvidia::gxf::Tensor>();
    auto gxf_tensor2 = std::make_shared<nvidia::gxf::Tensor>();
    auto dtype = nvidia::gxf::PrimitiveType::kUnsigned8;
    nvidia::gxf::Shape tensor_shape{{100, 100}};
    const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(dtype);
    auto strides = nvidia::gxf::ComputeTrivialStrides(tensor_shape, bytes_per_element);
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kSystem;

    // allocate two tensors of the specified shape and data type
    auto result = gxf_tensor1->reshapeCustom(
        tensor_shape, dtype, bytes_per_element, strides, storage_type, allocator.value());
    if (!result) { HOLOSCAN_LOG_ERROR("failed to generate tensor1"); }
    result = gxf_tensor2->reshapeCustom(
        tensor_shape, dtype, bytes_per_element, strides, storage_type, allocator.value());
    if (!result) { HOLOSCAN_LOG_ERROR("failed to generate tensor2"); }

    // Create Holoscan tensors from the GXF tensors
    auto maybe_dl_ctx1 = (*gxf_tensor1).toDLManagedTensorContext();
    if (!maybe_dl_ctx1) {
      HOLOSCAN_LOG_ERROR(
          "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
    }
    std::shared_ptr<Tensor> holoscan_tensor1 = std::make_shared<Tensor>(maybe_dl_ctx1.value());
    auto maybe_dl_ctx2 = (*gxf_tensor2).toDLManagedTensorContext();
    if (!maybe_dl_ctx2) {
      HOLOSCAN_LOG_ERROR(
          "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
    }
    std::shared_ptr<Tensor> holoscan_tensor2 = std::make_shared<Tensor>(maybe_dl_ctx2.value());

    // populate TensorMap output with the Holoscan Tensors
    holoscan::TensorMap out_tensors;
    out_tensors.insert({"tensor1"s, holoscan_tensor1});
    out_tensors.insert({"tensor2"s, holoscan_tensor2});
    op_output.emit(out_tensors, "out");
    count_++;
  }

  void set_num_keys(int num_keys) { num_keys_ = num_keys; }

 private:
  int count_ = 1;
  int num_keys_ = 1;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

// This operator receives metadata and a TensorMap
class PingSingleRxTensorMapMetadataOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingSingleRxTensorMapMetadataOp)

  PingSingleRxTensorMapMetadataOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<holoscan::TensorMap>("in1"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value1 = op_input.receive<holoscan::TensorMap>("in1").value();
    auto meta = metadata();
    HOLOSCAN_LOG_INFO("{} metadata has {} keys", name(), meta->size());
    if (is_metadata_enabled()) {
      HOLOSCAN_LOG_INFO("tx label: {}", meta->get<std::string>("PingTxTensorMapMetadataOp.label"));
    }
  }

 private:
  int count_ = 1;
};

}  // namespace

/* Metadata test app that broadcasts to 3 different forwarding ops, some of which add
 * additional metadata. The 3 parallel paths then converge to a single receiver.
 */
class OperatorMetadataMergeApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxMetadataOp>("tx", make_condition<CountCondition>(count_));
    auto fwd = make_operator<ForwardOp>("fwd_no_meta");
    auto fwd_with_meta = make_operator<ForwardAddMetadataOp>("fwd_with_meta");
    auto fwd_with_meta2 = make_operator<ForwardAddMetadataOp2>("fwd_with_meta2");
    auto rx = make_operator<PingThreeRxMetadataOp>("rx");

    tx->metadata_policy(MetadataPolicy::kRaise);
    fwd->metadata_policy(MetadataPolicy::kRaise);
    fwd_with_meta->metadata_policy(MetadataPolicy::kRaise);
    fwd_with_meta->metadata_policy(MetadataPolicy::kRaise);
    // need kReject or kUpdate since multiple ports will have the same tx metadata
    rx->metadata_policy(rx_metadata_policy_);

    add_flow(tx, fwd);
    add_flow(tx, fwd_with_meta);
    add_flow(tx, fwd_with_meta2);
    add_flow(fwd, rx, {{"out", "in1"}});
    add_flow(fwd_with_meta, rx, {{"out", "in2"}});
    add_flow(fwd_with_meta2, rx, {{"out", "in3"}});
  }

  void set_count(int count) { count_ = count; }
  void set_rx_policy(MetadataPolicy policy) { rx_metadata_policy_ = policy; }

 private:
  int count_ = 10;
  MetadataPolicy rx_metadata_policy_ = MetadataPolicy::kUpdate;
};

/// Simple app broadcasting metadata from one tx op to 4 rx ops
class OperatorMetadataBroadcastApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxMetadataOp>("tx", make_condition<CountCondition>(count_));
    auto rx1 = make_operator<PingSingleRxMetadataOp>("rx1");
    auto rx2 = make_operator<PingSingleRxMetadataOp>("rx2");
    auto rx3 = make_operator<PingSingleRxMetadataOp>("rx3");
    auto rx4 = make_operator<PingSingleRxMetadataOp>("rx4");

    add_flow(tx, rx1);
    add_flow(tx, rx2);
    add_flow(tx, rx3);
    add_flow(tx, rx4);
  }

  void set_count(int count) { count_ = count; }

 private:
  int count_ = 10;
};

/// Simple tx -> gxf_fwd -> rx app, where gxf_fwd is a GXF Forewrd Codelet
class GXFOperatorMetadataApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxMetadataOp>("tx", make_condition<CountCondition>(count_));
    auto gxf_fwd = make_operator<GXFForwardCodeletOp>("gxf_fwd");
    auto rx = make_operator<PingSingleRxMetadataOp>("rx");

    add_flow(tx, gxf_fwd);
    add_flow(gxf_fwd, rx);
  }

  void set_count(int count) { count_ = count; }

 private:
  int count_ = 10;
};

/// Simple tx -> fwd -> rx app, but with a large metedata dictionary
class OperatorLargeMetadataApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxMetadataOp>("tx", make_condition<CountCondition>(count_));
    tx->set_num_keys(num_keys_);
    auto fwd = make_operator<ForwardOp>("fwd_no_meta");
    auto rx = make_operator<holoscan::ops::PingRxOp>("rx");

    add_flow(tx, fwd);
    add_flow(fwd, rx);
  }

  void set_count(int count) { count_ = count; }
  void set_num_keys(int num_keys) { num_keys_ = num_keys; }

 private:
  int count_ = 10;
  int num_keys_ = 1000;
};

/// Simple tx -> rx app, where the data being sent is a TensorMap
class TensorMapOperatorsMetadataApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx =
        make_operator<PingTxTensorMapMetadataOp>("tx", make_condition<CountCondition>(count_));
    auto rx = make_operator<PingSingleRxTensorMapMetadataOp>("rx");

    add_flow(tx, rx);
  }

  void set_count(int count) { count_ = count; }

 private:
  int count_ = 10;
};

TEST(OperatorMetadataApps, TestOperatorMetadataMergeApp) {
  auto app = make_application<OperatorMetadataMergeApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("rx metadata has 3 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("tx label: my title") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("fwd date: 2024-07-16") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find(fmt::format("fwd2 value: {}", count)) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("value1 == value2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(OperatorMetadataApps, TestOperatorMetadataMergeAppPolicyRaise) {
  auto app = make_application<OperatorMetadataMergeApp>();
  int count = 3;
  app->set_count(count);
  app->set_rx_policy(MetadataPolicy::kRaise);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Key 'PingTxMetadataOp.label' already exists") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(OperatorMetadataApps, TestOperatorMetadataMergeAppTrackingDisabled) {
  auto app = make_application<OperatorMetadataMergeApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(false);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("rx metadata has 0 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("value1 == value2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

// This app verifies that metadata passes through GXF Codelet-based operators as expected
TEST(OperatorMetadataApps, TestGXFOperatorMetadataBroadcastApp) {
  auto app = make_application<OperatorMetadataBroadcastApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("rx1 metadata has 1 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("rx2 metadata has 1 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("rx3 metadata has 1 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("rx4 metadata has 1 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

// This app verifies that metadata passes through GXF Codelet-based operators as expected
TEST(OperatorMetadataApps, TestGXFOperatorMetadataApp) {
  auto app = make_application<GXFOperatorMetadataApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("rx metadata has 1 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("tx label: my title") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(OperatorMetadataApps, TestGXFOperatorMetadataAppTrackingDisabled) {
  auto app = make_application<GXFOperatorMetadataApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(false);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("rx metadata has 0 keys") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

// This app tests case with many objects in the metadata dictionary
TEST(OperatorMetadataApps, TestOperatorLargeMetadataApp) {
  auto app = make_application<OperatorLargeMetadataApp>();
  app->set_num_keys(1000);
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

// This app tests sending metadata in an entity alongside a TensorMap object
TEST(OperatorMetadataApps, TestTensorMapOperatorsMetadataApp) {
  auto app = make_application<TensorMapOperatorsMetadataApp>();
  int count = 3;
  app->set_count(count);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

}  // namespace holoscan
