// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone binary for cross-process CUDA IPC integration testing.
//
// Usage:
//   native_buffer_ipc_app --role publisher --count 5
//   native_buffer_ipc_app --role subscriber --count 5
//
// The publisher allocates GPU tensors filled with a deterministic pattern and
// emits them over DDS pub/sub.  The subscriber receives tensors, copies them
// back to host, and verifies the pattern.  Exit code 0 = success.
//
// Designed to be launched as two separate processes by a test script, avoiding
// the fork() issues that plague single-process multi-role tests.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gxf/std/tensor.hpp>

#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>
#include <holoscan/holoscan.hpp>
#include "holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp"

namespace {

constexpr const char* kTopicName = "native_buffer_ipc_test";
constexpr int kTensorRows = 4;
constexpr int kTensorCols = 8;
constexpr int kTensorElements = kTensorRows * kTensorCols;

float expected_value(int element_index) {
  return static_cast<float>(element_index + 1) * 1.5f;
}

// ---------------------------------------------------------------------------
// Publisher operator
// ---------------------------------------------------------------------------

class IpcTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IpcTxOp)
  IpcTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out").topic(kTopicName);
    spec.param(allocator_, "allocator", "Allocator", "GPU allocator");
    spec.param(target_count_, "target_count", "Count", "Messages to send", static_cast<int64_t>(5));
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override {
    resolve_tx();

    if (!pubsub_tx_ || !pubsub_tx_->has_matched_subscribers()) {
      ++wait_ticks_;
      if (wait_ticks_ % 50 == 1) {
        HOLOSCAN_LOG_INFO("tx: waiting for subscriber (tick={})", wait_ticks_);
      }
      return;
    }

    if (!logged_match_) {
      HOLOSCAN_LOG_INFO("tx: subscriber matched (count={})",
                        pubsub_tx_->matched_subscriber_count());
      logged_match_ = true;
      match_time_ = std::chrono::steady_clock::now();
    }

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - match_time_)
                          .count();
    if (elapsed_ms < kStabilizationMs)
      return;

    if (sent_ >= target_count_.get()) {
      if (++drain_ticks_ >= kDrainTicks)
        fragment()->stop_execution();
      return;
    }

    auto gxf_ctx = context.context();
    auto alloc =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_ctx, allocator_->gxf_cid());
    if (!alloc) {
      HOLOSCAN_LOG_ERROR("tx: failed to get allocator");
      return;
    }

    auto tensor = std::make_shared<nvidia::gxf::Tensor>();
    nvidia::gxf::Shape shape({kTensorRows, kTensorCols});
    auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
    auto rc = tensor->reshapeCustom(shape,
                                    nvidia::gxf::PrimitiveType::kFloat32,
                                    sizeof(float),
                                    strides,
                                    nvidia::gxf::MemoryStorageType::kDevice,
                                    alloc.value());
    if (!rc) {
      HOLOSCAN_LOG_ERROR("tx: reshapeCustom failed");
      return;
    }

    std::vector<float> host(kTensorElements);
    for (int i = 0; i < kTensorElements; ++i)
      host[i] = expected_value(i);
    auto err = cudaMemcpy(
        tensor->pointer(), host.data(), kTensorElements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("tx: cudaMemcpy H2D failed: {}", cudaGetErrorString(err));
      return;
    }

    auto dl = tensor->toDLManagedTensorContext();
    if (!dl) {
      HOLOSCAN_LOG_ERROR("tx: toDLManagedTensorContext failed");
      return;
    }
    auto* mb = static_cast<nvidia::gxf::MemoryBuffer*>(dl.value()->memory_ref.get());
    auto ht = std::make_shared<holoscan::Tensor>(dl.value(), mb);

    holoscan::TensorMap msg;
    msg.insert({"tensor", ht});
    op_output.emit(msg, "out");

    ++sent_;
    HOLOSCAN_LOG_INFO(
        "tx: sent {}/{} ({}x{} device)", sent_, target_count_.get(), kTensorRows, kTensorCols);
  }

 private:
  static constexpr int64_t kDrainTicks = 2;
  static constexpr int64_t kStabilizationMs = 1500;

  void resolve_tx() {
    if (pubsub_tx_)
      return;
    auto* s = spec();
    if (!s)
      return;
    auto it = s->outputs().find("out");
    if (it == s->outputs().end())
      return;
    pubsub_tx_ = std::dynamic_pointer_cast<holoscan::PubSubTransmitter>(it->second->connector());
  }

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  holoscan::Parameter<int64_t> target_count_;
  std::shared_ptr<holoscan::PubSubTransmitter> pubsub_tx_;
  int64_t sent_ = 0;
  int64_t wait_ticks_ = 0;
  int64_t drain_ticks_ = 0;
  bool logged_match_ = false;
  std::chrono::steady_clock::time_point match_time_;
};

// ---------------------------------------------------------------------------
// Subscriber operator — verifies data, sets exit code
// ---------------------------------------------------------------------------

class IpcRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IpcRxOp)
  IpcRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in").topic(kTopicName);
    spec.param(
        target_count_, "target_count", "Count", "Messages to receive", static_cast<int64_t>(5));
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe = op_input.receive<holoscan::TensorMap>("in");
    if (!maybe)
      return;

    auto tensor_map = maybe.value();
    if (tensor_map.empty()) {
      HOLOSCAN_LOG_ERROR("rx: received empty TensorMap");
      ++received_;
      ++mismatches_;
      return;
    }

    auto tensor_it = tensor_map.find("tensor");
    if (tensor_it == tensor_map.end()) {
      tensor_it = tensor_map.begin();
    }

    auto tensor = tensor_it->second;
    if (!tensor || !tensor->data()) {
      HOLOSCAN_LOG_ERROR("rx: null tensor or data pointer");
      ++received_;
      ++mismatches_;
      return;
    }

    int ndim = tensor->ndim();
    if (ndim != 2 || tensor->shape()[0] != kTensorRows || tensor->shape()[1] != kTensorCols) {
      if (ndim < 2) {
        HOLOSCAN_LOG_ERROR("rx: shape mismatch (rank={}, expected 2)", ndim);
      } else {
        HOLOSCAN_LOG_ERROR("rx: shape mismatch (rank={} dims={}x{}, expected {}x{})",
                           ndim,
                           tensor->shape()[0],
                           tensor->shape()[1],
                           kTensorRows,
                           kTensorCols);
      }
      ++received_;
      ++mismatches_;
      return;
    }

    std::vector<float> buf(kTensorElements);
    auto err = cudaMemcpy(
        buf.data(), tensor->data(), kTensorElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("rx: cudaMemcpy failed: {}", cudaGetErrorString(err));
      ++received_;
      ++mismatches_;
      return;
    }

    for (int i = 0; i < kTensorElements; ++i) {
      if (buf[i] != expected_value(i)) {
        HOLOSCAN_LOG_ERROR(
            "rx: data mismatch [{}] expected={} got={}", i, expected_value(i), buf[i]);
        ++mismatches_;
        break;
      }
    }

    ++received_;
    HOLOSCAN_LOG_INFO("rx: verified message {} OK", received_);
    if (received_ >= target_count_.get()) {
      fragment()->stop_execution();
    }
  }

  int received() const { return received_; }
  int mismatches() const { return mismatches_; }

 private:
  holoscan::Parameter<int64_t> target_count_;
  int received_ = 0;
  int mismatches_ = 0;
};

// ---------------------------------------------------------------------------
// Applications
// ---------------------------------------------------------------------------

class PubApp : public holoscan::Application {
 public:
  PubApp(int64_t count, int64_t timeout_ms, std::string native_buffer_policy)
      : count_(count),
        timeout_ms_(timeout_ms),
        native_buffer_policy_(std::move(native_buffer_policy)) {}

  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::FastDdsPubSubNetworkContext>(
        "pubsub_context", holoscan::Arg("native_buffer_policy", native_buffer_policy_));
  }

  void compose() override {
    using namespace holoscan;
    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", timeout_ms_)));

    add_operator(
        make_operator<IpcTxOp>("tx",
                               make_condition<PeriodicCondition>(
                                   "period",
                                   Arg{"recess_period", std::string("200ms")},
                                   Arg{"policy", PeriodicConditionPolicy::kNoCatchUpMissedTicks}),
                               Arg("allocator", make_resource<UnboundedAllocator>("alloc")),
                               Arg("target_count", count_)));
  }

 private:
  int64_t count_;
  int64_t timeout_ms_;
  std::string native_buffer_policy_;
};

class SubApp : public holoscan::Application {
 public:
  SubApp(int64_t count, int64_t timeout_ms, std::string native_buffer_policy)
      : count_(count),
        timeout_ms_(timeout_ms),
        native_buffer_policy_(std::move(native_buffer_policy)) {}

  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::FastDdsPubSubNetworkContext>(
        "pubsub_context", holoscan::Arg("native_buffer_policy", native_buffer_policy_));
  }

  void compose() override {
    using namespace holoscan;
    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", timeout_ms_)));

    rx_ = make_operator<IpcRxOp>(
        "rx", make_condition<CountCondition>("count", count_), Arg("target_count", count_));
    add_operator(rx_);
  }

  std::shared_ptr<IpcRxOp> rx() const { return rx_; }

 private:
  int64_t count_;
  int64_t timeout_ms_;
  std::string native_buffer_policy_;
  std::shared_ptr<IpcRxOp> rx_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string role;
  std::string native_buffer_policy = "preferred";
  std::string label;
  int64_t count = 5;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--role" && i + 1 < argc) {
      role = argv[++i];
    } else if (arg == "--count" && i + 1 < argc) {
      count = std::atol(argv[++i]);
    } else if (arg == "--native_buffer_policy" && i + 1 < argc) {
      native_buffer_policy = argv[++i];
    } else if (arg == "--label" && i + 1 < argc) {
      label = argv[++i];
    }
  }

  if (role != "publisher" && role != "subscriber") {
    std::cerr << "Usage: " << argv[0] << " --role <publisher|subscriber> [--count N]"
              << " [--native_buffer_policy disabled|preferred|required]"
              << " [--label NAME]\n";
    return 2;
  }

  const int64_t timeout_ms = std::max<int64_t>(10000, 2000 + count * 500);
  auto make_pass_prefix = [&label]() -> std::string {
    return label.empty() ? "PASS" : "PASS[" + label + "]";
  };
  auto make_fail_prefix = [&label]() -> std::string {
    return label.empty() ? "FAIL" : "FAIL[" + label + "]";
  };

  if (role == "publisher") {
    auto app = holoscan::make_application<PubApp>(count, timeout_ms, native_buffer_policy);
    app->run();
    return 0;
  }

  // subscriber
  auto app = holoscan::make_application<SubApp>(count, timeout_ms, native_buffer_policy);
  app->run();

  auto rx = app->rx();
  if (!rx) {
    std::cerr << "ERROR: rx operator is null\n";
    return 1;
  }

  int rc = 0;
  if (rx->received() != static_cast<int>(count)) {
    std::cerr << make_fail_prefix() << ": received " << rx->received() << "/" << count
              << " messages\n";
    rc = 1;
  }
  if (rx->mismatches() > 0) {
    std::cerr << make_fail_prefix() << ": " << rx->mismatches() << " data mismatches\n";
    rc = 1;
  }
  if (rc == 0) {
    std::cout << make_pass_prefix() << ": received " << rx->received() << "/" << count
              << " messages, 0 mismatches\n";
  }
  return rc;
}
