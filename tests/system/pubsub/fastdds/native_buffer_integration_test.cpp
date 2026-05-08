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

/// @file
/// Integration test for single-process DDS pub/sub with GPU tensors.
///
/// Loopback (single process):
///   Two operators (TxOp, RxOp) in the same fragment communicate GPU tensors
///   through DDS pub/sub.  Same-process CUDA IPC is ineligible, so the
///   byte-staging path (D2H -> serialize -> deserialize -> H2D) is used.
///
/// Cross-process CUDA IPC testing lives in a separate binary
/// (native_buffer_ipc_app) launched by test_native_buffer_ipc.sh.

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <gxf/std/tensor.hpp>

#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>
#include <holoscan/holoscan.hpp>
#include "holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp"

namespace {

constexpr const char* kTopicName = "native_buffer_test_tensor";
constexpr int kTensorRows = 4;
constexpr int kTensorCols = 8;
constexpr int kTensorElements = kTensorRows * kTensorCols;
constexpr int kMessageCount = 5;

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

float expected_value(int element_index) {
  return static_cast<float>(element_index + 1) * 1.5f;
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

class GpuTensorRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GpuTensorRxOp)
  GpuTensorRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in").topic(kTopicName);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe = op_input.receive<holoscan::TensorMap>("in");
    if (!maybe)
      return;

    auto tensor_map = maybe.value();
    if (tensor_map.empty()) {
      HOLOSCAN_LOG_ERROR("GpuTensorRxOp: received empty TensorMap");
      ++received_count_;
      ++mismatches_;
      return;
    }

    auto it = tensor_map.find("tensor");
    if (it == tensor_map.end()) {
      it = tensor_map.begin();
    }
    auto& ht = it->second;

    if (!ht->data()) {
      HOLOSCAN_LOG_ERROR("GpuTensorRxOp: null tensor data pointer");
      ++received_count_;
      ++mismatches_;
      return;
    }

    int ndim = ht->ndim();
    if (ndim != 2 || ht->shape()[0] != kTensorRows || ht->shape()[1] != kTensorCols) {
      if (ndim < 2) {
        HOLOSCAN_LOG_ERROR("GpuTensorRxOp: unexpected ndim={} (expected 2)", ndim);
      } else {
        HOLOSCAN_LOG_ERROR("GpuTensorRxOp: unexpected shape (ndim={}, dims={}x{})",
                           ndim,
                           ht->shape()[0],
                           ht->shape()[1]);
      }
      ++received_count_;
      ++mismatches_;
      return;
    }

    std::vector<float> host_data(kTensorElements);
    cudaError_t err = cudaMemcpy(
        host_data.data(), ht->data(), kTensorElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("GpuTensorRxOp: cudaMemcpy D2H failed: {}", cudaGetErrorString(err));
      ++received_count_;
      ++mismatches_;
      return;
    }

    for (int i = 0; i < kTensorElements; ++i) {
      if (host_data[i] != expected_value(i)) {
        HOLOSCAN_LOG_ERROR(
            "GpuTensorRxOp: data mismatch at index {} "
            "(expected={}, got={})",
            i,
            expected_value(i),
            host_data[i]);
        ++mismatches_;
        break;
      }
    }

    ++received_count_;
    HOLOSCAN_LOG_INFO("GpuTensorRxOp: verified message {} OK", received_count_);
  }

  int received_count() const { return received_count_; }
  int mismatches() const { return mismatches_; }

 private:
  int received_count_ = 0;
  int mismatches_ = 0;
};

class MatchedGpuTensorTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatchedGpuTensorTxOp)
  MatchedGpuTensorTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out").topic(kTopicName);
    spec.param(allocator_, "allocator", "Allocator", "Allocator for GPU tensors");
    spec.param(target_count_,
               "target_count",
               "Target Count",
               "Number of messages to emit",
               static_cast<int64_t>(kMessageCount));
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override {
    resolve_pubsub_transmitter();

    if (!pubsub_tx_ || !pubsub_tx_->has_matched_subscribers()) {
      ++wait_ticks_;
      if (wait_ticks_ % 50 == 1) {
        HOLOSCAN_LOG_INFO("{} waiting for matched subscriber (tick={}, matched={})",
                          name(),
                          wait_ticks_,
                          pubsub_tx_ ? pubsub_tx_->matched_subscriber_count() : 0UL);
      }
      return;
    }

    if (!logged_first_match_) {
      HOLOSCAN_LOG_INFO(
          "{} matched subscriber(s): {}", name(), pubsub_tx_->matched_subscriber_count());
      logged_first_match_ = true;
    }

    if (sent_count_ >= target_count_.get()) {
      if (++drain_ticks_ >= kDrainTicks) {
        fragment()->stop_execution();
      }
      return;
    }

    auto gxf_context = context.context();
    auto alloc_handle =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());
    if (!alloc_handle) {
      HOLOSCAN_LOG_ERROR("MatchedGpuTensorTxOp: failed to get allocator handle");
      return;
    }

    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
    nvidia::gxf::Shape shape({kTensorRows, kTensorCols});
    auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
    auto reshape = gxf_tensor->reshapeCustom(shape,
                                             nvidia::gxf::PrimitiveType::kFloat32,
                                             sizeof(float),
                                             strides,
                                             nvidia::gxf::MemoryStorageType::kDevice,
                                             alloc_handle.value());
    if (!reshape) {
      HOLOSCAN_LOG_ERROR("MatchedGpuTensorTxOp: reshapeCustom failed");
      return;
    }

    std::vector<float> host_data(kTensorElements);
    for (int i = 0; i < kTensorElements; ++i) {
      host_data[i] = expected_value(i);
    }
    auto err = cudaMemcpy(gxf_tensor->pointer(),
                          host_data.data(),
                          kTensorElements * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("MatchedGpuTensorTxOp: cudaMemcpy H2D failed: {}",
                         cudaGetErrorString(err));
      return;
    }

    auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR("MatchedGpuTensorTxOp: toDLManagedTensorContext failed");
      return;
    }
    auto dl_ctx = maybe_dl_ctx.value();
    auto* mem_buf = static_cast<nvidia::gxf::MemoryBuffer*>(dl_ctx->memory_ref.get());
    auto holoscan_tensor = std::make_shared<holoscan::Tensor>(dl_ctx, mem_buf);

    holoscan::TensorMap out_message;
    out_message.insert({"tensor", holoscan_tensor});
    op_output.emit(out_message, "out");

    ++sent_count_;
    HOLOSCAN_LOG_INFO("MatchedGpuTensorTxOp: sent message {} (shape={}x{}, device)",
                      sent_count_,
                      kTensorRows,
                      kTensorCols);
  }

 private:
  static constexpr int64_t kDrainTicks = 15;

  void resolve_pubsub_transmitter() {
    if (pubsub_tx_)
      return;
    auto* op_spec = spec();
    if (!op_spec)
      return;
    auto out_it = op_spec->outputs().find("out");
    if (out_it == op_spec->outputs().end())
      return;
    pubsub_tx_ =
        std::dynamic_pointer_cast<holoscan::PubSubTransmitter>(out_it->second->connector());
  }

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  holoscan::Parameter<int64_t> target_count_;
  std::shared_ptr<holoscan::PubSubTransmitter> pubsub_tx_;
  int64_t sent_count_ = 0;
  int64_t wait_ticks_ = 0;
  int64_t drain_ticks_ = 0;
  bool logged_first_match_ = false;
};

// ---------------------------------------------------------------------------
// Loopback application
// ---------------------------------------------------------------------------

class LoopbackApp : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::FastDdsPubSubNetworkContext>(
        "pubsub_context", holoscan::Arg("native_buffer_policy", std::string("disabled")));
  }

  void compose() override {
    using namespace holoscan;

    const int64_t deadlock_timeout_ms = 5000;
    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));

    tx_ = make_operator<MatchedGpuTensorTxOp>(
        "tx",
        make_condition<PeriodicCondition>(
            "tx_period",
            Arg{"recess_period", std::string("100ms")},
            Arg{"policy", PeriodicConditionPolicy::kNoCatchUpMissedTicks}),
        Arg("allocator", make_resource<UnboundedAllocator>("tx_alloc")),
        Arg("target_count", static_cast<int64_t>(kMessageCount)));

    rx_ = make_operator<GpuTensorRxOp>(
        "rx", make_condition<CountCondition>("rx_count", static_cast<int64_t>(kMessageCount)));

    add_operator(tx_);
    add_operator(rx_);
  }

  std::shared_ptr<GpuTensorRxOp> rx() const { return rx_; }

 private:
  std::shared_ptr<MatchedGpuTensorTxOp> tx_;
  std::shared_ptr<GpuTensorRxOp> rx_;
};

}  // namespace

// =============================================================================
// Loopback — single process, GPU tensors via byte-staging
// =============================================================================

TEST(NativeBufferIntegrationTest, LoopbackGpuTensorByteStagingRoundTrip) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto app = holoscan::make_application<LoopbackApp>();
  app->run();

  auto rx = app->rx();
  ASSERT_NE(rx, nullptr);

  EXPECT_EQ(rx->received_count(), kMessageCount)
      << "Expected all " << kMessageCount << " GPU tensor messages via byte-staging path";
  EXPECT_EQ(rx->mismatches(), 0) << "Data mismatches detected in byte-staging round-trip";

  HOLOSCAN_LOG_INFO("Loopback test: received {}/{} messages, {} mismatches",
                    rx->received_count(),
                    kMessageCount,
                    rx->mismatches());
}
