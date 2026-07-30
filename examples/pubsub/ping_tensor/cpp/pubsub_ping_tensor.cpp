/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp>
#include <holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp>
#include <holoscan/pubsub/runtime/conditions/publisher_available.hpp>
#include <holoscan/pubsub/runtime/conditions/subscriber_available.hpp>

#include <holoscan/pubsub/runtime/conditions/pending_export_condition.hpp>

#include "holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp"

namespace {

constexpr const char* kTopicName = "ping_tensor_topic";
constexpr uint64_t kMaxPendingNativeExports = 4;

enum class AppRole {
  kPublisher,
  kSubscriber,
};

std::string to_string(AppRole role) {
  switch (role) {
    case AppRole::kPublisher:
      return "publisher";
    case AppRole::kSubscriber:
      return "subscriber";
  }
  return "unknown";
}

std::optional<bool> get_boolean_arg(const std::vector<std::string>& args, const std::string& name) {
  if (std::find(args.begin(), args.end(), name) != std::end(args)) {
    return true;
  }
  return {};
}

std::optional<int32_t> get_int32_arg(const std::vector<std::string>& args,
                                     const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    try {
      return std::stoi(*loc);
    } catch (const std::exception&) {
      HOLOSCAN_LOG_ERROR("Unable to parse provided argument '{}'", name);
      return {};
    }
  }
  return {};
}

std::optional<int64_t> get_int64_arg(const std::vector<std::string>& args,
                                     const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    try {
      return std::stoll(*loc);
    } catch (const std::exception&) {
      HOLOSCAN_LOG_ERROR("Unable to parse provided argument '{}'", name);
      return {};
    }
  }
  return {};
}

std::optional<std::string> get_str_arg(const std::vector<std::string>& args,
                                       const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    return *loc;
  }
  return {};
}

std::optional<AppRole> parse_role(const std::string& value) {
  if (value == "publisher") {
    return AppRole::kPublisher;
  }
  if (value == "subscriber") {
    return AppRole::kSubscriber;
  }
  return {};
}

class PubSubAppBase : public holoscan::Application {
 public:
  PubSubAppBase(const std::string& native_buffer_policy, bool use_eager_acquire)
      : native_buffer_policy_(native_buffer_policy), use_eager_acquire_(use_eager_acquire) {}

 protected:
  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    using namespace holoscan;
    constexpr const char* kNetworkContextName = "pubsub_context";

    HOLOSCAN_LOG_INFO(
        "Creating Fast-DDS pub/sub network context '{}' "
        "(native_buffer_policy='{}', eager_acquire={})",
        kNetworkContextName,
        native_buffer_policy_,
        use_eager_acquire_);
    return make_network_context<FastDdsPubSubNetworkContext>(
        kNetworkContextName,
        Arg("native_buffer_policy", native_buffer_policy_),
        Arg("native_buffer_use_eager_acquire", use_eager_acquire_));
  }

  const std::string& native_buffer_policy() const { return native_buffer_policy_; }

 private:
  std::string native_buffer_policy_;
  bool use_eager_acquire_ = false;
};

class SubscriberRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SubscriberRxOp)
  SubscriberRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in").topic(kTopicName);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe_in_message = op_input.receive<holoscan::TensorMap>("in");
    if (!maybe_in_message) {
      return;
    }

    auto in_message = maybe_in_message.value();
    for (auto& [key, tensor] : in_message) {
      if (!tensor || tensor->data() == nullptr) {
        HOLOSCAN_LOG_WARN("rx received tensor '{}' with null data", key);
        continue;
      }
      ++received_count_;
      HOLOSCAN_LOG_INFO("rx received message {}: Tensor key: '{}', shape: ({})",
                        received_count_,
                        key,
                        fmt::join(tensor->shape(), ", "));
    }
  }

 private:
  int received_count_ = 0;
};

class SubscriberApp : public PubSubAppBase {
 public:
  SubscriberApp(const std::string& native_buffer_policy, bool use_eager_acquire, int64_t rx_count,
                int64_t deadlock_timeout_ms)
      : PubSubAppBase(native_buffer_policy, use_eager_acquire),
        rx_count_(rx_count),
        deadlock_timeout_ms_(deadlock_timeout_ms) {}

  void compose() override {
    using namespace holoscan;

    auto pubsub_network_context = create_pubsub_network_context();
    network_context(pubsub_network_context);

    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", deadlock_timeout_ms_)));

    auto pub_ready = make_condition<PublisherAvailableCondition>(
        "publisher_available",
        Arg("receiver", std::string("in")),
        Arg("poll_period_ms", static_cast<int64_t>(100)),
        Arg("latch_ready", true));

    std::shared_ptr<SubscriberRxOp> rx;
    if (rx_count_ >= 0) {
      rx = make_operator<SubscriberRxOp>(
          "rx", pub_ready, make_condition<CountCondition>("rx_count", rx_count_));
    } else {
      rx = make_operator<SubscriberRxOp>("rx", pub_ready);
    }
    add_operator(rx);
  }

 private:
  int64_t rx_count_ = 10;
  int64_t deadlock_timeout_ms_ = 4000;
};

class PublisherApp : public PubSubAppBase {
 public:
  PublisherApp(const std::string& native_buffer_policy, bool use_eager_acquire, bool gpu_tensor,
               int64_t tx_count, int32_t batch_size, int32_t rows, int32_t columns,
               int32_t channels, std::string data_type, int64_t tx_period_ms,
               int64_t deadlock_timeout_ms)
      : PubSubAppBase(native_buffer_policy, use_eager_acquire),
        gpu_tensor_(gpu_tensor),
        tx_count_(tx_count),
        batch_size_(batch_size),
        rows_(rows),
        columns_(columns),
        channels_(channels),
        data_type_(std::move(data_type)),
        tx_period_ms_(tx_period_ms),
        deadlock_timeout_ms_(deadlock_timeout_ms) {}

  void compose() override {
    using namespace holoscan;

    auto pubsub_network_context = create_pubsub_network_context();
    network_context(pubsub_network_context);

    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", deadlock_timeout_ms_)));

    auto tx_period =
        make_condition<PeriodicCondition>("tx_period", std::chrono::milliseconds(tx_period_ms_));
    auto tx_count = make_condition<CountCondition>("tx_count", tx_count_);
    auto sub_ready = make_condition<SubscriberAvailableCondition>(
        "subscriber_available",
        Arg("transmitter", std::string("out")),
        Arg("min_subscriber_count", static_cast<uint64_t>(1)),
        Arg("poll_period_ms", static_cast<int64_t>(100)),
        Arg("stabilization_ms", static_cast<int64_t>(500)),
        Arg("latch_ready", true));

    std::shared_ptr<holoscan::PendingExportCondition> pending_export_cond;
    if (native_buffer_policy() != "disabled") {
      pending_export_cond = make_condition<holoscan::PendingExportCondition>(
          "pending_export_cond", Arg("max_pending", kMaxPendingNativeExports));
      pending_export_cond->adapter_resolver(
          [pubsub_network_context]() -> std::shared_ptr<NativeBufferProtocolAdapter> {
            if (auto dds_ctx = std::dynamic_pointer_cast<FastDdsPubSubNetworkContext>(
                    pubsub_network_context)) {
              return dds_ctx->native_buffer_adapter();
            }
            return std::shared_ptr<NativeBufferProtocolAdapter>{};
          });
    }

    std::shared_ptr<ops::PingTensorTxOp> tx;
    if (pending_export_cond) {
      tx = make_operator<ops::PingTensorTxOp>(
          "tx",
          tx_count,
          tx_period,
          sub_ready,
          pending_export_cond,
          Arg("storage_type", std::string{gpu_tensor_ ? "device" : "system"}),
          Arg("batch_size", batch_size_),
          Arg("rows", rows_),
          Arg("columns", columns_),
          Arg("channels", channels_),
          Arg("data_type", data_type_));
    } else {
      tx = make_operator<ops::PingTensorTxOp>(
          "tx",
          tx_count,
          tx_period,
          sub_ready,
          Arg("storage_type", std::string{gpu_tensor_ ? "device" : "system"}),
          Arg("batch_size", batch_size_),
          Arg("rows", rows_),
          Arg("columns", columns_),
          Arg("channels", channels_),
          Arg("data_type", data_type_));
    }

    auto& outputs = tx->spec()->outputs();
    auto out_it = outputs.find("out");
    if (out_it != outputs.end()) {
      out_it->second->connector(IOSpec::ConnectorType::kPubSub, Arg("topic_name", kTopicName));
    }

    add_operator(tx);
  }

 private:
  bool gpu_tensor_ = false;
  int64_t tx_count_ = 10;
  int32_t batch_size_ = 0;
  int32_t rows_ = 32;
  int32_t columns_ = 64;
  int32_t channels_ = 0;
  std::string data_type_{"uint8_t"};
  int64_t tx_period_ms_ = 100;
  int64_t deadlock_timeout_ms_ = 5000;
};

}  // namespace

int main() {
  try {
    std::cout
        << "Additional supported arguments: \n"
        << "  --role ROLE         Process role: {'publisher', 'subscriber'}.\n"
        << "  --gpu               If specified, tensors will be in GPU memory.\n"
        << "  --count COUNT       Message count: publisher send count or subscriber receive\n"
        << "                      target (default: 10). Negative = run indefinitely.\n"
        << "  --batch_size BATCH  The batch size of the tensor (dimension omitted if 0).\n"
        << "  --rows ROWS         The number of rows in the tensor.\n"
        << "  --columns COLUMNS   The number of columns in the tensor.\n"
        << "  --channels CHANNELS The number of channels in the tensor (dimension omitted if 0).\n"
        << "  --data_type TYPE    Tensor element type.\n"
        << "  --native_buffer_policy POLICY\n"
        << "                      Native buffer policy: {'disabled', 'preferred', 'required'}\n"
        << "                      (default: 'preferred').\n"
        << "  --eager             Subscriber-side option: use holoipc\n"
        << "                      acquire_pointer_eager() for CUDA IPC imports on Fast-DDS.\n"
        << "                      Ignored by publisher role. Disabled by default.\n"
        << "  --track             Enable Data Flow Tracking output (default: disabled).\n"
        << "  --tx_period_ms MS   Publisher delay between tensor messages (default: 100 ms).\n"
        << '\n';

    auto args_app = holoscan::make_application<holoscan::Application>();
    std::vector<std::string>& remaining_args = args_app->argv();

    const auto role_arg = get_str_arg(remaining_args, "--role");
    if (!role_arg) {
      throw std::runtime_error(
          "Missing required --role argument. Expected 'publisher' or 'subscriber'.");
    }
    const auto parsed_role = parse_role(*role_arg);
    if (!parsed_role) {
      throw std::runtime_error("Invalid --role value. Expected 'publisher' or 'subscriber'.");
    }
    const AppRole role = *parsed_role;

    const bool tensor_on_gpu = get_boolean_arg(remaining_args, "--gpu").value_or(false);
    const int64_t count = get_int64_arg(remaining_args, "--count").value_or(10);
    const int32_t batch_size = get_int32_arg(remaining_args, "--batch_size").value_or(0);
    const int32_t rows = get_int32_arg(remaining_args, "--rows").value_or(32);
    const int32_t columns = get_int32_arg(remaining_args, "--columns").value_or(64);
    const int32_t channels = get_int32_arg(remaining_args, "--channels").value_or(0);
    const std::string data_type = get_str_arg(remaining_args, "--data_type").value_or("uint8_t");
    const int64_t tx_period_ms =
        std::max<int64_t>(1, get_int64_arg(remaining_args, "--tx_period_ms").value_or(100));
    const std::string native_buffer_policy =
        get_str_arg(remaining_args, "--native_buffer_policy").value_or("preferred");
    const bool use_eager_acquire = get_boolean_arg(remaining_args, "--eager").value_or(false);
    const bool enable_tracking = get_boolean_arg(remaining_args, "--track").value_or(false);

    constexpr int64_t kPublisherDeadlockTimeoutMs = 300'000;
    constexpr int64_t kSubscriberDeadlockTimeoutMs = 300'000;
    const int64_t deadlock_timeout_ms =
        (role == AppRole::kPublisher) ? kPublisherDeadlockTimeoutMs : kSubscriberDeadlockTimeoutMs;

    HOLOSCAN_LOG_INFO(
        "Running pubsub_ping_tensor role='{}' with backend='fastdds', tensors on {}, count={}, "
        "tx_period_ms={}, native_buffer_policy='{}', eager_acquire={}, track={}, "
        "deadlock_timeout_ms={}",
        to_string(role),
        tensor_on_gpu ? "GPU" : "host",
        count,
        tx_period_ms,
        native_buffer_policy,
        use_eager_acquire,
        enable_tracking,
        deadlock_timeout_ms);

    if (role == AppRole::kPublisher) {
      auto app = holoscan::make_application<PublisherApp>(native_buffer_policy,
                                                          use_eager_acquire,
                                                          tensor_on_gpu,
                                                          count,
                                                          batch_size,
                                                          rows,
                                                          columns,
                                                          channels,
                                                          data_type,
                                                          tx_period_ms,
                                                          deadlock_timeout_ms);
      if (!app) {
        throw std::runtime_error("Failed to create publisher app");
      }
      holoscan::DataFlowTracker* tracker = nullptr;
      if (enable_tracking) {
        tracker = &app->track(1, 0, 0);
      }
      app->run();
      if (tracker) {
        tracker->print();
      }
    } else {
      auto app = holoscan::make_application<SubscriberApp>(
          native_buffer_policy, use_eager_acquire, count, deadlock_timeout_ms);
      if (!app) {
        throw std::runtime_error("Failed to create subscriber app");
      }
      holoscan::DataFlowTracker* tracker = nullptr;
      if (enable_tracking) {
        tracker = &app->track(1, 0, 0);
      }
      app->run();
      if (tracker) {
        tracker->print();
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "pubsub_ping_tensor failed: " << e.what() << '\n';
    return 1;
  }
}
