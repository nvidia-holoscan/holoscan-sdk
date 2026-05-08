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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include <holoscan/pubsub/runtime/conditions/pending_export_condition.hpp>

#include "holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp"

namespace {

constexpr const char* kTopicName = "video_replayer_topic";
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

class SubscriberApp : public PubSubAppBase {
 public:
  SubscriberApp(const std::string& native_buffer_policy, bool use_eager_acquire,
                int64_t deadlock_timeout_ms)
      : PubSubAppBase(native_buffer_policy, use_eager_acquire),
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
        Arg("receiver", std::string("receivers")),
        Arg("poll_period_ms", static_cast<int64_t>(100)),
        Arg("latch_ready", true));

    auto holoviz = make_operator<ops::HolovizOp>("holoviz", pub_ready, from_config("holoviz"));

    // Directly bind Holoviz's kAnySize input to Pub/Sub. Use the same receiver
    // queue settings that the old relay used so incoming frames are popped
    // rather than stalling on overflow. This is a temporary explicit connector
    // workaround until kAnySize queue policy handling is fixed more generally.
    auto& inputs = holoviz->spec()->inputs();
    auto in_it = inputs.find("receivers");
    if (in_it != inputs.end()) {
      in_it->second->connector(IOSpec::ConnectorType::kPubSub,
                               Arg("topic_name", std::string{kTopicName}),
                               Arg("capacity", static_cast<uint64_t>(2)),
                               Arg("policy", static_cast<uint64_t>(0)));  // pop
    }

    add_operator(holoviz);
  }

 private:
  int64_t deadlock_timeout_ms_ = 3600000;
};

class PublisherApp : public PubSubAppBase {
 public:
  PublisherApp(const std::string& native_buffer_policy, bool use_eager_acquire,
               int64_t deadlock_timeout_ms, bool disable_pending_export_condition)
      : PubSubAppBase(native_buffer_policy, use_eager_acquire),
        deadlock_timeout_ms_(deadlock_timeout_ms),
        disable_pending_export_condition_(disable_pending_export_condition) {}

  void compose() override {
    using namespace holoscan;

    auto pubsub_network_context = create_pubsub_network_context();
    network_context(pubsub_network_context);

    scheduler(
        make_scheduler<EventBasedScheduler>("scheduler",
                                            Arg("worker_thread_number", static_cast<int64_t>(2)),
                                            Arg("stop_on_deadlock_timeout", deadlock_timeout_ms_)));

    ArgList replayer_args;
    auto* env_data_dir = std::getenv("HOLOSCAN_INPUT_PATH");
    if (env_data_dir && env_data_dir[0] != '\0') {
      replayer_args.add(
          Arg("directory", (std::filesystem::path(env_data_dir) / "racerx").string()));
    }
    replayer_args.add(Arg(
        "allocator", make_resource<RMMAllocator>("rmm_allocator", from_config("rmm_allocator"))));

    auto sub_ready = make_condition<SubscriberAvailableCondition>(
        "subscriber_available",
        Arg("transmitter", std::string("output")),
        Arg("min_subscriber_count", static_cast<uint64_t>(1)),
        Arg("poll_period_ms", static_cast<int64_t>(100)),
        Arg("stabilization_ms", static_cast<int64_t>(500)),
        Arg("latch_ready", true));

    std::shared_ptr<PendingExportCondition> pending_export_cond;
    if (!disable_pending_export_condition_ && native_buffer_policy() != "disabled") {
      pending_export_cond = make_condition<PendingExportCondition>(
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

    std::shared_ptr<ops::VideoStreamReplayerOp> replayer;
    if (pending_export_cond) {
      replayer = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", sub_ready, pending_export_cond, from_config("replayer"), replayer_args);
    } else {
      replayer = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", sub_ready, from_config("replayer"), replayer_args);
    }

    // Set pub/sub connector on the replayer output port so frames are published on the topic.
    auto& outputs = replayer->spec()->outputs();
    auto out_it = outputs.find("output");
    if (out_it != outputs.end()) {
      out_it->second->connector(IOSpec::ConnectorType::kPubSub,
                                Arg("topic_name", std::string{kTopicName}));
    }

    add_operator(replayer);
  }

 private:
  int64_t deadlock_timeout_ms_ = 10000;
  bool disable_pending_export_condition_ = false;
};

}  // namespace

int main() {
  try {
    std::cout
        << "Additional supported arguments:\n"
        << "  --role ROLE         Process role: {'publisher', 'subscriber'}.\n"
        << "  --config PATH       Path to the YAML config file (default: video_replayer.yaml\n"
        << "                      next to this binary).\n"
        << "  --native_buffer_policy POLICY\n"
        << "                      Native buffer policy: {'disabled', 'preferred', 'required'}\n"
        << "                      (default: 'preferred').\n"
        << "  --disable_pending_export_condition\n"
        << "                      Publisher-side debug option: skip PendingExportCondition\n"
        << "                      even when native buffers are enabled.\n"
        << "  --track            Enable Holoscan data flow tracking for this run.\n"
        << "                      Disabled by default.\n"
        << "  --eager             Subscriber-side option: use holoipc\n"
        << "                      acquire_pointer_eager() for CUDA IPC imports on Fast-DDS.\n"
        << "                      Ignored by publisher role. Disabled by default.\n"
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

    const std::string native_buffer_policy =
        get_str_arg(remaining_args, "--native_buffer_policy").value_or("preferred");
    const bool use_eager_acquire = get_boolean_arg(remaining_args, "--eager").value_or(false);
    const bool enable_tracking = get_boolean_arg(remaining_args, "--track").value_or(false);
    const bool disable_pending_export_condition =
        get_boolean_arg(remaining_args, "--disable_pending_export_condition").value_or(false);

    // Resolve config file path: use --config arg, or look for video_replayer.yaml next to binary.
    std::string config_path;
    if (const auto config_arg = get_str_arg(remaining_args, "--config")) {
      config_path = *config_arg;
    } else {
      config_path =
          (std::filesystem::canonical("/proc/self/exe").parent_path() / "video_replayer.yaml")
              .string();
    }

    HOLOSCAN_LOG_INFO(
        "Running pubsub_video_replayer role='{}' with backend='fastdds', "
        "native_buffer_policy='{}', eager_acquire={}, track={}, "
        "disable_pending_export_condition={}, config='{}'",
        to_string(role),
        native_buffer_policy,
        use_eager_acquire,
        enable_tracking,
        disable_pending_export_condition,
        config_path);

    // Use a long deadlock timeout for both roles so startup gating on availability conditions
    // does not terminate the app before a peer is present.
    constexpr int64_t kPublisherDeadlockTimeoutMs = 300'000;
    constexpr int64_t kSubscriberDeadlockTimeoutMs = 300'000;

    if (role == AppRole::kPublisher) {
      auto app = holoscan::make_application<PublisherApp>(native_buffer_policy,
                                                          use_eager_acquire,
                                                          kPublisherDeadlockTimeoutMs,
                                                          disable_pending_export_condition);
      app->config(config_path);
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
          native_buffer_policy, use_eager_acquire, kSubscriberDeadlockTimeoutMs);
      app->config(config_path);
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
    std::cerr << "pubsub_video_replayer failed: " << e.what() << '\n';
    return 1;
  }
}
