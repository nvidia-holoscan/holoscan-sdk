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

#include <ucxx/api.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/pose_tree/pose_tree.hpp>
#include <holoscan/pose_tree/pose_tree_ucx_client.hpp>
#include <holoscan/pose_tree/pose_tree_ucx_common.hpp>

#include "distributed_app_fixture.hpp"

namespace holoscan {

namespace {

constexpr auto kSnapshotSendTimeout = std::chrono::milliseconds(5000);

void wait_single_request(const std::shared_ptr<ucxx::Worker>& worker,
                         const std::shared_ptr<ucxx::Request>& request,
                         std::chrono::milliseconds timeout, std::chrono::microseconds poll_sleep) {
  const auto start_time = std::chrono::steady_clock::now();
  while (!request->isCompleted()) {
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      throw std::runtime_error("UCX request timed out");
    }
    worker->progress();
    std::this_thread::sleep_for(poll_sleep);
  }
  request->checkError();
}

class UcxOrderingTestServer {
 public:
  explicit UcxOrderingTestServer(uint16_t port) : port_(port) {}

  ~UcxOrderingTestServer() { stop(); }

  UcxOrderingTestServer(const UcxOrderingTestServer&) = delete;
  UcxOrderingTestServer& operator=(const UcxOrderingTestServer&) = delete;

  void start() {
    if (running_.exchange(true)) {
      return;
    }
    server_thread_ = std::thread(&UcxOrderingTestServer::run, this);

    std::unique_lock<std::mutex> lk(mutex_);
    constexpr auto kReadyTimeout = std::chrono::seconds(5);
    if (!ready_cv_.wait_for(lk, kReadyTimeout, [this] { return ready_; })) {
      lk.unlock();
      stop();
      throw std::runtime_error("UCX test server failed to become ready");
    }
    if (startup_failed_) {
      const auto err = startup_error_;
      lk.unlock();
      stop();
      throw std::runtime_error("UCX test server startup failed: " + err);
    }
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
  }

  bool wait_for_sequence_sent(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mutex_);
    return sequence_sent_cv_.wait_for(lk, timeout, [this] { return sequence_sent_.load(); });
  }

 private:
  static void connection_callback(ucp_conn_request_h req, void* arg) {
    auto* self = static_cast<UcxOrderingTestServer*>(arg);
    try {
      auto ep = self->listener_->createEndpointFromConnRequest(req, true);
      self->endpoint_ = std::move(ep);
    } catch (const std::exception&) {
      // Best-effort: test will time out if connection cannot be established.
    }
  }

  void run() {
    try {
      context_ = ucxx::createContext({}, UCP_FEATURE_AM);
      worker_ = context_->createWorker();
      listener_ = worker_->createListener(port_, connection_callback, this);

      // Receive subscribe message from the client.
      ucxx::AmReceiverCallbackInfo subscribe_callback_info("AMServer", MSG_SUBSCRIBE);
      worker_->registerAmReceiverCallback(
          std::move(subscribe_callback_info), [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
            try {
              req->checkError();
              auto buffer = req->getRecvBuffer();
              const auto* data = static_cast<const uint8_t*>(buffer->data());
              const auto size = buffer->getSize();
              if (size != sizeof(SubscribeMessage)) {
                return;
              }
              SubscribeMessage sub_msg{};
              std::memcpy(&sub_msg, data, sizeof(sub_msg));
              want_snapshot_.store(sub_msg.request_snapshot != 0);
              got_subscribe_.store(true);
            } catch (const std::exception&) {
              // Best-effort: test will time out if subscribe handling fails.
            }
          });

      // Receive close message from the client (best-effort, ignore).
      ucxx::AmReceiverCallbackInfo close_callback_info("AMServer", MSG_CLOSE);
      worker_->registerAmReceiverCallback(std::move(close_callback_info),
                                          [](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
                                            try {
                                              req->checkError();
                                            } catch (const std::exception&) {
                                            }
                                          });

      // Receive snapshot-ack message from the client (best-effort, ignore).
      ucxx::AmReceiverCallbackInfo snapshot_ack_info("AMServer", MSG_SNAPSHOT_ACK);
      worker_->registerAmReceiverCallback(std::move(snapshot_ack_info),
                                          [](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
                                            try {
                                              req->checkError();
                                            } catch (const std::exception&) {
                                            }
                                          });
    } catch (const std::exception& e) {
      {
        std::lock_guard<std::mutex> lk(mutex_);
        startup_failed_ = true;
        startup_error_ = e.what();
        ready_ = true;
      }
      ready_cv_.notify_one();
      return;
    }

    {
      std::lock_guard<std::mutex> lk(mutex_);
      ready_ = true;
    }
    ready_cv_.notify_one();

    while (running_.load()) {
      worker_->progress();

      if (!sequence_sent_.load() && endpoint_ && got_subscribe_.load() && want_snapshot_.load()) {
        try {
          send_test_sequence();
          {
            std::lock_guard<std::mutex> lk(mutex_);
            sequence_sent_ = true;
          }
          sequence_sent_cv_.notify_one();
        } catch (const std::exception&) {
          // Let the test time out / fail on the client side assertions.
        }
      }

      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    listener_.reset();
    endpoint_.reset();
    worker_.reset();
    context_.reset();
  }

  void send_test_sequence() {
    // 1) Send a single DELTA_FRAME_CREATED for "sun" so the client creates it before the snapshot.
    DeltaMessage delta_msg = create_pose_tree_frame_delta(1, "sun");
    auto delta_buf = std::make_shared<std::vector<char>>(sizeof(delta_msg));
    std::memcpy(delta_buf->data(), &delta_msg, sizeof(delta_msg));
    ucxx::AmReceiverCallbackInfo delta_callback("AMClient", MSG_DELTA);
    auto delta_req = endpoint_->amSend(
        delta_buf->data(), delta_buf->size(), UCS_MEMORY_TYPE_HOST, delta_callback);
    wait_single_request(
        worker_, delta_req, std::chrono::milliseconds(2000), std::chrono::microseconds(10));

    // Give the client time to process the delta before sending the snapshot.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 2) Send a snapshot that includes:
    //    - frames: "sun"(id=1), "earth"(id=2)
    //    - edge: sun->earth at time=1.0
    std::vector<FrameInfo> frames;
    frames.reserve(2);
    FrameInfo sun_info{};
    sun_info.frame_id = 1;
    std::snprintf(sun_info.name, sizeof(sun_info.name), "%s", "sun");
    frames.push_back(sun_info);
    FrameInfo earth_info{};
    earth_info.frame_id = 2;
    std::snprintf(earth_info.name, sizeof(earth_info.name), "%s", "earth");
    frames.push_back(earth_info);

    std::vector<EdgeData> edges;
    edges.reserve(1);
    EdgeData edge{};
    edge.lhs_frame = 1;
    edge.rhs_frame = 2;
    edge.time = 1.0;
    serialize_pose3d(Pose3d::identity(), edge);
    edges.push_back(edge);

    auto snapshot_buf = std::make_shared<std::vector<char>>(serialize_snapshot(frames, edges));
    ucxx::AmReceiverCallbackInfo snapshot_callback("AMClient", MSG_SNAPSHOT_DATA);
    auto snapshot_req = endpoint_->amSend(
        snapshot_buf->data(), snapshot_buf->size(), UCS_MEMORY_TYPE_HOST, snapshot_callback);
    wait_single_request(worker_, snapshot_req, kSnapshotSendTimeout, std::chrono::microseconds(10));
  }

  const uint16_t port_;

  std::atomic<bool> running_{false};
  std::thread server_thread_;

  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Listener> listener_;
  // Note: endpoint_ is assigned in connection_callback (invoked by worker_->progress())
  // and read in run(), both on the server thread, so no additional synchronization needed.
  std::shared_ptr<ucxx::Endpoint> endpoint_;

  std::atomic<bool> got_subscribe_{false};
  std::atomic<bool> want_snapshot_{false};

  std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable sequence_sent_cv_;
  bool ready_{false};
  bool startup_failed_{false};
  std::string startup_error_;
  std::atomic<bool> sequence_sent_{false};
};

}  // namespace

TEST_F(DistributedApp, PoseTreeUcxSnapshotDeltaOrderingRepro) {
  const auto port = static_cast<uint16_t>(candidates_.at(0));

  UcxOrderingTestServer server(port);
  server.start();

  auto pose_tree = std::make_shared<PoseTree>();
  ASSERT_TRUE(pose_tree->init(256, 4096, 16384, 16, 1024, 4, 64));

  PoseTreeUCXClientConfig client_config;
  client_config.request_timeout_ms = 5000;
  client_config.request_poll_sleep_us = 10;
  client_config.worker_progress_sleep_us = 10;
  auto client = std::make_unique<PoseTreeUCXClient>(pose_tree, client_config);

  ASSERT_TRUE(client->connect("127.0.0.1", port, /*request_snapshot=*/true));
  ASSERT_TRUE(server.wait_for_sequence_sent(kSnapshotSendTimeout));

  // Wait for the snapshot to be applied on the client side.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pose_tree->find_frame("earth")) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(pose_tree->find_frame("sun"));
  ASSERT_TRUE(pose_tree->find_frame("earth"));

  // Expected behavior after fixing the snapshot/delta ordering robustness:
  // the edge in the snapshot should be applied, so the query should succeed.
  // Without the fix, the snapshot handler would fail to rebuild the remote_to_local_frame_id
  // mapping for "sun" (since it already exists from the delta), leaving the mapping incomplete
  // and causing edge lookups to fail (sun_to_earth_pose would be nullopt).
  auto sun_to_earth_pose = pose_tree->get("sun", "earth", 1.0);
  ASSERT_TRUE(sun_to_earth_pose.has_value());
  const auto& pose = sun_to_earth_pose.value();
  EXPECT_NEAR(pose.translation.norm(), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(pose.rotation.angle()), 0.0, 1e-12);

  // Disconnect client before stopping server to ensure clean shutdown order.
  client->disconnect();
  server.stop();
}

}  // namespace holoscan
