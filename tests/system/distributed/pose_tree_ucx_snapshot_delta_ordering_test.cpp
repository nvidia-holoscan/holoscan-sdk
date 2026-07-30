/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <holoscan/pose_tree/pose_tree_ucx_server.hpp>

#include "distributed_app_fixture.hpp"

namespace holoscan {

namespace {

constexpr auto kSnapshotSendTimeout = std::chrono::milliseconds(5000);
constexpr auto kDeltaPropagationTimeout = std::chrono::milliseconds(2000);
constexpr size_t kInitialSnapshotFrameCount = 2048;
constexpr uint64_t kInitialSnapshotFrameIdBase = 100000;

std::string make_snapshot_frame_name(size_t index) {
  char name[PoseTree::kFrameNameMaximumLength + 1];
  std::snprintf(name, sizeof(name), "snapshot_frame_%05zu", index);
  return std::string(name);
}

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

class UcxRawDeltaClient {
 public:
  UcxRawDeltaClient(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {}

  ~UcxRawDeltaClient() { disconnect(); }

  void connect() {
    if (endpoint_) {
      return;
    }

    context_ = ucxx::createContext({}, UCP_FEATURE_AM);
    worker_ = context_->createWorker();

    ucxx::AmReceiverCallbackInfo config_callback_info("AMClient", MSG_DISTRIBUTED_CONFIG);
    worker_->registerAmReceiverCallback(
        std::move(config_callback_info), [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
          try {
            req->checkError();
            auto buffer = req->getRecvBuffer();
            const auto* data = static_cast<const uint8_t*>(buffer->data());
            const auto size = buffer->getSize();
            if (size != sizeof(DistributedConfig)) {
              return;
            }
            std::memcpy(&config_, data, sizeof(config_));
            config_received_ = true;
          } catch (const std::exception&) {
            // Best-effort: waits will time out if config delivery fails.
          }
        });

    ucxx::AmReceiverCallbackInfo delta_callback_info("AMClient", MSG_DELTA);
    worker_->registerAmReceiverCallback(
        std::move(delta_callback_info), [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
          try {
            req->checkError();
            auto buffer = req->getRecvBuffer();
            const auto* data = static_cast<const uint8_t*>(buffer->data());
            const auto size = buffer->getSize();
            if (size != sizeof(DeltaMessage)) {
              return;
            }
            DeltaMessage delta_msg{};
            std::memcpy(&delta_msg, data, sizeof(delta_msg));
            switch (delta_msg.delta_type) {
              case DELTA_FRAME_CREATED:
                ++received_frame_count_;
                break;
              case DELTA_EDGE_SET:
                received_edge_pairs_.emplace_back(delta_msg.data.edge_data.lhs_frame,
                                                  delta_msg.data.edge_data.rhs_frame);
                break;
            }
          } catch (const std::exception&) {
            // Best-effort: waits will time out if delta delivery fails.
          }
        });

    ucxx::AmReceiverCallbackInfo close_callback_info("AMClient", MSG_CLOSE);
    worker_->registerAmReceiverCallback(std::move(close_callback_info),
                                        [](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
                                          try {
                                            req->checkError();
                                          } catch (const std::exception&) {
                                          }
                                        });

    endpoint_ = worker_->createEndpointFromHostname(host_, port_, true);

    SubscribeMessage subscribe_msg{};
    subscribe_msg.request_snapshot = 0;
    ucxx::AmReceiverCallbackInfo subscribe_callback_info("AMServer", MSG_SUBSCRIBE);
    auto subscribe_req = endpoint_->amSend(
        &subscribe_msg, sizeof(subscribe_msg), UCS_MEMORY_TYPE_HOST, subscribe_callback_info);
    wait_single_request(
        worker_, subscribe_req, std::chrono::milliseconds(2000), std::chrono::microseconds(10));

    if (!wait_for_config(std::chrono::milliseconds(2000))) {
      throw std::runtime_error("Raw UCX client did not receive distributed config");
    }
  }

  void disconnect() {
    if (!endpoint_) {
      worker_.reset();
      context_.reset();
      return;
    }

    try {
      uint8_t close_msg = 0;
      ucxx::AmReceiverCallbackInfo close_callback_info("AMServer", MSG_CLOSE);
      auto close_req = endpoint_->amSend(
          &close_msg, sizeof(close_msg), UCS_MEMORY_TYPE_HOST, close_callback_info);
      wait_single_request(
          worker_, close_req, std::chrono::milliseconds(500), std::chrono::microseconds(10));
    } catch (const std::exception&) {
      // Best-effort: shutdown should not fail the test.
    }

    endpoint_.reset();
    worker_.reset();
    context_.reset();
  }

  const DistributedConfig& config() const { return config_; }

  void send_frame_delta(uint64_t frame_id, const char* frame_name) {
    send_delta(create_pose_tree_frame_delta(frame_id, frame_name));
  }

  void send_edge_delta(uint64_t lhs_frame, uint64_t rhs_frame, double time, const Pose3d& pose) {
    EdgeData edge_data{};
    edge_data.lhs_frame = lhs_frame;
    edge_data.rhs_frame = rhs_frame;
    edge_data.time = time;
    serialize_pose3d(pose, edge_data);
    send_delta(create_pose_tree_edge_delta(edge_data));
  }

  bool wait_for_config(std::chrono::milliseconds timeout) {
    return progress_until(timeout, [this] { return config_received_; });
  }

  bool wait_for_remote_frame_count(size_t expected_count, std::chrono::milliseconds timeout) {
    return progress_until(
        timeout, [this, expected_count] { return received_frame_count_ >= expected_count; });
  }

  bool wait_for_remote_edge_count(size_t expected_count, std::chrono::milliseconds timeout) {
    return progress_until(
        timeout, [this, expected_count] { return received_edge_pairs_.size() >= expected_count; });
  }

  std::vector<std::pair<uint64_t, uint64_t>> received_edge_pairs() const {
    return received_edge_pairs_;
  }

 private:
  template <typename Predicate>
  bool progress_until(std::chrono::milliseconds timeout, Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      worker_->progress();
      if (predicate()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    worker_->progress();
    return predicate();
  }

  void send_delta(const DeltaMessage& delta_msg) {
    auto delta_buf = std::make_shared<std::vector<char>>(sizeof(delta_msg));
    std::memcpy(delta_buf->data(), &delta_msg, sizeof(delta_msg));
    ucxx::AmReceiverCallbackInfo delta_callback_info("AMServer", MSG_DELTA);
    auto delta_req = endpoint_->amSend(
        delta_buf->data(), delta_buf->size(), UCS_MEMORY_TYPE_HOST, delta_callback_info);
    wait_single_request(
        worker_, delta_req, std::chrono::milliseconds(2000), std::chrono::microseconds(10));
  }

  std::string host_;
  uint16_t port_;

  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Endpoint> endpoint_;

  DistributedConfig config_{};
  bool config_received_{false};
  size_t received_frame_count_{0};
  std::vector<std::pair<uint64_t, uint64_t>> received_edge_pairs_;
};

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

  mutable std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable sequence_sent_cv_;
  bool ready_{false};
  bool startup_failed_{false};
  std::string startup_error_;
  std::atomic<bool> sequence_sent_{false};
};

class UcxInitialSnapshotSyncTestServer {
 public:
  explicit UcxInitialSnapshotSyncTestServer(uint16_t port) : port_(port) {}

  ~UcxInitialSnapshotSyncTestServer() { stop(); }

  UcxInitialSnapshotSyncTestServer(const UcxInitialSnapshotSyncTestServer&) = delete;
  UcxInitialSnapshotSyncTestServer& operator=(const UcxInitialSnapshotSyncTestServer&) = delete;

  void start() {
    if (running_.exchange(true)) {
      return;
    }
    server_thread_ = std::thread(&UcxInitialSnapshotSyncTestServer::run, this);

    std::unique_lock<std::mutex> lk(mutex_);
    constexpr auto kReadyTimeout = std::chrono::seconds(5);
    if (!ready_cv_.wait_for(lk, kReadyTimeout, [this] { return ready_; })) {
      lk.unlock();
      stop();
      throw std::runtime_error("UCX snapshot sync server failed to become ready");
    }
    if (startup_failed_) {
      const auto err = startup_error_;
      lk.unlock();
      stop();
      throw std::runtime_error("UCX snapshot sync server startup failed: " + err);
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
    return state_cv_.wait_for(
        lk, timeout, [this] { return sequence_sent_.load() || sequence_failed_.load(); });
  }

  bool snapshot_ack_received() const { return snapshot_ack_received_.load(); }

  bool wait_for_snapshot_ack(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mutex_);
    return state_cv_.wait_for(lk, timeout, [this] { return snapshot_ack_received_.load(); });
  }

  bool wait_for_edge_result(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mutex_);
    return state_cv_.wait_for(
        lk, timeout, [this] { return edge_set_applied_.load() || edge_set_failed_.load(); });
  }

  bool edge_set_failed() const { return edge_set_failed_.load(); }

  bool wait_for_edge_results(size_t expected_count, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mutex_);
    return state_cv_.wait_for(
        lk, timeout, [this, expected_count] { return edge_result_count_ >= expected_count; });
  }

  size_t edge_failure_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return failed_edge_pairs_.size();
  }

  std::vector<std::pair<holoscan::PoseTree::frame_t, holoscan::PoseTree::frame_t>>
  failed_edge_pairs() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return failed_edge_pairs_;
  }

  std::shared_ptr<PoseTree> pose_tree() const { return pose_tree_; }

 private:
  static void connection_callback(ucp_conn_request_h req, void* arg) {
    auto* self = static_cast<UcxInitialSnapshotSyncTestServer*>(arg);
    try {
      auto ep = self->listener_->createEndpointFromConnRequest(req, true);
      self->endpoint_ = std::move(ep);
    } catch (const std::exception&) {
      // Best-effort: test will time out if connection cannot be established.
    }
  }

  void handle_delta_message(const DeltaMessage& delta_msg) {
    switch (delta_msg.delta_type) {
      case DELTA_FRAME_CREATED: {
        const auto frame_id = delta_msg.data.frame_data.frame_id;
        const std::string_view frame_name = delta_msg.data.frame_data.name;
        auto find_result = pose_tree_->find_frame(frame_name);
        if (!find_result.has_value()) {
          auto result = pose_tree_->create_frame_with_id(frame_id, frame_name);
          if (!result) {
            throw std::runtime_error("Failed to create frame from client delta");
          }
        }
        break;
      }
      case DELTA_EDGE_SET: {
        const auto& edge_data = delta_msg.data.edge_data;
        auto pose = deserialize_pose3d(edge_data);
        auto set_result =
            pose_tree_->set(edge_data.lhs_frame, edge_data.rhs_frame, edge_data.time, pose);
        if (set_result) {
          edge_set_applied_ = true;
        } else {
          edge_set_failed_ = true;
          std::lock_guard<std::mutex> lk(mutex_);
          failed_edge_pairs_.emplace_back(edge_data.lhs_frame, edge_data.rhs_frame);
        }
        ++edge_result_count_;
        state_cv_.notify_all();
        break;
      }
    }
  }

  void run() {
    try {
      pose_tree_ = std::make_shared<PoseTree>();
      if (!pose_tree_->init(32768, 131072, 262144, 16, 1024, 4, 64)) {
        throw std::runtime_error("Failed to initialize PoseTree for snapshot sync test server");
      }

      context_ = ucxx::createContext({}, UCP_FEATURE_AM);
      worker_ = context_->createWorker();
      listener_ = worker_->createListener(port_, connection_callback, this);

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

      ucxx::AmReceiverCallbackInfo delta_callback_info("AMServer", MSG_DELTA);
      worker_->registerAmReceiverCallback(
          std::move(delta_callback_info), [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
            try {
              req->checkError();
              auto buffer = req->getRecvBuffer();
              const auto* data = static_cast<const uint8_t*>(buffer->data());
              const auto size = buffer->getSize();
              if (size != sizeof(DeltaMessage)) {
                return;
              }
              DeltaMessage delta_msg{};
              std::memcpy(&delta_msg, data, sizeof(delta_msg));
              handle_delta_message(delta_msg);
            } catch (const std::exception&) {
              // Best-effort: test will fail on missing edge propagation.
            }
          });

      ucxx::AmReceiverCallbackInfo close_callback_info("AMServer", MSG_CLOSE);
      worker_->registerAmReceiverCallback(std::move(close_callback_info),
                                          [](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
                                            try {
                                              req->checkError();
                                            } catch (const std::exception&) {
                                            }
                                          });

      ucxx::AmReceiverCallbackInfo snapshot_ack_info("AMServer", MSG_SNAPSHOT_ACK);
      worker_->registerAmReceiverCallback(std::move(snapshot_ack_info),
                                          [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
                                            try {
                                              req->checkError();
                                              snapshot_ack_received_ = true;
                                              state_cv_.notify_all();
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

      if (!sequence_attempted_.load() && endpoint_ && got_subscribe_.load()) {
        sequence_attempted_ = true;
        try {
          send_test_sequence();
          {
            std::lock_guard<std::mutex> lk(mutex_);
            sequence_sent_ = true;
          }
          state_cv_.notify_all();
        } catch (const std::exception&) {
          {
            std::lock_guard<std::mutex> lk(mutex_);
            sequence_failed_ = true;
          }
          state_cv_.notify_all();
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
    DistributedConfig config{};
    config.start_frame_id = 1;
    config.increment = 1024;

    auto config_buf = std::make_shared<std::vector<char>>(sizeof(config));
    std::memcpy(config_buf->data(), &config, sizeof(config));
    ucxx::AmReceiverCallbackInfo config_callback("AMClient", MSG_DISTRIBUTED_CONFIG);
    auto config_req = endpoint_->amSend(
        config_buf->data(), config_buf->size(), UCS_MEMORY_TYPE_HOST, config_callback);
    wait_single_request(
        worker_, config_req, std::chrono::milliseconds(2000), std::chrono::microseconds(10));

    std::vector<FrameInfo> frames;
    frames.reserve(kInitialSnapshotFrameCount);
    for (size_t i = 0; i < kInitialSnapshotFrameCount; ++i) {
      FrameInfo frame{};
      frame.frame_id = kInitialSnapshotFrameIdBase + i;
      const auto frame_name = make_snapshot_frame_name(i);
      std::snprintf(frame.name, sizeof(frame.name), "%s", frame_name.c_str());
      frames.push_back(frame);
    }

    auto snapshot_buf = std::make_shared<std::vector<char>>(serialize_snapshot(frames, {}));
    ucxx::AmReceiverCallbackInfo snapshot_callback("AMClient", MSG_SNAPSHOT_DATA);
    auto snapshot_req = endpoint_->amSend(
        snapshot_buf->data(), snapshot_buf->size(), UCS_MEMORY_TYPE_HOST, snapshot_callback);
    wait_single_request(worker_, snapshot_req, kSnapshotSendTimeout, std::chrono::microseconds(10));
  }

  const uint16_t port_;

  std::atomic<bool> running_{false};
  std::thread server_thread_;

  std::shared_ptr<PoseTree> pose_tree_;
  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Listener> listener_;
  std::shared_ptr<ucxx::Endpoint> endpoint_;

  std::atomic<bool> got_subscribe_{false};
  std::atomic<bool> want_snapshot_{false};
  std::atomic<bool> sequence_attempted_{false};
  std::atomic<bool> sequence_sent_{false};
  std::atomic<bool> sequence_failed_{false};
  std::atomic<bool> snapshot_ack_received_{false};
  std::atomic<bool> edge_set_applied_{false};
  std::atomic<bool> edge_set_failed_{false};
  std::atomic<size_t> edge_result_count_{0};

  mutable std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable state_cv_;
  bool ready_{false};
  bool startup_failed_{false};
  std::string startup_error_;
  std::vector<std::pair<holoscan::PoseTree::frame_t, holoscan::PoseTree::frame_t>>
      failed_edge_pairs_;
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

TEST_F(DistributedApp, PoseTreeUcxServerTranslatesClientFrameIdsAcrossSessions) {
  const auto port = static_cast<uint16_t>(candidates_.at(0));

  auto server_pose_tree = std::make_shared<PoseTree>();
  ASSERT_TRUE(server_pose_tree->init(32768, 131072, 262144, 16, 1024, 4, 64));

  PoseTreeUCXServerConfig server_config;
  server_config.worker_progress_sleep_us = 10;
  server_config.maximum_clients = 1024;
  PoseTreeUCXServer server(server_pose_tree, server_config);
  ASSERT_TRUE(server.start(port));

  UcxRawDeltaClient raw_client("127.0.0.1", port);
  ASSERT_NO_THROW(raw_client.connect());
  EXPECT_EQ(raw_client.config().start_frame_id, 1U);
  EXPECT_EQ(raw_client.config().increment, 1024U);

  UcxRawDeltaClient priming_client("127.0.0.1", port);
  ASSERT_NO_THROW(priming_client.connect());
  EXPECT_EQ(priming_client.config().start_frame_id, 2U);
  EXPECT_EQ(priming_client.config().increment, 1024U);

  const uint64_t sun_server_id = priming_client.config().start_frame_id;
  const uint64_t earth_server_id = sun_server_id + priming_client.config().increment;
  const uint64_t moon_server_id = earth_server_id + priming_client.config().increment;

  priming_client.send_frame_delta(sun_server_id, "sun");
  priming_client.send_frame_delta(earth_server_id, "earth");
  priming_client.send_frame_delta(moon_server_id, "moon");
  ASSERT_TRUE(raw_client.wait_for_remote_frame_count(3, kDeltaPropagationTimeout));

  UcxRawDeltaClient observing_client("127.0.0.1", port);
  ASSERT_NO_THROW(observing_client.connect());
  EXPECT_EQ(observing_client.config().start_frame_id, 3U);
  EXPECT_EQ(observing_client.config().increment, 1024U);

  testing::internal::CaptureStderr();

  raw_client.send_frame_delta(1, "sun");
  raw_client.send_frame_delta(1025, "earth");
  raw_client.send_frame_delta(2049, "moon");
  raw_client.send_edge_delta(1, 1025, 1.0, Pose3d::identity());
  raw_client.send_edge_delta(1025, 2049, 1.0, Pose3d::identity());

  EXPECT_TRUE(observing_client.wait_for_remote_edge_count(2, kDeltaPropagationTimeout));

  const auto received_edge_pairs = observing_client.received_edge_pairs();
  EXPECT_GE(received_edge_pairs.size(), 2U);
  if (received_edge_pairs.size() >= 2U) {
    EXPECT_EQ(received_edge_pairs[0], std::make_pair(sun_server_id, earth_server_id));
    EXPECT_EQ(received_edge_pairs[1], std::make_pair(earth_server_id, moon_server_id));
  }

  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(log_output.find("PoseTreeUCXServer: set failed"), std::string::npos) << log_output;
  EXPECT_EQ(log_output.find("Pose frame UID 1 or UID 1025 not found"), std::string::npos)
      << log_output;
  EXPECT_EQ(log_output.find("Pose frame UID 1025 or UID 2049 not found"), std::string::npos)
      << log_output;

  observing_client.disconnect();
  priming_client.disconnect();
  raw_client.disconnect();
  ASSERT_TRUE(server.stop());
}

TEST_F(DistributedApp, PoseTreeUcxConnectWaitsForInitialSnapshotBeforeLocalCallbacks) {
  const auto port = static_cast<uint16_t>(candidates_.at(0));

  UcxInitialSnapshotSyncTestServer server(port);
  server.start();

  auto pose_tree = std::make_shared<PoseTree>();
  ASSERT_TRUE(pose_tree->init(32768, 131072, 262144, 16, 1024, 4, 64));

  PoseTreeUCXClientConfig client_config;
  client_config.request_timeout_ms = 5000;
  client_config.request_poll_sleep_us = 10;
  client_config.worker_progress_sleep_us = 10;
  auto client = std::make_unique<PoseTreeUCXClient>(pose_tree, client_config);

  ASSERT_TRUE(client->connect("127.0.0.1", port, /*request_snapshot=*/true));
  ASSERT_TRUE(server.wait_for_sequence_sent(kSnapshotSendTimeout));

  const auto last_snapshot_frame_name = make_snapshot_frame_name(kInitialSnapshotFrameCount - 1);

  EXPECT_TRUE(pose_tree->find_frame(last_snapshot_frame_name))
      << "connect() returned before the last snapshot frame was installed locally";

  ASSERT_TRUE(pose_tree->create_frame("sun"));
  ASSERT_TRUE(pose_tree->create_frame("earth"));
  ASSERT_TRUE(pose_tree->create_frame("moon"));

  ASSERT_TRUE(server.wait_for_snapshot_ack(kSnapshotSendTimeout));

  ASSERT_TRUE(pose_tree->set("sun", "earth", 1.0, Pose3d::identity()));
  ASSERT_TRUE(pose_tree->set("earth", "moon", 1.0, Pose3d::identity()));
  ASSERT_TRUE(server.wait_for_edge_results(2, kSnapshotSendTimeout));
  EXPECT_FALSE(server.edge_set_failed());
  EXPECT_EQ(server.edge_failure_count(), 0U);

  auto sun_to_earth_pose = server.pose_tree()->get("sun", "earth", 1.0);
  ASSERT_TRUE(sun_to_earth_pose.has_value());
  EXPECT_NEAR(sun_to_earth_pose.value().translation.norm(), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(sun_to_earth_pose.value().rotation.angle()), 0.0, 1e-12);

  auto earth_to_moon_pose = server.pose_tree()->get("earth", "moon", 1.0);
  ASSERT_TRUE(earth_to_moon_pose.has_value());
  EXPECT_NEAR(earth_to_moon_pose.value().translation.norm(), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(earth_to_moon_pose.value().rotation.angle()), 0.0, 1e-12);

  client->disconnect();
  server.stop();
}

}  // namespace holoscan
