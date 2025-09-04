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

#include "holoscan/pose_tree/pose_tree_ucx_server.hpp"

#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <ucxx/api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "holoscan/logger/logger.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_common.hpp"

namespace holoscan {

struct PoseTreeUCXServer::ServerImpl {
  std::shared_ptr<ucxx::Context> context;
  std::shared_ptr<ucxx::Worker> worker;
  std::shared_ptr<ucxx::Listener> listener;
  std::shared_ptr<holoscan::PoseTree> pose_tree;
  PoseTree::InitParameters pose_tree_init_params;

  struct ClientSession;
  std::list<std::unique_ptr<ClientSession>> clients;
  std::mutex clients_mtx;
  int next_client_id{0};
  int64_t maximum_clients_;

  struct PendingRequest {
    std::shared_ptr<ucxx::Request> request;
    int client_id;
    std::shared_ptr<std::vector<char>> buffer;
    std::shared_ptr<ucxx::Endpoint> endpoint;

    PendingRequest(std::shared_ptr<ucxx::Request> req, int id,
                   std::shared_ptr<std::vector<char>> buf = nullptr,
                   std::shared_ptr<ucxx::Endpoint> ep = nullptr)
        : request(std::move(req)), client_id(id), buffer(std::move(buf)), endpoint(std::move(ep)) {}
  };
  std::list<PendingRequest> pending_requests;
  std::mutex pending_requests_mtx;

  // Internal version that assumes pending_requests_mtx is already locked
  void addPendingRequestLocked(std::shared_ptr<ucxx::Request> request, int client_id,
                               std::shared_ptr<std::vector<char>> buffer = nullptr,
                               std::shared_ptr<ucxx::Endpoint> endpoint = nullptr) {
    pending_requests.emplace_back(
        std::move(request), client_id, std::move(buffer), std::move(endpoint));
  }

  void addPendingRequest(std::shared_ptr<ucxx::Request> request, int client_id,
                         std::shared_ptr<std::vector<char>> buffer = nullptr,
                         std::shared_ptr<ucxx::Endpoint> endpoint = nullptr) {
    std::lock_guard<std::mutex> lk(pending_requests_mtx);
    addPendingRequestLocked(std::move(request), client_id, std::move(buffer), std::move(endpoint));
  }

  void removeClients(const std::vector<int>& failed_client_ids) {
    if (failed_client_ids.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lk(clients_mtx);
    for (int failed_id : failed_client_ids) {
      auto client_it = std::find_if(clients.begin(), clients.end(), [failed_id](const auto& c) {
        return c->client_id_ == failed_id;
      });
      if (client_it != clients.end()) {
        (*client_it)->disconnected_ = true;
        clients.erase(client_it);
      }
    }
  }

  void processCompletedRequests() {
    // Lock both mutexes at the beginning to maintain consistent lock ordering
    std::scoped_lock lock(clients_mtx, pending_requests_mtx);

    auto it = pending_requests.begin();
    while (it != pending_requests.end()) {
      if (it->request->isCompleted()) {
        try {
          it->request->checkError();
        } catch (const ucxx::ConnectionResetError&) {
          HOLOSCAN_LOG_ERROR("PoseTreeUCXServer: Connection reset error on client {}",
                             it->client_id);
          if (it->client_id != -1) {
            // Remove client directly since we already hold the clients_mtx
            auto client_it = std::find_if(
                clients.begin(), clients.end(), [client_id = it->client_id](const auto& c) {
                  return c->client_id_ == client_id;
                });
            if (client_it != clients.end()) {
              (*client_it)->disconnected_ = true;
              clients.erase(client_it);
            }
          }
        } catch (const std::exception&) {
          HOLOSCAN_LOG_ERROR("PoseTreeUCXServer: Exception on client {}", it->client_id);
          if (it->client_id != -1) {
            // Remove client directly since we already hold the clients_mtx
            auto client_it = std::find_if(
                clients.begin(), clients.end(), [client_id = it->client_id](const auto& c) {
                  return c->client_id_ == client_id;
                });
            if (client_it != clients.end()) {
              (*client_it)->disconnected_ = true;
              clients.erase(client_it);
            }
          }
        }
        it = pending_requests.erase(it);
      } else {
        ++it;
      }
    }
  }
  void broadcast_frame_created(holoscan::PoseTree::frame_t frame_id, const std::string& frame_name,
                               ClientSession* origin_session);
  void broadcast_edge_set(holoscan::PoseTree::frame_t lhs, holoscan::PoseTree::frame_t rhs,
                          double time, const holoscan::Pose3d& lhs_T_rhs,
                          ClientSession* origin_session);

  struct ClientSession {
    ClientSession(std::shared_ptr<ucxx::Endpoint> ep, ServerImpl* server_impl, int client_id);
    void handle_subscribe(bool want_snap);
    void handle_delta_message(const DeltaMessage& delta_msg);
    void send_snapshot();
    void send_config();

    std::shared_ptr<ucxx::Endpoint> ep_;
    ServerImpl* server_impl_;
    int client_id_;
    std::atomic<bool> disconnected_{false};
  };

  ClientSession* find_client_session(ucp_ep_h ep) {
    std::lock_guard<std::mutex> lk(clients_mtx);
    for (auto& client : clients) {
      if (client->ep_->getHandle() == ep) {
        return client.get();
      }
    }
    return nullptr;
  }
};

// NOLINTBEGIN(whitespace/indent_namespace)
PoseTreeUCXServer::ServerImpl::ClientSession::ClientSession(
    std::shared_ptr<ucxx::Endpoint> ep, PoseTreeUCXServer::ServerImpl* server_impl, int client_id)
    : ep_{std::move(ep)}, server_impl_{server_impl}, client_id_{client_id} {}
// NOLINTEND(whitespace/indent_namespace)

void PoseTreeUCXServer::ServerImpl::ClientSession::handle_subscribe(bool want_snap) {
  send_config();
  if (want_snap) {
    send_snapshot();
  }
}

void PoseTreeUCXServer::ServerImpl::ClientSession::handle_delta_message(
    const DeltaMessage& delta_msg) {
  std::string frame_name_str;
  holoscan::PoseTree::frame_t frame_id;
  bool frame_created = false;

  holoscan::PoseTree::frame_t lhs, rhs;
  double time;
  holoscan::Pose3d pose;
  bool edge_set = false;

  // client: Update the server's PoseTree instance.
  HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: handle_delta_message: delta_msg.delta_type = {}",
                     static_cast<int>(delta_msg.delta_type));
  switch (delta_msg.delta_type) {
    case DELTA_FRAME_CREATED: {
      frame_name_str = delta_msg.data.frame_data.name;
      frame_id = delta_msg.data.frame_data.frame_id;

      auto find_result = server_impl_->pose_tree->find_frame(frame_name_str);
      if (!find_result.has_value()) {
        auto result = server_impl_->pose_tree->create_frame_with_id(frame_id, frame_name_str);
        if (result) {
          frame_id = result.value();
          frame_created = true;
        }
      }
      break;
    }
    case DELTA_EDGE_SET: {
      const auto& edge_data = delta_msg.data.edge_data;
      pose = deserialize_pose3d(edge_data);
      auto set_result = server_impl_->pose_tree->set(
          edge_data.lhs_frame, edge_data.rhs_frame, edge_data.time, pose);
      if (set_result) {
        lhs = edge_data.lhs_frame;
        rhs = edge_data.rhs_frame;
        time = edge_data.time;
        edge_set = true;
      } else {
        HOLOSCAN_LOG_ERROR("PoseTreeUCXServer: set failed");
      }
      break;
    }
    default:
      break;
  }

  if (frame_created) {
    server_impl_->broadcast_frame_created(frame_id, frame_name_str, this);
  }
  if (edge_set) {
    server_impl_->broadcast_edge_set(lhs, rhs, time, pose, this);
  }
}

void PoseTreeUCXServer::ServerImpl::ClientSession::send_snapshot() {
  std::vector<FrameInfo> frames;
  std::vector<EdgeData> edges;
  std::vector<holoscan::PoseTree::frame_t> frame_uids;
  frame_uids.reserve(server_impl_->pose_tree_init_params.number_frames);
  HOLOSCAN_LOG_DEBUG("send_snapshot: pose_tree_ = 0x{:x}",
                     reinterpret_cast<uint64_t>(server_impl_->pose_tree.get()));
  if (server_impl_->pose_tree->get_frame_uids(frame_uids)) {
    for (auto frame_id : frame_uids) {
      if (auto name_result = server_impl_->pose_tree->get_frame_name(frame_id)) {
        FrameInfo info;
        info.frame_id = frame_id;
        std::strncpy(info.name, std::string(name_result.value()).c_str(), sizeof(info.name) - 1);
        info.name[sizeof(info.name) - 1] = '\0';
        frames.push_back(info);
      }
    }
  }
  std::vector<std::pair<holoscan::PoseTree::frame_t, holoscan::PoseTree::frame_t>> edge_uids;
  edge_uids.reserve(server_impl_->pose_tree_init_params.number_edges * 2);
  if (server_impl_->pose_tree->get_edge_uids(edge_uids)) {
    for (const auto& edge_pair : edge_uids) {
      if (auto latest_result =
              server_impl_->pose_tree->get_latest(edge_pair.first, edge_pair.second)) {
        EdgeData info;
        info.lhs_frame = edge_pair.first;
        info.rhs_frame = edge_pair.second;
        info.time = latest_result.value().second;
        serialize_pose3d(latest_result.value().first, info);
        edges.push_back(info);
      }
    }
  }

  auto buf = std::make_shared<std::vector<char>>(serialize_snapshot(frames, edges));
  ucxx::AmReceiverCallbackInfo callbackInfo("AMClient", MSG_SNAPSHOT_DATA);
  auto request = ep_->amSend(buf->data(), buf->size(), UCS_MEMORY_TYPE_HOST, callbackInfo);
  server_impl_->addPendingRequest(std::move(request), client_id_, buf, ep_);
  HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: send_snapshot: frames.size() = {}", frames.size());
  HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: send_snapshot: edges.size() = {}", edges.size());
}

void PoseTreeUCXServer::ServerImpl::ClientSession::send_config() {
  DistributedConfig msg;
  msg.start_frame_id = client_id_ + 1;
  msg.increment = server_impl_->maximum_clients_;

  auto msg_buffer = std::make_shared<std::vector<char>>(sizeof(msg));
  std::memcpy(msg_buffer->data(), &msg, sizeof(msg));

  ucxx::AmReceiverCallbackInfo callbackInfo("AMClient", MSG_DISTRIBUTED_CONFIG);
  auto request =
      ep_->amSend(msg_buffer->data(), msg_buffer->size(), UCS_MEMORY_TYPE_HOST, callbackInfo);
  server_impl_->addPendingRequest(std::move(request), client_id_, msg_buffer, ep_);
}

void PoseTreeUCXServer::ServerImpl::broadcast_frame_created(holoscan::PoseTree::frame_t frame_id,
                                                            const std::string& frame_name,
                                                            ClientSession* origin_session) {
  HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: Broadcasting frame created: {}", frame_name);

  // Use the helper method for safe initialization
  DeltaMessage delta_msg = create_pose_tree_frame_delta(frame_id, frame_name.c_str());

  auto msg_buffer = std::make_shared<std::vector<char>>(sizeof(delta_msg));
  std::memcpy(msg_buffer->data(), &delta_msg, sizeof(delta_msg));

  ucxx::AmReceiverCallbackInfo callbackInfo("AMClient", MSG_DELTA);
  std::lock_guard<std::mutex> lk(clients_mtx);
  for (auto& client : clients) {
    if (client.get() == origin_session || client->disconnected_) {
      continue;
    }
    HOLOSCAN_LOG_TRACE("client.get()={}, origin_session={}, disconnected={}",
                       client.get()->client_id_,
                       origin_session->client_id_,
                       client->disconnected_.load());
    HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: Broadcasting frame created to client {}",
                       client->client_id_);
    auto request = client->ep_->amSend(
        msg_buffer->data(), msg_buffer->size(), UCS_MEMORY_TYPE_HOST, callbackInfo);
    addPendingRequest(std::move(request), client->client_id_, msg_buffer, client->ep_);
  }
}

void PoseTreeUCXServer::ServerImpl::broadcast_edge_set(holoscan::PoseTree::frame_t lhs,
                                                       holoscan::PoseTree::frame_t rhs, double time,
                                                       const holoscan::Pose3d& lhs_T_rhs,
                                                       ClientSession* origin_session) {
  HOLOSCAN_LOG_TRACE(
      "PoseTreeUCXServer: Broadcasting edge set from {} to {} at time {}", lhs, rhs, time);

  // Create EdgeData first, then use helper method
  EdgeData edge_data;
  edge_data.lhs_frame = lhs;
  edge_data.rhs_frame = rhs;
  edge_data.time = time;
  serialize_pose3d(lhs_T_rhs, edge_data);

  DeltaMessage delta_msg = create_pose_tree_edge_delta(edge_data);

  auto msg_buffer = std::make_shared<std::vector<char>>(sizeof(delta_msg));
  std::memcpy(msg_buffer->data(), &delta_msg, sizeof(delta_msg));

  ucxx::AmReceiverCallbackInfo callbackInfo("AMClient", MSG_DELTA);
  std::lock_guard<std::mutex> lk(clients_mtx);
  for (auto& client : clients) {
    if (client.get() == origin_session || client->disconnected_) {
      continue;
    }
    HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: Broadcasting edge set to client {}", client->client_id_);
    auto request = client->ep_->amSend(
        msg_buffer->data(), msg_buffer->size(), UCS_MEMORY_TYPE_HOST, callbackInfo);
    addPendingRequest(std::move(request), client->client_id_, msg_buffer, client->ep_);
  }
}

PoseTreeUCXServer::PoseTreeUCXServer(std::shared_ptr<PoseTree> pose_tree,
                                     PoseTreeUCXServerConfig config)
    : impl_(std::make_unique<ServerImpl>()),
      pose_tree_(std::move(pose_tree)),
      config_(std::move(config)) {
  if (!pose_tree_) {
    throw std::runtime_error("PoseTree pointer is null.");
  }
  auto params = pose_tree_->get_init_parameters();
  if (!params) {
    throw std::runtime_error("PoseTree must be initialized before creating PoseTreeUCXServer.");
  }
  pose_tree_init_params_ = params.value();
  // Create new PoseTree object to avoid shared object with worker
  // TODO(gbae): This is a hack to avoid shared object with worker. We should find a better way to
  // do this.
  pose_tree_ = std::make_shared<PoseTree>();
  pose_tree_->init(pose_tree_init_params_.number_frames,
                   pose_tree_init_params_.number_edges,
                   pose_tree_init_params_.history_length,
                   pose_tree_init_params_.default_number_edges,
                   pose_tree_init_params_.default_history_length,
                   pose_tree_init_params_.edges_chunk_size,
                   pose_tree_init_params_.history_chunk_size);
  impl_->pose_tree = pose_tree_;
  impl_->pose_tree_init_params = pose_tree_init_params_;
  impl_->maximum_clients_ = config_.maximum_clients;
}
PoseTreeUCXServer::~PoseTreeUCXServer() {
  if (running_) {  // best-effort safety net
    try {
      auto result = stop();
      if (!result) {
        HOLOSCAN_LOG_ERROR("PoseTreeUCXServer destructor: stop failed: {}",
                           error_to_str(result.error()));
      }
    } catch (...) {
      // swallow – nothrow dtor
    }
  }
}

PoseTreeUCXServer::expected<void> PoseTreeUCXServer::start(uint16_t port) {
  if (running_) {
    return unexpected(Error::kAlreadyRunning);
  }

  if (port == 0) {
    return unexpected(Error::kInvalidArgument);
  }

  port_ = port;
  running_ = true;
  ready_ = false;

  try {
    server_thread_ = std::thread(&PoseTreeUCXServer::run, this);
  } catch (const std::exception& e) {
    running_ = false;
    HOLOSCAN_LOG_ERROR("Failed to start server thread: {}", e.what());
    return unexpected(Error::kStartupFailed);
  }

  std::unique_lock<std::mutex> lk(ready_mutex_);
  ready_cv_.wait(lk, [this] { return ready_.load(); });

  return {};
}

PoseTreeUCXServer::expected<void> PoseTreeUCXServer::stop() {
  if (!running_) {
    return {};  // Already stopped, not an error
  }

  running_ = false;

  if (server_thread_.joinable()) {
    try {
      server_thread_.join();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Failed to join server thread: {}", e.what());
      return unexpected(Error::kInternalError);
    }
  }

  return {};
}

const char* PoseTreeUCXServer::error_to_str(Error error) {
  switch (error) {
    case Error::kAlreadyRunning:
      return "Server is already running";
    case Error::kInvalidArgument:
      return "Invalid argument provided";
    case Error::kStartupFailed:
      return "Failed to start server";
    case Error::kNotRunning:
      return "Server is not running";
    case Error::kShutdownTimeout:
      return "Server shutdown timed out";
    case Error::kInternalError:
      return "Internal error";
    default:
      return "Unknown error";
  }
}

void connection_callback(ucp_conn_request_h req, void* arg) {
  auto server = static_cast<PoseTreeUCXServer*>(arg);
  auto ep = server->impl_->listener->createEndpointFromConnRequest(req, true);

  std::lock_guard<std::mutex> lk(server->impl_->clients_mtx);
  int client_id = server->impl_->next_client_id++;  // Now protected by mutex
  server->impl_->clients.emplace_back(
      std::make_unique<PoseTreeUCXServer::ServerImpl::ClientSession>(
          std::move(ep), server->impl_.get(), client_id));
  HOLOSCAN_LOG_DEBUG("New client connected: {}", client_id);
}

void PoseTreeUCXServer::run() {
  impl_->context = ucxx::createContext({}, UCP_FEATURE_AM);
  impl_->worker = impl_->context->createWorker();

  impl_->listener = impl_->worker->createListener(port_, connection_callback, this);

  auto am_receiver = [this](
                         std::shared_ptr<ucxx::Request> req, ucp_ep_h sender_ep, uint16_t msg_id) {
    if (!running_) {
      return;
    }
    try {
      req->checkError();

      auto* client_session = impl_->find_client_session(sender_ep);
      if (!client_session) {
        return;
      }

      if (msg_id == MSG_CLOSE) {
        HOLOSCAN_LOG_DEBUG("PoseTreeUCXServer: Client {} disconnected", client_session->client_id_);
        impl_->removeClients({client_session->client_id_});
        return;
      }

      auto buffer = req->getRecvBuffer();
      auto data = static_cast<const uint8_t*>(buffer->data());
      auto size = buffer->getSize();
      if (size == 0) {
        return;
      }
      HOLOSCAN_LOG_TRACE("PoseTreeUCXServer: Received message: msg_id = {} (client_id={})",
                         msg_id,
                         client_session->client_id_);

      switch (msg_id) {
        case MSG_SUBSCRIBE: {
          if (size == sizeof(SubscribeMessage)) {
            SubscribeMessage sub_msg;
            std::memcpy(&sub_msg, data, sizeof(SubscribeMessage));
            client_session->handle_subscribe(sub_msg.request_snapshot != 0);
          }
          break;
        }
        case MSG_DELTA: {
          if (size == sizeof(DeltaMessage)) {
            DeltaMessage delta_msg;
            std::memcpy(&delta_msg, data, sizeof(DeltaMessage));
            client_session->handle_delta_message(delta_msg);
          }
          break;
        }
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("PoseTreeUCXServer: AM receiver threw: {}", e.what());
    }
  };

  ucxx::AmReceiverCallbackInfo subscribe_callback_info("AMServer", MSG_SUBSCRIBE);
  impl_->worker->registerAmReceiverCallback(
      std::move(subscribe_callback_info),
      [am_receiver](auto req, auto ep) { am_receiver(std::move(req), ep, MSG_SUBSCRIBE); });
  ucxx::AmReceiverCallbackInfo delta_callback_info("AMServer", MSG_DELTA);
  impl_->worker->registerAmReceiverCallback(
      std::move(delta_callback_info),
      [am_receiver](auto req, auto ep) { am_receiver(std::move(req), ep, MSG_DELTA); });
  ucxx::AmReceiverCallbackInfo close_callback_info("AMServer", MSG_CLOSE);
  impl_->worker->registerAmReceiverCallback(
      std::move(close_callback_info),
      [am_receiver](auto req, auto ep) { am_receiver(std::move(req), ep, MSG_CLOSE); });

  {
    std::lock_guard<std::mutex> lk(ready_mutex_);
    ready_ = true;
  }
  ready_cv_.notify_one();

  while (running_) {
    impl_->worker->progress();
    impl_->processCompletedRequests();
    std::this_thread::sleep_for(std::chrono::microseconds(config_.worker_progress_sleep_us));
  }

  {
    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    if (!impl_->clients.empty()) {
      HOLOSCAN_LOG_DEBUG("PoseTreeUCXServer: Sending shutdown to {} clients...",
                         impl_->clients.size());
      ucxx::AmReceiverCallbackInfo close_info("AMClient", MSG_CLOSE);
      for (auto& client : impl_->clients) {
        if (!client->disconnected_) {
          auto request = client->ep_->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST, close_info);
          impl_->addPendingRequest(std::move(request), client->client_id_, nullptr, client->ep_);
        }
      }
    }
  }

  auto shutdown_start_time = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - shutdown_start_time <
         std::chrono::milliseconds(config_.shutdown_timeout_ms)) {
    impl_->worker->progress();
    impl_->processCompletedRequests();
    bool pending;
    {
      std::lock_guard<std::mutex> lk(impl_->pending_requests_mtx);
      pending = !impl_->pending_requests.empty();
    }
    if (!pending) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.shutdown_poll_sleep_ms));
  }

  impl_->listener.reset();
  {
    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    impl_->clients.clear();
  }
}

}  // namespace holoscan
