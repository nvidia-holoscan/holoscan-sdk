/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/conditions/gxf/cuda_stream.hpp>

#include <cuda_runtime.h>
#include <gxf/core/gxf.h>

#include <string>
#include <vector>

#include <gxf/cuda/cuda_stream.hpp>
#include <gxf/cuda/cuda_stream_id.hpp>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/executor.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

void CudaStreamCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Receiver",
             "(Legacy API) Single receiver port to monitor for CUDA streams. "
             "Cannot be used together with 'receivers'.",
             ParameterFlag::kOptional);
  spec.param(receivers_,
             "receivers",
             "Receivers",
             "Receiver port(s) to monitor for CUDA streams. Works with both regular ports "
             "(by exact name) and multi-receiver ports (IOSpec::kAnySize, by base name). "
             "Cannot be used together with 'receiver'.",
             ParameterFlag::kOptional);
  spec.param(check_all_messages_,
             "check_all_messages",
             "Check All Messages",
             "If true, checks ALL messages in the queue(s) for CudaStreamId components "
             "and waits for all associated streams. If false, only checks the first message "
             "per receiver.",
             true);
}

void CudaStreamCondition::initialize() {
  // Call parent initialize to register parameters
  Condition::initialize();

  // Validate that exactly one of 'receiver' or 'receivers' is specified
  bool has_receiver = receiver_.has_value();
  bool has_receivers = receivers_.has_value();

  if (has_receiver && has_receivers) {
    throw std::runtime_error(
        "CudaStreamCondition: cannot specify both 'receiver' and 'receivers' parameters. "
        "Use 'receiver' for a single port (legacy API) or 'receivers' for one or more ports.");
  }

  if (!has_receiver && !has_receivers) {
    throw std::runtime_error(
        "CudaStreamCondition: must specify either 'receiver' or 'receivers' parameter. "
        "Use 'receiver' for a single port (legacy API) or 'receivers' for one or more ports.");
  }

  // Log deprecation warning if legacy 'receiver' parameter is used (only once)
  if (has_receiver) {
    static bool deprecation_warning_logged = false;
    if (!deprecation_warning_logged) {
      HOLOSCAN_LOG_WARN(
          "CudaStreamCondition: the 'receiver' parameter is deprecated. "
          "Please use 'receivers' instead (e.g., Arg(\"receivers\", \"port_name\") in C++ "
          "or receivers=\"port_name\" in Python).");
      deprecation_warning_logged = true;
    }
  }
}

void* CudaStreamCondition::get_gxf_context() const {
  if (fragment_ == nullptr) {
    HOLOSCAN_LOG_ERROR("CudaStreamCondition: fragment is null");
    return nullptr;
  }
  return fragment_->executor().context();
}

void CUDART_CB CudaStreamCondition::cuda_host_callback(void* user_data) {
  auto* data = reinterpret_cast<CallbackData*>(user_data);
  auto* self = data->condition;

  // Decrement pending count atomically
  size_t remaining = self->pending_callbacks_.fetch_sub(1) - 1;

  if (remaining == 0) {
    // All callbacks have fired - transition to DATA_AVAILABLE
    auto expected = State::CALLBACKS_REGISTERED;
    if (self->state_.compare_exchange_strong(expected, State::DATA_AVAILABLE)) {
      // Notify the scheduler that this condition has changed
      self->notify_scheduler();
    }
  }

  delete data;
}

std::vector<std::shared_ptr<Receiver>> CudaStreamCondition::get_all_receivers() const {
  std::vector<std::shared_ptr<Receiver>> result;

  // Check legacy 'receiver' parameter first
  if (receiver_.has_value()) {
    auto recv = receiver_.try_get();
    if (recv.has_value() && recv.value()) {
      result.push_back(recv.value());
    }
    return result;
  }

  // Check 'receivers' parameter
  if (!receivers_.has_value()) {
    return {};
  }
  auto recvs = receivers_.try_get();
  if (!recvs.has_value()) {
    return {};
  }
  // Filter out any null receivers
  for (auto& recv : recvs.value()) {
    if (recv) {
      result.push_back(recv);
    }
  }
  return result;
}

void CudaStreamCondition::register_callbacks_for_single_receiver(nvidia::gxf::Receiver* gxf_recv,
                                                                 gxf_context_t gxf_context,
                                                                 size_t& streams_found) {
  if (!gxf_recv) {
    return;
  }

  // Sync back stage to main stage
  auto sync_result = gxf_recv->sync();
  if (!sync_result) {
    HOLOSCAN_LOG_WARN("CudaStreamCondition: failed to sync receiver");
  }

  size_t queue_size = gxf_recv->size();
  if (queue_size == 0) {
    return;
  }

  // Determine how many messages to check based on the check_all_messages parameter
  size_t messages_to_check = check_all_messages_.get() ? queue_size : 1;

  // Iterate through messages in the queue
  for (size_t i = 0; i < messages_to_check; ++i) {
    auto maybe_entity = gxf_recv->peek(static_cast<int32_t>(i));
    if (!maybe_entity) {
      HOLOSCAN_LOG_WARN("CudaStreamCondition: failed to peek message at index {}", i);
      continue;
    }

    auto entity = maybe_entity.value();

    // Find ALL CudaStreamId components in this entity using findAllHeap to minimize stack usage
    auto maybe_stream_ids = entity.findAllHeap<nvidia::gxf::CudaStreamId>();
    if (!maybe_stream_ids) {
      // No CudaStreamId components in this message - that's okay, treat as ready
      continue;
    }

    auto& stream_ids = maybe_stream_ids.value();
    for (size_t j = 0; j < stream_ids.size(); ++j) {
      auto stream_id_handle = stream_ids.at(j);
      if (!stream_id_handle) {
        continue;
      }

      auto stream_id = stream_id_handle.value();
      if (stream_id->stream_cid == kNullUid) {
        continue;
      }

      // Get the cudaStream_t from the CudaStreamId
      auto maybe_stream_handle =
          nvidia::gxf::Handle<nvidia::gxf::CudaStream>::Create(gxf_context, stream_id->stream_cid);
      if (!maybe_stream_handle) {
        HOLOSCAN_LOG_WARN("CudaStreamCondition: failed to create CudaStream handle for cid {}",
                          stream_id->stream_cid);
        continue;
      }

      auto maybe_cuda_stream = maybe_stream_handle.value()->stream();
      if (!maybe_cuda_stream) {
        HOLOSCAN_LOG_WARN("CudaStreamCondition: failed to get cudaStream_t from CudaStream");
        continue;
      }

      cudaStream_t cuda_stream = maybe_cuda_stream.value();

      // Register host callback on this stream
      auto* callback_data = new CallbackData{this};
      pending_callbacks_.fetch_add(1);
      streams_found++;

      cudaError_t result = cudaLaunchHostFunc(cuda_stream, cuda_host_callback, callback_data);
      if (result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("CudaStreamCondition: failed to register CUDA host callback: {}",
                           cudaGetErrorString(result));
        pending_callbacks_.fetch_sub(1);
        streams_found--;  // Correct the count so debug logs reflect actual registered callbacks
        delete callback_data;
      }
    }
  }
}

void CudaStreamCondition::register_callbacks_for_receivers() {
  auto all_receivers = get_all_receivers();
  if (all_receivers.empty()) {
    HOLOSCAN_LOG_ERROR(
        "CudaStreamCondition: no receivers configured. "
        "Specify either 'receiver' (single port) or 'receivers' (one or more ports) parameter.");
    return;
  }

  gxf_context_t gxf_context = static_cast<gxf_context_t>(get_gxf_context());
  if (gxf_context == nullptr) {
    HOLOSCAN_LOG_ERROR("CudaStreamCondition: GXF context is null");
    return;
  }

  size_t streams_found = 0;

  for (auto& recv : all_receivers) {
    auto* gxf_recv = recv->get();
    register_callbacks_for_single_receiver(gxf_recv, gxf_context, streams_found);
  }

  HOLOSCAN_LOG_DEBUG("CudaStreamCondition: registered {} callbacks across {} receivers",
                     streams_found,
                     all_receivers.size());
}

void CudaStreamCondition::update_state(int64_t timestamp) {
  auto current_state = state_.load();

  // Already registered callbacks or data available - nothing to do
  if (current_state == State::CALLBACKS_REGISTERED || current_state == State::DATA_AVAILABLE) {
    return;
  }

  // State is UNSET - check for new messages across all receivers
  auto all_receivers = get_all_receivers();
  if (all_receivers.empty()) {
    return;
  }

  // Check if any receiver has messages
  size_t total_queue_size = 0;
  for (auto& recv : all_receivers) {
    auto* gxf_recv = recv->get();
    if (!gxf_recv) {
      continue;
    }

    // Sync and check for messages
    auto sync_result = gxf_recv->sync();
    if (!sync_result) {
      HOLOSCAN_LOG_WARN("CudaStreamCondition: failed to sync receiver in update_state");
    }

    total_queue_size += gxf_recv->size();
  }

  if (total_queue_size == 0) {
    // No messages available in any receiver
    return;
  }

  // Register callbacks for all messages across all receivers
  register_callbacks_for_receivers();

  if (pending_callbacks_.load() == 0) {
    // No streams found in any message - immediately ready
    state_.store(State::DATA_AVAILABLE);
    last_state_change_ = timestamp;
  } else {
    // Callbacks registered - wait for them
    state_.store(State::CALLBACKS_REGISTERED);
    last_state_change_ = timestamp;
  }
}

void CudaStreamCondition::check([[maybe_unused]] int64_t timestamp, SchedulingStatusType* type,
                                int64_t* target_timestamp) const {
  if (type == nullptr) {
    throw std::runtime_error("CudaStreamCondition::check received nullptr for type");
  }
  if (target_timestamp == nullptr) {
    throw std::runtime_error("CudaStreamCondition::check received nullptr for target_timestamp");
  }

  auto current_state = state_.load();
  switch (current_state) {
    case State::DATA_AVAILABLE:
      *type = SchedulingStatusType::kReady;
      *target_timestamp = last_state_change_;
      break;
    case State::CALLBACKS_REGISTERED:
      *type = SchedulingStatusType::kWaitEvent;
      *target_timestamp = last_state_change_;
      break;
    case State::UNSET:
    default:
      *type = SchedulingStatusType::kWait;
      *target_timestamp = last_state_change_;
      break;
  }
}

void CudaStreamCondition::on_execute(int64_t timestamp) {
  // After compute() runs, reset state for next batch of messages
  // The messages should have been consumed by the operator
  state_.store(State::UNSET);
  pending_callbacks_.store(0);
  last_state_change_ = timestamp;
}

}  // namespace holoscan
