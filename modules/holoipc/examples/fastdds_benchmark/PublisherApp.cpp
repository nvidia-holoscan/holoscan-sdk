// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file PublisherApp.cpp
 *
 */

#include "PublisherApp.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fastdds/dds/core/policy/QosPolicies.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>

#include <gen/Buffer.hpp>
#include "common.hpp"
#include "holoscan/ipc/context.hpp"

using namespace eprosima::fastdds::dds;

namespace holoscan {
namespace ipc {
namespace fastdds_benchmark {

namespace {

constexpr size_t kDescriptorPoolSize = 16;
//! History depth: startup bursts + TRANSIENT_LOCAL cache for late-matching readers.
constexpr int32_t kDataWriterHistoryDepth = 64;

}  // namespace

void PublisherApp::ParticipantDeleter::operator()(DomainParticipant* p) const {
  if (p != nullptr) {
    p->delete_contained_entities();
    DomainParticipantFactory::get_instance()->delete_participant(p);
  }
}

PublisherApp::PublisherApp(const CLIParser::publisher_config& config, const std::string& topic_name)
    : participant_(nullptr),
      publisher_(nullptr),
      topic_(nullptr),
      writer_(nullptr),
      type_(make_buffer_sample_typesupport(config.use_accel_buffer)),
      matched_(0),
      samples_(config.samples),
      expected_matches_(config.matched),
      stop_(false),
      message_size_(config.message_size),
      use_accel_buffer_(config.use_accel_buffer) {
  auto factory = DomainParticipantFactory::get_instance();
  DomainParticipant* raw =
      factory->create_participant_with_default_profile(nullptr, StatusMask::none());
  if (raw == nullptr) {
    throw std::runtime_error("Participant initialization failed");
  }
  participant_.reset(raw);

  context_ = make_context<holoscan::ipc::transport::fastdds::FastDdsTransport>(participant_.get());
  type_.register_type(participant_.get());

  // Create the publisher
  PublisherQos pub_qos = PUBLISHER_QOS_DEFAULT;
  participant_->get_default_publisher_qos(pub_qos);
  publisher_ = participant_->create_publisher(pub_qos, nullptr, StatusMask::none());
  if (publisher_ == nullptr) {
    throw std::runtime_error("Publisher initialization failed");
  }

  // Create the topic
  TopicQos topic_qos = TOPIC_QOS_DEFAULT;
  participant_->get_default_topic_qos(topic_qos);
  topic_ = participant_->create_topic(topic_name, type_.get_type_name(), topic_qos);
  if (topic_ == nullptr) {
    throw std::runtime_error("Topic initialization failed");
  }

  // Create the data writer: reliable + transient-local so samples written just before match
  // remain available to a reader that finishes discovery shortly after (see also DataReader QoS).
  DataWriterQos writer_qos;
  publisher_->get_default_datawriter_qos(writer_qos);
  writer_qos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  writer_qos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
  writer_qos.history().kind = KEEP_LAST_HISTORY_QOS;
  writer_qos.history().depth = kDataWriterHistoryDepth;
  writer_ = publisher_->create_datawriter(topic_, writer_qos, this, StatusMask::all());
  if (writer_ == nullptr) {
    throw std::runtime_error("DataWriter initialization failed");
  }

  if (use_accel_buffer_) {
    descriptor_pool_.resize(kDescriptorPoolSize);
  }

  cuda_message_buffer_ = make_cuda_device_buffer(static_cast<size_t>(message_size_));
  cuda_device_memset(cuda_message_buffer_, 0xA5, static_cast<size_t>(message_size_));
}

PublisherApp::~PublisherApp() {
  if (participant_.get() != nullptr) {
    context_.reset();
    participant_.reset();
  }
}

void PublisherApp::on_publication_matched(DataWriter* /*writer*/,
                                          const PublicationMatchedStatus& info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (info.current_count_change == 1) {
    matched_ = info.current_count < 0 ? 0 : static_cast<std::size_t>(info.current_count);
    cv_.notify_one();
  } else if (info.current_count_change == -1) {
    matched_ = info.current_count < 0 ? 0 : static_cast<std::size_t>(info.current_count);
  } else {
    std::cerr << info.current_count_change
              << " is not a valid value for PublicationMatchedStatus current count change"
              << std::endl;
  }
}

void PublisherApp::run() {
  std::size_t sample_count = 0;
  while (!is_stopped() && ((samples_ == 0) || (sample_count < samples_))) {
    const std::size_t index_sent = sample_count;
    if (publish(index_sent)) {
      sample_count++;

      if (sample_count == 1u) {
        ReturnCode_t acked = RETCODE_ERROR;
        do {
          Duration_t acked_wait{1, 0};
          acked = writer_->wait_for_acknowledgments(acked_wait);
        } while (acked != RETCODE_OK && !is_stopped());
      }
    }
    // Wait for period or stop event
    std::unique_lock<std::mutex> period_lock(mutex_);
    cv_.wait_for(
        period_lock, std::chrono::milliseconds(period_ms_), [&]() { return is_stopped(); });
  }
}

bool PublisherApp::publish(std::size_t sample_index) {
  bool ret = false;
  // Wait for the data endpoints discovery
  std::unique_lock<std::mutex> matched_lock(mutex_);
  cv_.wait(matched_lock, [&]() {
    // at least one has been discovered
    return ((matched_ >= expected_matches_) || is_stopped());
  });

  if (!is_stopped()) {
    if (use_accel_buffer_) {
      // Allocate CUDA memory for this sample; wrap immediately so all return paths free it.
      std::shared_ptr<void> cuda_ptr = make_cuda_device_buffer(static_cast<size_t>(message_size_));

      cuda_device_memcpy_device_to_device(
          cuda_ptr, cuda_message_buffer_, static_cast<size_t>(message_size_));

      cuda_buffer_latency_log("pub", "accel", static_cast<uint32_t>(sample_index));

      std::shared_ptr<holoscan::ipc::transport::fastdds::gen::PointerDescriptor> descriptor =
          context_->share_pointer(cuda_ptr,
                                  holoscan::ipc::transport::fastdds::gen::HandleType::CUDA_IPC);
      if (!descriptor) {
        std::cerr << "share_pointer failed" << std::endl;
        return false;
      }
      const size_t slot = descriptor_pool_index_;
      descriptor_pool_[slot] = std::move(descriptor);
      descriptor_pool_index_ = (slot + 1) % kDescriptorPoolSize;

      AccelBuffer sample;
      sample.descriptor(*descriptor_pool_[slot]);
      sample.size(message_size_);
      sample.index(static_cast<unsigned long>(sample_index));
      ret = (RETCODE_OK == writer_->write(&sample));
    } else {
      std::vector<uint8_t> host_data(static_cast<size_t>(message_size_));
      cuda_buffer_latency_log("pub", "host", static_cast<uint32_t>(sample_index));
      const cudaError_t memcpy_err = cudaMemcpy(host_data.data(),
                                                cuda_message_buffer_.get(),
                                                static_cast<size_t>(message_size_),
                                                cudaMemcpyDeviceToHost);
      if (memcpy_err != cudaSuccess) {
        std::cerr << "[Publisher] cudaMemcpy (device to host) failed: "
                  << cudaGetErrorString(memcpy_err) << std::endl;
        return false;
      }
      Buffer sample;
      sample.data(std::move(host_data));
      sample.index(static_cast<unsigned long>(sample_index));
      ret = (RETCODE_OK == writer_->write(&sample));
    }
  }
  return ret;
}

bool PublisherApp::is_stopped() {
  return stop_.load();
}

void PublisherApp::stop() {
  stop_.store(true);
  cv_.notify_one();
}

}  // namespace fastdds_benchmark
}  // namespace ipc
}  // namespace holoscan
