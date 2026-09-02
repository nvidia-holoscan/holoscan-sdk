// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ListenerSubscriberApp.cpp
 *
 */

#include "ListenerSubscriberApp.hpp"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/qos/SubscriberQos.hpp>

#include <gen/CudaBufferPubSubTypes.hpp>
#include "Application.hpp"
#include "holoscan/ipc/context.hpp"

using namespace eprosima::fastdds::dds;

namespace holoscan {
namespace ipc {
namespace dds_example {

void ListenerSubscriberApp::ParticipantDeleter::operator()(DomainParticipant* p) const {
  if (p != nullptr) {
    p->delete_contained_entities();
    DomainParticipantFactory::get_instance()->delete_participant(p);
  }
}

ListenerSubscriberApp::ListenerSubscriberApp(const CLIParser::subscriber_config& config,
                                             const std::string& topic_name)
    : participant_(nullptr),
      subscriber_(nullptr),
      topic_(nullptr),
      reader_(nullptr),
      type_(new CudaBufferPubSubType()),
      samples_(config.samples),
      received_samples_(0),
      stop_(false) {
  auto factory = DomainParticipantFactory::get_instance();
  DomainParticipant* raw =
      factory->create_participant_with_default_profile(nullptr, StatusMask::none());
  if (raw == nullptr) {
    throw std::runtime_error("Participant initialization failed");
  }
  participant_.reset(raw);

  context_ = make_context<holoscan::ipc::transport::fastdds::FastDdsTransport>(participant_.get());
  type_.register_type(participant_.get());

  // Create the subscriber
  SubscriberQos sub_qos = SUBSCRIBER_QOS_DEFAULT;
  participant_->get_default_subscriber_qos(sub_qos);
  subscriber_ = participant_->create_subscriber(sub_qos, nullptr, StatusMask::none());
  if (subscriber_ == nullptr) {
    throw std::runtime_error("Subscriber initialization failed");
  }

  // Create the topic
  TopicQos topic_qos = TOPIC_QOS_DEFAULT;
  participant_->get_default_topic_qos(topic_qos);
  topic_ = participant_->create_topic(topic_name, type_.get_type_name(), topic_qos);
  if (topic_ == nullptr) {
    throw std::runtime_error("Topic initialization failed");
  }

  // Create the reader
  DataReaderQos reader_qos = DATAREADER_QOS_DEFAULT;
  subscriber_->get_default_datareader_qos(reader_qos);
  reader_ = subscriber_->create_datareader(topic_, reader_qos, this, StatusMask::all());
  if (reader_ == nullptr) {
    throw std::runtime_error("DataReader initialization failed");
  }
}

ListenerSubscriberApp::~ListenerSubscriberApp() {
  if (participant_.get() != nullptr) {
    {
      std::lock_guard<std::mutex> lock(cuda_buffer_mutex_);
      std::queue<CudaBuffer>().swap(cuda_buffer_queue_);
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      while (!buffer_queue_.empty()) {
        buffer_queue_.pop();
      }
    }
    context_.reset();
    type_.reset();
    participant_.reset();
  }
}

void ListenerSubscriberApp::on_subscription_matched(DataReader* /*reader*/,
                                                    const SubscriptionMatchedStatus& info) {
  if (info.current_count_change == 1) {
    std::cout << "Subscriber matched." << std::endl;
  } else if (info.current_count_change == -1) {
    std::cout << "Subscriber unmatched." << std::endl;
  } else {
    std::cout << info.current_count_change
              << " is not a valid value for SubscriptionMatchedStatus current count change"
              << std::endl;
  }
}

void ListenerSubscriberApp::on_data_available(DataReader* reader) {
  SampleInfo info;
  CudaBuffer sample;
  while ((!is_stopped()) && (RETCODE_OK == reader->take_next_sample(&sample, &info))) {
    if ((info.instance_state == ALIVE_INSTANCE_STATE) && info.valid_data && context_) {
      {
        std::lock_guard<std::mutex> lock(cuda_buffer_mutex_);
        cuda_buffer_queue_.push(sample);
      }
      cuda_buffer_cv_.notify_one();
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        received_samples_++;
        if (samples_ > 0 && received_samples_ >= samples_) {
          stop();
        }
      }
    }
  }
}

void ListenerSubscriberApp::process_buffer(const std::shared_ptr<void>& received_buffer,
                                           size_t size) {
  // Read content from GPU memory (size is application-known) without holding queue_mutex_: D2H
  // copy can be slow and should not block other threads that update the buffer queue.
  if (size == 0) {
    std::cout << "[Subscriber] Received empty buffer" << std::endl;
    return;
  }
  std::vector<char> host_buffer(size);
  cudaError_t err =
      cudaMemcpy(host_buffer.data(), received_buffer.get(), size, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    // Publisher writes a null-terminated string in the first bytes; the rest of the allocation
    // is padding. Do not use `size` as the string length: building/printing a 4MiB std::string
    // stalls the terminal for seconds per sample.
    const size_t text_len = strnlen(host_buffer.data(), size);
    std::string content(host_buffer.data(), text_len);
    std::cout << "[Subscriber] Read from GPU: \"" << content << "\"" << std::endl;
  } else {
    std::cerr << "[Subscriber] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
  }

  std::lock_guard<std::mutex> lock(queue_mutex_);

  // If queue is full, remove oldest buffer (will trigger RELEASED message automatically)
  if (buffer_queue_.size() >= MAX_QUEUE_SIZE) {
    std::cout << "Queue full (" << MAX_QUEUE_SIZE << " buffers), removing oldest buffer"
              << std::endl;
    buffer_queue_.pop();
  }

  // Add new buffer to queue
  buffer_queue_.push(received_buffer);
  std::cout << "Queue size: " << buffer_queue_.size() << "/" << MAX_QUEUE_SIZE << std::endl;
}

void ListenerSubscriberApp::run() {
  size_t processed_count = 0;
  for (;;) {
    CudaBuffer item;
    {
      std::unique_lock<std::mutex> lk(cuda_buffer_mutex_);
      cuda_buffer_cv_.wait(lk, [this] { return is_stopped() || !cuda_buffer_queue_.empty(); });
      if (is_stopped() && cuda_buffer_queue_.empty()) {
        break;
      }
      if (!cuda_buffer_queue_.empty()) {
        item = cuda_buffer_queue_.front();
        cuda_buffer_queue_.pop();
      } else {
        continue;
      }
    }
    if (!context_) {
      continue;
    }
    const size_t buf_size = static_cast<size_t>(item.size());
    auto future = context_->acquire_pointer(item.descriptor());
    try {
      std::shared_ptr<void> cuda_ptr = future.get();
      if (!cuda_ptr) {
        continue;
      }
      processed_count++;
      std::cout << "Buffer with size: " << buf_size << " bytes, device_ptr: " << cuda_ptr.get()
                << " (sample " << processed_count << ") ACQUIRED" << std::endl;
      process_buffer(cuda_ptr, buf_size);
    } catch (const std::exception& e) {
      // Context shutting down, invalid future, or NACK from publisher.
      std::cerr << "[Subscriber] Failed to acquire pointer: " << e.what() << std::endl;
    }
  }
}

bool ListenerSubscriberApp::is_stopped() {
  return stop_.load();
}

void ListenerSubscriberApp::stop() {
  stop_.store(true);
  terminate_cv_.notify_all();
  cuda_buffer_cv_.notify_all();
}

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan
