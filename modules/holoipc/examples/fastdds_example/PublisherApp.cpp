// Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file PublisherApp.cpp
 *
 */

#include "PublisherApp.hpp"

#include <cuda_runtime.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>

#include <gen/CudaBufferPubSubTypes.hpp>
#include "holoscan/ipc/context.hpp"

using namespace eprosima::fastdds::dds;

namespace {
constexpr size_t kDescriptorPoolSize = 16;
}  // namespace

namespace holoscan {
namespace ipc {
namespace dds_example {

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
      type_(new CudaBufferPubSubType()),
      matched_(0),
      samples_(config.samples),
      expected_matches_(config.matched),
      stop_(false),
      message_(config.message) {
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

  // Create the data writer
  DataWriterQos writer_qos = DATAWRITER_QOS_DEFAULT;
  publisher_->get_default_datawriter_qos(writer_qos);
  writer_qos.history().depth = 5;
  writer_ = publisher_->create_datawriter(topic_, writer_qos, this, StatusMask::all());
  if (writer_ == nullptr) {
    throw std::runtime_error("DataWriter initialization failed");
  }

  descriptor_pool_.resize(kDescriptorPoolSize);
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
    std::cout << "Publisher matched." << std::endl;
    cv_.notify_one();
  } else if (info.current_count_change == -1) {
    matched_ = info.current_count < 0 ? 0 : static_cast<std::size_t>(info.current_count);
    std::cout << "Publisher unmatched." << std::endl;
  } else {
    std::cout << info.current_count_change
              << " is not a valid value for PublicationMatchedStatus current count change"
              << std::endl;
  }
}

void PublisherApp::run() {
  std::size_t sample_count = 0;
  while (!is_stopped() && ((samples_ == 0) || (sample_count < samples_))) {
    if (publish()) {
      sample_count++;
      std::cout << "Buffer with size: " << cuda_buffer_size_ << " bytes (sample " << sample_count
                << ") SENT" << std::endl;

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

bool PublisherApp::publish() {
  bool ret = false;
  // Wait for the data endpoints discovery
  std::unique_lock<std::mutex> matched_lock(mutex_);
  cv_.wait(matched_lock, [&]() {
    // at least one has been discovered
    return ((matched_ >= expected_matches_) || is_stopped());
  });

  if (!is_stopped()) {
    // Allocate CUDA memory for this sample; wrap immediately so all return paths free it.
    void* raw_ptr = nullptr;
    cudaError_t err = cudaMalloc(&raw_ptr, cuda_buffer_size_);
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    std::shared_ptr<void> cuda_ptr(raw_ptr, [](void* ptr) {
      std::cout << "Freeing CUDA memory at " << ptr << std::endl;
      cudaFree(ptr);
    });

    std::cout << "Allocated CUDA memory at " << raw_ptr << " (" << cuda_buffer_size_ << " bytes)"
              << std::endl;

    const size_t message_bytes = message_.size() + 1;
    if (cuda_buffer_size_ < message_bytes) {
      std::cerr << "Buffer size (" << cuda_buffer_size_ << ") too small for message ("
                << message_bytes << " bytes)" << std::endl;
      return false;
    }
    // Write message to GPU memory
    err = cudaMemcpy(cuda_ptr.get(), message_.c_str(), message_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpy (write) failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }

    // Read back and verify
    std::vector<char> host_buffer(cuda_buffer_size_);
    err = cudaMemcpy(host_buffer.data(), cuda_ptr.get(), cuda_buffer_size_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpy (read) failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    host_buffer[cuda_buffer_size_ - 1] = '\0';  // Ensure null termination

    std::cout << "[Publisher] Wrote to GPU: \"" << host_buffer.data() << "\"" << std::endl;

    // Keep descriptor alive in a fixed-size pool until the slot is reused
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

    CudaBuffer sample;
    sample.descriptor(*descriptor_pool_[slot]);
    sample.size(static_cast<uint32_t>(cuda_buffer_size_));
    ret = (RETCODE_OK == writer_->write(&sample));
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

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan
