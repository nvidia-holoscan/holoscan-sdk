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
 * @file WaitsetSubscriberApp.cpp
 *
 */

#include "WaitsetSubscriberApp.hpp"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastdds/dds/core/condition/GuardCondition.hpp>
#include <fastdds/dds/core/condition/WaitSet.hpp>
#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/qos/SubscriberQos.hpp>

#include <gen/CudaBufferPubSubTypes.hpp>
#include "Application.hpp"
#include "CLIParser.hpp"
#include "holoscan/ipc/context.hpp"

using namespace eprosima::fastdds::dds;

namespace holoscan {
namespace ipc {
namespace dds_example {

void WaitsetSubscriberApp::ParticipantDeleter::operator()(DomainParticipant* p) const {
  if (p != nullptr) {
    p->delete_contained_entities();
    DomainParticipantFactory::get_instance()->delete_participant(p);
  }
}

WaitsetSubscriberApp::WaitsetSubscriberApp(const CLIParser::subscriber_config& config,
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
  reader_ = subscriber_->create_datareader(topic_, reader_qos, nullptr, StatusMask::all());
  if (reader_ == nullptr) {
    throw std::runtime_error("DataReader initialization failed");
  }

  // Prepare a wait-set
  wait_set_.attach_condition(reader_->get_statuscondition());
  wait_set_.attach_condition(terminate_condition_);
}

WaitsetSubscriberApp::~WaitsetSubscriberApp() {
  if (participant_.get() != nullptr) {
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

void WaitsetSubscriberApp::run() {
  while (!is_stopped()) {
    ConditionSeq triggered_conditions;
    ReturnCode_t ret_code =
        wait_set_.wait(triggered_conditions, eprosima::fastdds::dds::c_TimeInfinite);
    if (RETCODE_OK != ret_code) {
      std::cerr << "Error waiting for conditions" << std::endl;
      continue;
    }
    for (Condition* cond : triggered_conditions) {
      StatusCondition* status_cond = dynamic_cast<StatusCondition*>(cond);
      if (nullptr != status_cond) {
        Entity* entity = status_cond->get_entity();
        StatusMask changed_statuses = entity->get_status_changes();
        if (changed_statuses.is_active(StatusMask::subscription_matched())) {
          SubscriptionMatchedStatus status_;
          reader_->get_subscription_matched_status(status_);
          if (status_.current_count_change == 1) {
            std::cout << "Waitset Subscriber matched." << std::endl;
          } else if (status_.current_count_change == -1) {
            std::cout << "Waitset Subscriber unmatched." << std::endl;
          } else {
            std::cout << status_.current_count_change
                      << " is not a valid value for SubscriptionMatchedStatus current "
                         "count change"
                      << std::endl;
          }
        }
        if (changed_statuses.is_active(StatusMask::data_available()) && context_) {
          SampleInfo info;
          CudaBuffer sample;
          while ((!is_stopped()) && (RETCODE_OK == reader_->take_next_sample(&sample, &info))) {
            if ((info.instance_state == ALIVE_INSTANCE_STATE) && info.valid_data) {
              try {
                std::shared_ptr<void> cuda_ptr =
                    context_->acquire_pointer(sample.descriptor()).get();
                if (!cuda_ptr) {
                  continue;
                }
                const size_t buf_size = static_cast<size_t>(sample.size());
                received_samples_++;
                std::cout << "Buffer with size: " << buf_size
                          << " bytes, device_ptr: " << cuda_ptr.get() << " (sample "
                          << received_samples_ << ") ACQUIRED" << std::endl;
                process_buffer(cuda_ptr, buf_size);
                if (samples_ > 0 && (received_samples_ >= samples_)) {
                  stop();
                }
              } catch (const std::exception& e) {
                // Context shutting down, invalid future, or NACK from publisher.
                std::cerr << "[Subscriber] Failed to acquire pointer: " << e.what() << std::endl;
              }
            }
          }
        }
      }
    }
  }
}

bool WaitsetSubscriberApp::is_stopped() {
  return stop_.load();
}

void WaitsetSubscriberApp::stop() {
  stop_.store(true);
  terminate_condition_.set_trigger_value(true);
}

void WaitsetSubscriberApp::process_buffer(const std::shared_ptr<void>& received_buffer,
                                          size_t size) {
  // D2H copy can be slow; do not hold queue_mutex_ across cudaMemcpy (see ListenerSubscriberApp).
  if (size == 0) {
    std::cout << "[Subscriber] Received empty buffer" << std::endl;
    return;
  }
  std::vector<char> host_buffer(size);
  cudaError_t err =
      cudaMemcpy(host_buffer.data(), received_buffer.get(), size, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    // Match ListenerSubscriberApp: log only the null-terminated prefix, not the full allocation.
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

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan
