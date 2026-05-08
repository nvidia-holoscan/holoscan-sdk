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
 * @file ListenerSubscriberApp.cpp
 *
 */

#include "ListenerSubscriberApp.hpp"

#include <cuda_runtime.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <fastdds/dds/core/policy/QosPolicies.hpp>
#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/qos/SubscriberQos.hpp>

#include <gen/Buffer.hpp>
#include "Application.hpp"
#include "common.hpp"
#include "holoscan/ipc/context.hpp"

using namespace eprosima::fastdds::dds;

namespace holoscan {
namespace ipc {
namespace fastdds_benchmark {

namespace {

//! Match writer: reliable + transient-local + keep-last so early samples are not lost on late sync.
constexpr int32_t kDataReaderHistoryDepth = 64;

}  // namespace

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
      type_(make_buffer_sample_typesupport(config.use_accel_buffer)),
      message_size_(config.message_size),
      use_accel_buffer_(config.use_accel_buffer),
      use_eager_acquire_(config.use_eager_acquire),
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

  context_ = make_context<transport::fastdds::FastDdsTransport>(participant_.get());
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

  // Create the reader: same durability/reliability as writer so TRANSIENT_LOCAL history is
  // delivered.
  DataReaderQos reader_qos;
  subscriber_->get_default_datareader_qos(reader_qos);
  reader_qos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  reader_qos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
  reader_qos.history().kind = KEEP_LAST_HISTORY_QOS;
  reader_qos.history().depth = kDataReaderHistoryDepth;
  reader_ = subscriber_->create_datareader(topic_, reader_qos, this, StatusMask::all());
  if (reader_ == nullptr) {
    throw std::runtime_error("DataReader initialization failed");
  }

  cuda_message_buffer_ = make_cuda_device_buffer(static_cast<size_t>(message_size_));
}

ListenerSubscriberApp::~ListenerSubscriberApp() {
  if (participant_.get() != nullptr) {
    {
      std::lock_guard<std::mutex> lock(cuda_buffer_mutex_);
      std::queue<ReceivedSampleVariant>().swap(cuda_buffer_queue_);
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
  if (info.current_count_change != 1 && info.current_count_change != -1) {
    std::cerr << info.current_count_change
              << " is not a valid value for SubscriptionMatchedStatus current count change"
              << std::endl;
  }
}

void ListenerSubscriberApp::on_data_available(DataReader* reader) {
  SampleInfo info;
  // Drain the reader in this callback. Do not test is_stopped() on the loop condition: after the
  // Nth sample we call stop(), and any remaining take_next_sample calls in the same (or a later)
  // invocation must still run or samples stay stuck in the DataReader and the worker never exits.
  if (use_accel_buffer_) {
    AccelBuffer sample;
    while (RETCODE_OK == reader->take_next_sample(&sample, &info)) {
      if ((info.instance_state == ALIVE_INSTANCE_STATE) && info.valid_data && context_) {
        {
          std::lock_guard<std::mutex> lock(cuda_buffer_mutex_);
          cuda_buffer_queue_.push(ReceivedSampleVariant(std::move(sample)));
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
  } else {
    Buffer sample;
    while (RETCODE_OK == reader->take_next_sample(&sample, &info)) {
      if ((info.instance_state == ALIVE_INSTANCE_STATE) && info.valid_data) {
        {
          std::lock_guard<std::mutex> lock(cuda_buffer_mutex_);
          cuda_buffer_queue_.push(ReceivedSampleVariant(std::move(sample)));
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
}

void ListenerSubscriberApp::process_buffer(const std::shared_ptr<void>& received_buffer,
                                           size_t size) {
  if (size == 0) {
    return;
  }
  if (size > static_cast<size_t>(message_size_)) {
    std::cerr << "[Subscriber] sample size " << size << " exceeds message_size_ (" << message_size_
              << " bytes)" << std::endl;
    return;
  }

  std::lock_guard<std::mutex> lock(queue_mutex_);

  // If queue is full, remove oldest buffer (will trigger RELEASED message automatically)
  if (buffer_queue_.size() >= MAX_QUEUE_SIZE) {
    buffer_queue_.pop();
  }

  // Add new buffer to queue
  buffer_queue_.push(received_buffer);
}

void ListenerSubscriberApp::process_buffer(const Buffer& host_sample) {
  const size_t size = host_sample.data().size();
  if (size == 0) {
    return;
  }
  if (!cuda_message_buffer_) {
    std::cerr << "[Subscriber] cuda_message_buffer_ is not allocated" << std::endl;
    return;
  }
  if (size > static_cast<size_t>(message_size_)) {
    std::cerr << "[Subscriber] Buffer payload size " << size << " exceeds cuda_message_buffer_ ("
              << message_size_ << " bytes)" << std::endl;
    return;
  }
  const cudaError_t err = cudaMemcpy(
      cuda_message_buffer_.get(), host_sample.data().data(), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[Subscriber] cudaMemcpy (host to device) failed: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }
  cuda_buffer_latency_log("sub", "host", host_sample.index());
}

void ListenerSubscriberApp::run() {
  for (;;) {
    ReceivedSampleVariant item;
    {
      std::unique_lock<std::mutex> lk(cuda_buffer_mutex_);
      cuda_buffer_cv_.wait(lk, [this] { return is_stopped() || !cuda_buffer_queue_.empty(); });
      if (is_stopped() && cuda_buffer_queue_.empty()) {
        break;
      }
      if (!cuda_buffer_queue_.empty()) {
        item = std::move(cuda_buffer_queue_.front());
        cuda_buffer_queue_.pop();
      } else {
        continue;
      }
    }
    std::visit(
        [this](auto&& sample) {
          using T = std::decay_t<decltype(sample)>;
          if constexpr (std::is_same_v<T, AccelBuffer>) {
            if (!context_) {
              return;
            }
            const uint32_t msg_index = sample.index();
            if (msg_index != next_expected_msg_index_) {
              std::cerr << "[Subscriber] AccelBuffer index gap: expected "
                        << next_expected_msg_index_ << ", got " << msg_index << std::endl;
              next_expected_msg_index_ = msg_index + 1;
            } else {
              ++next_expected_msg_index_;
            }
            const size_t buf_size = static_cast<size_t>(sample.size());
            std::shared_ptr<void> cuda_ptr;
            if (use_eager_acquire_) {
              cuda_ptr = context_->acquire_pointer_eager(sample.descriptor());
            } else {
              auto future = context_->acquire_pointer(sample.descriptor());
              try {
                cuda_ptr = future.get();
              } catch (const std::exception&) {
                return;
              }
            }
            if (!cuda_ptr) {
              return;
            }
            // Latency end (accel): subscriber sees CUDA payload; before staging D2D
            // into cuda_message_buffer_ (README, CUDA buffer latency).
            cuda_buffer_latency_log("sub", "accel", msg_index);
            process_buffer(cuda_ptr, buf_size);
          } else {
            const uint32_t msg_index = sample.index();
            if (msg_index != next_expected_msg_index_) {
              std::cerr << "[Subscriber] Buffer index gap: expected " << next_expected_msg_index_
                        << ", got " << msg_index << std::endl;
              next_expected_msg_index_ = msg_index + 1;
            } else {
              ++next_expected_msg_index_;
            }
            process_buffer(sample);
          }
        },
        item);
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

}  // namespace fastdds_benchmark
}  // namespace ipc
}  // namespace holoscan
