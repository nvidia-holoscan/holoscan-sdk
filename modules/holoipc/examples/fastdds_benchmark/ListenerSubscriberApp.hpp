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
 * @file ListenerSubscriberApp.hpp
 *
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_LISTENERSUBSCRIBERAPP_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_LISTENERSUBSCRIBERAPP_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <variant>

#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <gen/Buffer.hpp>
#include <gen/BufferPubSubTypes.hpp>
#include "Application.hpp"
#include "CLIParser.hpp"
#include "holoscan/ipc/context.hpp"
#include "holoscan/ipc/transport/fastdds/fast_dds_transport.hpp"

namespace holoscan {
namespace ipc {
namespace fastdds_benchmark {

namespace fastdds = ::eprosima::fastdds::dds;

using ::holoscan::ipc::Context;
using ::holoscan::ipc::make_context;
using ::holoscan::ipc::transport::fastdds::FastDdsTransport;

class ListenerSubscriberApp : public Application, public fastdds::DataReaderListener {
 public:
  ListenerSubscriberApp(const CLIParser::subscriber_config& config, const std::string& topic_name);

  ~ListenerSubscriberApp();

  //! Subscription callback
  void on_data_available(fastdds::DataReader* reader) override;

  //! Subscriber matched method
  void on_subscription_matched(fastdds::DataReader* reader,
                               const fastdds::SubscriptionMatchedStatus& info) override;

  //! Run subscriber
  void run() override;

  //! Trigger the end of execution
  void stop() override;

 private:
  //! Return the current state of execution
  bool is_stopped();

  //! Accel path: retain acquired device pointer in `buffer_queue_` for IPC lifecycle (no staging
  //! D2D).
  void process_buffer(const std::shared_ptr<void>& received_buffer, size_t size);

  //! Plain `Buffer` path: copy host payload into `cuda_message_buffer_` (host-to-device).
  void process_buffer(const Buffer& host_sample);

  struct ParticipantDeleter {
    void operator()(fastdds::DomainParticipant* p) const;
  };
  using ParticipantPtr = std::unique_ptr<fastdds::DomainParticipant, ParticipantDeleter>;
  ParticipantPtr participant_;

  fastdds::Subscriber* subscriber_;

  fastdds::Topic* topic_;

  fastdds::DataReader* reader_;

  fastdds::TypeSupport type_;
  std::shared_ptr<Context<FastDdsTransport>> context_;

  uint32_t message_size_;

  //! AccelBuffer (`--accel`) vs plain Buffer (default).
  bool use_accel_buffer_;

  //! Accel path: `acquire_pointer_eager` vs `acquire_pointer` (`--eager`).
  bool use_eager_acquire_;

  //! Local device buffer of `message_size_` bytes (reserved for upcoming use).
  std::shared_ptr<void> cuda_message_buffer_;

  std::size_t samples_;

  std::size_t received_samples_;

  //! Next expected sample `index()` (0-based) for AccelBuffer or Buffer; gap logging.
  uint32_t next_expected_msg_index_{0};

  std::atomic<bool> stop_;

  mutable std::mutex terminate_cv_mtx_;

  std::condition_variable terminate_cv_;

  //! Pending DDS sample: `AccelBuffer` (IPC) or plain `Buffer` (host octets).
  using ReceivedSampleVariant = std::variant<AccelBuffer, Buffer>;

  // Pending samples for the worker thread (`run`).
  std::queue<ReceivedSampleVariant> cuda_buffer_queue_;
  std::mutex cuda_buffer_mutex_;
  std::condition_variable cuda_buffer_cv_;

  // Bounded queue for CUDA buffers
  static constexpr size_t MAX_QUEUE_SIZE = 10;
  std::queue<std::shared_ptr<void>> buffer_queue_;
  std::mutex queue_mutex_;
};

}  // namespace fastdds_benchmark
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_LISTENERSUBSCRIBERAPP_HPP
