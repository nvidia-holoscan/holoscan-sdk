// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ListenerSubscriberApp.hpp
 *
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_LISTENERSUBSCRIBERAPP_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_LISTENERSUBSCRIBERAPP_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>

#include <fastdds/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <gen/CudaBuffer.hpp>
#include <gen/CudaBufferPubSubTypes.hpp>
#include "Application.hpp"
#include "CLIParser.hpp"
#include "holoscan/ipc/context.hpp"
#include "holoscan/ipc/transport/fastdds/fast_dds_transport.hpp"

namespace holoscan {
namespace ipc {
namespace dds_example {

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

  //! Process incoming buffer and add to queue (size from CudaBuffer.size()).
  void process_buffer(const std::shared_ptr<void>& received_buffer, size_t size);

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

  std::size_t samples_;

  std::size_t received_samples_;

  std::atomic<bool> stop_;

  mutable std::mutex terminate_cv_mtx_;

  std::condition_variable terminate_cv_;

  // CudaBuffer queue: pending (descriptor + size) for processing
  std::queue<CudaBuffer> cuda_buffer_queue_;
  std::mutex cuda_buffer_mutex_;
  std::condition_variable cuda_buffer_cv_;

  // Bounded queue for CUDA buffers
  static constexpr size_t MAX_QUEUE_SIZE = 10;
  std::queue<std::shared_ptr<void>> buffer_queue_;
  std::mutex queue_mutex_;
};

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_LISTENERSUBSCRIBERAPP_HPP
