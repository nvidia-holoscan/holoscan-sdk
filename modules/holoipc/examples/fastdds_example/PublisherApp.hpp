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
 * @file PublisherApp.hpp
 *
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_PUBLISHERAPP_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_PUBLISHERAPP_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <fastdds/dds/core/status/PublicationMatchedStatus.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <gen/CudaBufferPubSubTypes.hpp>
#include "Application.hpp"
#include "CLIParser.hpp"
#include "CudaDataWriterListener.hpp"
#include "holoscan/ipc/context.hpp"
#include "holoscan/ipc/transport/fastdds/fast_dds_transport.hpp"

namespace holoscan {
namespace ipc {
namespace dds_example {

namespace fastdds = ::eprosima::fastdds::dds;

using ::holoscan::ipc::Context;
using ::holoscan::ipc::make_context;
using ::holoscan::ipc::transport::fastdds::FastDdsTransport;

class PublisherApp : public Application, public CudaDataWriterListener {
 public:
  PublisherApp(const CLIParser::publisher_config& config, const std::string& topic_name);

  ~PublisherApp();

  //! Publisher matched method
  void on_publication_matched(fastdds::DataWriter* writer,
                              const fastdds::PublicationMatchedStatus& info) override;

  //! Run publisher
  void run() override;

  //! Stop publisher
  void stop() override;

 private:
  //! Return the current state of execution
  bool is_stopped();

  //! Publish a sample
  bool publish();

  struct ParticipantDeleter {
    void operator()(fastdds::DomainParticipant* p) const;
  };
  using ParticipantPtr = std::unique_ptr<fastdds::DomainParticipant, ParticipantDeleter>;
  ParticipantPtr participant_;

  fastdds::Publisher* publisher_;

  fastdds::Topic* topic_;

  fastdds::DataWriter* writer_;

  fastdds::TypeSupport type_;
  std::shared_ptr<Context<FastDdsTransport>> context_;

  std::size_t matched_;

  std::size_t samples_;

  std::size_t expected_matches_;

  std::mutex mutex_;

  std::condition_variable cv_;

  std::atomic<bool> stop_;

  const uint32_t period_ms_ = 100;  // in ms
  /** 4 MiB: large enough that avoiding a host copy via CUDA IPC is meaningful for the demo. */
  static constexpr size_t cuda_buffer_size_ = static_cast<size_t>(4) * 1024 * 1024;
  std::string message_;  // Message to write to GPU memory

  /** Fixed-size pool; each entry keeps a descriptor alive until the slot is reused. */
  std::vector<std::shared_ptr<holoscan::ipc::transport::fastdds::gen::PointerDescriptor>>
      descriptor_pool_;
  size_t descriptor_pool_index_ = 0;
};

}  // namespace dds_example
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOIPC_EXAMPLES_FASTDDS_EXAMPLE_PUBLISHERAPP_HPP
