/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/application.hpp>
#include <holoscan/core/arg.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/gxf/async_buffer_receiver.hpp>
#include <holoscan/core/resources/gxf/async_buffer_transmitter.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>
#include <holoscan/core/resources/gxf/condition_combiner.hpp>
#include <holoscan/core/resources/gxf/cpu_thread.hpp>
#include <holoscan/core/resources/gxf/cuda_green_context.hpp>
#include <holoscan/core/resources/gxf/cuda_green_context_pool.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/double_buffer_receiver.hpp>
#include <holoscan/core/resources/gxf/double_buffer_transmitter.hpp>
#include <holoscan/core/resources/gxf/manual_clock.hpp>
#include <holoscan/core/resources/gxf/pubsub_receiver.hpp>
#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>
#include <holoscan/core/resources/gxf/realtime_clock.hpp>
#include <holoscan/core/resources/gxf/rmm_allocator.hpp>
#include <holoscan/core/resources/gxf/serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/std_component_serializer.hpp>
#include <holoscan/core/resources/gxf/std_entity_serializer.hpp>
#include <holoscan/core/resources/gxf/stream_ordered_allocator.hpp>
#include <holoscan/core/resources/gxf/synthetic_clock.hpp>
#include <holoscan/core/resources/gxf/system_resources.hpp>
#include <holoscan/core/resources/gxf/ucx_component_serializer.hpp>
#include <holoscan/core/resources/gxf/ucx_entity_serializer.hpp>
#include <holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp>
#include <holoscan/core/resources/gxf/ucx_receiver.hpp>
#include <holoscan/core/resources/gxf/ucx_serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/ucx_transmitter.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include "../config.hpp"
#include "../utils.hpp"
#include "common/assert.hpp"

using namespace std::string_literals;

namespace holoscan {

using ResourceClassesWithGXFContext = TestWithGXFContext;

TEST_F(ResourceClassesWithGXFContext, TestBlockMemoryPool) {
  const std::string name{"block-memory-pool"};
  ArgList arglist{
      Arg{"storage_type", static_cast<int32_t>(1)},
      Arg{"block_size", static_cast<uint64_t>(1024 * 1024 * 16)},
      Arg{"num_blocks", static_cast<uint64_t>(1)},
  };
  auto resource = F.make_resource<BlockMemoryPool>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<BlockMemoryPool>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::BlockMemoryPool"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestBlockMemoryPoolDefaultConstructor) {
  auto resource = F.make_resource<BlockMemoryPool>();
}

TEST_F(ResourceClassesWithGXFContext, TestCudaStreamPool) {
  const std::string name{"cuda-stream-pool"};
  ArgList arglist{
      Arg{"dev_id", static_cast<int32_t>(0)},
      Arg{"stream_flags", static_cast<uint32_t>(0)},
      Arg{"stream_priority", static_cast<int32_t>(0)},
      Arg{"reserved_size", static_cast<uint32_t>(1)},
      Arg{"max_size", static_cast<uint32_t>(5)},
  };
  auto resource = F.make_resource<CudaStreamPool>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CudaStreamPool>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CudaStreamPool"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestCudaStreamPoolDefaultConstructor) {
  auto resource = F.make_resource<CudaStreamPool>();
}

TEST_F(ResourceClassesWithGXFContext, TestCudaGreenContextPool) {
  const std::string name{"cuda-green-context-pool"};
  std::vector<uint32_t> sms_per_partition{4, 4};
  ArgList arglist{
      Arg{"dev_id", static_cast<int32_t>(0)},
      Arg{"green_context_flags", static_cast<uint32_t>(cudaStreamNonBlocking)},
      Arg{"num_partitions", static_cast<uint32_t>(2)},
      Arg{"sms_per_partition", sms_per_partition},
  };
  auto resource = F.make_resource<CudaGreenContextPool>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CudaGreenContextPool>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CudaGreenContextPool"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestCudaGreenContextPoolDefaultConstructor) {
  auto resource = F.make_resource<CudaGreenContextPool>();
}

TEST_F(ResourceClassesWithGXFContext, TestCudaGreenContext) {
  const std::string name{"cuda-green-context"};
  auto pool = F.make_resource<CudaGreenContextPool>("pool");
  ArgList arglist{
      Arg{"cuda_green_context_pool", pool},
      Arg{"index", static_cast<int32_t>(0)},
      Arg{"nvtx_identifier", std::string("test_context")},
  };
  auto resource = F.make_resource<CudaGreenContext>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CudaGreenContext>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CudaGreenContext"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestCudaGreenContextDefaultConstructor) {
  auto resource = F.make_resource<CudaGreenContext>();
}

TEST_F(ResourceClassesWithGXFContext, TestRMMAllocator) {
  const std::string name{"rmm-pool"};
  ArgList arglist{
      Arg{"device_memory_initial_size", std::string{"10MB"}},
      Arg{"device_memory_max_size", std::string{"20MB"}},
      Arg{"host_memory_initial_size", std::string{"10MB"}},
      Arg{"host_memory_max_size", std::string{"20MB"}},
      Arg{"dev_id", static_cast<int32_t>(0)},
  };
  auto resource = F.make_resource<RMMAllocator>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<RMMAllocator>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::RMMAllocator"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestRMMAllocatorDefaultConstructor) {
  auto resource = F.make_resource<RMMAllocator>();
}

TEST_F(ResourceClassesWithGXFContext, TestStreamOrderedAllocator) {
  const std::string name{"rmm-pool"};
  ArgList arglist{
      Arg{"device_memory_initial_size", std::string{"10MB"}},
      Arg{"device_memory_max_size", std::string{"20MB"}},
      Arg{"release_threshold", std::string{"4MB"}},
      Arg{"dev_id", static_cast<int32_t>(0)},
  };
  auto resource = F.make_resource<StreamOrderedAllocator>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<StreamOrderedAllocator>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::StreamOrderedAllocator"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestStreamOrderedAllocatorDefaultConstructor) {
  auto resource = F.make_resource<StreamOrderedAllocator>();
}

TEST_F(ResourceClassesWithGXFContext, TestDoubleBufferReceiver) {
  const std::string name{"receiver"};
  ArgList arglist{
      Arg{"capacity", 1UL},
      Arg{"policy", 2UL},
  };
  auto resource = F.make_resource<DoubleBufferReceiver>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<DoubleBufferReceiver>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::DoubleBufferReceiver"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestDoubleBufferReceiverDefaultConstructor) {
  auto resource = F.make_resource<DoubleBufferReceiver>();
}

TEST_F(ResourceClassesWithGXFContext, TestAsyncBufferReceiver) {
  const std::string name{"receiver"};
  auto resource = F.make_resource<AsyncBufferReceiver>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<AsyncBufferReceiver>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "holoscan::HoloscanAsyncBufferReceiver"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestDoubleBufferTransmitter) {
  const std::string name{"transmitter"};
  ArgList arglist{
      Arg{"capacity", 1UL},
      Arg{"policy", 2UL},
  };
  auto resource = F.make_resource<DoubleBufferTransmitter>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<DoubleBufferTransmitter>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::DoubleBufferTransmitter"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestDoubleBufferTransmitterDefaultConstructor) {
  auto resource = F.make_resource<DoubleBufferTransmitter>();
}

TEST_F(ResourceClassesWithGXFContext, TestAsyncBufferTransmitter) {
  const std::string name{"transmitter"};
  auto resource = F.make_resource<AsyncBufferTransmitter>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<AsyncBufferTransmitter>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "holoscan::HoloscanAsyncBufferTransmitter"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubReceiver) {
  const std::string name{"pubsub-receiver"};
  ArgList arglist{
      Arg{"topic_name", std::string("/test/topic")},
      Arg{"capacity", 1UL},
      Arg{"policy", 2UL},
  };
  auto resource = F.make_resource<PubSubReceiver>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<PubSubReceiver>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::PubSubReceiver"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubReceiverDefaultConstructor) {
  auto resource = F.make_resource<PubSubReceiver>();
  // Verify it was created successfully (no crash)
  EXPECT_NE(resource, nullptr);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubTransmitter) {
  const std::string name{"pubsub-transmitter"};
  ArgList arglist{
      Arg{"topic_name", std::string("/test/topic")},
      Arg{"capacity", 1UL},
      Arg{"policy", 2UL},
  };
  auto resource = F.make_resource<PubSubTransmitter>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<PubSubTransmitter>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::PubSubTransmitter"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubTransmitterDefaultConstructor) {
  auto resource = F.make_resource<PubSubTransmitter>();
  // Verify it was created successfully (no crash)
  EXPECT_NE(resource, nullptr);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubReceiverQosGetterSetter) {
  auto resource = F.make_resource<PubSubReceiver>("pubsub-receiver");

  EXPECT_FALSE(resource->qos().has_value());

  nvidia::gxf::QoSProfile qos_profile = nvidia::gxf::QoSProfile::Default();
  qos_profile.set_reliability(nvidia::gxf::ReliabilityPolicy::kReliable)
      .set_durability(nvidia::gxf::DurabilityPolicy::kTransientLocal)
      .set_history(nvidia::gxf::HistoryPolicy::kKeepAll, 0);
  resource->qos(qos_profile);

  ASSERT_TRUE(resource->qos().has_value());
  EXPECT_EQ(resource->qos()->reliability, nvidia::gxf::ReliabilityPolicy::kReliable);
  EXPECT_EQ(resource->qos()->durability, nvidia::gxf::DurabilityPolicy::kTransientLocal);
  EXPECT_EQ(resource->qos()->history, nvidia::gxf::HistoryPolicy::kKeepAll);
}

TEST_F(ResourceClassesWithGXFContext, TestPubSubTransmitterQosGetterSetter) {
  auto resource = F.make_resource<PubSubTransmitter>("pubsub-transmitter");

  EXPECT_FALSE(resource->qos().has_value());

  nvidia::gxf::QoSProfile qos_profile = nvidia::gxf::QoSProfile::Default();
  qos_profile.set_reliability(nvidia::gxf::ReliabilityPolicy::kReliable)
      .set_durability(nvidia::gxf::DurabilityPolicy::kTransientLocal)
      .set_history(nvidia::gxf::HistoryPolicy::kKeepAll, 0);
  resource->qos(qos_profile);

  ASSERT_TRUE(resource->qos().has_value());
  EXPECT_EQ(resource->qos()->reliability, nvidia::gxf::ReliabilityPolicy::kReliable);
  EXPECT_EQ(resource->qos()->durability, nvidia::gxf::DurabilityPolicy::kTransientLocal);
  EXPECT_EQ(resource->qos()->history, nvidia::gxf::HistoryPolicy::kKeepAll);
}

TEST_F(ResourceClassesWithGXFContext, TestStdComponentSerializer) {
  const std::string name{"std-component-serializer"};
  auto resource = F.make_resource<StdComponentSerializer>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<StdComponentSerializer>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::StdComponentSerializer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestStdComponentSerializerDefaultConstructor) {
  auto resource = F.make_resource<StdComponentSerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUnboundedAllocator) {
  const std::string name{"unbounded"};
  auto resource = F.make_resource<UnboundedAllocator>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UnboundedAllocator>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::UnboundedAllocator"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUnboundedAllocatorGXFComponentMethods) {
  const std::string name{"unbounded"};
  auto resource = F.make_resource<UnboundedAllocator>(name);

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  auto gxf_typename = resource->gxf_typename();
  auto context = resource->gxf_context();
  auto cid = resource->gxf_cid();
  auto eid = resource->gxf_eid();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
}

TEST_F(ResourceClassesWithGXFContext, TestUnboundedAllocatorAllocation) {
  // For the base Allocator, this always returns true
  const std::string name{"unbounded"};
  auto resource = F.make_resource<UnboundedAllocator>(name);

  int nbytes = 1024 * 1024;
  resource->initialize();
  bool is_avail = resource->is_available(nbytes);
  EXPECT_EQ(is_avail, true);

  if (is_avail) {
    auto ptr = resource->allocate(nbytes, MemoryStorageType::kHost);
    resource->free(ptr);
  }
}

TEST_F(ResourceClassesWithGXFContext, TestUnboundedAllocatorDefaultConstructor) {
  auto resource = F.make_resource<UnboundedAllocator>();
}

TEST_F(ResourceClassesWithGXFContext, TestStdEntitySerializer) {
  const std::string name{"video-stream-serializer"};
  auto resource = F.make_resource<StdEntitySerializer>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<StdEntitySerializer>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::StdEntitySerializer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestStdEntitySerializerDefaultConstructor) {
  auto resource = F.make_resource<StdEntitySerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestReceiver) {
  const std::string name{"receiver"};
  auto resource = F.make_resource<Receiver>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<Receiver>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::Receiver"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestReceiverDefaultConstructor) {
  auto resource = F.make_resource<Receiver>();
}

TEST_F(ResourceClassesWithGXFContext, TestTransmitter) {
  const std::string name{"transmitter"};
  auto resource = F.make_resource<Transmitter>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<Transmitter>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::Transmitter"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestTransmitterDefaultConstructor) {
  auto resource = F.make_resource<Transmitter>();
}

TEST_F(ResourceClassesWithGXFContext, TestAllocator) {
  const std::string name{"allocator"};
  auto resource = F.make_resource<Allocator>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<Allocator>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::Allocator"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);

  // For the base Allocator, this always returns false
  EXPECT_EQ(resource->is_available(1024), false);

  // For the base Allocator, allocate and free exist but don't do anything
  EXPECT_EQ(resource->allocate(1024, MemoryStorageType::kHost), nullptr);
  resource->free(nullptr);
}

TEST_F(ResourceClassesWithGXFContext, TestAllocatorDefaultConstructor) {
  auto resource = F.make_resource<Allocator>();
}

TEST_F(ResourceClassesWithGXFContext, TestRealtimeClock) {
  const std::string name{"realtime"};
  ArgList arglist{Arg{"initial_time_offset", 0.0},
                  Arg{"initial_time_scale", 1.0},
                  Arg{"use_time_since_epoch", false}};
  auto resource = F.make_resource<RealtimeClock>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<RealtimeClock>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::RealtimeClock"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestRealtimeClockDefaultConstructor) {
  auto resource = F.make_resource<RealtimeClock>();
}

TEST_F(ResourceClassesWithGXFContext, TestManualClock) {
  const std::string name{"realtime"};
  ArgList arglist{Arg{"initial_timestamp", static_cast<int64_t>(0)}};
  auto resource = F.make_resource<ManualClock>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<ManualClock>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::ManualClock"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestManualClockDefaultConstructor) {
  auto resource = F.make_resource<ManualClock>();
}

TEST_F(ResourceClassesWithGXFContext, TestSyntheticClock) {
  const std::string name{"realtime"};
  ArgList arglist{Arg{"initial_timestamp", static_cast<int64_t>(0)}};
  auto resource = F.make_resource<SyntheticClock>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<SyntheticClock>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::SyntheticClock"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestSyntheticClockDefaultConstructor) {
  auto resource = F.make_resource<SyntheticClock>();
}
TEST_F(ResourceClassesWithGXFContext, TestSerializationBuffer) {
  const std::string name{"serialization_buffer"};
  ArgList arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
      Arg{"buffer_size", static_cast<size_t>(16 * 1024 * 1024)},
  };
  auto resource = F.make_resource<SerializationBuffer>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<SerializationBuffer>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::SerializationBuffer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestSerializationBufferDefaultConstructor) {
  auto resource = F.make_resource<SerializationBuffer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxSerializationBuffer) {
  const std::string name{"serialization_buffer"};
  ArgList arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
      Arg{"buffer_size", static_cast<size_t>(16 * 1024 * 1024)},
  };
  auto resource = F.make_resource<UcxSerializationBuffer>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxSerializationBuffer>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::UcxSerializationBuffer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxSerializationBufferDefaultConstructor) {
  auto resource = F.make_resource<UcxSerializationBuffer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxComponentSerializer) {
  const std::string name{"ucx_component_serializer"};
  ArgList arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
  };
  auto resource = F.make_resource<UcxComponentSerializer>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxComponentSerializer>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::UcxComponentSerializer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxComponentSerializerDefaultConstructor) {
  auto resource = F.make_resource<UcxComponentSerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxHoloscanComponentSerializer) {
  const std::string name{"ucx_holoscan_component_serializer"};
  ArgList arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
  };
  auto resource = F.make_resource<UcxHoloscanComponentSerializer>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxHoloscanComponentSerializer>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::UcxHoloscanComponentSerializer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxHoloscanComponentSerializerDefaultConstructor) {
  auto resource = F.make_resource<UcxHoloscanComponentSerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxEntitySerializer) {
  const std::string name{"entity_serializer"};

  auto component_serializer = F.make_resource<UcxEntitySerializer>(
      name, Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")});
  std::vector<std::shared_ptr<holoscan::Resource>> component_serializers{component_serializer};

  ArgList arglist{
      Arg{"component_serializers", component_serializers},
      Arg{"verbose_warning,", false},
  };
  auto resource = F.make_resource<UcxEntitySerializer>(name, arglist);

  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxEntitySerializer>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::UcxEntitySerializer"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxEntitySerializerDefaultConstructor) {
  auto resource = F.make_resource<UcxEntitySerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxReceiver) {
  const std::string name{"receiver"};

  ArgList buffer_arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
      Arg{"buffer_size", static_cast<size_t>(16 * 1024 * 1024)},
  };
  auto buffer = F.make_resource<UcxSerializationBuffer>("buffer", buffer_arglist);

  ArgList arglist{
      Arg{"buffer", buffer},
      Arg{"capacity", static_cast<uint64_t>(1)},
      Arg{"policy", static_cast<uint64_t>(2)},
      Arg{"address", std::string("0.0.0.0")},
      Arg{"port", static_cast<int32_t>(13337)},
  };
  auto resource = F.make_resource<UcxReceiver>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxReceiver>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "holoscan::HoloscanUcxReceiver"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxReceiverDefaultConstructor) {
  auto resource = F.make_resource<UcxReceiver>();
}

TEST_F(ResourceClassesWithGXFContext, TestUcxTransmitter) {
  const std::string name{"transmitter"};

  ArgList buffer_arglist{
      Arg{"allocator", F.make_resource<UnboundedAllocator>("unbounded_alloc")},
      Arg{"buffer_size", static_cast<size_t>(16 * 1024 * 1024)},
  };
  auto buffer = F.make_resource<UcxSerializationBuffer>("buffer", buffer_arglist);

  ArgList arglist{
      Arg{"buffer", buffer},
      Arg{"capacity", static_cast<uint64_t>(1)},
      Arg{"policy", static_cast<uint64_t>(2)},
      Arg{"address", std::string("10.0.0.20")},
      Arg{"port", static_cast<uint32_t>(13337)},
      Arg{"local_address", std::string("0.0.0.0")},
      Arg{"local_port", static_cast<uint32_t>(0)},
  };
  auto resource = F.make_resource<UcxTransmitter>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<UcxTransmitter>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "holoscan::HoloscanUcxTransmitter"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestUcxTransmitterDefaultConstructor) {
  auto resource = F.make_resource<UcxTransmitter>();
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThread) {
  const std::string name{"thread0"};
  Arg pin_arg{"pin_operator", true};
  auto resource = F.make_resource<CPUThread>(name, pin_arg);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CPUThread>(pin_arg)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CPUThread"s);
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThreadDefaultConstructor) {
  auto resource = F.make_resource<CPUThread>();
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThreadWithPinCores) {
  const std::string name{"thread0"};
  std::vector<uint32_t> pin_cores{0, 2};
  ArgList arglist{Arg{"pin_operator", true}, Arg{"pin_cores", pin_cores}};
  auto resource = F.make_resource<CPUThread>(name, arglist);

  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CPUThread>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CPUThread"s);
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThreadWithRealtimeScheduling) {
  const std::string name{"thread0"};
  std::vector<uint32_t> pin_cores{0, 2};
  ArgList arglist{Arg{"pin_operator", true},
                  Arg{"pin_cores", pin_cores},
                  Arg{"sched_policy", "SCHED_DEADLINE"},
                  Arg{"sched_priority", static_cast<uint32_t>(1)},
                  Arg{"sched_runtime", static_cast<uint64_t>(1000000)},
                  Arg{"sched_deadline", static_cast<uint64_t>(1000000000)},
                  Arg{"sched_period", static_cast<uint64_t>(1000000000)}};
  auto resource = F.make_resource<CPUThread>(name, arglist);

  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CPUThread>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CPUThread"s);

  // Have to initialize to be able to retrieve and check the parameter values
  // (this is expected to cause a warning due to the manual initialization)
  resource->initialize();

  EXPECT_EQ(resource->pinned(), true);
  EXPECT_EQ(resource->pin_cores().size(), 2);
  EXPECT_EQ(resource->sched_policy().value(), SchedulingPolicy::kDeadline);
  EXPECT_EQ(resource->sched_priority().value(), 1);
  EXPECT_EQ(resource->sched_runtime().value(), 1000000);
  EXPECT_EQ(resource->sched_deadline().value(), 1000000000);
  EXPECT_EQ(resource->sched_period().value(), 1000000000);
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThreadWithRealtimeSchedulingFromYAMLEnum) {
  const std::string config_file = test_config.get_test_data_file("threadpool_config.yaml");
  auto app = make_application<Application>();
  app->config(config_file);

  const std::string name{"thread0"};
  std::vector<uint32_t> pin_cores{0, 2};
  ArgList arglist = app->from_config("realtime_threadpool_enum");
  auto resource = F.make_resource<CPUThread>(name, arglist);

  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CPUThread>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CPUThread"s);

  // Have to initialize to be able to retrieve and check the parameter values
  // (this is expected to cause a warning due to the manual initialization)
  resource->initialize();
  const auto& maybe_policy = resource->sched_policy();
  EXPECT_TRUE(maybe_policy);
  if (maybe_policy) {
    EXPECT_EQ(maybe_policy.value(), SchedulingPolicy::kRoundRobin);
  }
  const auto& maybe_priority = resource->sched_priority();
  EXPECT_TRUE(maybe_priority);
  if (maybe_policy) {
    EXPECT_EQ(maybe_priority.value(), 2);
  }
  const auto& cores = resource->pin_cores();
  auto sz = cores.size();
  EXPECT_EQ(sz, 2);
  if (sz > 0) {
    EXPECT_EQ(cores[0], 0);
  }
  if (sz > 1) {
    EXPECT_EQ(cores[1], 2);
  }
}

TEST_F(ResourceClassesWithGXFContext, TestCPUThreadWithRealtimeSchedulingFromYAMLString) {
  const std::string config_file = test_config.get_test_data_file("threadpool_config.yaml");
  auto app = make_application<Application>();
  app->config(config_file);

  const std::string name{"thread0"};
  std::vector<uint32_t> pin_cores{0, 2};
  ArgList arglist = app->from_config("realtime_threadpool_string");
  auto resource = F.make_resource<CPUThread>(name, arglist);

  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<CPUThread>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::CPUThread"s);

  // Have to initialize to be able to retrieve and check the parameter values
  // (this is expected to cause a warning due to the manual initialization)
  resource->initialize();
  const auto& maybe_policy = resource->sched_policy();
  EXPECT_TRUE(maybe_policy);
  if (maybe_policy) {
    EXPECT_EQ(maybe_policy.value(), SchedulingPolicy::kFirstInFirstOut);
  }
  const auto& maybe_priority = resource->sched_priority();
  EXPECT_TRUE(maybe_priority);
  if (maybe_policy) {
    EXPECT_EQ(maybe_priority.value(), 2);
  }
  const auto& cores = resource->pin_cores();
  auto sz = cores.size();
  EXPECT_EQ(sz, 2);
  if (sz > 0) {
    EXPECT_EQ(cores[0], 0);
  }
  if (sz > 1) {
    EXPECT_EQ(cores[1], 2);
  }
}

TEST_F(ResourceClassesWithGXFContext, TestGPUDevice) {
  const std::string name{"dev0"};
  Arg id_arg{"dev_id", static_cast<int32_t>(0)};
  auto resource = F.make_resource<GPUDevice>(name, id_arg);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<GPUDevice>(id_arg)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::GPUDevice"s);
}

TEST_F(ResourceClassesWithGXFContext, TestGPUDeviceDefaultConstructor) {
  auto resource = F.make_resource<GPUDevice>();
}

TEST_F(ResourceClassesWithGXFContext, TestThreadPool) {
  const std::string name{"thread-pool0"};
  ArgList arglist{
      Arg{"initial_size", static_cast<int64_t>(5)},
      Arg{"priority", static_cast<int64_t>(2)},
  };
  auto resource = F.make_resource<ThreadPool>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<ThreadPool>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::ThreadPool"s);
}

TEST_F(ResourceClassesWithGXFContext, TestThreadPoolDefaultConstructor) {
  auto resource = F.make_resource<ThreadPool>();
}

TEST_F(ResourceClassesWithGXFContext, TestResourceUniqueDefaultNames) {
  // Test that different resource types get unique default names when created without
  // an explicit name parameter. This prevents naming conflicts when multiple unnamed
  // resources of different types are added to the same operator or fragment and ensures
  // C++ API consistency with Python API.

  // Create resources without specifying names
  auto async_buffer_receiver = F.make_resource<AsyncBufferReceiver>();
  auto async_buffer_transmitter = F.make_resource<AsyncBufferTransmitter>();
  auto block_pool = F.make_resource<BlockMemoryPool>(
      Arg{"storage_type", 1}, Arg{"block_size", 1024UL}, Arg{"num_blocks", 10UL});
  auto cpu_thread = F.make_resource<CPUThread>();
  auto cuda_green_context = F.make_resource<CudaGreenContext>();
  auto cuda_green_context_pool = F.make_resource<CudaGreenContextPool>();
  auto cuda_stream_pool = F.make_resource<CudaStreamPool>();
  auto double_buffer_receiver = F.make_resource<DoubleBufferReceiver>();
  auto double_buffer_transmitter = F.make_resource<DoubleBufferTransmitter>();
  auto gpu_device = F.make_resource<GPUDevice>();
  auto pubsub_receiver = F.make_resource<PubSubReceiver>();
  auto pubsub_transmitter = F.make_resource<PubSubTransmitter>();
  auto manual_clock = F.make_resource<ManualClock>();
  auto or_condition_combiner = F.make_resource<OrConditionCombiner>();
  auto realtime_clock = F.make_resource<RealtimeClock>();
  auto rmm_allocator = F.make_resource<RMMAllocator>();
  auto serialization_buffer = F.make_resource<SerializationBuffer>();
  auto std_component_serializer = F.make_resource<StdComponentSerializer>();
  auto std_entity_serializer = F.make_resource<StdEntitySerializer>();
  auto stream_ordered_allocator = F.make_resource<StreamOrderedAllocator>();
  auto synthetic_clock = F.make_resource<SyntheticClock>();
  auto thread_pool = F.make_resource<ThreadPool>();
  auto ucx_component_serializer = F.make_resource<UcxComponentSerializer>();
  auto ucx_entity_serializer = F.make_resource<UcxEntitySerializer>();
  auto ucx_holoscan_component_serializer = F.make_resource<UcxHoloscanComponentSerializer>();
  auto ucx_receiver = F.make_resource<UcxReceiver>();
  auto ucx_serialization_buffer = F.make_resource<UcxSerializationBuffer>();
  auto ucx_transmitter = F.make_resource<UcxTransmitter>();
  auto unbounded = F.make_resource<UnboundedAllocator>();

  // Verify each resource has the expected default name matching Python API
  EXPECT_EQ(async_buffer_receiver->name(), "async_buffer_receiver");
  EXPECT_EQ(async_buffer_transmitter->name(), "async_buffer_transmitter");
  EXPECT_EQ(block_pool->name(), "block_memory_pool");
  EXPECT_EQ(cpu_thread->name(), "cpu_thread");
  EXPECT_EQ(cuda_green_context->name(), "cuda_green_context");
  EXPECT_EQ(cuda_green_context_pool->name(), "cuda_green_context_pool");
  EXPECT_EQ(cuda_stream_pool->name(), "cuda_stream_pool");
  EXPECT_EQ(double_buffer_receiver->name(), "double_buffer_receiver");
  EXPECT_EQ(double_buffer_transmitter->name(), "double_buffer_transmitter");
  EXPECT_EQ(gpu_device->name(), "gpu_device");
  EXPECT_EQ(manual_clock->name(), "manual_clock");
  EXPECT_EQ(pubsub_receiver->name(), "pubsub_receiver");
  EXPECT_EQ(pubsub_transmitter->name(), "pubsub_transmitter");
  EXPECT_EQ(or_condition_combiner->name(), "or_condition_combiner");
  EXPECT_EQ(realtime_clock->name(), "realtime_clock");
  EXPECT_EQ(rmm_allocator->name(), "rmm_pool");
  EXPECT_EQ(serialization_buffer->name(), "serialization_buffer");
  EXPECT_EQ(stream_ordered_allocator->name(), "stream_ordered_allocator");
  EXPECT_EQ(std_component_serializer->name(), "standard_component_serializer");
  EXPECT_EQ(std_entity_serializer->name(), "standard_entity_serializer");
  EXPECT_EQ(synthetic_clock->name(), "synthetic_clock");
  EXPECT_EQ(thread_pool->name(), "thread_pool");
  EXPECT_EQ(ucx_component_serializer->name(), "ucx_component_serializer");
  EXPECT_EQ(ucx_entity_serializer->name(), "ucx_entity_serializer");
  EXPECT_EQ(ucx_holoscan_component_serializer->name(), "ucx_holoscan_component_serializer");
  EXPECT_EQ(ucx_receiver->name(), "ucx_receiver");
  EXPECT_EQ(ucx_serialization_buffer->name(), "ucx_serialization_buffer");
  EXPECT_EQ(ucx_transmitter->name(), "ucx_transmitter");
  EXPECT_EQ(unbounded->name(), "unbounded_allocator");
}

}  // namespace holoscan
