/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../utils.hpp"
#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/resources/gxf/rmm_allocator.hpp"
#include "holoscan/core/resources/gxf/serialization_buffer.hpp"
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"
#include "holoscan/core/resources/gxf/std_entity_serializer.hpp"
#include "holoscan/core/resources/gxf/stream_ordered_allocator.hpp"
#include "holoscan/core/resources/gxf/ucx_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_receiver.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"
#include "holoscan/core/resources/gxf/ucx_transmitter.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

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
      Arg{"release_threadhold", std::string{"0B"}},
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

  auto gxf_typename = resource->gxf_typename();
  auto context = resource->gxf_context();
  auto cid = resource->gxf_cid();
  auto eid = resource->gxf_eid();
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
  ArgList arglist{
      Arg{"initial_time_offset", 0.0},
      Arg{"initial_time_scale", 1.0},
      Arg{"use_time_since_epoch", false},
  };
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
  ArgList arglist{
      Arg{"initial_timestamp", static_cast<int64_t>(0)},
  };
  auto resource = F.make_resource<ManualClock>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<ManualClock>(arglist)));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::ManualClock"s);
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestManualClockDefaultConstructor) {
  auto resource = F.make_resource<ManualClock>();
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
}  // namespace holoscan
