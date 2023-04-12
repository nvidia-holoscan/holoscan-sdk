/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "../config.hpp"
#include "../utils.hpp"
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
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"
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
}

TEST_F(ResourceClassesWithGXFContext, TestCudaStreamPoolDefaultConstructor) {
  auto resource = F.make_resource<CudaStreamPool>();
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

TEST_F(ResourceClassesWithGXFContext, TestVideoStreamSerializer) {
  const std::string name{"video-stream-serializer"};
  auto resource = F.make_resource<VideoStreamSerializer>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<VideoStreamSerializer>()));
  EXPECT_EQ(std::string(resource->gxf_typename()),
            "nvidia::holoscan::stream_playback::VideoStreamSerializer"s);
}

TEST_F(ResourceClassesWithGXFContext, TestVideoStreamSerializerDefaultConstructor) {
  auto resource = F.make_resource<VideoStreamSerializer>();
}

TEST_F(ResourceClassesWithGXFContext, TestReceiver) {
  const std::string name{"receiver"};
  auto resource = F.make_resource<Receiver>(name);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<Receiver>()));
  EXPECT_EQ(std::string(resource->gxf_typename()), "nvidia::gxf::Receiver"s);
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

  // For the base Allocator, this always returns false
  EXPECT_EQ(resource->is_available(1024), false);

  // For the base Allocator, allocate and free exist but don't do anything
  EXPECT_EQ(resource->allocate(1024, MemoryStorageType::kHost), nullptr);
  resource->free(nullptr);
}

TEST_F(ResourceClassesWithGXFContext, TestAllocatorDefaultConstructor) {
  auto resource = F.make_resource<Allocator>();
}

}  // namespace holoscan
