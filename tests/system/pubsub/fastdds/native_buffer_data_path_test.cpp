// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>

#include <gxf/core/entity.hpp>
#include <gxf/pubsub/cuda_ipc_descriptor.hpp>
#include <gxf/pubsub/cuda_ipc_eligibility.hpp>
#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/gid.hpp>
#include <gxf/pubsub/pubsub_context.hpp>
#include <gxf/pubsub/pubsub_native_buffer.hpp>
#include <gxf/std/tensor.hpp>
#include <gxf/std/timestamp.hpp>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_discovery.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_native_buffer_adapter.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_serializer.hpp"
#include "holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp"

#include "config.hpp"
#include "utils.hpp"

namespace {

using eprosima::fastdds::dds::DomainParticipant;
using eprosima::fastdds::dds::DomainParticipantFactory;
using eprosima::fastdds::dds::DomainParticipantQos;

// NDPL wire format constants (must match fastdds_serializer.cpp)
constexpr uint32_t kNdplMagic = 0x4C50444E;  // "NDPL" little-endian
constexpr uint8_t kNdplVersion = 1;

bool has_cuda_device() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

std::vector<uint8_t> make_ndpl_payload(const std::string& protocol_name, uint16_t num_tensors = 0,
                                       const std::vector<uint8_t>& sideband = {}) {
  assert(protocol_name.size() <= std::numeric_limits<uint16_t>::max());
  assert(sideband.size() <= std::numeric_limits<uint32_t>::max());
  std::vector<uint8_t> payload;
  const uint16_t protocol_name_len = static_cast<uint16_t>(protocol_name.size());
  const uint32_t sideband_size = static_cast<uint32_t>(sideband.size());

  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&kNdplMagic, sizeof(kNdplMagic));
  append(&kNdplVersion, sizeof(kNdplVersion));
  append(&protocol_name_len, sizeof(protocol_name_len));
  append(protocol_name.data(), protocol_name.size());
  append(&num_tensors, sizeof(num_tensors));
  append(&sideband_size, sizeof(sideband_size));
  if (!sideband.empty()) {
    append(sideband.data(), sideband.size());
  }

  return payload;
}

std::vector<uint8_t> make_ndpl_payload_with_tensor(const std::string& protocol_name,
                                                   const std::vector<uint8_t>& descriptor_bytes,
                                                   const std::string& tensor_name,
                                                   const std::vector<uint8_t>& sideband = {}) {
  auto payload = make_ndpl_payload(protocol_name, 1, sideband);
  assert(descriptor_bytes.size() <= std::numeric_limits<uint32_t>::max());
  assert(tensor_name.size() <= std::numeric_limits<uint16_t>::max());
  const uint32_t descriptor_size = static_cast<uint32_t>(descriptor_bytes.size());
  const uint16_t tensor_name_len = static_cast<uint16_t>(tensor_name.size());

  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&descriptor_size, sizeof(descriptor_size));
  append(descriptor_bytes.data(), descriptor_bytes.size());
  append(&tensor_name_len, sizeof(tensor_name_len));
  append(tensor_name.data(), tensor_name.size());
  return payload;
}

class MockNativeProtocolAdapter : public holoscan::NativeBufferProtocolAdapter {
 public:
  bool is_initialized() const override { return true; }

  const std::string& default_protocol_name() const override { return default_protocol_name_; }

  bool supports_protocol(const std::string& protocol_name) const override {
    return protocol_name == "mock_ipc" || protocol_name == "mock_ipc_alt";
  }

  bool can_export_tensor(const nvidia::gxf::Tensor& tensor) const override {
    return tensor.storage_type() == nvidia::gxf::MemoryStorageType::kHost;
  }

  uint8_t descriptor_format_version() const override { return 7; }

  nvidia::gxf::Expected<std::vector<uint8_t>> export_tensor(
      void* device_ptr, const std::shared_ptr<void>& device_ptr_owner,
      const holoscan::NativeTensorMetadata& tensor_info, const std::string& protocol_name,
      uint32_t sequence_num) override {
    (void)device_ptr;
    (void)device_ptr_owner;
    (void)tensor_info;
    (void)sequence_num;
    if (!supports_protocol(protocol_name)) {
      return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
    }
    last_export_protocol_ = protocol_name;
    export_count_++;
    return std::vector<uint8_t>{protocol_name == "mock_ipc" ? uint8_t{0xA1} : uint8_t{0xB1}};
  }

  nvidia::gxf::Expected<holoscan::ImportedNativeTensor> import_tensor_generic(
      const std::vector<uint8_t>& descriptor_bytes, const std::string& protocol_name,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) override {
    (void)timeout;
    if (!supports_protocol(protocol_name)) {
      return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
    }
    const uint8_t expected_marker = protocol_name == "mock_ipc" ? uint8_t{0xA1} : uint8_t{0xB1};
    if (descriptor_bytes.size() != 1 || descriptor_bytes[0] != expected_marker) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }

    last_import_protocol_ = protocol_name;
    import_count_++;

    auto data = std::shared_ptr<float>(new float[4]{1.0f, 2.0f, 3.0f, 4.0f},
                                       std::default_delete<float[]>());
    holoscan::ImportedNativeTensor imported;
    imported.metadata.shape = {2, 2};
    imported.metadata.dtype = "float32";
    imported.metadata.storage_type = nvidia::gxf::MemoryStorageType::kHost;
    imported.metadata.bytes_per_element = sizeof(float);
    imported.mapped_ptr = std::shared_ptr<void>(data, data.get());
    return imported;
  }

  const std::string& last_export_protocol() const { return last_export_protocol_; }
  const std::string& last_import_protocol() const { return last_import_protocol_; }
  size_t export_count() const { return export_count_; }
  size_t import_count() const { return import_count_; }

 private:
  const std::string default_protocol_name_{"mock_ipc"};
  std::string last_export_protocol_;
  std::string last_import_protocol_;
  size_t export_count_{0};
  size_t import_count_{0};
};

// =============================================================================
// Discovery NativeBufferCapability Serialize/Parse Tests
// =============================================================================
//
// These tests verify the NBUF wire format used for advertising native buffer
// capabilities via DDS participant UserData. No DDS participant or CUDA needed.
// =============================================================================

TEST(DiscoveryNativeCapabilityTest, SerializeParseRoundTrip) {
  nvidia::gxf::NativeBufferCapability original;
  original.native_buffer_protocols = {"cuda_ipc"};
  original.memory_domain = "same_host_gpu";
  original.gpu_device_uuid = "GPU-12345678-1234-1234-1234-123456789abc";
  original.native_buffer_profile = "cuda_ipc_same_gpu_v1";
  original.descriptor_format_version = 1;
  original.host_id = "nbuf-host-roundtrip";

  auto serialized = holoscan::FastDdsDiscovery::serialize_native_capability(original);
  ASSERT_GT(serialized.size(), 0u);

  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(serialized);

  EXPECT_EQ(parsed.supports_native_buffers(), original.supports_native_buffers());
  ASSERT_EQ(parsed.native_buffer_protocols.size(), original.native_buffer_protocols.size());
  EXPECT_EQ(parsed.native_buffer_protocols[0], "cuda_ipc");
  EXPECT_EQ(parsed.memory_domain, original.memory_domain);
  EXPECT_EQ(parsed.gpu_device_uuid, original.gpu_device_uuid);
  EXPECT_EQ(parsed.native_buffer_profile, original.native_buffer_profile);
  EXPECT_EQ(parsed.descriptor_format_version, original.descriptor_format_version);
  EXPECT_EQ(parsed.host_id, original.host_id);
}

TEST(DiscoveryNativeCapabilityTest, EmptyCapabilityRoundTrip) {
  nvidia::gxf::NativeBufferCapability original;
  // All defaults: supports=false, no protocols, empty strings, version=0

  auto serialized = holoscan::FastDdsDiscovery::serialize_native_capability(original);
  ASSERT_GT(serialized.size(), 0u);

  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(serialized);

  EXPECT_FALSE(parsed.supports_native_buffers());
  EXPECT_TRUE(parsed.native_buffer_protocols.empty());
  EXPECT_TRUE(parsed.memory_domain.empty());
  EXPECT_TRUE(parsed.gpu_device_uuid.empty());
  EXPECT_TRUE(parsed.native_buffer_profile.empty());
  EXPECT_EQ(parsed.descriptor_format_version, 0u);
  EXPECT_TRUE(parsed.host_id.empty());
}

TEST(DiscoveryNativeCapabilityTest, MultipleProtocolsRoundTrip) {
  nvidia::gxf::NativeBufferCapability original;
  original.native_buffer_protocols = {"cuda_ipc", "cuda_vmm", "host_shm"};
  original.memory_domain = "same_host_gpu";
  original.gpu_device_uuid = "GPU-AABBCCDD-1122-3344-5566";
  original.native_buffer_profile = "cuda_ipc_same_gpu_v1";
  original.descriptor_format_version = 2;

  auto serialized = holoscan::FastDdsDiscovery::serialize_native_capability(original);
  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(serialized);

  EXPECT_TRUE(parsed.supports_native_buffers());
  ASSERT_EQ(parsed.native_buffer_protocols.size(), 3u);
  EXPECT_EQ(parsed.native_buffer_protocols[0], "cuda_ipc");
  EXPECT_EQ(parsed.native_buffer_protocols[1], "cuda_vmm");
  EXPECT_EQ(parsed.native_buffer_protocols[2], "host_shm");
  EXPECT_EQ(parsed.descriptor_format_version, 2u);
}

TEST(DiscoveryNativeCapabilityTest, TruncatedDataReturnsDefaults) {
  // Provide only the magic bytes (truncated before version)
  std::vector<uint8_t> truncated = {0x4E, 0x42, 0x55, 0x46};  // "NBUF" magic

  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(truncated);

  // Should return defaults since parsing fails partway through
  EXPECT_FALSE(parsed.supports_native_buffers());
  EXPECT_TRUE(parsed.native_buffer_protocols.empty());
}

TEST(DiscoveryNativeCapabilityTest, WrongMagicReturnsDefaults) {
  std::vector<uint8_t> bad_magic = {0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x01, 0x00};

  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(bad_magic);

  EXPECT_FALSE(parsed.supports_native_buffers());
}

TEST(DiscoveryNativeCapabilityTest, EmptyInputReturnsDefaults) {
  std::vector<uint8_t> empty;

  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(empty);

  EXPECT_FALSE(parsed.supports_native_buffers());
}

TEST(DiscoveryNativeCapabilityTest, UuidFieldRoundTrip) {
  // Test with full 40-char canonical UUID (the field size)
  nvidia::gxf::NativeBufferCapability cap;
  cap.native_buffer_protocols = {"cuda_ipc"};
  cap.gpu_device_uuid = "GPU-12345678-1234-1234-1234-123456789abc";  // 40 chars

  auto serialized = holoscan::FastDdsDiscovery::serialize_native_capability(cap);
  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(serialized);

  EXPECT_EQ(parsed.gpu_device_uuid, cap.gpu_device_uuid);
}

TEST(DiscoveryNativeCapabilityTest, ShortUuidPaddedCorrectly) {
  nvidia::gxf::NativeBufferCapability cap;
  cap.native_buffer_protocols = {"cuda_ipc"};
  cap.gpu_device_uuid = "GPU-ABCD";  // Much shorter than 40 bytes

  auto serialized = holoscan::FastDdsDiscovery::serialize_native_capability(cap);
  auto parsed = holoscan::FastDdsDiscovery::parse_native_capability(serialized);

  EXPECT_EQ(parsed.gpu_device_uuid, "GPU-ABCD");
}

// =============================================================================
// NDPL Payload Wire Format Tests
// =============================================================================
//
// These tests directly construct and validate the NDPL (Native Descriptor
// PayLoad) binary format used by FastDdsSerializer for native descriptor transport.
// =============================================================================

TEST(NDPLPayloadFormatTest, MagicAndVersionConstants) {
  // Verify the NDPL magic is "NDPL" in little-endian
  EXPECT_EQ(kNdplMagic, 0x4C50444Eu);

  const auto* magic_bytes = reinterpret_cast<const uint8_t*>(&kNdplMagic);
  EXPECT_EQ(magic_bytes[0], 'N');
  EXPECT_EQ(magic_bytes[1], 'D');
  EXPECT_EQ(magic_bytes[2], 'P');
  EXPECT_EQ(magic_bytes[3], 'L');

  EXPECT_EQ(kNdplVersion, 1u);
}

TEST(NDPLPayloadFormatTest, MinimalHeaderSize) {
  // NDPL header: [4]magic + [1]version + [2]protocol_len + [2]num_tensors + [4]sideband_size
  constexpr size_t kMinHeaderSize =
      sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint32_t);
  EXPECT_EQ(kMinHeaderSize, 13u);
}

TEST(NDPLPayloadFormatTest, ConstructValidEmptyPayload) {
  // Build an NDPL payload with 0 GPU tensors and 0 sideband bytes
  constexpr const char* kProtocolName = "cuda_ipc";
  const uint16_t protocol_name_len = static_cast<uint16_t>(std::strlen(kProtocolName));
  std::vector<uint8_t> payload;
  payload.reserve(13 + protocol_name_len);

  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&kNdplMagic, sizeof(kNdplMagic));
  append(&kNdplVersion, sizeof(kNdplVersion));
  append(&protocol_name_len, sizeof(protocol_name_len));
  append(kProtocolName, protocol_name_len);
  uint16_t num_tensors = 0;
  append(&num_tensors, sizeof(num_tensors));
  uint32_t sideband_size = 0;
  append(&sideband_size, sizeof(sideband_size));

  EXPECT_EQ(payload.size(), 13u + protocol_name_len);

  // Verify we can parse the header back
  uint32_t magic = 0;
  std::memcpy(&magic, payload.data(), sizeof(magic));
  EXPECT_EQ(magic, kNdplMagic);

  uint8_t version = payload[4];
  EXPECT_EQ(version, kNdplVersion);

  uint16_t parsed_protocol_len = 0;
  std::memcpy(&parsed_protocol_len, payload.data() + 5, sizeof(parsed_protocol_len));
  EXPECT_EQ(parsed_protocol_len, protocol_name_len);
  std::string parsed_protocol(reinterpret_cast<const char*>(payload.data() + 7),
                              parsed_protocol_len);
  EXPECT_EQ(parsed_protocol, kProtocolName);

  uint16_t parsed_tensors = 0;
  std::memcpy(&parsed_tensors, payload.data() + 7 + parsed_protocol_len, sizeof(parsed_tensors));
  EXPECT_EQ(parsed_tensors, 0u);

  uint32_t parsed_sideband = 0;
  std::memcpy(&parsed_sideband,
              payload.data() + 7 + parsed_protocol_len + sizeof(parsed_tensors),
              sizeof(parsed_sideband));
  EXPECT_EQ(parsed_sideband, 0u);
}

TEST(NDPLPayloadFormatTest, ConstructPayloadWithSidebandOnly) {
  // Build an NDPL payload with 0 GPU tensors but non-empty sideband
  constexpr const char* kProtocolName = "cuda_ipc";
  const uint16_t protocol_name_len = static_cast<uint16_t>(std::strlen(kProtocolName));
  std::vector<uint8_t> sideband_data = {0x01, 0x02, 0x03, 0x04, 0x05};

  std::vector<uint8_t> payload;
  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&kNdplMagic, sizeof(kNdplMagic));
  append(&kNdplVersion, sizeof(kNdplVersion));
  append(&protocol_name_len, sizeof(protocol_name_len));
  append(kProtocolName, protocol_name_len);
  uint16_t num_tensors = 0;
  append(&num_tensors, sizeof(num_tensors));
  uint32_t sideband_size = static_cast<uint32_t>(sideband_data.size());
  append(&sideband_size, sizeof(sideband_size));
  append(sideband_data.data(), sideband_data.size());

  EXPECT_EQ(payload.size(), 13u + protocol_name_len + sideband_data.size());

  // Parse back and verify sideband content
  uint32_t parsed_sideband_size = 0;
  std::memcpy(&parsed_sideband_size,
              payload.data() + 7 + protocol_name_len + sizeof(num_tensors),
              sizeof(parsed_sideband_size));
  EXPECT_EQ(parsed_sideband_size, 5u);

  const uint8_t* sideband_ptr = payload.data() + 13 + protocol_name_len;
  EXPECT_EQ(std::memcmp(sideband_ptr, sideband_data.data(), sideband_data.size()), 0);
}

TEST(NDPLPayloadFormatTest, ConstructPayloadWithTensorDescriptor) {
  // Build an NDPL payload with 1 GPU tensor containing an HIPC descriptor
  // We use a dummy HIPC descriptor for testing the framing layer
  constexpr const char* kProtocolName = "cuda_ipc";
  const uint16_t protocol_name_len = static_cast<uint16_t>(std::strlen(kProtocolName));

  // Create a mock HIPC descriptor (just some bytes)
  nvidia::gxf::CudaIpcDescriptor desc;
  std::memset(&desc.mem_handle, 0xAB, sizeof(desc.mem_handle));
  desc.has_event_handle = false;
  desc.gpu_device_uuid = "GPU-TEST-UUID";
  desc.gpu_device_id = 0;
  desc.byte_size = 1024;
  desc.tensor_info.shape = {1, 3, 8, 8};
  desc.tensor_info.strides = {768, 256, 32, 4};
  desc.tensor_info.dtype = "float32";
  desc.tensor_info.storage_type = 1;
  desc.tensor_info.bytes_per_element = 4;
  desc.producing_timestamp_ns = 12345;
  desc.sequence_number = 1;

  auto maybe_hipc = nvidia::gxf::serialize_cuda_ipc_descriptor(desc);
  ASSERT_TRUE(maybe_hipc) << "Failed to serialize test HIPC descriptor";
  auto& hipc_bytes = maybe_hipc.value();

  std::string tensor_name = "test_tensor";

  // Build NDPL
  std::vector<uint8_t> payload;
  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&kNdplMagic, sizeof(kNdplMagic));
  append(&kNdplVersion, sizeof(kNdplVersion));
  append(&protocol_name_len, sizeof(protocol_name_len));
  append(kProtocolName, protocol_name_len);
  uint16_t num_tensors = 1;
  append(&num_tensors, sizeof(num_tensors));
  uint32_t sideband_size = 0;
  append(&sideband_size, sizeof(sideband_size));

  // Tensor entry: [4]desc_size [D]hipc_bytes [2]name_len [K]name
  assert(hipc_bytes.size() <= std::numeric_limits<uint32_t>::max());
  assert(tensor_name.size() <= std::numeric_limits<uint16_t>::max());
  uint32_t descriptor_size = static_cast<uint32_t>(hipc_bytes.size());
  uint16_t name_len = static_cast<uint16_t>(tensor_name.size());

  append(&descriptor_size, sizeof(descriptor_size));
  append(hipc_bytes.data(), hipc_bytes.size());
  append(&name_len, sizeof(name_len));
  append(tensor_name.data(), tensor_name.size());

  // Parse the embedded HIPC descriptor back from the payload
  // Skip NDPL header + tensor entry header (4 bytes)
  size_t hipc_offset = 13 + protocol_name_len + sizeof(uint32_t);
  auto maybe_parsed =
      nvidia::gxf::deserialize_cuda_ipc_descriptor(payload.data() + hipc_offset, descriptor_size);
  ASSERT_TRUE(maybe_parsed);

  auto& parsed = maybe_parsed.value();
  EXPECT_EQ(parsed.byte_size, 1024u);
  EXPECT_EQ(parsed.gpu_device_uuid, "GPU-TEST-UUID");
  ASSERT_EQ(parsed.tensor_info.shape.size(), 4u);
  EXPECT_EQ(parsed.tensor_info.shape[0], 1);
  EXPECT_EQ(parsed.tensor_info.shape[1], 3);
  EXPECT_EQ(parsed.tensor_info.dtype, "float32");
  EXPECT_EQ(parsed.sequence_number, 1u);

  // Parse the tensor name from the payload
  size_t name_offset = hipc_offset + descriptor_size;
  uint16_t parsed_name_len = 0;
  std::memcpy(&parsed_name_len, payload.data() + name_offset, sizeof(parsed_name_len));
  EXPECT_EQ(parsed_name_len, tensor_name.size());

  std::string parsed_name(
      reinterpret_cast<const char*>(payload.data() + name_offset + sizeof(uint16_t)),
      parsed_name_len);
  EXPECT_EQ(parsed_name, tensor_name);
}

TEST(NDPLPayloadFormatTest, MultiTensorPayloadLayout) {
  // Build an NDPL payload with 3 GPU tensor entries to verify multi-tensor framing
  constexpr const char* kProtocolName = "cuda_ipc";
  const uint16_t protocol_name_len = static_cast<uint16_t>(std::strlen(kProtocolName));
  constexpr int kNumTensors = 3;
  std::vector<std::string> tensor_names = {"rgb_frame", "depth_map", "confidence"};
  std::vector<uint64_t> byte_sizes = {1920 * 1080 * 3, 1920 * 1080 * 4, 1920 * 1080};

  std::vector<std::vector<uint8_t>> hipc_blobs;
  for (int i = 0; i < kNumTensors; ++i) {
    nvidia::gxf::CudaIpcDescriptor desc;
    std::memset(&desc.mem_handle, static_cast<int>(0xA0 + i), sizeof(desc.mem_handle));
    desc.has_event_handle = false;
    desc.gpu_device_uuid = "GPU-MULTI-TEST";
    desc.gpu_device_id = 0;
    desc.byte_size = byte_sizes[i];
    desc.tensor_info.shape = {1080, 1920, (i == 0) ? int64_t{3} : int64_t{1}};
    desc.tensor_info.strides = {};
    desc.tensor_info.dtype = (i == 1) ? "float32" : "uint8";
    desc.tensor_info.storage_type = 1;
    desc.tensor_info.bytes_per_element = (i == 1) ? uint64_t{4} : uint64_t{1};
    desc.producing_timestamp_ns = 100 * (i + 1);
    desc.sequence_number = i;

    auto maybe_hipc = nvidia::gxf::serialize_cuda_ipc_descriptor(desc);
    ASSERT_TRUE(maybe_hipc);
    hipc_blobs.push_back(std::move(maybe_hipc.value()));
  }

  // Build NDPL payload
  std::vector<uint8_t> payload;
  auto append = [&payload](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), bytes, bytes + size);
  };

  append(&kNdplMagic, sizeof(kNdplMagic));
  append(&kNdplVersion, sizeof(kNdplVersion));
  append(&protocol_name_len, sizeof(protocol_name_len));
  append(kProtocolName, protocol_name_len);
  uint16_t num_tensors = kNumTensors;
  append(&num_tensors, sizeof(num_tensors));
  uint32_t sideband_size = 0;
  append(&sideband_size, sizeof(sideband_size));

  for (int i = 0; i < kNumTensors; ++i) {
    assert(hipc_blobs[i].size() <= std::numeric_limits<uint32_t>::max());
    assert(tensor_names[i].size() <= std::numeric_limits<uint16_t>::max());
    uint32_t desc_size = static_cast<uint32_t>(hipc_blobs[i].size());
    uint16_t name_len = static_cast<uint16_t>(tensor_names[i].size());

    append(&desc_size, sizeof(desc_size));
    append(hipc_blobs[i].data(), hipc_blobs[i].size());
    append(&name_len, sizeof(name_len));
    append(tensor_names[i].data(), tensor_names[i].size());
  }

  // Walk through the payload and verify each tensor entry can be parsed
  size_t offset = 13 + protocol_name_len;  // skip NDPL header
  for (int i = 0; i < kNumTensors; ++i) {
    uint32_t desc_size = 0;
    std::memcpy(&desc_size, payload.data() + offset, sizeof(desc_size));
    offset += sizeof(desc_size);

    auto maybe_desc =
        nvidia::gxf::deserialize_cuda_ipc_descriptor(payload.data() + offset, desc_size);
    ASSERT_TRUE(maybe_desc) << "Failed to parse tensor " << i;
    offset += desc_size;

    EXPECT_EQ(maybe_desc.value().byte_size, byte_sizes[i]);
    EXPECT_EQ(maybe_desc.value().sequence_number, static_cast<uint32_t>(i));

    uint16_t name_len = 0;
    std::memcpy(&name_len, payload.data() + offset, sizeof(name_len));
    offset += sizeof(name_len);

    std::string name(reinterpret_cast<const char*>(payload.data() + offset), name_len);
    EXPECT_EQ(name, tensor_names[i]);
    offset += name_len;
  }

  EXPECT_EQ(offset, payload.size());
}

// =============================================================================
// FastDdsNativeBufferAdapter Lifecycle Tests
// =============================================================================
//
// These tests verify the adapter's lifecycle management: initialization,
// shutdown, pending export tracking, and stale eviction. Tests requiring
// a DDS participant create a lightweight one for the adapter.
// =============================================================================

class DDSNativeBufferAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto* factory = DomainParticipantFactory::get_instance();
    DomainParticipantQos qos;
    factory->get_default_participant_qos(qos);
    // Use a high domain ID to avoid collision with other tests
    participant_ = factory->create_participant(99, qos);
    ASSERT_NE(participant_, nullptr);
  }

  void TearDown() override {
    adapter_.reset();
    if (participant_) {
      auto* factory = DomainParticipantFactory::get_instance();
      factory->delete_participant(participant_);
      participant_ = nullptr;
    }
  }

  DomainParticipant* participant_ = nullptr;
  std::unique_ptr<holoscan::FastDdsNativeBufferAdapter> adapter_ =
      std::make_unique<holoscan::FastDdsNativeBufferAdapter>();
};

TEST_F(DDSNativeBufferAdapterTest, InitializeAndShutdown) {
  EXPECT_FALSE(adapter_->is_initialized());

  auto result = adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred);
  ASSERT_TRUE(result) << "initialize() failed";
  EXPECT_TRUE(adapter_->is_initialized());
  EXPECT_EQ(adapter_->policy(), nvidia::gxf::NativeBufferPolicy::kPreferred);

  adapter_->shutdown();
  EXPECT_FALSE(adapter_->is_initialized());
}

TEST_F(DDSNativeBufferAdapterTest, InitializeWithDisabledPolicy) {
  auto result = adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kDisabled);
  ASSERT_TRUE(result) << "initialize() with kDisabled failed";
  EXPECT_TRUE(adapter_->is_initialized());
  EXPECT_EQ(adapter_->policy(), nvidia::gxf::NativeBufferPolicy::kDisabled);
}

TEST_F(DDSNativeBufferAdapterTest, InitializeWithRequiredPolicy) {
  auto result = adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kRequired);
  ASSERT_TRUE(result) << "initialize() with kRequired failed";
  EXPECT_TRUE(adapter_->is_initialized());
  EXPECT_EQ(adapter_->policy(), nvidia::gxf::NativeBufferPolicy::kRequired);
}

TEST_F(DDSNativeBufferAdapterTest, DoubleInitializeWarns) {
  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));
  EXPECT_TRUE(adapter_->is_initialized());

  // Second init should succeed (logs a warning) without crashing
  auto result = adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kRequired);
  EXPECT_TRUE(result);
  EXPECT_TRUE(adapter_->is_initialized());
}

TEST_F(DDSNativeBufferAdapterTest, ShutdownWithoutInitialize) {
  EXPECT_FALSE(adapter_->is_initialized());
  adapter_->shutdown();  // Should not crash
  EXPECT_FALSE(adapter_->is_initialized());
}

TEST_F(DDSNativeBufferAdapterTest, InitializeNullParticipant) {
  auto result = adapter_->initialize(nullptr, nvidia::gxf::NativeBufferPolicy::kPreferred);
  EXPECT_FALSE(result);
  EXPECT_FALSE(adapter_->is_initialized());
}

TEST_F(DDSNativeBufferAdapterTest, PendingExportCountInitiallyZero) {
  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));

  EXPECT_EQ(adapter_->pending_export_count(), 0u);
}

TEST_F(DDSNativeBufferAdapterTest, EvictStaleExportsOnEmpty) {
  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));

  size_t evicted = adapter_->evict_stale_exports(std::chrono::milliseconds{100});
  EXPECT_EQ(evicted, 0u);
  EXPECT_EQ(adapter_->pending_export_count(), 0u);
}

TEST_F(DDSNativeBufferAdapterTest, ExportTensorFailsWhenNotInitialized) {
  EXPECT_FALSE(adapter_->is_initialized());

  nvidia::gxf::CudaTensorDescriptor tensor_info;
  tensor_info.shape = {2, 3};
  tensor_info.dtype = "float32";
  tensor_info.bytes_per_element = 4;

  auto result = adapter_->export_tensor(nullptr, nullptr, tensor_info, "GPU-TEST", 0, 0);
  EXPECT_FALSE(result);
}

TEST_F(DDSNativeBufferAdapterTest, ImportTensorFailsWhenNotInitialized) {
  EXPECT_FALSE(adapter_->is_initialized());

  auto result = adapter_->import_tensor(std::vector<uint8_t>{});
  EXPECT_FALSE(result);
}

TEST_F(DDSNativeBufferAdapterTest, DestructorCleansUp) {
  // Initialize, then let the adapter go out of scope via reset
  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));
  EXPECT_TRUE(adapter_->is_initialized());

  adapter_.reset();  // Destructor should call shutdown()
}

// =============================================================================
// FastDdsNativeBufferAdapter GPU Export Tests (requires CUDA device)
// =============================================================================

TEST_F(DDSNativeBufferAdapterTest, ExportTensorProducesValidWrappedDescriptor) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));

  // Allocate a small GPU buffer
  void* device_ptr = nullptr;
  constexpr size_t kBufferSize = 1024;
  ASSERT_EQ(cudaMalloc(&device_ptr, kBufferSize), cudaSuccess);
  ASSERT_NE(device_ptr, nullptr);

  // Wrap in shared_ptr for lifetime tracking
  auto device_owner = std::shared_ptr<void>(device_ptr, [](void* p) { cudaFree(p); });

  nvidia::gxf::CudaTensorDescriptor tensor_info;
  tensor_info.shape = {1, 256};
  tensor_info.strides = {1024, 4};
  tensor_info.dtype = "float32";
  tensor_info.storage_type = 1;  // kDevice
  tensor_info.bytes_per_element = 4;

  // Query GPU UUID
  cudaDeviceProp props;
  ASSERT_EQ(cudaGetDeviceProperties(&props, 0), cudaSuccess);
  char uuid_str[64];
  auto* u = reinterpret_cast<const uint8_t*>(&props.uuid);
  snprintf(uuid_str,
           sizeof(uuid_str),
           "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           u[0],
           u[1],
           u[2],
           u[3],
           u[4],
           u[5],
           u[6],
           u[7],
           u[8],
           u[9],
           u[10],
           u[11],
           u[12],
           u[13],
           u[14],
           u[15]);

  auto result = adapter_->export_tensor(
      device_ptr, device_owner, tensor_info, std::string(uuid_str), 0, /*seq=*/42);
  ASSERT_TRUE(result) << "export_tensor() failed";

  // The result should be a valid HIPC wire-format blob
  auto& hipc_bytes = result.value();
  EXPECT_GT(hipc_bytes.size(), 0u);

  auto maybe_exported = holoscan::FastDdsNativeBufferAdapter::decode_exported_tensor(hipc_bytes);
  ASSERT_TRUE(maybe_exported);

  auto& desc = maybe_exported.value().gxf_descriptor;
  EXPECT_EQ(desc.byte_size, kBufferSize);
  EXPECT_EQ(desc.sequence_number, 42u);
  EXPECT_EQ(desc.tensor_info.dtype, "float32");
  ASSERT_EQ(desc.tensor_info.shape.size(), 2u);
  EXPECT_EQ(desc.tensor_info.shape[0], 1);
  EXPECT_EQ(desc.tensor_info.shape[1], 256);
  EXPECT_FALSE(maybe_exported.value().lifecycle_key.empty());
  EXPECT_FALSE(maybe_exported.value().lifecycle_reply_to_topic_name.empty());

  // Pending export count should have incremented
  EXPECT_EQ(adapter_->pending_export_count(), 1u);
}

TEST_F(DDSNativeBufferAdapterTest, ExportMultipleTensorsTracksPending) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));

  constexpr int kNumExports = 5;
  std::vector<std::shared_ptr<void>> device_owners;

  for (int i = 0; i < kNumExports; ++i) {
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 256), cudaSuccess);
    device_owners.push_back(std::shared_ptr<void>(device_ptr, [](void* p) { cudaFree(p); }));

    nvidia::gxf::CudaTensorDescriptor tensor_info;
    tensor_info.shape = {16, 16};
    tensor_info.strides = {64, 4};
    tensor_info.dtype = "float32";
    tensor_info.storage_type = 1;
    tensor_info.bytes_per_element = 4;

    auto result =
        adapter_->export_tensor(device_ptr, device_owners.back(), tensor_info, "GPU-TEST", 0, i);
    ASSERT_TRUE(result) << "export_tensor() failed for seq=" << i;
  }

  EXPECT_EQ(adapter_->pending_export_count(), static_cast<size_t>(kNumExports));
}

TEST_F(DDSNativeBufferAdapterTest, EvictStaleExportsRemovesOldEntries) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));

  // Export a tensor
  void* device_ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&device_ptr, 256), cudaSuccess);
  auto device_owner = std::shared_ptr<void>(device_ptr, [](void* p) { cudaFree(p); });

  nvidia::gxf::CudaTensorDescriptor tensor_info;
  tensor_info.shape = {8, 8};
  tensor_info.strides = {32, 4};
  tensor_info.dtype = "float32";
  tensor_info.storage_type = 1;
  tensor_info.bytes_per_element = 4;

  ASSERT_TRUE(adapter_->export_tensor(device_ptr, device_owner, tensor_info, "GPU-TEST", 0, 0));
  EXPECT_EQ(adapter_->pending_export_count(), 1u);

  // Evict with a very long max_age (should NOT evict)
  size_t evicted = adapter_->evict_stale_exports(std::chrono::milliseconds{60000});
  EXPECT_EQ(evicted, 0u);
  EXPECT_EQ(adapter_->pending_export_count(), 1u);

  // Sleep briefly, then evict with a very short max_age (should evict)
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  evicted = adapter_->evict_stale_exports(std::chrono::milliseconds{10});
  EXPECT_EQ(evicted, 1u);
  EXPECT_EQ(adapter_->pending_export_count(), 0u);
}

TEST_F(DDSNativeBufferAdapterTest, ExportTensorEvictsStaleEntriesOnNextExport) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  ASSERT_TRUE(adapter_->initialize(participant_, nvidia::gxf::NativeBufferPolicy::kPreferred));
  adapter_->set_export_ttl(std::chrono::milliseconds{10});

  auto make_tensor_info = []() {
    nvidia::gxf::CudaTensorDescriptor tensor_info;
    tensor_info.shape = {8, 8};
    tensor_info.strides = {32, 4};
    tensor_info.dtype = "float32";
    tensor_info.storage_type = 1;
    tensor_info.bytes_per_element = 4;
    return tensor_info;
  };

  void* device_ptr1 = nullptr;
  ASSERT_EQ(cudaMalloc(&device_ptr1, 256), cudaSuccess);
  auto owner1 = std::shared_ptr<void>(device_ptr1, [](void* p) { cudaFree(p); });
  ASSERT_TRUE(adapter_->export_tensor(device_ptr1, owner1, make_tensor_info(), "GPU-TEST", 0, 1));
  EXPECT_EQ(adapter_->pending_export_count(), 1u);

  std::this_thread::sleep_for(std::chrono::milliseconds{30});

  void* device_ptr2 = nullptr;
  ASSERT_EQ(cudaMalloc(&device_ptr2, 256), cudaSuccess);
  auto owner2 = std::shared_ptr<void>(device_ptr2, [](void* p) { cudaFree(p); });
  ASSERT_TRUE(adapter_->export_tensor(device_ptr2, owner2, make_tensor_info(), "GPU-TEST", 0, 2));

  // The second export performs opportunistic TTL cleanup, so only the fresh
  // export should remain pending.
  EXPECT_EQ(adapter_->pending_export_count(), 1u);
}

// =============================================================================
// FastDdsSerializer Native Descriptor Tests
// =============================================================================
//
// These tests verify the serializer's native descriptor interface, including
// the supports_native_descriptors() flag and export/import behavior.
// The full round-trip with GPU tensors requires CUDA + adapter.
// =============================================================================

class DDSSerializerNativeTest : public holoscan::TestWithGXFContext {
 protected:
  void SetUp() override {
    holoscan::TestWithGXFContext::SetUp();
    serializer_ = std::make_unique<holoscan::FastDdsSerializer>();

    holoscan::ArgList args{};
    allocator_ = F.make_resource<holoscan::UnboundedAllocator>("test_allocator", args);
    allocator_->initialize();
  }

  void TearDown() override {
    allocator_.reset();
    serializer_.reset();
    holoscan::TestWithGXFContext::TearDown();
  }

  gxf_context_t context() { return F.executor().context(); }

  nvidia::gxf::Handle<nvidia::gxf::Allocator> get_allocator_handle() {
    return nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context(), allocator_->gxf_cid())
        .value();
  }

  std::unique_ptr<holoscan::FastDdsSerializer> serializer_;
  std::shared_ptr<holoscan::UnboundedAllocator> allocator_;
};

TEST_F(DDSSerializerNativeTest, DoesNotSupportNativeDescriptorsWithoutAdapter) {
  EXPECT_FALSE(serializer_->supports_native_descriptors());
}

TEST_F(DDSSerializerNativeTest, SupportsNativeDescriptorsWithAdapter) {
  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  serializer_->set_native_buffer_adapter(adapter);

  EXPECT_TRUE(serializer_->supports_native_descriptors());
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorFailsWithoutAdapter) {
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      maybe_entity.value().eid(), context(), get_allocator_handle(), payload);
  EXPECT_FALSE(result);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorRejectsUnsupportedProtocol) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(92, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("gpu_tensor");
  ASSERT_TRUE(maybe_tensor);
  nvidia::gxf::Shape shape({2, 2});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(maybe_tensor.value()->reshapeCustom(shape,
                                                  nvidia::gxf::PrimitiveType::kFloat32,
                                                  sizeof(float),
                                                  strides,
                                                  nvidia::gxf::MemoryStorageType::kDevice,
                                                  get_allocator_handle()));

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload, "host_shm");
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error(), GXF_NOT_IMPLEMENTED);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorUsesRequestedMockProtocol) {
  auto adapter = std::make_shared<MockNativeProtocolAdapter>();
  serializer_->set_native_buffer_adapter(adapter);

  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("host_tensor");
  ASSERT_TRUE(maybe_tensor);
  nvidia::gxf::Shape shape({2, 2});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(maybe_tensor.value()->reshapeCustom(shape,
                                                  nvidia::gxf::PrimitiveType::kFloat32,
                                                  sizeof(float),
                                                  strides,
                                                  nvidia::gxf::MemoryStorageType::kHost,
                                                  get_allocator_handle()));

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload, "mock_ipc");
  ASSERT_TRUE(result);

  EXPECT_EQ(adapter->export_count(), 1u);
  EXPECT_EQ(adapter->last_export_protocol(), "mock_ipc");
  EXPECT_EQ(payload.protocol_name, "mock_ipc");
  EXPECT_EQ(payload.descriptor_format_version, 7u);

  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  EXPECT_EQ(protocol_name_len, payload.protocol_name.size());

  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 1u);
}

TEST_F(DDSSerializerNativeTest, ImportNativeDescriptorFailsWithoutAdapter) {
  nvidia::gxf::NativeDescriptorPayload payload;
  payload.descriptor_bytes = {0x00};  // dummy

  auto result = serializer_->import_native_descriptor(payload, context(), get_allocator_handle());
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error(), GXF_NOT_IMPLEMENTED);
}

TEST_F(DDSSerializerNativeTest, ImportNativeDescriptorUsesPayloadProtocolWithMockAdapter) {
  auto adapter = std::make_shared<MockNativeProtocolAdapter>();
  serializer_->set_native_buffer_adapter(adapter);

  nvidia::gxf::NativeDescriptorPayload payload;
  payload.protocol_name = "mock_ipc";
  payload.descriptor_format_version = 7;
  payload.descriptor_bytes = make_ndpl_payload_with_tensor("mock_ipc", {0xA1}, "mock_tensor");

  auto result = serializer_->import_native_descriptor(payload, context(), get_allocator_handle());
  ASSERT_TRUE(result);

  EXPECT_EQ(adapter->import_count(), 1u);
  EXPECT_EQ(adapter->last_import_protocol(), "mock_ipc");

  auto maybe_tensor = result.value().get<nvidia::gxf::Tensor>("mock_tensor");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();
  EXPECT_EQ(tensor->rank(), 2u);
  EXPECT_EQ(tensor->shape().dimension(0), 2);
  EXPECT_EQ(tensor->shape().dimension(1), 2);
  EXPECT_EQ(tensor->storage_type(), nvidia::gxf::MemoryStorageType::kHost);

  float* values = reinterpret_cast<float*>(tensor->pointer());
  ASSERT_NE(values, nullptr);
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[3], 4.0f);
}

TEST_F(DDSSerializerNativeTest, ImportNativeDescriptorRejectsUnsupportedProtocol) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(91, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  nvidia::gxf::NativeDescriptorPayload payload;
  payload.protocol_name = "host_shm";
  payload.descriptor_bytes = make_ndpl_payload("host_shm");

  auto result = serializer_->import_native_descriptor(payload, context(), get_allocator_handle());
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error(), GXF_NOT_IMPLEMENTED);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ImportNativeDescriptorRejectsProtocolMismatch) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(90, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  nvidia::gxf::NativeDescriptorPayload payload;
  payload.protocol_name = "cuda_ipc";
  payload.descriptor_bytes = make_ndpl_payload("host_shm");

  auto result = serializer_->import_native_descriptor(payload, context(), get_allocator_handle());
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error(), GXF_PUBSUB_SERIALIZATION_FAILED);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ImportNativeDescriptorRejectsMissingPayloadProtocol) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(89, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  nvidia::gxf::NativeDescriptorPayload payload;
  payload.descriptor_bytes = make_ndpl_payload("cuda_ipc");

  auto result = serializer_->import_native_descriptor(payload, context(), get_allocator_handle());
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error(), GXF_PUBSUB_SERIALIZATION_FAILED);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorHostTensorOnly) {
  // With adapter set but only host tensors in the entity,
  // export_native_descriptor should return GXF_NOT_IMPLEMENTED
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available (needed for adapter init)";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(98, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  // Create entity with a host-only tensor
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("host_tensor");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();

  nvidia::gxf::Shape shape({4, 4});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(tensor->reshapeCustom(shape,
                                    nvidia::gxf::PrimitiveType::kFloat32,
                                    sizeof(float),
                                    strides,
                                    nvidia::gxf::MemoryStorageType::kHost,
                                    get_allocator_handle()));

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);

  // Should fail because there are no GPU tensors
  EXPECT_FALSE(result);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorMixedHostAndGpuTensorsPreferredUsesSideband) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(94, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_gpu_tensor = entity.add<nvidia::gxf::Tensor>("gpu_tensor");
  ASSERT_TRUE(maybe_gpu_tensor);
  auto gpu_tensor = maybe_gpu_tensor.value();
  nvidia::gxf::Shape shape({2, 2});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(gpu_tensor->reshapeCustom(shape,
                                        nvidia::gxf::PrimitiveType::kFloat32,
                                        sizeof(float),
                                        strides,
                                        nvidia::gxf::MemoryStorageType::kDevice,
                                        get_allocator_handle()));

  auto maybe_host_tensor = entity.add<nvidia::gxf::Tensor>("host_tensor");
  ASSERT_TRUE(maybe_host_tensor);
  auto host_tensor = maybe_host_tensor.value();
  ASSERT_TRUE(host_tensor->reshapeCustom(shape,
                                         nvidia::gxf::PrimitiveType::kFloat32,
                                         sizeof(float),
                                         strides,
                                         nvidia::gxf::MemoryStorageType::kHost,
                                         get_allocator_handle()));

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);
  ASSERT_TRUE(result);
  EXPECT_EQ(payload.protocol_name, "cuda_ipc");

  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 1u);

  uint32_t sideband_size = 0;
  std::memcpy(&sideband_size,
              payload.descriptor_bytes.data() + 7 + protocol_name_len + sizeof(num_tensors),
              sizeof(sideband_size));
  EXPECT_GT(sideband_size, 0u);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorMixedHostAndGpuTensorsRequiredUsesSideband) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(93, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kRequired));
  serializer_->set_native_buffer_adapter(adapter);

  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_gpu_tensor = entity.add<nvidia::gxf::Tensor>("gpu_tensor");
  ASSERT_TRUE(maybe_gpu_tensor);
  auto gpu_tensor = maybe_gpu_tensor.value();
  nvidia::gxf::Shape shape({2, 2});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(gpu_tensor->reshapeCustom(shape,
                                        nvidia::gxf::PrimitiveType::kFloat32,
                                        sizeof(float),
                                        strides,
                                        nvidia::gxf::MemoryStorageType::kDevice,
                                        get_allocator_handle()));

  auto maybe_host_tensor = entity.add<nvidia::gxf::Tensor>("host_tensor");
  ASSERT_TRUE(maybe_host_tensor);
  auto host_tensor = maybe_host_tensor.value();
  ASSERT_TRUE(host_tensor->reshapeCustom(shape,
                                         nvidia::gxf::PrimitiveType::kFloat32,
                                         sizeof(float),
                                         strides,
                                         nvidia::gxf::MemoryStorageType::kHost,
                                         get_allocator_handle()));

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);
  ASSERT_TRUE(result);
  EXPECT_EQ(payload.protocol_name, "cuda_ipc");

  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 1u);

  uint32_t sideband_size = 0;
  std::memcpy(&sideband_size,
              payload.descriptor_bytes.data() + 7 + protocol_name_len + sizeof(num_tensors),
              sizeof(sideband_size));
  EXPECT_GT(sideband_size, 0u);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorWithGpuTensor) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(97, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  // Create entity with a GPU tensor
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("gpu_tensor");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();

  nvidia::gxf::Shape shape({2, 4});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(tensor->reshapeCustom(shape,
                                    nvidia::gxf::PrimitiveType::kFloat32,
                                    sizeof(float),
                                    strides,
                                    nvidia::gxf::MemoryStorageType::kDevice,
                                    get_allocator_handle()));

  // Fill with test data
  std::vector<float> host_data(8);
  for (int i = 0; i < 8; ++i)
    host_data[i] = static_cast<float>(i) * 2.0f;
  ASSERT_EQ(cudaMemcpy(tensor->pointer(),
                       host_data.data(),
                       host_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);
  ASSERT_TRUE(result) << "export_native_descriptor() failed";

  // Verify the NDPL payload structure
  ASSERT_GE(payload.descriptor_bytes.size(), 13u + payload.protocol_name.size());
  EXPECT_EQ(payload.protocol_name, "cuda_ipc");

  uint32_t magic = 0;
  std::memcpy(&magic, payload.descriptor_bytes.data(), sizeof(magic));
  EXPECT_EQ(magic, kNdplMagic);

  uint8_t version = payload.descriptor_bytes[4];
  EXPECT_EQ(version, kNdplVersion);

  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  EXPECT_EQ(protocol_name_len, payload.protocol_name.size());

  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 1u);

  EXPECT_EQ(payload.source_entity_uid, entity.eid());

  // Pending export should be tracked
  EXPECT_GE(adapter->pending_export_count(), 1u);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorMultipleGpuTensors) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(96, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  // Create entity with 3 GPU tensors
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  std::vector<std::string> tensor_names = {"rgb", "depth", "normals"};
  for (const auto& name : tensor_names) {
    auto maybe_tensor = entity.add<nvidia::gxf::Tensor>(name.c_str());
    ASSERT_TRUE(maybe_tensor);
    auto tensor = maybe_tensor.value();

    nvidia::gxf::Shape shape({4, 4});
    auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
    ASSERT_TRUE(tensor->reshapeCustom(shape,
                                      nvidia::gxf::PrimitiveType::kFloat32,
                                      sizeof(float),
                                      strides,
                                      nvidia::gxf::MemoryStorageType::kDevice,
                                      get_allocator_handle()));
  }

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);
  ASSERT_TRUE(result) << "export_native_descriptor() failed";

  // Verify 3 tensors in the NDPL header
  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 3u);

  // All 3 exports should be tracked
  EXPECT_EQ(adapter->pending_export_count(), 3u);

  adapter->shutdown();
  factory->delete_participant(participant);
}

TEST_F(DDSSerializerNativeTest, ExportNativeDescriptorMixedComponents) {
  // Entity with a GPU tensor + Timestamp + MetadataDictionary
  // The non-GPU components should be in the sideband
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  auto* factory = DomainParticipantFactory::get_instance();
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);
  auto* participant = factory->create_participant(95, qos);
  ASSERT_NE(participant, nullptr);

  auto adapter = std::make_shared<holoscan::FastDdsNativeBufferAdapter>();
  ASSERT_TRUE(adapter->initialize(participant, nvidia::gxf::NativeBufferPolicy::kPreferred));
  serializer_->set_native_buffer_adapter(adapter);

  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  // Add GPU tensor
  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("image");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();
  nvidia::gxf::Shape shape({8, 8});
  auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float));
  ASSERT_TRUE(tensor->reshapeCustom(shape,
                                    nvidia::gxf::PrimitiveType::kFloat32,
                                    sizeof(float),
                                    strides,
                                    nvidia::gxf::MemoryStorageType::kDevice,
                                    get_allocator_handle()));

  // Add timestamp
  auto maybe_ts = entity.add<nvidia::gxf::Timestamp>("ts");
  ASSERT_TRUE(maybe_ts);
  maybe_ts.value()->acqtime = 12345;
  maybe_ts.value()->pubtime = 67890;

  // Add metadata
  auto maybe_meta = entity.add<holoscan::MetadataDictionary>("meta");
  ASSERT_TRUE(maybe_meta);
  maybe_meta.value()->set("frame_id", 42);

  nvidia::gxf::NativeDescriptorPayload payload;
  auto result = serializer_->export_native_descriptor(
      entity.eid(), context(), get_allocator_handle(), payload);
  ASSERT_TRUE(result) << "export_native_descriptor() with mixed components failed";

  // Should have 1 GPU tensor
  uint16_t protocol_name_len = 0;
  std::memcpy(&protocol_name_len, payload.descriptor_bytes.data() + 5, sizeof(protocol_name_len));
  uint16_t num_tensors = 0;
  std::memcpy(
      &num_tensors, payload.descriptor_bytes.data() + 7 + protocol_name_len, sizeof(num_tensors));
  EXPECT_EQ(num_tensors, 1u);

  // Sideband should be non-empty (contains Timestamp + MetadataDictionary)
  uint32_t sideband_size = 0;
  std::memcpy(&sideband_size,
              payload.descriptor_bytes.data() + 7 + protocol_name_len + sizeof(num_tensors),
              sizeof(sideband_size));
  EXPECT_GT(sideband_size, 0u);

  adapter->shutdown();
  factory->delete_participant(participant);
}

// =============================================================================
// CUDA IPC Eligibility Tests (no CUDA needed, pure metadata comparison)
// =============================================================================

TEST(CudaIpcEligibilityTest, SameGpuSameHostIsEligible) {
  nvidia::gxf::EndpointInfo publisher;
  publisher.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  publisher.native_buffer_capability.memory_domain = "same_host_gpu";
  publisher.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";
  publisher.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  nvidia::gxf::EndpointInfo subscriber;
  subscriber.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  subscriber.native_buffer_capability.memory_domain = "same_host_gpu";
  subscriber.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";
  subscriber.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kPreferred,
                                                        /*is_same_process=*/false);

  EXPECT_TRUE(result.eligible());
}

TEST(CudaIpcEligibilityTest, DifferentGpuIsIneligible) {
  nvidia::gxf::EndpointInfo publisher;
  publisher.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  publisher.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";
  publisher.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  nvidia::gxf::EndpointInfo subscriber;
  subscriber.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  subscriber.native_buffer_capability.gpu_device_uuid = "GPU-BBBB";
  subscriber.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kPreferred,
                                                        /*is_same_process=*/false);

  EXPECT_FALSE(result.eligible());
}

TEST(CudaIpcEligibilityTest, SameProcessIsIneligible) {
  nvidia::gxf::EndpointInfo publisher;
  publisher.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  publisher.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";
  publisher.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  nvidia::gxf::EndpointInfo subscriber = publisher;

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kPreferred,
                                                        /*is_same_process=*/true);

  // Same-process should be ineligible for CUDA IPC (use shared memory instead)
  EXPECT_FALSE(result.eligible());
}

TEST(CudaIpcEligibilityTest, DisabledPolicyAlwaysIneligible) {
  nvidia::gxf::EndpointInfo publisher;
  publisher.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  publisher.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";
  publisher.native_buffer_capability.native_buffer_profile = "cuda_ipc_same_gpu_v1";

  nvidia::gxf::EndpointInfo subscriber = publisher;

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kDisabled,
                                                        /*is_same_process=*/false);

  EXPECT_FALSE(result.eligible());
}

TEST(CudaIpcEligibilityTest, PublisherNotSupportingNativeBuffers) {
  nvidia::gxf::EndpointInfo publisher;
  // publisher has no protocols (default empty)

  nvidia::gxf::EndpointInfo subscriber;
  subscriber.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  subscriber.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kPreferred,
                                                        /*is_same_process=*/false);

  EXPECT_FALSE(result.eligible());
}

TEST(CudaIpcEligibilityTest, NoCommonProtocolIsIneligible) {
  nvidia::gxf::EndpointInfo publisher;
  publisher.native_buffer_capability.native_buffer_protocols = {"cuda_vmm"};
  publisher.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";

  nvidia::gxf::EndpointInfo subscriber;
  subscriber.native_buffer_capability.native_buffer_protocols = {"cuda_ipc"};
  subscriber.native_buffer_capability.gpu_device_uuid = "GPU-AAAA";

  auto result = nvidia::gxf::check_cuda_ipc_eligibility(publisher,
                                                        subscriber,
                                                        nvidia::gxf::NativeBufferPolicy::kPreferred,
                                                        /*is_same_process=*/false);

  EXPECT_FALSE(result.eligible());
}

}  // namespace
