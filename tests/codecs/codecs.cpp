/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "./codecs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/argument_setter.hpp>
#include <holoscan/core/codecs.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/operators/holoviz/codecs.hpp>
#include "./mock_allocator.hpp"
#include "./mock_serialization_buffer.hpp"

using std::string_literals::operator""s;

namespace holoscan {

struct OversizedCodecElement {
  std::array<std::byte, 256> data;
};

class CodecTestEndpoint : public Endpoint {
 public:
  bool is_read_available() override { return true; }
  bool is_write_available() override { return true; }

  expected<size_t, RuntimeError> write(const void* data, size_t size) override {
    const size_t call = write_calls_++;
    if (call == fail_write_call_) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "injected write error"));
    }
    const auto* bytes = static_cast<const std::byte*>(data);
    output_.insert(output_.end(), bytes, bytes + size);
    return size;
  }

  expected<size_t, RuntimeError> read(void* data, size_t size) override {
    max_read_size_ = std::max(max_read_size_, size);
    const size_t call = read_calls_++;
    const size_t read_size = call == short_read_call_ && size > 0 ? size - 1 : size;
    if (read_size > input_.size() - read_offset_) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "injected truncated input"));
    }
    std::memcpy(data, input_.data() + read_offset_, read_size);
    read_offset_ += read_size;
    return read_size;
  }

  expected<void, RuntimeError> write_ptr(const void*, size_t, MemoryStorageType) override {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "write_ptr is unsupported by this test endpoint"));
  }

  template <typename T>
  void append_input(const T& value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    input_.insert(input_.end(), bytes, bytes + sizeof(T));
  }

  void set_fail_write_call(size_t call) { fail_write_call_ = call; }
  void set_short_read_call(size_t call) { short_read_call_ = call; }
  size_t max_read_size() const { return max_read_size_; }

 private:
  std::vector<std::byte> input_;
  std::vector<std::byte> output_;
  size_t read_offset_ = 0;
  size_t read_calls_ = 0;
  size_t write_calls_ = 0;
  size_t fail_write_call_ = std::numeric_limits<size_t>::max();
  size_t short_read_call_ = std::numeric_limits<size_t>::max();
  size_t max_read_size_ = 0;
};

template <typename dataT>
void codec_compare(dataT& value, size_t buffer_size = 4096, bool omit_size_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<dataT>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    EXPECT_EQ(maybe_size.value(), sizeof(value));
  }

  auto maybe_value = codec<dataT>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result, value);
}

template <typename dataT>
void codec_shared_compare(std::shared_ptr<dataT> value, size_t buffer_size = 4096,
                          bool omit_size_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<std::shared_ptr<dataT>>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    EXPECT_EQ(maybe_size.value(), sizeof(*value));
  }

  auto maybe_value = codec<std::shared_ptr<dataT>>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(*result, *value);
}

template <typename dataT>
void codec_data_blob_compare(dataT& data, size_t buffer_size = 4096) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<dataT>::serialize(data, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  // expect data.size() of the data blob + sizeof(ContiguousDataHeader)

  size_t bytes_per_element = data.size() > 0 ? sizeof(data[0]) : 1;
  size_t data_size_bytes = data.size() * bytes_per_element;
  EXPECT_EQ(maybe_size.value(), data_size_bytes + sizeof(holoscan::ContiguousDataHeader));

  auto maybe_data = codec<dataT>::deserialize(endpoint.get());
  auto result = maybe_data.value();
  EXPECT_EQ(typeid(result), typeid(data));
  EXPECT_EQ(result, data);
}

template <typename dataT>
void codec_shared_data_blob_compare(std::shared_ptr<dataT> data, size_t buffer_size = 4096) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<std::shared_ptr<dataT>>::serialize(data, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  size_t bytes_per_element = data->size() > 0 ? sizeof(data->at(0)) : 1;
  size_t data_size_bytes = data->size() * bytes_per_element;
  EXPECT_EQ(maybe_size.value(), data_size_bytes + sizeof(holoscan::ContiguousDataHeader));

  auto maybe_data = codec<std::shared_ptr<dataT>>::deserialize(endpoint.get());
  auto result = maybe_data.value();
  EXPECT_EQ(typeid(result), typeid(data));
  EXPECT_EQ(*result, *data);
}

template <typename dataT>
void codec_vector_compare(dataT& value, size_t buffer_size = 4096, bool omit_size_check = false,
                          bool omit_values_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<dataT>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    size_t expected_size = sizeof(holoscan::ContiguousDataHeader);
    if (value.size() > 0) {
      expected_size += value.size() * sizeof(value[0]);
    }
    EXPECT_EQ(maybe_size.value(), expected_size);
  }

  auto maybe_value = codec<dataT>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result.size(), value.size());

  if (!omit_values_check) {
    for (size_t i = 0; i < value.size(); i++) {
      EXPECT_EQ(result[i], value[i]);
    }
  }
}

template <typename dataT>
void codec_shared_vector_compare(std::shared_ptr<dataT> value, size_t buffer_size = 4096,
                                 bool omit_size_check = false, bool omit_values_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<std::shared_ptr<dataT>>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    size_t expected_size = sizeof(holoscan::ContiguousDataHeader);
    if (value->size() > 0) {
      expected_size += value->size() * sizeof(value->at(0));
    }
    EXPECT_EQ(maybe_size.value(), expected_size);
  }
  auto maybe_value = codec<std::shared_ptr<dataT>>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result->size(), value->size());

  if (!omit_values_check) {
    for (size_t i = 0; i < value->size(); i++) {
      EXPECT_EQ(result->at(i), value->at(i));
    }
  }
}

// TODO(unknown): update size check here
template <typename dataT>
void codec_vector_vector_compare(dataT& vectors, size_t buffer_size = 4096,
                                 bool omit_size_check = true, bool omit_values_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<dataT>::serialize(vectors, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    throw std::runtime_error("size check not implemented for this type");
  }

  auto maybe_vectors = codec<dataT>::deserialize(endpoint.get());
  auto result = maybe_vectors.value();
  EXPECT_EQ(typeid(result), typeid(vectors));
  EXPECT_EQ(result.size(), vectors.size());

  // Check values for nested vector type
  if (!omit_values_check) {
    for (size_t j = 0; j < vectors.size(); j++) {
      auto vec = vectors[j];
      auto res = result[j];
      EXPECT_EQ(typeid(vec), typeid(res));
      EXPECT_EQ(vec.size(), res.size());
      for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], res[i]);
      }
    }
  }
}

// TODO(unknown): update size check here
template <typename dataT>
void codec_shared_vector_vector_compare(std::shared_ptr<dataT> vectors, size_t buffer_size = 4096,
                                        bool omit_size_check = true,
                                        bool omit_values_check = false) {
  // need buffer_size large enough to hold any tested type
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      buffer_size, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<std::shared_ptr<dataT>>::serialize(vectors, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  if (!omit_size_check) {
    throw std::runtime_error("size check not implemented for this type");
  }

  auto maybe_vectors = codec<std::shared_ptr<dataT>>::deserialize(endpoint.get());
  auto result = maybe_vectors.value();
  EXPECT_EQ(typeid(result), typeid(vectors));
  EXPECT_EQ(result->size(), vectors->size());

  // Check values for nested vector type
  if (!omit_values_check) {
    for (size_t j = 0; j < vectors->size(); j++) {
      auto vec = vectors->at(j);
      auto res = result->at(j);
      EXPECT_EQ(typeid(vec), typeid(res));
      EXPECT_EQ(vec.size(), res.size());
      for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], res[i]);
      }
    }
  }
}

TEST(Codecs, TestBool) {
  bool value = true;
  codec_compare<bool>(value);
}

TEST(Codecs, TestFloat) {
  float value = 5.0;
  codec_compare<float>(value);
}

TEST(Codecs, TestDouble) {
  double value = 5.0;
  codec_compare<double>(value);
}

TEST(Codecs, TestDoubleShared) {
  auto value = std::make_shared<double>(5.0);
  codec_shared_compare<double>(value);
}

TEST(Codecs, TestInt8) {
  int8_t value = 8;
  codec_compare<int8_t>(value);
}

TEST(Codecs, TestInt16) {
  int16_t value = 8;
  codec_compare<int16_t>(value);
}

TEST(Codecs, TestInt32) {
  int32_t value = 8;
  codec_compare<int32_t>(value);
}

TEST(Codecs, TestInt64) {
  int64_t value = 8;
  codec_compare<int64_t>(value);
}

TEST(Codecs, TestUInt8) {
  uint8_t value = 8;
  codec_compare<uint8_t>(value);
}

TEST(Codecs, TestUInt16) {
  uint16_t value = 8;
  codec_compare<uint16_t>(value);
}

TEST(Codecs, TestUInt32) {
  uint32_t value = 8;
  codec_compare<uint32_t>(value);
}

TEST(Codecs, TestUInt64) {
  uint64_t value = 8;
  codec_compare<uint64_t>(value);
}

TEST(Codecs, TestString) {
  std::string value{"abcdefgh"s};
  codec_data_blob_compare<std::string>(value);
}

TEST(Codecs, TestStringShared) {
  auto value = std::make_shared<std::string>("abcdefgh"s);
  codec_shared_data_blob_compare<std::string>(value);
}

TEST(Codecs, TestComplexFloat) {
  std::complex<float> value{1.0, 1.5};
  codec_compare<std::complex<float>>(value);
}

TEST(Codecs, TestComplexDouble) {
  std::complex<double> value{1.0, 1.5};
  codec_compare<std::complex<double>>(value);
}

TEST(Codecs, TestComplexDoubleShared) {
  auto value = std::make_shared<std::complex<double>>(5.0, 2.0);
  codec_shared_compare<std::complex<double>>(value);
}

TEST(Codecs, TestVectorBool) {
  // choose a length here that is not a multiple of 8
  std::vector<bool> value{0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1};

  // serialize
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(128, holoscan::MemoryStorageType::kSystem);
  auto maybe_size = codec<std::vector<bool>>::serialize(value, endpoint.get());

  // Check that serialization used bit packing as expected.
  // Stores a size_t for number of unit8_t elements after bit packing in addition to the packed bits
  size_t expected_size = sizeof(size_t) + (value.size() + 7) / 8;
  EXPECT_EQ(maybe_size.value(), expected_size);

  // deserialize and verify roundtrip result
  auto maybe_deserialized = codec<std::vector<bool>>::deserialize(endpoint.get());
  auto result = maybe_deserialized.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result.size(), value.size());
  for (size_t i = 0; i < value.size(); i++) {
    EXPECT_EQ(result[i], value[i]);
  }
}

TEST(Codecs, TestVectorInt32) {
  std::vector<int32_t> value{1, 2, 3, 4, 5};
  codec_vector_compare<std::vector<int32_t>>(value);
}

TEST(Codecs, TestEmptyVectorInt32) {
  std::vector<int32_t> value;
  codec_vector_compare<std::vector<int32_t>>(value);
}

TEST(Codecs, TestEmptyBinaryBlobAllowsOversizedElementType) {
  std::vector<OversizedCodecElement> value;
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto serialized = codec<std::vector<OversizedCodecElement>>::serialize(value, endpoint.get());
  ASSERT_TRUE(serialized);
  EXPECT_EQ(serialized.value(), sizeof(ContiguousDataHeader));

  auto deserialized = codec<std::vector<OversizedCodecElement>>::deserialize(endpoint.get());
  ASSERT_TRUE(deserialized);
  EXPECT_TRUE(deserialized.value().empty());
}

TEST(Codecs, TestNonEmptyBinaryBlobRejectsOversizedElementType) {
  std::vector<OversizedCodecElement> value(1);
  CodecTestEndpoint endpoint;

  auto result = codec<std::vector<OversizedCodecElement>>::serialize(value, &endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestBinaryBlobRejectsElementSizeMismatch) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{3, sizeof(uint16_t)};
  endpoint.append_input(header);

  auto result = codec<std::vector<uint32_t>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestBinaryBlobRejectsByteSizeOverflow) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{std::numeric_limits<size_t>::max(), sizeof(uint32_t)};
  endpoint.append_input(header);

  auto result = codec<std::vector<uint32_t>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
  EXPECT_EQ(endpoint.max_read_size(), sizeof(ContiguousDataHeader));
}

TEST(Codecs, TestBinaryBlobReadsHugeTruncatedPayloadInBoundedChunks) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{std::numeric_limits<size_t>::max() / sizeof(uint32_t),
                              sizeof(uint32_t)};
  endpoint.append_input(header);

  auto result = codec<std::vector<uint32_t>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
  EXPECT_LE(endpoint.max_read_size(), 64 * 1024);
}

TEST(Codecs, TestFixedArrayRejectsElementCountMismatch) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{15, sizeof(float)};
  endpoint.append_input(header);

  auto result = codec<std::array<float, 16>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestFixedArrayRejectsElementSizeMismatch) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{16, sizeof(double)};
  endpoint.append_input(header);

  auto result = codec<std::array<float, 16>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestBinaryBlobRejectsShortRead) {
  CodecTestEndpoint endpoint;
  ContiguousDataHeader header{2, sizeof(uint32_t)};
  std::array<uint32_t, 2> payload{1, 2};
  endpoint.append_input(header);
  endpoint.append_input(payload);
  endpoint.set_short_read_call(1);

  auto result = codec<std::vector<uint32_t>>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestVectorUInt8) {
  std::vector<uint8_t> value{1, 2, 3, 4, 5};
  codec_vector_compare<std::vector<uint8_t>>(value);
}

TEST(Codecs, TestComplexUInt8Shared) {
  size_t sz = 10;
  auto value = std::make_shared<std::vector<uint8_t>>(sz);
  for (size_t i = 0; i < 10; i++) {
    value->at(i) = i;
  }
  codec_shared_vector_compare<std::vector<uint8_t>>(value);
}

TEST(Codecs, TestHolovizOpCameraPose) {
  // Test the camera pose type used by HolovizOp (std::shared_ptr<std::array<float, 16>>)
  auto value = std::make_shared<std::array<float, 16>>();
  value->at(0) = 1.0;
  value->at(5) = 2.0;
  codec_shared_vector_compare<std::array<float, 16>>(value);
}

TEST(Codecs, TestVectorFloat) {
  std::vector<float> value{1.0, 2.0, 3.0, 4.0, 5.0};
  codec_vector_compare<std::vector<float>>(value);
}

TEST(Codecs, TestVectorComplexFloat) {
  std::vector<std::complex<float>> value{{1.0, 1.5}, {2.0, 0.0}, {0.0, 3.0}};
  codec_vector_compare<std::vector<std::complex<float>>>(value);
}

TEST(Codecs, TestVectorString) {
  std::vector<std::string> value{"abcd"s, "ef"s, "g"s, "hijklm"s};
  codec_vector_vector_compare<std::vector<std::string>>(value, 4096);
}

TEST(Codecs, TestVectorVectorFloat) {
  std::vector<std::vector<float>> value{{1.0, 2.0, 3.0, 4.0, 5.0}, {3.0, 4.0}, {5.0, 6.0, 8.0}};
  codec_vector_vector_compare<std::vector<std::vector<float>>>(value, 4096);
}

TEST(Codecs, TestSharedVectorVectorFloat) {
  std::vector<std::vector<float>> v{{1.0, 2.0, 3.0, 4.0, 5.0}, {3.0, 4.0}, {5.0, 6.0, 8.0}};
  auto value = std::make_shared<std::vector<std::vector<float>>>(v);
  codec_shared_vector_vector_compare<std::vector<std::vector<float>>>(value, 4096);
}

TEST(Codecs, TestVectorVectorString) {
  std::vector<std::vector<std::string>> value{{"ab"s, "cd"s, "ef"s}, {"g"s, "hijkl"s}, {"mno"s}};
  codec_vector_vector_compare<std::vector<std::vector<std::string>>>(value, 4096);
}

TEST(Codecs, TestVectorVectorBool) {
  std::vector<std::vector<bool>> bvecs;

  std::vector<bool> v1{0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1};
  std::vector<bool> v2{1, 1, 0, 1, 1, 0, 1};
  std::vector<bool> v3(1001, true);
  bvecs.push_back(v1);
  bvecs.push_back(v2);
  bvecs.push_back(v3);

  size_t expected_size = sizeof(size_t);  // number of vectors
  // add bit-packed serialization size of each vector
  for (auto& v : bvecs) {
    expected_size += sizeof(size_t) + (v.size() + 7) / 8;
  }

  // serialize to buffer of exactly expected_size (exception thrown if expected_size is too small)
  auto endpoint = std::make_shared<MockUcxSerializationBuffer>(
      expected_size, holoscan::MemoryStorageType::kSystem);
  auto maybe_size = codec<std::vector<std::vector<bool>>>::serialize(bvecs, endpoint.get());
  EXPECT_EQ(maybe_size.value(), expected_size);

  // deserialize and verify roundtrip result
  auto maybe_deserialized = codec<std::vector<std::vector<bool>>>::deserialize(endpoint.get());
  auto result = maybe_deserialized.value();
  EXPECT_EQ(typeid(result), typeid(bvecs));
  EXPECT_EQ(result.size(), bvecs.size());
  for (size_t j = 0; j < bvecs.size(); j++) {
    auto vec = bvecs[j];
    auto res = result[j];
    EXPECT_EQ(typeid(vec), typeid(res));
    EXPECT_EQ(vec.size(), res.size());
    for (size_t i = 0; i < vec.size(); i++) {
      EXPECT_EQ(res[i], vec[i]);
    }
  }
}

TEST(Codecs, TestCustomTrivialSerializer) {
  // codecs.hpp defines a custom serializer for a Coordinate type
  // We verify proper roundtrip serialization and deserialization of that type

  Coordinate value{1.0, 2.0, 3.5};
  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<Coordinate>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  EXPECT_EQ(maybe_size.value(), sizeof(value));

  auto maybe_value = codec<Coordinate>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result.x, value.x);
  EXPECT_EQ(result.y, value.y);
  EXPECT_EQ(result.z, value.z);
}

TEST(Codecs, TestCustomTrivialSerializer2) {
  // codecs.hpp defines a custom serializer for a struct with Mixed type data members
  // We verify proper roundtrip serialization and deserialization of that type/

  std::vector ivec{1, 2, 3, 4, 5};
  MixedType value{1, 2.5, ivec.data(), -8};
  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<MixedType>::serialize(value, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));
  EXPECT_EQ(maybe_size.value(), sizeof(value));

  auto maybe_value = codec<MixedType>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(value));
  EXPECT_EQ(result.a, value.a);
  EXPECT_EQ(result.b, value.b);
  EXPECT_EQ(result.c, value.c);
  EXPECT_EQ(result.d, value.d);
}

TEST(Codecs, TestViewSerializer) {
  ops::HolovizOp::InputSpec::View v1;
  v1.offset_x_ = 0.1;
  v1.offset_y_ = 0.1;
  v1.width_ = 0.8;
  v1.height_ = 0.6;
  v1.matrix_ = std::array<float, 16>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<ops::HolovizOp::InputSpec::View>::serialize(v1, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  auto maybe_value = codec<ops::HolovizOp::InputSpec::View>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(v1));
  EXPECT_EQ(result.offset_x_, v1.offset_x_);
  EXPECT_EQ(result.offset_y_, v1.offset_y_);
  EXPECT_EQ(result.width_, v1.width_);
  EXPECT_EQ(result.height_, v1.height_);
  ASSERT_TRUE(result.matrix_.has_value());
  for (size_t i = 0; i < 16; i++) {
    if ((i % 5) == 0) {
      EXPECT_EQ(v1.matrix_.value()[i], 1);
    } else {
      EXPECT_EQ(v1.matrix_.value()[i], 0);
    }
  }
}

TEST(Codecs, TestViewSerializerNoMatrix) {
  ops::HolovizOp::InputSpec::View v1{0.2, 0.1, 0.6, 0.8};

  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<ops::HolovizOp::InputSpec::View>::serialize(v1, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  auto maybe_value = codec<ops::HolovizOp::InputSpec::View>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(v1));
  EXPECT_EQ(result.offset_x_, v1.offset_x_);
  EXPECT_EQ(result.offset_y_, v1.offset_y_);
  EXPECT_EQ(result.width_, v1.width_);
  EXPECT_EQ(result.height_, v1.height_);
  ASSERT_FALSE(result.matrix_.has_value());
}

TEST(Codecs, TestViewDeserializerPropagatesMatrixShortRead) {
  CodecTestEndpoint endpoint;
  const float offset_x = 0.1F;
  const float offset_y = 0.2F;
  const float width = 0.8F;
  const float height = 0.6F;
  const bool has_matrix = true;
  const ContiguousDataHeader header{16, sizeof(float)};
  const std::array<float, 16> matrix{};
  endpoint.append_input(offset_x);
  endpoint.append_input(offset_y);
  endpoint.append_input(width);
  endpoint.append_input(height);
  endpoint.append_input(has_matrix);
  endpoint.append_input(header);
  endpoint.append_input(matrix);
  endpoint.set_short_read_call(6);

  auto result = codec<ops::HolovizOp::InputSpec::View>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestViewDeserializerRejectsFirstScalarShortRead) {
  CodecTestEndpoint endpoint;
  const float offset_x = 0.1F;
  endpoint.append_input(offset_x);
  endpoint.set_short_read_call(0);

  auto result = codec<ops::HolovizOp::InputSpec::View>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestViewSerializerPropagatesWriteError) {
  CodecTestEndpoint endpoint;
  endpoint.set_fail_write_call(0);
  ops::HolovizOp::InputSpec::View view{0.2, 0.1, 0.6, 0.8};

  auto result = codec<ops::HolovizOp::InputSpec::View>::serialize(view, &endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestVectorViewSerializer) {
  ops::HolovizOp::InputSpec::View v1{0.2, 0.1, 0.6, 0.8};
  ops::HolovizOp::InputSpec::View v2{0.1, 0.1, 0.7, 0.8};
  v2.matrix_ = std::array<float, 16>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  std::vector<ops::HolovizOp::InputSpec::View> views;
  views.reserve(2);
  views.push_back(v1);
  views.push_back(v2);

  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(512, holoscan::MemoryStorageType::kSystem);

  auto maybe_size =
      codec<std::vector<ops::HolovizOp::InputSpec::View>>::serialize(views, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  auto maybe_value =
      codec<std::vector<ops::HolovizOp::InputSpec::View>>::deserialize(endpoint.get());
  auto result = maybe_value.value();
  EXPECT_EQ(typeid(result), typeid(views));
  EXPECT_EQ(result[0].offset_x_, views[0].offset_x_);
  EXPECT_EQ(result[1].offset_x_, views[1].offset_x_);
  EXPECT_EQ(result[0].offset_y_, views[0].offset_y_);
  EXPECT_EQ(result[1].offset_y_, views[1].offset_y_);
  EXPECT_EQ(result[0].width_, views[0].width_);
  EXPECT_EQ(result[1].width_, views[1].width_);
  EXPECT_EQ(result[0].height_, views[0].height_);
  EXPECT_EQ(result[1].height_, views[1].height_);
  ASSERT_FALSE(result[0].matrix_.has_value());
  ASSERT_TRUE(result[1].matrix_.has_value());
  for (size_t i = 0; i < 16; i++) {
    if ((i % 5) == 0) {
      EXPECT_EQ(result[1].matrix_.value()[i], 1);
    } else {
      EXPECT_EQ(result[1].matrix_.value()[i], 0);
    }
  }
}

TEST(Codecs, TestInputSpec) {
  std::string tensor_name{"video"};

  ops::HolovizOp::InputSpec spec{tensor_name, ops::HolovizOp::InputType::COLOR};

  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(4096, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<ops::HolovizOp::InputSpec>::serialize(spec, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  auto maybe_spec = codec<ops::HolovizOp::InputSpec>::deserialize(endpoint.get());
  auto result = maybe_spec.value();
  EXPECT_EQ(typeid(result), typeid(spec));
  EXPECT_EQ(result.tensor_name_, tensor_name);
  EXPECT_EQ(result.type_, ops::HolovizOp::InputType::COLOR);
}

TEST(Codecs, TestInputSpecCodecPropagatesEndpointErrors) {
  CodecTestEndpoint write_endpoint;
  write_endpoint.set_fail_write_call(0);
  ops::HolovizOp::InputSpec spec{"video", ops::HolovizOp::InputType::COLOR};

  auto write_result = codec<ops::HolovizOp::InputSpec>::serialize(spec, &write_endpoint);
  EXPECT_FALSE(write_result);

  CodecTestEndpoint read_endpoint;
  auto read_result = codec<ops::HolovizOp::InputSpec>::deserialize(&read_endpoint);
  EXPECT_FALSE(read_result);
}

TEST(Codecs, TestInputSpecDeserializerRejectsFirstScalarShortRead) {
  CodecTestEndpoint endpoint;
  const ContiguousDataHeader tensor_name_header{5, sizeof(char)};
  const std::array<char, 5> tensor_name{'v', 'i', 'd', 'e', 'o'};
  const ops::HolovizOp::InputType input_type = ops::HolovizOp::InputType::COLOR;
  endpoint.append_input(tensor_name_header);
  endpoint.append_input(tensor_name);
  endpoint.append_input(input_type);
  endpoint.set_short_read_call(2);

  auto result = codec<ops::HolovizOp::InputSpec>::deserialize(&endpoint);

  EXPECT_FALSE(result);
}

TEST(Codecs, TestVectorInputSpec) {
  std::string tensor_name1{"video1"};
  ops::HolovizOp::InputSpec spec1{tensor_name1, ops::HolovizOp::InputType::COLOR};

  std::string tensor_name2{"video1"};
  ops::HolovizOp::InputSpec spec2{tensor_name2, ops::HolovizOp::InputType::COLOR};

  std::vector<ops::HolovizOp::InputSpec> specs{spec1, spec2};

  // need buffer_size large enough to hold any tested type
  auto endpoint =
      std::make_shared<MockUcxSerializationBuffer>(4096, holoscan::MemoryStorageType::kSystem);

  auto maybe_size = codec<std::vector<ops::HolovizOp::InputSpec>>::serialize(specs, endpoint.get());
  EXPECT_EQ(typeid(maybe_size.value()), typeid(size_t));

  auto maybe_specs = codec<std::vector<ops::HolovizOp::InputSpec>>::deserialize(endpoint.get());
  auto result = maybe_specs.value();
  EXPECT_EQ(typeid(result), typeid(specs));
  EXPECT_EQ(result[0].tensor_name_, tensor_name1);
  EXPECT_EQ(result[1].tensor_name_, tensor_name2);
  EXPECT_EQ(result[0].type_, spec1.type_);
  EXPECT_EQ(result[1].type_, spec2.type_);
}
}  // namespace holoscan
