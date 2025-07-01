/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <complex>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <typeindex>
#include <vector>

#include "./codecs.hpp"
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/gxf/codec_registry.hpp"
#include "holoscan/core/codecs.hpp"
#include "holoscan/core/expected.hpp"

using std::string_literals::operator""s;

namespace holoscan {

TEST(CodecRegistry, TestAddCodec) {
  auto codec_registry = gxf::CodecRegistry::get_instance();

  // add existing custom Coordinate type codec (defined in codecs.hpp)
  codec_registry.add_codec<Coordinate>("Coordinate"s);

  // get the added Coordinate codec
  auto& c = codec_registry.get_codec("Coordinate"s);
  EXPECT_EQ(typeid(c), typeid(holoscan::gxf::CodecRegistry::none_codec));
}

TEST(CodecRegistry, TestAddCodec2) {
  auto codec_registry = gxf::CodecRegistry::get_instance();

  struct IntPair {
    int x;
    int y;
  };

  // add a non-functional dummy codec for the type above
  codec_registry.add_codec<IntPair>(gxf::CodecRegistry::none_codec, "none-1");

  // alternative API to add a non-functional dummy codec for the type above
  codec_registry.add_codec(
      std::type_index(typeid(IntPair)), gxf::CodecRegistry::none_codec, "none-2");
}

TEST(CodecRegistry, TestNameToTypeIndex) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  double d = 5.0;
  auto maybe_index = codec_registry.name_to_index("double"s);
  ASSERT_TRUE(maybe_index);
  ASSERT_TRUE(maybe_index.has_value());
  EXPECT_EQ(maybe_index.value(), std::type_index(typeid(d)));
}

TEST(CodecRegistry, TestNameToTypeIndexInvalid) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  auto maybe_index2 = codec_registry.name_to_index("non-existent");
  ASSERT_FALSE(maybe_index2);
  ASSERT_FALSE(maybe_index2.has_value());
  auto err_msg = std::string(maybe_index2.error().what());
  ASSERT_TRUE(err_msg.find("No codec") != std::string::npos);
}

TEST(CodecRegistry, TestTypeIndexToName) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  double d = 5.0;
  auto maybe_name = codec_registry.index_to_name(std::type_index(typeid(d)));
  ASSERT_TRUE(maybe_name);
  ASSERT_TRUE(maybe_name.has_value());
  EXPECT_EQ(maybe_name.value(), "double"s);
}

TEST(CodecRegistry, TestTypeIndexToNameInvalid) {
  auto codec_registry = gxf::CodecRegistry::get_instance();

  // create a type for which no codec has been registered
  std::set<std::vector<std::complex<float>>> no_codec;

  auto maybe_name2 = codec_registry.index_to_name(std::type_index(typeid(no_codec)));
  ASSERT_FALSE(maybe_name2);
  ASSERT_FALSE(maybe_name2.has_value());
  auto err_msg = std::string(maybe_name2.error().what());
  ASSERT_TRUE(err_msg.find("No codec for type") != std::string::npos);
}

TEST(CodecRegistry, TestGetSerializerFromString) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  auto& s = codec_registry.get_serializer("int16_t"s);
  EXPECT_EQ(typeid(s), typeid(holoscan::gxf::CodecRegistry::none_serialize));
}

TEST(CodecRegistry, TestGetDeSerializerFromString) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  auto& d = codec_registry.get_deserializer("int64_t"s);
  EXPECT_EQ(typeid(d), typeid(holoscan::gxf::CodecRegistry::none_deserialize));
}

TEST(CodecRegistry, TestGetCodecFromString) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  auto& [s, d] = codec_registry.get_codec("std::string"s);
  EXPECT_EQ(typeid(s), typeid(holoscan::gxf::CodecRegistry::none_serialize));
  EXPECT_EQ(typeid(d), typeid(holoscan::gxf::CodecRegistry::none_deserialize));
}

TEST(CodecRegistry, TestGetSerializerFromTypeIndex) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  uint8_t i;
  auto& s = codec_registry.get_serializer(std::type_index(typeid(i)));
  EXPECT_EQ(typeid(s), typeid(holoscan::gxf::CodecRegistry::none_serialize));
}

TEST(CodecRegistry, TestGetDeSerializerFromTypeIndex) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  std::vector<float> v;
  auto& d = codec_registry.get_deserializer(std::type_index(typeid(v)));
  EXPECT_EQ(typeid(d), typeid(holoscan::gxf::CodecRegistry::none_deserialize));
}

TEST(CodecRegistry, TestGetCodecFromTypeIndex) {
  auto codec_registry = gxf::CodecRegistry::get_instance();
  std::string s1;
  auto& [s, d] = codec_registry.get_codec(std::type_index(typeid(s1)));
  EXPECT_EQ(typeid(s), typeid(holoscan::gxf::CodecRegistry::none_serialize));
  EXPECT_EQ(typeid(d), typeid(holoscan::gxf::CodecRegistry::none_deserialize));
}

}  // namespace holoscan
