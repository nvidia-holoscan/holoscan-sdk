/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/metadata.hpp"

#include "dummy_classes.hpp"

using namespace std::string_literals;

namespace holoscan {

TEST(MetadataObject, TestConstructors) {
  // value-based constructor
  MetadataObject metadata_int{DummyIntClass{15}};
  EXPECT_EQ(typeid(DummyIntClass), metadata_int.value().type());

  auto metadata_int_val = std::any_cast<DummyIntClass>(metadata_int.value());
  EXPECT_EQ(metadata_int_val, DummyIntClass{15});

  // default constructor
  MetadataObject metadata_int2;
  auto int_obj = DummyIntClass{20};
  metadata_int2.set_value(int_obj);
  EXPECT_EQ(typeid(DummyIntClass), metadata_int2.value().type());

  auto metadata_int2_val = std::any_cast<DummyIntClass>(metadata_int2.value());
  EXPECT_EQ(metadata_int2_val, int_obj);

  // set value of an existing object to a different type
  int32_t i = 5;
  metadata_int2.set_value(i);
  EXPECT_EQ(typeid(int32_t), metadata_int2.value().type());
  EXPECT_EQ(std::any_cast<int32_t>(metadata_int2.value()), i);
}

TEST(MetadataObject, TestSharedPtrAny) {
  // value-based constructor
  auto shared_obj = std::make_shared<DummyIntClass>(15);
  MetadataObject metadata_shared{shared_obj};
  EXPECT_EQ(typeid(std::shared_ptr<DummyIntClass>), metadata_shared.value().type());

  // can use as<T>() method for shared_ptr types
  auto val = metadata_shared.as<DummyIntClass>();
  EXPECT_EQ(*val, *shared_obj);

  // update the underlying value
  shared_obj->set(20);
  EXPECT_EQ(*val, *shared_obj);

  // equivalent to use any_cast to shared_ptr<T> on the value
  auto val2 = std::any_cast<std::shared_ptr<DummyIntClass>>(metadata_shared.value());
  EXPECT_EQ(*val, *val2);
}

TEST(MetadataObject, TestArgConversion) {
  std::vector<float> vec{1.0, 2.0, 4.0};
  MetadataObject obj{vec};
  EXPECT_EQ(typeid(vec), obj.value().type());

  // can assign MetadataObject.value() to Arg
  Arg a{"v"};
  // need any_cast here if we want ArgType enums to be set correctly
  a = std::any_cast<std::vector<float>>(obj.value());
  auto arg_type = a.arg_type();
  EXPECT_EQ(arg_type.element_type(), ArgElementType::kFloat32);
  EXPECT_EQ(arg_type.container_type(), ArgContainerType::kVector);

  // // can pass Arg.value() to MetadataObject.set_value() without std::any_cast
  MetadataObject obj2;
  obj2.set_value(a.value());
  EXPECT_EQ(typeid(vec), obj2.value().type());
  auto vec2 = std::any_cast<std::vector<float>>(obj2.value());
  EXPECT_EQ(vec[0], vec2[0]);
  EXPECT_EQ(vec[1], vec2[1]);
  EXPECT_EQ(vec[2], vec2[2]);
}

TEST(MetadataDictionary, TestConstructor) {
  MetadataDictionary d{};

  std::vector<std::string> keys = d.keys();
  EXPECT_EQ(keys.size(), 0);
}

TEST(MetadataDictionary, TestBracketOperator) {
  MetadataDictionary d{};

  d["patient name"] = std::make_shared<MetadataObject>("John Doe"s);
  d["date"] = std::make_shared<MetadataObject>("2024-06-27");
  d["age"] = std::make_shared<MetadataObject>(25);

  std::vector<std::string> keys = d.keys();
  EXPECT_EQ(keys.size(), 3);

  // have to cast base type to derived type to get a value of the expected type
  auto shared_obj_ptr = d["patient name"];
  EXPECT_EQ(std::any_cast<std::string>(shared_obj_ptr->value()), "John Doe"s);

  auto shared_obj_ptr2 = d["date"];
  EXPECT_EQ(std::any_cast<const char*>(shared_obj_ptr2->value()), "2024-06-27"s);

  auto shared_obj_ptr3 = d["age"];
  EXPECT_EQ(std::any_cast<int>(shared_obj_ptr3->value()), 25);
}

TEST(MetadataDictionary, TestBracketOperatorOnConstDictionary) {
  MetadataDictionary d{};

  d["patient name"] = std::make_shared<MetadataObject>("John Doe"s);
  d["date"] = std::make_shared<MetadataObject>("2024-06-27");
  d["age"] = std::make_shared<MetadataObject>(25);

  const MetadataDictionary d2{d};

  std::vector<std::string> keys = d2.keys();
  EXPECT_EQ(keys.size(), 3);

  // const pointer bracket object for const dictionary
  const MetadataObject* obj_ptr = d2["patient name"];
  auto any_value = obj_ptr->value();
  EXPECT_EQ(std::any_cast<std::string>(any_value), "John Doe"s);
}

TEST(MetadataDictionary, TestGet) {
  MetadataDictionary d{};

  d["patient name"] = std::make_shared<MetadataObject>("John Doe"s);
  d["date"] = std::make_shared<MetadataObject>("2024-06-27");
  d["age"] = std::make_shared<MetadataObject>(25);

  std::vector<std::string> keys = d.keys();
  EXPECT_EQ(keys.size(), 3);

  // runtime error if invalid key is requested
  EXPECT_THROW(d.get("invalid_key"), std::runtime_error);
  EXPECT_THROW(d.get<int>("invalid_key"), std::runtime_error);

  // get<T> has a two argument version that returns a default value if the key doesn't exist
  int default_value = 5;
  EXPECT_EQ(d.get<int>("invalid_key", default_value), default_value);
  // return type can be inferred from the provided default
  EXPECT_EQ(d.get("invalid_key", default_value), default_value);

  // have to cast base type to derived type to get a value of the expected type
  std::shared_ptr<MetadataObject> obj_ptr = d.get("patient name");
  EXPECT_EQ(std::any_cast<std::string>(obj_ptr->value()), "John Doe"s);
}

TEST(MetadataDictionary, TestGetTemplate) {
  MetadataDictionary d{};

  d["patient name"] = std::make_shared<MetadataObject>("John Doe"s);
  d["date"] = std::make_shared<MetadataObject>("2024-06-27");
  d["age"] = std::make_shared<MetadataObject>(25);

  std::vector<std::string> keys = d.keys();
  EXPECT_EQ(keys.size(), 3);

  // have to cast base type to derived type to get a value of the expected type
  EXPECT_EQ(d.get<std::string>("patient name"), "John Doe"s);
  EXPECT_EQ(d.get<const char*>("date"), "2024-06-27");
  EXPECT_EQ(d.get<int>("age"), 25);
}

TEST(MetadataDictionary, TestMetadataPolicy) {
  MetadataDictionary d{};

  d.set("patient name", "John Doe"s);
  EXPECT_EQ(d.policy(), MetadataPolicy::kRaise);

  // raise if new value is provided for existing key
  EXPECT_THROW(d.set("patient name", "Mr. Smith"s), std::runtime_error);

  // reject new value
  d.policy(MetadataPolicy::kReject);
  d.set("patient name", "Mr. Smith"s);
  EXPECT_EQ(d.get<std::string>("patient name"), "John Doe"s);

  // update using new metadata object
  auto shared_obj = d.get("patient name");
  d.policy(MetadataPolicy::kUpdate);
  d.set("patient name", "Mr. Smith"s);
  EXPECT_EQ(d.get<std::string>("patient name"), "Mr. Smith"s);
  EXPECT_EQ(std::any_cast<std::string>(shared_obj->value()), "John Doe"s);

  // update existing metadata object in-place
  shared_obj = d.get("patient name");
  d.policy(MetadataPolicy::kInplaceUpdate);
  d.set("patient name", "Mr. Nobody"s);
  EXPECT_EQ(d.get<std::string>("patient name"), "Mr. Nobody"s);
  EXPECT_EQ(std::any_cast<std::string>(shared_obj->value()), "Mr. Nobody"s);
}

TEST(MetadataDictionary, TestMetadataUpdatePolicyUpdate) {
  MetadataDictionary d{};
  d.set("key1", 5);
  d.set("key2", 10);

  MetadataDictionary d2{};
  d2.set("key1", 3);
  d2.set("key3", 20);

  d.policy(MetadataPolicy::kUpdate);
  d.update(d2);
  EXPECT_EQ(d.size(), 3);
  EXPECT_EQ(d.get<int>("key1"), 3);
  EXPECT_EQ(d.get<int>("key2"), 10);
  EXPECT_EQ(d.get<int>("key3"), 20);
}

TEST(MetadataDictionary, TestMetadataUpdatePolicyReject) {
  MetadataDictionary d{};
  d.set("key1", 5);
  d.set("key2", 10);

  MetadataDictionary d2{};
  d2.set("key1", 3);
  d2.set("key3", 20);

  d.policy(MetadataPolicy::kReject);
  d.update(d2);
  EXPECT_EQ(d.size(), 3);
  EXPECT_EQ(d.get<int>("key1"), 5);
  EXPECT_EQ(d.get<int>("key2"), 10);
  EXPECT_EQ(d.get<int>("key3"), 20);
}

TEST(MetadataDictionary, TestMetadataUpdatePolicyRaise) {
  MetadataDictionary d{};
  d.set("key1", 5);
  d.set("key2", 10);

  MetadataDictionary d2{};
  d2.set("key1", 3);
  d2.set("key3", 20);

  d.policy(MetadataPolicy::kRaise);
  EXPECT_THROW(d.update(d2), std::runtime_error);
}

}  // namespace holoscan
