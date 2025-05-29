/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

namespace holoscan {

using IOSpecWithGXFContext = TestWithGXFContext;

TEST(IOSpec, TestIOSpecInitialize) {
  OperatorSpec op_spec = OperatorSpec();

  // kInput
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));
  EXPECT_EQ(spec.op_spec(), &op_spec);
  EXPECT_EQ(spec.name(), "a");
  EXPECT_EQ(spec.io_type(), IOSpec::IOType::kInput);
  EXPECT_EQ(spec.typeinfo(), &typeid(holoscan::gxf::Entity));
  EXPECT_EQ(spec.queue_size(), 1);

  // kOutput
  IOSpec out_spec{
      &op_spec, std::string("b"), IOSpec::IOType::kOutput, &typeid(holoscan::gxf::Entity)};
  EXPECT_EQ(out_spec.name(), "b");
  EXPECT_EQ(out_spec.io_type(), IOSpec::IOType::kOutput);
}

TEST(IOSpec, TestIOSpecConditionMessageAvailable) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));
  EXPECT_EQ(spec.conditions().size(), 0);

  // kMessageAvailable with no params
  spec.condition(ConditionType::kMessageAvailable);

  // kMessageAvailable with one params
  spec.condition(ConditionType::kMessageAvailable, Arg("a", static_cast<size_t>(5)));

  // KMessageAvailable with two params
  spec.condition(ConditionType::kMessageAvailable,
                 Arg("a", static_cast<size_t>(5)),
                 Arg("b", static_cast<size_t>(10)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 3);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
  EXPECT_EQ(condition_pairs[2].second->args().size(), 2);
}

TEST(IOSpec, TestIOSpecConditionDownstreamMessageAffordable) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  // kMessageDownstreamAffordable with no params
  spec.condition(ConditionType::kDownstreamMessageAffordable);

  // kMessageDownstreamAffordable with one params
  spec.condition(ConditionType::kDownstreamMessageAffordable, Arg("a", static_cast<size_t>(5)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 2);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
}

TEST(IOSpec, TestIOSpecConditionCount) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  // kCount with no params
  spec.condition(ConditionType::kCount);

  // kCount with one param
  spec.condition(ConditionType::kCount, Arg("a", static_cast<int64_t>(5)));
  auto condition_pairs = spec.conditions();
  // count condition will log an error and not actually be added to conditions
  EXPECT_EQ(condition_pairs.size(), 0);
}

TEST(IOSpec, TestIOSpecConditionBoolean) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  // kBoolean with no params
  spec.condition(ConditionType::kBoolean);

  // kBoolean with one param
  spec.condition(ConditionType::kBoolean, Arg("a", static_cast<bool>(true)));
  auto condition_pairs = spec.conditions();
  // boolean condition will log an error and not actually be added to conditions
  EXPECT_EQ(condition_pairs.size(), 0);
}

TEST(IOSpec, TestIOSpecConditionNone) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  spec.condition(ConditionType::kNone);
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 1);
  EXPECT_EQ(condition_pairs[0].second, nullptr);
}

TEST_F(TestWithGXFContext, TESTIOSpecResource) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));
  const std::string name{"unbounded"};
  auto resource = F.make_resource<UnboundedAllocator>(name);
  spec.connector(resource);
}

TEST(IOSpec, TestIOSpecConnectorNone) {
  OperatorSpec op_spec = OperatorSpec();
  std::vector<IOSpec::IOType> io_types{IOSpec::IOType::kInput, IOSpec::IOType::kOutput};
  for (auto& io_type : io_types) {
    IOSpec spec = IOSpec(&op_spec, std::string("a"), io_type, &typeid(holoscan::gxf::Entity));

    spec.connector(IOSpec::ConnectorType::kDefault);
    EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDefault);
    EXPECT_EQ(spec.connector(), nullptr);
  }
}

TEST(IOSpec, TestIOSpecConnectorDoubleBufferReceiver) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  // no arguments
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer);
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  ASSERT_TRUE(spec.connector() != nullptr);
  EXPECT_EQ(typeid(spec.connector()), typeid(std::make_shared<Resource>()));
  auto receiver = std::dynamic_pointer_cast<DoubleBufferReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferReceiver"));

  // single argument
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer, Arg("capacity", 2));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  receiver = std::dynamic_pointer_cast<DoubleBufferReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferReceiver"));

  // two arguments
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer, Arg("capacity", 2), Arg("policy", 1));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  receiver = std::dynamic_pointer_cast<DoubleBufferReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferReceiver"));

  // arglist
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer,
                 ArgList{Arg("capacity", 2), Arg("policy", 1)});
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  receiver = std::dynamic_pointer_cast<DoubleBufferReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferReceiver"));
}

TEST(IOSpec, TestIOSpecConnectorUcxReceiver) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  // no arguments
  spec.connector(IOSpec::ConnectorType::kUCX);
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  ASSERT_TRUE(spec.connector() != nullptr);
  EXPECT_EQ(typeid(spec.connector()), typeid(std::make_shared<Resource>()));
  auto receiver = std::dynamic_pointer_cast<UcxReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()), std::string("holoscan::HoloscanUcxReceiver"));

  // two arguments
  spec.connector(IOSpec::ConnectorType::kUCX, Arg("capacity", 2), Arg("policy", 1));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  receiver = std::dynamic_pointer_cast<UcxReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()), std::string("holoscan::HoloscanUcxReceiver"));

  // arglist
  spec.connector(IOSpec::ConnectorType::kUCX,
                 ArgList{Arg("capacity", 2),
                         Arg("policy", 1),
                         Arg("address", "0.0.0.0"),
                         Arg("port", static_cast<uint32_t>(13337))});
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  receiver = std::dynamic_pointer_cast<UcxReceiver>(spec.connector());
  EXPECT_EQ(std::string(receiver->gxf_typename()), std::string("holoscan::HoloscanUcxReceiver"));
}

TEST(IOSpec, TestIOSpecConnectorDoubleBufferTransmitter) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kOutput, &typeid(holoscan::gxf::Entity));

  // no arguments
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer);
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  ASSERT_TRUE(spec.connector() != nullptr);
  EXPECT_EQ(typeid(spec.connector()), typeid(std::make_shared<Resource>()));
  auto transmitter = std::dynamic_pointer_cast<DoubleBufferTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferTransmitter"));

  // single argument
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer, Arg("capacity", 2));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  transmitter = std::dynamic_pointer_cast<DoubleBufferTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferTransmitter"));

  // two arguments
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer, Arg("capacity", 2), Arg("policy", 1));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  transmitter = std::dynamic_pointer_cast<DoubleBufferTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferTransmitter"));

  // arglist
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer,
                 ArgList{Arg("capacity", 2), Arg("policy", 1)});
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kDoubleBuffer);
  transmitter = std::dynamic_pointer_cast<DoubleBufferTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("nvidia::gxf::DoubleBufferTransmitter"));
}

TEST(IOSpec, TestIOSpecConnectorUcxTransmitter) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kOutput, &typeid(holoscan::gxf::Entity));

  // no arguments
  spec.connector(IOSpec::ConnectorType::kUCX);
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  ASSERT_TRUE(spec.connector() != nullptr);
  EXPECT_EQ(typeid(spec.connector()), typeid(std::make_shared<Resource>()));
  auto transmitter = std::dynamic_pointer_cast<UcxTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("holoscan::HoloscanUcxTransmitter"));

  // two arguments
  spec.connector(IOSpec::ConnectorType::kUCX, Arg("capacity", 2), Arg("policy", 1));
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  transmitter = std::dynamic_pointer_cast<UcxTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("holoscan::HoloscanUcxTransmitter"));

  // arglist
  spec.connector(IOSpec::ConnectorType::kUCX,
                 ArgList{Arg("capacity", 2),
                         Arg("policy", 1),
                         Arg("receiver_address", "10.0.0.20"),
                         Arg("port", static_cast<uint32_t>(13337)),
                         Arg("local_address", "0.0.0.0"),
                         Arg("local_port", static_cast<uint32_t>(0))});
  EXPECT_EQ(spec.connector_type(), IOSpec::ConnectorType::kUCX);
  transmitter = std::dynamic_pointer_cast<UcxTransmitter>(spec.connector());
  EXPECT_EQ(std::string(transmitter->gxf_typename()),
            std::string("holoscan::HoloscanUcxTransmitter"));
}

TEST(IOSpec, TestIOSpecQueueSize) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(
      &op_spec, std::string("receivers"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));

  EXPECT_EQ(spec.queue_size(), 1);

  spec.queue_size(0);
  EXPECT_EQ(spec.queue_size(), 0);

  spec.queue_size(-7);
  EXPECT_EQ(spec.queue_size(), -7);
}

TEST_F(IOSpecWithGXFContext, TestIOSpecDescription) {
  F.name("fragment1");
  OperatorSpec op_spec = OperatorSpec(&F);
  IOSpec spec = IOSpec(
      &op_spec, std::string("output1"), IOSpec::IOType::kOutput, &typeid(holoscan::gxf::Entity));
  std::string entity_typename = typeid(holoscan::gxf::Entity).name();

  // Condition added in this way will not yet have had a fragment or component spec assigned
  spec.condition(ConditionType::kMessageAvailable, Arg("min_size", static_cast<uint64_t>(5)));
  auto cond = spec.conditions()[0].second;
  // manually set name and fragment for the condition
  cond->name("message_available");
  cond->fragment(&F);
  // intentionally not connecting a component spec in this case

  // Connector added in this way will not yet have had a fragment or component spec assigned
  spec.connector(IOSpec::ConnectorType::kDoubleBuffer, Arg("capacity", 2));
  // manually populate attributes here (as would normally be done in gxf_executor.cpp)
  auto conn = spec.connector();
  conn->name("receiver");
  conn->fragment(&F);
  auto connector_spec = std::make_shared<ComponentSpec>(&F);
  conn->setup(*connector_spec);
  conn->spec(std::move(connector_spec));

  std::string description = fmt::format(R"(name: output1
io_type: kOutput
typeinfo_name: {}
connector_type: kDoubleBuffer
connector:
  id: -1
  name: receiver
  fragment: fragment1
  args:
    - name: capacity
      type: int32_t
      value: 2
  type: native
  spec:
    fragment: fragment1
    params:
      - name: policy
        type: uint64_t
        description: "0: pop, 1: reject, 2: fault"
        flag: kNone
      - name: capacity
        type: uint64_t
        description: ""
        flag: kNone
  gxf_eid: 0
  gxf_cid: 0
  gxf_typename: nvidia::gxf::DoubleBufferTransmitter
conditions:
  - id: -1
    name: message_available
    fragment: fragment1
    args:
      - name: min_size
        type: uint64_t
        value: 5
    component_type: native
    spec: ~
    gxf_eid: 0
    gxf_cid: 0
    gxf_typename: nvidia::gxf::MessageAvailableSchedulingTerm
    type: kMessageAvailable)",
                                        entity_typename);
  EXPECT_EQ(spec.description(), description);
}

TEST(IOSpec, TestIOSpecInvalidPortName) {
  // "." character is not allowed in IOSPec names
  OperatorSpec op_spec = OperatorSpec();

  EXPECT_THROW(
      { IOSpec spec = IOSpec(&op_spec, std::string("a.b"), IOSpec::IOType::kInput); },
      std::invalid_argument);

  EXPECT_THROW(
      {
        IOSpec spec = IOSpec(
            &op_spec, std::string("a.b"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));
      },
      std::invalid_argument);
}
}  // namespace holoscan
