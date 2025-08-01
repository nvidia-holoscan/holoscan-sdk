/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <utility>

#include <holoscan/holoscan.hpp>

class MyServiceResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MyServiceResource, holoscan::Resource)

  MyServiceResource() = default;

  void setup(holoscan::ComponentSpec& spec) override {
    spec.param(int_param_, "int_value", "Integer parameter", "Integer parameter for testing", {});
  }
  [[nodiscard]] int value() const { return int_param_.get(); }

 private:
  holoscan::Parameter<int> int_param_;
};

class MyResourceManagerSelfRef : public holoscan::Resource, public holoscan::FragmentService {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MyResourceManagerSelfRef, holoscan::Resource)

  MyResourceManagerSelfRef() = default;

  [[nodiscard]] std::shared_ptr<Resource> resource() const override { return resource_.lock(); }

  void resource(const std::shared_ptr<Resource>& resource) override { resource_ = resource; }

  void setup(holoscan::ComponentSpec& spec) override {
    spec.param(int_param_, "int_value", "Integer parameter", "Integer parameter for testing", {});
  }

  [[nodiscard]] int value() const { return int_param_.get(); }

 private:
  holoscan::Parameter<int> int_param_;
  std::weak_ptr<Resource> resource_;  ///< Weak reference to the managed resource (self)
};

// NOLINTNEXTLINE(fuchsia-multiple-inheritance)
class MyResourceManagerEnabledShared
    : public holoscan::Resource,
      public holoscan::FragmentService,
      public std::enable_shared_from_this<MyResourceManagerEnabledShared> {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MyResourceManagerEnabledShared, holoscan::Resource)

  MyResourceManagerEnabledShared() = default;
  ~MyResourceManagerEnabledShared() override = default;
  MyResourceManagerEnabledShared(const MyResourceManagerEnabledShared&) = delete;
  MyResourceManagerEnabledShared& operator=(const MyResourceManagerEnabledShared&) = delete;
  MyResourceManagerEnabledShared(MyResourceManagerEnabledShared&&) = delete;
  MyResourceManagerEnabledShared& operator=(MyResourceManagerEnabledShared&&) = delete;

  [[nodiscard]] std::shared_ptr<Resource> resource() const override {
    return std::const_pointer_cast<MyResourceManagerEnabledShared>(shared_from_this());
  }

  void resource([[maybe_unused]] const std::shared_ptr<Resource>& resource) override {}

  void setup(holoscan::ComponentSpec& spec) override {
    spec.param(int_param_, "int_value", "Integer parameter", "Integer parameter for testing", {});
  }

  [[nodiscard]] int value() const { return int_param_.get(); }

 private:
  holoscan::Parameter<int> int_param_;
};

class MyOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MyOp)

  MyOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {}

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("MyOp::compute() executed");

    // 1,2) Fragment service resource can be retrieved via the base FragmentService type
    auto my_service = service<holoscan::DefaultFragmentService>("my_service");
    HOLOSCAN_LOG_INFO("MyService value (via FragmentService): {}",
                      my_service->resource<MyServiceResource>()->value());

    // 1,2) Fragment service resource can also be retrieved directly via the resource type
    auto my_service_resource = service<MyServiceResource>("my_service");
    HOLOSCAN_LOG_INFO("MyService value (via Resource): {}", my_service_resource->value());

    // 3) Fragment service resource can be retrieved via the resource type inheriting both
    //    holoscan::Resource and holoscan::FragmentService interfaces (self-reference)
    auto my_service_resource_selfref =
        service<MyResourceManagerSelfRef>("my_service_resource_selfref");
    HOLOSCAN_LOG_INFO("MyResourceManager value (via MyResourceManagerSelfRef): {}",
                      my_service_resource_selfref->value());

    // 4) Fragment service resource can be retrieved via the resource type inheriting both
    //    holoscan::Resource and holoscan::FragmentService interfaces (shared_from_this())
    auto my_service_resource_enabled_shared =
        service<MyResourceManagerEnabledShared>("my_resource_manager_enabled_shared");
    HOLOSCAN_LOG_INFO("MyResourceManager value (via MyResourceManagerEnabledShared): {}",
                      my_service_resource_enabled_shared->value());
  }
};

class FragmentServiceApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // WARNING: The Fragment Service API is currently experimental in Holoscan SDK.
    // Future SDK releases may introduce breaking changes to this API.

    // 1,2) Create a service resource with an integer value of 20
    auto my_service_resource =
        make_resource<MyServiceResource>("my_service", Arg("int_value") = 20);

    // // 1) Fragment service can be registered with a DefaultFragmentService instance (created with
    // //    a holoscan::Resource object)
    // auto my_service = std::make_shared<DefaultFragmentService>(my_service_resource);
    // register_service(my_service);

    // 2) Fragment service can also be registered directly with a Holoscan Resource object
    register_service(my_service_resource);

    // 3) Create and register a fragment service that inherits from both Resource and
    //    FragmentService.
    //    The service maintains a weak reference to itself that is initialized during
    //    registration.
    auto my_service_resource_selfref = make_resource<MyResourceManagerSelfRef>(
        "my_service_resource_selfref", Arg("int_value") = 30);
    register_service(my_service_resource_selfref);

    // 4) Create and register a fragment service that inherits from both Resource and
    //    FragmentService.
    //    The service maintains a shared reference to itself that is initialized during
    //    registration.
    auto my_resource_manager_enabled_shared = make_resource<MyResourceManagerEnabledShared>(
        "my_resource_manager_enabled_shared", Arg("int_value") = 40);
    register_service(my_resource_manager_enabled_shared);

    auto my_op = make_operator<MyOp>("my_op", make_condition<holoscan::CountCondition>(1));
    add_operator(my_op);
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<FragmentServiceApp>();
  app->run();

  return 0;
}
