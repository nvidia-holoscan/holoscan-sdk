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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../utils.hpp"
#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

using namespace std::string_literals;

namespace holoscan {

using DataLoggerResourceClassesWithGXFContext = TestWithGXFContext;
using ResourceClassesWithGXFContext = TestWithGXFContext;

TEST_F(DataLoggerResourceClassesWithGXFContext, TestBasicConsoleLogger) {
  const std::string name{"console-logger"};
  ArgList arglist{Arg{"log_inputs", true},
                  Arg{"log_outputs", true},
                  Arg{"log_metadata", true},
                  Arg{"log_tensor_data_contents", true},
                  Arg{"allowlist_patterns", std::vector<std::string>{}},
                  Arg{"denylist_patterns", std::vector<std::string>{".*op3.*", ".*op5.*"}}};
  auto resource = F.make_resource<data_loggers::BasicConsoleLogger>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource), typeid(std::make_shared<data_loggers::BasicConsoleLogger>(arglist)));
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(DataLoggerResourceClassesWithGXFContext, TestBasicConsoleLoggerDefaultConstructor) {
  auto resource = F.make_resource<data_loggers::BasicConsoleLogger>();
}

TEST_F(ResourceClassesWithGXFContext, TestSimpleTextSerializer) {
  const std::string name{"text-serializer"};
  ArgList arglist{Arg{"max_elements", static_cast<int64_t>(10)},
                  Arg{"max_metadata_items", static_cast<int64_t>(10)},
                  Arg{"log_python_object_contents", true}};
  auto resource = F.make_resource<data_loggers::SimpleTextSerializer>(name, arglist);
  EXPECT_EQ(resource->name(), name);
  EXPECT_EQ(typeid(resource),
            typeid(std::make_shared<data_loggers::SimpleTextSerializer>(arglist)));
  EXPECT_TRUE(resource->description().find("name: " + name) != std::string::npos);
}

TEST_F(ResourceClassesWithGXFContext, TestSimpleTextSerializerDefaultConstructor) {
  auto resource = F.make_resource<data_loggers::SimpleTextSerializer>();
}
}  // namespace holoscan
