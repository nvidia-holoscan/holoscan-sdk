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

#ifndef HOLOSCAN_CORE_FORWARD_DEF_HPP
#define HOLOSCAN_CORE_FORWARD_DEF_HPP

#include <cinttypes>
namespace holoscan {

class Application;
class Arg;
class ArgumentSetter;
enum class ArgContainerType : uint8_t;
enum class ArgElementType;
class ArgList;
class ArgType;
class Condition;
enum class ConditionType;
class Config;
class Component;
class ComponentSpec;
class ExecutionContext;
class ExtensionManager;
class Executor;
class Fragment;
enum class FlowType;
class Graph;
class GXFParameterAdaptor;
class InputContext;
class IOSpec;
class Logger;
class Message;
class Operator;
class OperatorSpec;
class OutputContext;

template <typename ValueT>
class MetaParameter;

template <typename ValueT>
using Parameter = MetaParameter<ValueT>;

class ParameterWrapper;
enum class ParameterFlag;
class Resource;

// holoscan::gxf
namespace gxf {
class Entity;
class GXFComponent;
class GXFCondition;
class GXFInputContext;
class GXFOutputContext;
class GXFResource;
class GXFExtensionManager;
}  // namespace gxf

// holoscan::ops
namespace ops {
class GXFOperator;
}

// Conditions
class BooleanCondition;
class CountCondition;
class DownstreamMessageAffordableCondition;
class MessageAvailableCondition;

// Resources
class Allocator;
class BlockMemoryPool;
class CudaStreamPool;
class StdComponentSerializer;
class UnboundedAllocator;
class VideoStreamSerializer;

// Domain objects
class Tensor;

}  // namespace holoscan

namespace YAML {
class Node;
}  // namespace YAML

#endif /* HOLOSCAN_CORE_FORWARD_DEF_HPP */
