/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_IO_SPEC_HPP
#define PYHOLOSCAN_CORE_IO_SPEC_HPP

#include <pybind11/pybind11.h>

#include <unordered_map>

#include <holoscan/core/io_spec.hpp>

namespace py = pybind11;

namespace holoscan {

void init_io_spec(py::module_&);

static const std::unordered_map<IOSpec::IOType, const char*> io_type_namemap{
    {IOSpec::IOType::kInput, "INPUT"},
    {IOSpec::IOType::kOutput, "OUTPUT"},
};

static const std::unordered_map<IOSpec::ConnectorType, const char*> connector_type_namemap{
    {IOSpec::ConnectorType::kDefault, "DEFAULT"},
    {IOSpec::ConnectorType::kDoubleBuffer, "DOUBLE_BUFFER"},
    {IOSpec::ConnectorType::kAsyncBuffer, "ASYNC_BUFFER"},
    {IOSpec::ConnectorType::kUCX, "UCX"},
    {IOSpec::ConnectorType::kPubSub, "PUBSUB"},
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_IO_SPEC_HPP */
