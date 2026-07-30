/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_component.hpp>
#include <holoscan/core/gxf/gxf_network_context.hpp>
#include <holoscan/core/network_contexts/gxf/ucx_context.hpp>
#include <holoscan/core/resources/gxf/ucx_entity_serializer.hpp>
#ifdef HOLOSCAN_PYTHON_HAS_FASTDDS_PUBSUB_CONTEXT
#include <holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp>
#endif
#include "../core/component_util.hpp"
#include "./network_contexts_pydoc.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the network context.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the network context's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on
 * Fragment::make_network_context<NetworkContextT>
 */

class PyUcxContext : public UcxContext {
 public:
  /* Inherit the constructors */
  using UcxContext::UcxContext;

  // Define a constructor that fully initializes the object.
  explicit PyUcxContext(Fragment* fragment,
                        std::shared_ptr<UcxEntitySerializer> serializer = nullptr,
                        const std::string& name = network_context_default_name_v<UcxContext>) {
    if (serializer) {
      this->add_arg(Arg{"serializer", serializer});
    }
    init_component_base(this, fragment, name);
  }
};

#ifdef HOLOSCAN_PYTHON_HAS_FASTDDS_PUBSUB_CONTEXT
class PyFastDdsPubSubNetworkContext : public FastDdsPubSubNetworkContext {
 public:
  using FastDdsPubSubNetworkContext::FastDdsPubSubNetworkContext;

  explicit PyFastDdsPubSubNetworkContext(
      Fragment* fragment, const std::string& native_buffer_policy = "preferred",
      int64_t native_buffer_acquire_timeout_ms = 500, int64_t native_buffer_export_ttl_ms = 5000,
      bool native_buffer_use_eager_acquire = false,
      const std::string& name = network_context_default_name_v<FastDdsPubSubNetworkContext>)
      : FastDdsPubSubNetworkContext(
            ArgList{Arg{"native_buffer_policy", native_buffer_policy},
                    Arg{"native_buffer_acquire_timeout_ms", native_buffer_acquire_timeout_ms},
                    Arg{"native_buffer_export_ttl_ms", native_buffer_export_ttl_ms},
                    Arg{"native_buffer_use_eager_acquire", native_buffer_use_eager_acquire}}) {
    init_component_base(this, fragment, name);
  }
};
#endif
// End of trampoline classes for handling Python kwargs

PYBIND11_MODULE(_network_contexts, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK NetworkContext Python Bindings
        -------------------------------------------
        .. currentmodule:: _network_contexts
    )pbdoc";

  py::class_<UcxContext,
             PyUcxContext,
             gxf::GXFNetworkContext,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<UcxContext>>(m, "UcxContext", doc::UcxContext::doc_UcxContext)
      .def(py::init<Fragment*, std::shared_ptr<UcxEntitySerializer>, const std::string&>(),
           "fragment"_a,
           "serializer"_a = nullptr,
           "name"_a = std::string(network_context_default_name_v<UcxContext>),
           doc::UcxContext::doc_UcxContext_python);

#ifdef HOLOSCAN_PYTHON_HAS_FASTDDS_PUBSUB_CONTEXT
  py::class_<FastDdsPubSubNetworkContext,
             PyFastDdsPubSubNetworkContext,
             gxf::GXFNetworkContext,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<FastDdsPubSubNetworkContext>>(
      m,
      "FastDdsPubSubNetworkContext",
      doc::FastDdsPubSubNetworkContext::doc_FastDdsPubSubNetworkContext)
      .def(py::init<Fragment*, const std::string&, int64_t, int64_t, bool, const std::string&>(),
           "fragment"_a,
           "native_buffer_policy"_a = "preferred"s,
           "native_buffer_acquire_timeout_ms"_a = int64_t{500},
           "native_buffer_export_ttl_ms"_a = int64_t{5000},
           "native_buffer_use_eager_acquire"_a = false,
           "name"_a = std::string(network_context_default_name_v<FastDdsPubSubNetworkContext>),
           doc::FastDdsPubSubNetworkContext::doc_FastDdsPubSubNetworkContext_python);
#endif
}  // PYBIND11_MODULE
}  // namespace holoscan
