/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_authoring.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/errors.hpp>

#include "dl_converter.hpp"
#include "graph_pydoc.hpp"
#include "module_name.hpp"
#include "native_io.hpp"

namespace holoscan {
namespace {

[[noreturn]] void throw_topic_error(std::string_view field) {
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument,
            "AUTHORING_TOPIC_OPTION_TYPE_MISMATCH field_name=" + std::string{field}}};
}

[[noreturn]] void throw_tensor_parameter_error(std::string_view token, std::string_view field) {
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument, std::string{token} + " field_name=" + std::string{field}}};
}

template <typename T>
T unsigned_option(nb::handle value, std::string_view field) {
  static_assert(std::is_unsigned_v<T>);
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_topic_error(field);
  }
  const unsigned long long result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() || result > std::numeric_limits<T>::max()) {
    PyErr_Clear();
    throw_topic_error(field);
  }
  return static_cast<T>(result);
}

template <typename T>
std::optional<std::vector<T>> vector_option(nb::handle value, std::string_view field) {
  if (value.is_none()) {
    return std::nullopt;
  }
  try {
    return nb::cast<std::vector<T>>(value);
  } catch (const nb::cast_error&) {
    throw_topic_error(field);
  }
}

template <typename T>
nb::object optional_tuple(const std::optional<std::vector<T>>& values) {
  if (!values) {
    return nb::none();
  }
  nb::tuple_builder builder(values->size());
  for (const T& value : *values) {
    builder.put(value);
  }
  return builder.commit();
}

template <typename EnumT>
EnumT enum_option(nb::handle value, std::string_view field) {
  try {
    return nb::cast<EnumT>(value);
  } catch (const nb::cast_error&) {
    throw_topic_error(field);
  }
}

}  // namespace

DLDataType python_tensor_dtype(nb::handle value) {
  if (nb::isinstance<DLDataType>(value)) {
    return nb::cast<DLDataType>(value);
  }
  constexpr std::array named_dtypes{
      std::pair<std::string_view, DLDataType>{"bool", {kDLBool, 8U, 1U}},
      std::pair<std::string_view, DLDataType>{"uint8", {kDLUInt, 8U, 1U}},
      std::pair<std::string_view, DLDataType>{"int8", {kDLInt, 8U, 1U}},
      std::pair<std::string_view, DLDataType>{"uint16", {kDLUInt, 16U, 1U}},
      std::pair<std::string_view, DLDataType>{"int16", {kDLInt, 16U, 1U}},
      std::pair<std::string_view, DLDataType>{"uint32", {kDLUInt, 32U, 1U}},
      std::pair<std::string_view, DLDataType>{"int32", {kDLInt, 32U, 1U}},
      std::pair<std::string_view, DLDataType>{"uint64", {kDLUInt, 64U, 1U}},
      std::pair<std::string_view, DLDataType>{"int64", {kDLInt, 64U, 1U}},
      std::pair<std::string_view, DLDataType>{"float16", {kDLFloat, 16U, 1U}},
      std::pair<std::string_view, DLDataType>{"float32", {kDLFloat, 32U, 1U}},
      std::pair<std::string_view, DLDataType>{"float64", {kDLFloat, 64U, 1U}},
      std::pair<std::string_view, DLDataType>{"bfloat16", {kDLBfloat, 16U, 1U}},
      std::pair<std::string_view, DLDataType>{"complex64", {kDLComplex, 64U, 1U}},
      std::pair<std::string_view, DLDataType>{"complex128", {kDLComplex, 128U, 1U}},
  };
  if (nb::isinstance<nb::str>(value)) {
    const std::string spelling = nb::cast<std::string>(value);
    for (const auto& [name, dtype] : named_dtypes) {
      if (name == spelling) {
        return dtype;
      }
    }
  }
  try {
    const nb::object dtype = nb::module_::import_("numpy").attr("dtype")(value);
    auto result = dldatatype_from_typestr(nb::cast<std::string>(dtype.attr("str")));
    if (result) {
      return *result;
    }
  } catch (const nb::python_error&) {
    PyErr_Clear();
    // A failed best-effort NumPy cast falls through to the loaded Torch dtype table.
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (const nb::cast_error&) {
  }
  constexpr std::array torch_dtypes{
      std::pair<const char*, DLDataType>{"bool", {kDLBool, 8U, 1U}},
      std::pair<const char*, DLDataType>{"uint8", {kDLUInt, 8U, 1U}},
      std::pair<const char*, DLDataType>{"int8", {kDLInt, 8U, 1U}},
      std::pair<const char*, DLDataType>{"int16", {kDLInt, 16U, 1U}},
      std::pair<const char*, DLDataType>{"int32", {kDLInt, 32U, 1U}},
      std::pair<const char*, DLDataType>{"int64", {kDLInt, 64U, 1U}},
      std::pair<const char*, DLDataType>{"float16", {kDLFloat, 16U, 1U}},
      std::pair<const char*, DLDataType>{"float32", {kDLFloat, 32U, 1U}},
      std::pair<const char*, DLDataType>{"float64", {kDLFloat, 64U, 1U}},
      std::pair<const char*, DLDataType>{"bfloat16", {kDLBfloat, 16U, 1U}},
      std::pair<const char*, DLDataType>{"complex64", {kDLComplex, 64U, 1U}},
      std::pair<const char*, DLDataType>{"complex128", {kDLComplex, 128U, 1U}},
  };
  const nb::dict modules = nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
  const nb::str torch_name{"torch"};
  if (modules.contains(torch_name)) {
    const nb::object torch = nb::borrow<nb::object>(modules[torch_name]);
    for (const auto& [name, dtype] : torch_dtypes) {
      if (nb::hasattr(torch, name) && value.ptr() == torch.attr(name).ptr()) {
        return dtype;
      }
    }
  }
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_DTYPE_INVALID field_name=dtype"}};
}

PyTensorBounds::PyTensorBounds(nb::handle max_rank, nb::handle max_shape, nb::handle max_byte_span,
                               nb::handle max_abs_byte_strides) {
  set_max_rank(max_rank);
  set_max_shape(max_shape);
  set_max_byte_span(max_byte_span);
  set_max_abs_byte_strides(max_abs_byte_strides);
}

TensorBounds PyTensorBounds::value() const noexcept {
  TensorBounds result{.max_rank = max_rank_, .max_byte_span = max_byte_span_};
  if (max_shape_) {
    result.max_shape = TensorShapeView{*max_shape_};
  }
  if (max_abs_byte_strides_) {
    result.max_abs_byte_strides = TensorMaxAbsByteStridesView{*max_abs_byte_strides_};
  }
  return result;
}

const std::optional<std::uint32_t>& PyTensorBounds::max_rank() const noexcept {
  return max_rank_;
}

void PyTensorBounds::set_max_rank(nb::handle value) {
  max_rank_ = value.is_none()
                  ? std::optional<std::uint32_t>{}
                  : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(value, "max_rank")};
}

nb::object PyTensorBounds::max_shape() const {
  return optional_tuple(max_shape_);
}

void PyTensorBounds::set_max_shape(nb::handle value) {
  max_shape_ = vector_option<std::int64_t>(value, "max_shape");
}

const std::optional<std::uint64_t>& PyTensorBounds::max_byte_span() const noexcept {
  return max_byte_span_;
}

void PyTensorBounds::set_max_byte_span(nb::handle value) {
  max_byte_span_ =
      value.is_none()
          ? std::optional<std::uint64_t>{}
          : std::optional<std::uint64_t>{unsigned_option<std::uint64_t>(value, "max_byte_span")};
}

nb::object PyTensorBounds::max_abs_byte_strides() const {
  return optional_tuple(max_abs_byte_strides_);
}

void PyTensorBounds::set_max_abs_byte_strides(nb::handle value) {
  max_abs_byte_strides_ = vector_option<std::uint64_t>(value, "max_abs_byte_strides");
}

namespace {

[[nodiscard]] TensorOutputStorage python_tensor_output_storage(nb::handle value) {
  if (nb::isinstance<TensorOutputStorage>(value)) {
    return nb::cast<TensorOutputStorage>(value);
  }
  if (nb::isinstance<nb::str>(value)) {
    const std::string spelling = nb::cast<std::string>(value);
    if (spelling == "pass_through") {
      return TensorOutputStorage::kPassThrough;
    }
    if (spelling == "runtime_pool") {
      return TensorOutputStorage::kRuntimePool;
    }
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument,
                           "PYTHON_TENSOR_OUTPUT_STORAGE_INVALID "
                           "accepted=pass_through,runtime_pool field_name=storage"}};
}

}  // namespace

PyTensorInputSpec::PyTensorInputSpec(nb::handle memory_kind, nb::handle dtype, nb::handle rank,
                                     nb::handle strides, nb::handle bounds,
                                     nb::handle memory_conversion) {
  try {
    representation_.memory_kind =
        memory_kind.is_none() ? std::optional<MemoryKind>{}
                              : std::optional<MemoryKind>{nb::cast<MemoryKind>(memory_kind)};
    representation_.dtype = dtype.is_none() ? std::optional<DLDataType>{}
                                            : std::optional<DLDataType>{python_tensor_dtype(dtype)};
    representation_.rank =
        rank.is_none() ? std::optional<std::uint32_t>{}
                       : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(rank, "rank")};
    representation_.strides =
        strides.is_none()
            ? std::optional<TensorStridePolicy>{}
            : std::optional<TensorStridePolicy>{nb::cast<TensorStridePolicy>(strides)};
    if (!bounds.is_none()) {
      bounds_ = nb::cast<PyTensorBounds>(bounds);
    }
    if (!memory_conversion.is_none()) {
      memory_conversion_ = nb::cast<MemoryConversionPolicy>(memory_conversion);
    }
  } catch (const nb::cast_error&) {
    throw_topic_error("tensor_input_spec");
  }
}

TensorInputSpec PyTensorInputSpec::value() const noexcept {
  return TensorInputSpec{
      .representation = representation_,
      .bounds = bounds_ ? std::optional<TensorBounds>{bounds_->value()} : std::nullopt,
      .memory_conversion = memory_conversion_,
  };
}

const TensorRepresentation& PyTensorInputSpec::representation() const noexcept {
  return representation_;
}

const PyTensorBounds* PyTensorInputSpec::bounds() const noexcept {
  return bounds_ ? &*bounds_ : nullptr;
}

const std::optional<MemoryConversionPolicy>& PyTensorInputSpec::memory_conversion() const noexcept {
  return memory_conversion_;
}

PyTensorOutputSpec::PyTensorOutputSpec(nb::handle storage, nb::handle memory_kind, nb::handle dtype,
                                       nb::handle rank, nb::handle strides, nb::handle bounds,
                                       nb::handle transfer)
    : storage_(python_tensor_output_storage(storage)) {
  try {
    representation_.memory_kind =
        memory_kind.is_none() ? std::optional<MemoryKind>{}
                              : std::optional<MemoryKind>{nb::cast<MemoryKind>(memory_kind)};
  } catch (const nb::cast_error&) {
    throw_topic_error("memory_kind");
  }
  representation_.dtype = dtype.is_none() ? std::optional<DLDataType>{}
                                          : std::optional<DLDataType>{python_tensor_dtype(dtype)};
  representation_.rank =
      rank.is_none() ? std::optional<std::uint32_t>{}
                     : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(rank, "rank")};
  try {
    representation_.strides =
        strides.is_none()
            ? std::optional<TensorStridePolicy>{}
            : std::optional<TensorStridePolicy>{nb::cast<TensorStridePolicy>(strides)};
  } catch (const nb::cast_error&) {
    throw_topic_error("strides");
  }
  try {
    if (!bounds.is_none()) {
      bounds_ = nb::cast<PyTensorBounds>(bounds);
    }
  } catch (const nb::cast_error&) {
    throw_topic_error("bounds");
  }
  try {
    if (!transfer.is_none()) {
      transfer_ = nb::cast<TensorTransferPolicy>(transfer);
    }
  } catch (const nb::cast_error&) {
    throw_topic_error("transfer");
  }
}

TensorOutputSpec PyTensorOutputSpec::value() const noexcept {
  return TensorOutputSpec{
      .representation = representation_,
      .bounds = bounds_ ? std::optional<TensorBounds>{bounds_->value()} : std::nullopt,
      .storage = storage_,
      .transfer = transfer_,
  };
}

TensorOutputStorage PyTensorOutputSpec::storage() const noexcept {
  return storage_;
}

const TensorRepresentation& PyTensorOutputSpec::representation() const noexcept {
  return representation_;
}

const PyTensorBounds* PyTensorOutputSpec::bounds() const noexcept {
  return bounds_ ? &*bounds_ : nullptr;
}

const TensorTransferPolicy* PyTensorOutputSpec::transfer() const noexcept {
  return transfer_ ? &*transfer_ : nullptr;
}

PyTensorBoundaryContract::PyTensorBoundaryContract(
    TensorRepresentation representation, PyTensorBounds bounds,
    std::optional<TensorTransferPolicy> transfer,
    std::optional<MemoryConversionPolicy> memory_conversion)
    : representation_(std::move(representation)),
      bounds_(std::move(bounds)),
      transfer_(std::move(transfer)),
      memory_conversion_(memory_conversion) {}

TensorBoundaryContract PyTensorBoundaryContract::value() const noexcept {
  return TensorBoundaryContract{.representation = representation_,
                                .bounds = bounds_.value(),
                                .transfer = transfer_,
                                .memory_conversion = memory_conversion_};
}

TensorRepresentation* PyTensorBoundaryContract::representation() noexcept {
  return &representation_;
}

void PyTensorBoundaryContract::set_representation(TensorRepresentation value) {
  representation_ = std::move(value);
}

PyTensorBounds* PyTensorBoundaryContract::bounds() noexcept {
  return &bounds_;
}

void PyTensorBoundaryContract::set_bounds(PyTensorBounds value) {
  bounds_ = std::move(value);
}

TensorTransferPolicy* PyTensorBoundaryContract::transfer() noexcept {
  return transfer_ ? &*transfer_ : nullptr;
}

void PyTensorBoundaryContract::set_transfer(std::optional<TensorTransferPolicy> value) {
  transfer_ = std::move(value);
}

const std::optional<MemoryConversionPolicy>& PyTensorBoundaryContract::memory_conversion()
    const noexcept {
  return memory_conversion_;
}

void PyTensorBoundaryContract::set_memory_conversion(std::optional<MemoryConversionPolicy> value) {
  memory_conversion_ = value;
}

// This function intentionally keeps one declarative view of the related tensor authoring types.
// Splitting the binding chain would obscure the Python surface without simplifying runtime logic.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void init_tensor_authoring(nb::module_& module) {
  auto memory_conversion_policy_type = python::bind_enum_in_module<MemoryConversionPolicy>(
      module, "holoscan.tensor", "MemoryConversionPolicy");
  memory_conversion_policy_type.value("DEFAULT", MemoryConversionPolicy::kDefault)
      .value("DISABLE", MemoryConversionPolicy::kDisable)
      .value("ALLOW_IF_SAFE", MemoryConversionPolicy::kAllowIfSafe)
      .value("REQUIRE", MemoryConversionPolicy::kRequire);
  auto memory_kind_type =
      python::bind_enum_in_module<MemoryKind>(module, "holoscan.tensor", "MemoryKind");
  memory_kind_type.value("UNKNOWN", MemoryKind::kUnknown)
      .value("HOST", MemoryKind::kHost)
      .value("PINNED_HOST", MemoryKind::kPinnedHost)
      .value("CUDA_DEVICE", MemoryKind::kCudaDevice)
      .value("CUDA_MANAGED", MemoryKind::kCudaManaged);
  auto tensor_stride_policy_type = python::bind_enum_in_module<TensorStridePolicy>(
      module, "holoscan.tensor", "TensorStridePolicy");
  tensor_stride_policy_type.value("CONTIGUOUS_ONLY", TensorStridePolicy::kContiguousOnly)
      .value("BOUNDED_STRIDED", TensorStridePolicy::kBoundedStrided);
  auto interface_tensor_emit_policy_type = python::bind_enum_in_module<InterfaceTensorEmitPolicy>(
      module, "holoscan.tensor", "InterfaceTensorEmitPolicy");
  interface_tensor_emit_policy_type
      .value("REQUIRE_EXPORTABLE", InterfaceTensorEmitPolicy::kRequireExportable)
      .value("COPY_IF_NEEDED", InterfaceTensorEmitPolicy::kCopyIfNeeded)
      .value("INLINE_SMALL_HOST_THEN_COPY_IF_NEEDED",
             InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded);
  auto tensor_representation_type = python::bind_class_in_module<TensorRepresentation>(
      module, "holoscan.tensor", "TensorRepresentation", doc::graph::kTensorRepresentation);
  tensor_representation_type
      .def(
          "__init__",
          [](TensorRepresentation* self,
             nb::handle memory_kind,
             nb::handle dtype,
             nb::handle rank,
             nb::handle strides) {
            try {
              new (self) TensorRepresentation{
                  .memory_kind = memory_kind.is_none()
                                     ? std::optional<MemoryKind>{}
                                     : std::optional<MemoryKind>{nb::cast<MemoryKind>(memory_kind)},
                  .dtype = dtype.is_none() ? std::optional<DLDataType>{}
                                           : std::optional<DLDataType>{nb::cast<DLDataType>(dtype)},
                  .rank = rank.is_none()
                              ? std::optional<std::uint32_t>{}
                              : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(
                                    rank, "rank")},
                  .strides = strides.is_none()
                                 ? std::optional<TensorStridePolicy>{}
                                 : std::optional<TensorStridePolicy>{nb::cast<TensorStridePolicy>(
                                       strides)},
              };
            } catch (const nb::cast_error&) {
              throw_topic_error("tensor_representation");
            }
          },
          nb::kw_only(),
          nb::arg("memory_kind") = nb::none(),
          nb::arg("dtype") = nb::none(),
          nb::arg("rank") = nb::none(),
          nb::arg("strides") = nb::none(),
          nb::sig("def __init__(self, *, memory_kind: holoscan.tensor.MemoryKind | None = None, "
                  "dtype: holoscan.tensor.DLDataType | None = None, rank: int | None = None, "
                  "strides: holoscan.tensor.TensorStridePolicy | None = None) -> None"))
      .def_rw("memory_kind", &TensorRepresentation::memory_kind)
      .def_rw("dtype", &TensorRepresentation::dtype)
      .def_rw("rank", &TensorRepresentation::rank)
      .def_rw("strides", &TensorRepresentation::strides);
  auto tensor_bounds_type = python::bind_class_in_module<PyTensorBounds>(
      module, "holoscan.tensor", "TensorBounds", doc::graph::kTensorBounds);
  tensor_bounds_type
      .def(nb::init<nb::handle, nb::handle, nb::handle, nb::handle>(),
           nb::kw_only(),
           nb::arg("max_rank") = nb::none(),
           nb::arg("max_shape") = nb::none(),
           nb::arg("max_byte_span") = nb::none(),
           nb::arg("max_abs_byte_strides") = nb::none(),
           nb::sig("def __init__(self, *, max_rank: int | None = None, "
                   "max_shape: typing.Sequence[int] | None = None, "
                   "max_byte_span: int | None = None, "
                   "max_abs_byte_strides: typing.Sequence[int] | None = None) -> None"))
      .def_prop_rw("max_rank",
                   &PyTensorBounds::max_rank,
                   &PyTensorBounds::set_max_rank,
                   nb::for_setter(nb::arg().none()))
      .def_prop_rw("max_shape",
                   &PyTensorBounds::max_shape,
                   &PyTensorBounds::set_max_shape,
                   nb::for_setter(nb::arg().none()))
      .def_prop_rw("max_byte_span",
                   &PyTensorBounds::max_byte_span,
                   &PyTensorBounds::set_max_byte_span,
                   nb::for_setter(nb::arg().none()))
      .def_prop_rw("max_abs_byte_strides",
                   &PyTensorBounds::max_abs_byte_strides,
                   &PyTensorBounds::set_max_abs_byte_strides,
                   nb::for_setter(nb::arg().none()));
  auto tensor_transfer_policy_type = python::bind_class_in_module<TensorTransferPolicy>(
      module, "holoscan.tensor", "TensorTransferPolicy", doc::graph::kTensorTransferPolicy);
  tensor_transfer_policy_type
      .def(
          "__init__",
          [](TensorTransferPolicy* self,
             nb::handle foreign_emit_policy,
             nb::handle max_inline_bytes,
             nb::handle max_staging_byte_span) {
            new (self) TensorTransferPolicy{
                .foreign_emit_policy = enum_option<InterfaceTensorEmitPolicy>(
                    foreign_emit_policy, "foreign_emit_policy"),
                .max_inline_bytes =
                    max_inline_bytes.is_none()
                        ? std::optional<std::uint64_t>{}
                        : std::optional<std::uint64_t>{unsigned_option<std::uint64_t>(
                              max_inline_bytes, "max_inline_bytes")},
                .max_staging_byte_span =
                    max_staging_byte_span.is_none()
                        ? std::optional<std::uint64_t>{}
                        : std::optional<std::uint64_t>{unsigned_option<std::uint64_t>(
                              max_staging_byte_span, "max_staging_byte_span")},
            };
          },
          nb::kw_only(),
          nb::arg("foreign_emit_policy") = InterfaceTensorEmitPolicy::kRequireExportable,
          nb::arg("max_inline_bytes") = nb::none(),
          nb::arg("max_staging_byte_span") = nb::none(),
          nb::sig("def __init__(self, *, foreign_emit_policy: "
                  "holoscan.tensor.InterfaceTensorEmitPolicy = ..., "
                  "max_inline_bytes: int | None = None, "
                  "max_staging_byte_span: int | None = None) -> None"))
      .def_rw("foreign_emit_policy", &TensorTransferPolicy::foreign_emit_policy)
      .def_rw("max_inline_bytes", &TensorTransferPolicy::max_inline_bytes)
      .def_rw("max_staging_byte_span", &TensorTransferPolicy::max_staging_byte_span);
  auto tensor_input_spec_type =
      python::bind_class_in_module<PyTensorInputSpec>(module, "holoscan.tensor", "InputSpec");
  tensor_input_spec_type
      .def(nb::init<nb::handle, nb::handle, nb::handle, nb::handle, nb::handle, nb::handle>(),
           nb::kw_only(),
           nb::arg("memory_kind") = nb::none(),
           nb::arg("dtype") = nb::none(),
           nb::arg("rank") = nb::none(),
           nb::arg("strides") = nb::none(),
           nb::arg("bounds") = nb::none(),
           nb::arg("memory_conversion") = nb::none())
      .def_prop_ro(
          "representation", &PyTensorInputSpec::representation, nb::rv_policy::reference_internal)
      .def_prop_ro("bounds", &PyTensorInputSpec::bounds, nb::rv_policy::reference_internal)
      .def_prop_ro("memory_conversion",
                   &PyTensorInputSpec::memory_conversion,
                   nb::rv_policy::reference_internal);
  auto tensor_output_storage_type =
      python::bind_enum_in_module<TensorOutputStorage>(module, "holoscan.tensor", "OutputStorage");
  tensor_output_storage_type.value("PASS_THROUGH", TensorOutputStorage::kPassThrough)
      .value("RUNTIME_POOL", TensorOutputStorage::kRuntimePool);
  auto tensor_output_spec_type =
      python::bind_class_in_module<PyTensorOutputSpec>(module, "holoscan.tensor", "OutputSpec");
  tensor_output_spec_type
      .def(nb::init<nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::handle>(),
           nb::kw_only(),
           nb::arg("storage") = "pass_through",
           nb::arg("memory_kind") = nb::none(),
           nb::arg("dtype") = nb::none(),
           nb::arg("rank") = nb::none(),
           nb::arg("strides") = nb::none(),
           nb::arg("bounds") = nb::none(),
           nb::arg("transfer") = nb::none())
      .def_prop_ro("storage", &PyTensorOutputSpec::storage)
      .def_prop_ro(
          "representation", &PyTensorOutputSpec::representation, nb::rv_policy::reference_internal)
      .def_prop_ro("bounds", &PyTensorOutputSpec::bounds, nb::rv_policy::reference_internal)
      .def_prop_ro("transfer", &PyTensorOutputSpec::transfer, nb::rv_policy::reference_internal);
  auto byte_span_expression_type = python::bind_class_in_module<TensorByteSpanExpression>(
      module, "holoscan.tensor", "ByteSpanExpression");
  auto byte_span_invalid_reason_type =
      python::bind_enum_in_module<TensorByteSpanExpressionInvalidReason>(
          module, "holoscan.tensor", "ByteSpanExpressionInvalidReason");
  byte_span_invalid_reason_type.value("NONE", TensorByteSpanExpressionInvalidReason::kNone)
      .value("EMPTY", TensorByteSpanExpressionInvalidReason::kEmpty)
      .value("NON_POSITIVE_CONSTANT", TensorByteSpanExpressionInvalidReason::kNonPositiveConstant)
      .value("INVALID_OPERAND", TensorByteSpanExpressionInvalidReason::kInvalidOperand)
      .value("NODE_LIMIT_EXCEEDED", TensorByteSpanExpressionInvalidReason::kNodeLimitExceeded)
      .value("DEPTH_LIMIT_EXCEEDED", TensorByteSpanExpressionInvalidReason::kDepthLimitExceeded)
      .value("ALIGNMENT_INVALID", TensorByteSpanExpressionInvalidReason::kAlignmentInvalid)
      .value("PROJECTION_INVALID", TensorByteSpanExpressionInvalidReason::kProjectionInvalid);
  byte_span_expression_type.def_prop_ro("invalid_reason", &TensorByteSpanExpression::invalid_reason)
      .def("multiplied_by", &TensorByteSpanExpression::multiplied_by, nb::arg("factor"))
      .def("ceiling_divided_by", &TensorByteSpanExpression::ceiling_divided_by, nb::arg("divisor"))
      .def("maximum_with", &TensorByteSpanExpression::maximum_with, nb::arg("other"))
      .def("aligned_to", &TensorByteSpanExpression::aligned_to, nb::arg("alignment"));
  auto memory_kind_expression_type = python::bind_class_in_module<TensorMemoryKindExpression>(
      module, "holoscan.tensor", "MemoryKindExpression");
  auto memory_kind_invalid_reason_type =
      python::bind_enum_in_module<TensorMemoryKindExpressionInvalidReason>(
          module, "holoscan.tensor", "MemoryKindExpressionInvalidReason");
  memory_kind_invalid_reason_type.value("NONE", TensorMemoryKindExpressionInvalidReason::kNone)
      .value("EMPTY", TensorMemoryKindExpressionInvalidReason::kEmpty)
      .value("MAPPING_EMPTY", TensorMemoryKindExpressionInvalidReason::kMappingEmpty)
      .value("MAPPING_LIMIT_EXCEEDED",
             TensorMemoryKindExpressionInvalidReason::kMappingLimitExceeded)
      .value("MAPPING_DUPLICATE", TensorMemoryKindExpressionInvalidReason::kMappingDuplicate)
      .value("MAPPING_TARGET_INVALID",
             TensorMemoryKindExpressionInvalidReason::kMappingTargetInvalid)
      .value("UNEXPECTED_MAPPING", TensorMemoryKindExpressionInvalidReason::kUnexpectedMapping);
  memory_kind_expression_type.def_prop_ro("invalid_reason",
                                          &TensorMemoryKindExpression::invalid_reason);
  python::bind_functions_in_module(module, "holoscan.tensor", [&] {
    module.def("byte_span", &tensor_byte_span, nb::arg("value"));
    module.def(
        "parameter_memory_kind",
        [](nb::handle descriptor, const nb::dict& mapping) {
          if (mapping.size() != 2U) {
            throw_tensor_parameter_error("PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID", "mapping");
          }
          for (const nb::handle item_handle : mapping.items()) {
            const nb::tuple item = nb::cast<nb::tuple>(item_handle);
            if (!nb::isinstance<nb::bool_>(item[0])) {
              throw_tensor_parameter_error("PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID", "mapping");
            }
          }
          const nb::bool_ false_key{false};
          const nb::bool_ true_key{true};
          if (!mapping.contains(false_key) || !mapping.contains(true_key)) {
            throw_tensor_parameter_error("PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID", "mapping");
          }
          try {
            const MemoryKind when_false = nb::cast<MemoryKind>(mapping[false_key]);
            const MemoryKind when_true = nb::cast<MemoryKind>(mapping[true_key]);
            if (when_false == MemoryKind::kUnknown || when_true == MemoryKind::kUnknown) {
              throw_tensor_parameter_error("PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID", "mapping");
            }
            return active_parameter_memory_kind(descriptor, when_false, when_true);
          } catch (const nb::cast_error&) {
            throw_tensor_parameter_error("PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID", "mapping");
          }
        },
        nb::arg("parameter"),
        nb::arg("mapping"));
    module.def(
        "parameter_byte_span",
        [](nb::handle descriptor) { return active_parameter_byte_span(descriptor, false); },
        nb::arg("parameter"));
    module.def(
        "parameter_contiguous_byte_span",
        [](nb::handle descriptor) { return active_parameter_byte_span(descriptor, true); },
        nb::arg("parameter"));
    module.def(
        "input_max_byte_span",
        [](const PyNativePort& input) { return input.input_max_byte_span(); },
        nb::arg("input"));
  });
  auto tensor_boundary_contract_type = python::bind_class_in_module<PyTensorBoundaryContract>(
      module, "holoscan.tensor", "TensorBoundaryContract", doc::graph::kTensorBoundaryContract);
  tensor_boundary_contract_type
      .def(nb::init<TensorRepresentation,
                    PyTensorBounds,
                    std::optional<TensorTransferPolicy>,
                    std::optional<MemoryConversionPolicy>>(),
           nb::kw_only(),
           nb::arg("representation"),
           nb::arg("bounds"),
           nb::arg("transfer") = nb::none(),
           nb::arg("memory_conversion") = nb::none(),
           nb::sig("def __init__(self, *, representation: "
                   "holoscan.tensor.TensorRepresentation, bounds: holoscan.tensor.TensorBounds, "
                   "transfer: holoscan.tensor.TensorTransferPolicy | None = None, "
                   "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None) -> "
                   "None"))
      .def_prop_rw("representation",
                   &PyTensorBoundaryContract::representation,
                   &PyTensorBoundaryContract::set_representation,
                   nb::rv_policy::reference_internal)
      .def_prop_rw("bounds",
                   &PyTensorBoundaryContract::bounds,
                   &PyTensorBoundaryContract::set_bounds,
                   nb::rv_policy::reference_internal)
      .def_prop_rw("transfer",
                   &PyTensorBoundaryContract::transfer,
                   &PyTensorBoundaryContract::set_transfer,
                   nb::for_setter(nb::arg().none()),
                   nb::rv_policy::reference_internal)
      .def_prop_rw("memory_conversion",
                   &PyTensorBoundaryContract::memory_conversion,
                   &PyTensorBoundaryContract::set_memory_conversion,
                   nb::for_setter(nb::arg().none()));
}

}  // namespace holoscan
