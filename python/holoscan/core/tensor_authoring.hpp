/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <vector>

#include <holoscan/core/tensor_contract.hpp>

namespace holoscan {

namespace nb = nanobind;

// Owns shape and stride arrays that the corresponding C++ TensorBounds exposes as borrowed views.
class PyTensorBounds {
 public:
  PyTensorBounds(nb::handle max_rank, nb::handle max_shape, nb::handle max_byte_span,
                 nb::handle max_abs_byte_strides);

  [[nodiscard]] TensorBounds value() const noexcept;
  [[nodiscard]] const std::optional<std::uint32_t>& max_rank() const noexcept;
  void set_max_rank(nb::handle value);
  [[nodiscard]] nb::object max_shape() const;
  void set_max_shape(nb::handle value);
  [[nodiscard]] const std::optional<std::uint64_t>& max_byte_span() const noexcept;
  void set_max_byte_span(nb::handle value);
  [[nodiscard]] nb::object max_abs_byte_strides() const;
  void set_max_abs_byte_strides(nb::handle value);

 private:
  std::optional<std::uint32_t> max_rank_;
  std::optional<std::uint64_t> max_byte_span_;
  std::optional<std::vector<std::int64_t>> max_shape_;
  std::optional<std::vector<std::uint64_t>> max_abs_byte_strides_;
};

// Owns the nested values backing one Python-authored C++ Tensor boundary contract.
class PyTensorBoundaryContract {
 public:
  PyTensorBoundaryContract(TensorRepresentation representation, PyTensorBounds bounds,
                           std::optional<TensorTransferPolicy> transfer,
                           std::optional<MemoryConversionPolicy> memory_conversion);

  [[nodiscard]] TensorBoundaryContract value() const noexcept;
  [[nodiscard]] TensorRepresentation* representation() noexcept;
  void set_representation(TensorRepresentation value);
  [[nodiscard]] PyTensorBounds* bounds() noexcept;
  void set_bounds(PyTensorBounds value);
  [[nodiscard]] TensorTransferPolicy* transfer() noexcept;
  void set_transfer(std::optional<TensorTransferPolicy> value);
  [[nodiscard]] const std::optional<MemoryConversionPolicy>& memory_conversion() const noexcept;
  void set_memory_conversion(std::optional<MemoryConversionPolicy> value);

 private:
  TensorRepresentation representation_;
  PyTensorBounds bounds_;
  std::optional<TensorTransferPolicy> transfer_;
  std::optional<MemoryConversionPolicy> memory_conversion_;
};

// Owns the optional nested values backing a Python-authored C++ Tensor input declaration.
class PyTensorInputSpec {
 public:
  PyTensorInputSpec(nb::handle memory_kind, nb::handle dtype, nb::handle rank, nb::handle strides,
                    nb::handle bounds, nb::handle memory_conversion);

  [[nodiscard]] TensorInputSpec value() const noexcept;
  [[nodiscard]] const TensorRepresentation& representation() const noexcept;
  [[nodiscard]] const PyTensorBounds* bounds() const noexcept;
  [[nodiscard]] const std::optional<MemoryConversionPolicy>& memory_conversion() const noexcept;

 private:
  TensorRepresentation representation_;
  std::optional<PyTensorBounds> bounds_;
  std::optional<MemoryConversionPolicy> memory_conversion_;
};

// Owns the optional nested values backing a Python-authored C++ Tensor output declaration.
class PyTensorOutputSpec {
 public:
  PyTensorOutputSpec(nb::handle storage, nb::handle memory_kind, nb::handle dtype, nb::handle rank,
                     nb::handle strides, nb::handle bounds, nb::handle transfer);

  [[nodiscard]] TensorOutputSpec value() const noexcept;
  [[nodiscard]] TensorOutputStorage storage() const noexcept;
  [[nodiscard]] const TensorRepresentation& representation() const noexcept;
  [[nodiscard]] const PyTensorBounds* bounds() const noexcept;
  [[nodiscard]] const TensorTransferPolicy* transfer() const noexcept;

 private:
  TensorRepresentation representation_;
  std::optional<PyTensorBounds> bounds_;
  TensorOutputStorage storage_{TensorOutputStorage::kPassThrough};
  std::optional<TensorTransferPolicy> transfer_;
};

[[nodiscard]] DLDataType python_tensor_dtype(nb::handle value);

void init_tensor_authoring(nb::module_& module);

}  // namespace holoscan
