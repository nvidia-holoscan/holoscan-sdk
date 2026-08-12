/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/predicate_input_view.hpp>

#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/core/port.hpp>

namespace holoscan::detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<Tensor>() noexcept {
  // Tensor descriptors travel through a typed sidecar rather than an inline/schema byte codec.
  // Runtime still needs one stable type-erased table for typed retention, predicate views, and
  // ordinary port binding without exposing descriptor or pool state through the public port API.
  static const RuntimePortTypeOps ops{
      .payload_size = sizeof(Tensor),
      .payload_alignment = alignof(Tensor),
      .wire_capacity = 0U,
      .sample_size = sizeof(Sample<Tensor>),
      .sample_alignment = alignof(Sample<Tensor>),
      .predicate_view_size = sizeof(PredicateInputView<Tensor>),
      .predicate_view_alignment = alignof(PredicateInputView<Tensor>),
      .codec_family = RuntimePayloadCodecFamily::kUnsupported,
      .payload_kind = PayloadKind::kDescriptorBacked,
      .inline_pod = false,
      .generic_loan_supported = false,
      .generic_loan_slot_size_bytes = 0U,
      .encode_payload = nullptr,
      .construct_sample = nullptr,
      .destroy_sample =
          [](void* storage) noexcept { std::destroy_at(static_cast<Sample<Tensor>*>(storage)); },
      .sample_payload = [](const void* sample) noexcept -> const void* {
        return &static_cast<const Sample<Tensor>*>(sample)->data;
      },
      .sample_metadata = [](const void* sample) noexcept -> const SampleMetadata* {
        return &static_cast<const Sample<Tensor>*>(sample)->metadata;
      },
      .construct_predicate_view =
          [](void* storage,
             const void* sample,
             const InputSelection selection,
             const std::uint64_t* active_generation,
             const std::uint64_t expected_generation) noexcept -> expected<void, Error> {
        if (storage == nullptr || sample == nullptr || active_generation == nullptr ||
            reinterpret_cast<std::uintptr_t>(storage) % alignof(PredicateInputView<Tensor>) != 0U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto view = make_runtime_predicate_input_view<Tensor>(PredicateInputBinding<Tensor>{
            .sample = static_cast<const Sample<Tensor>*>(sample),
            .selection = selection,
            .active_generation = active_generation,
            .expected_generation = expected_generation,
        });
        if (!view) {
          return make_unexpected(std::move(view).error());
        }
        std::construct_at(static_cast<PredicateInputView<Tensor>*>(storage), std::move(*view));
        return {};
      },
      .destroy_predicate_view =
          [](void* storage) noexcept {
            std::destroy_at(static_cast<PredicateInputView<Tensor>*>(storage));
          },
      .construct_value = nullptr,
      .destroy_value = nullptr,
      .bind_input =
          [](void* declaration, RuntimeInputBinding* binding) noexcept {
            RuntimePortBindingAccess::bind_input(*static_cast<Input<Tensor>*>(declaration),
                                                 binding);
          },
      .unbind_input =
          [](void* declaration, RuntimeInputBinding* binding) noexcept {
            RuntimePortBindingAccess::unbind_input(*static_cast<Input<Tensor>*>(declaration),
                                                   binding);
          },
      .bind_output =
          [](void* declaration, RuntimeOutputBinding* binding) noexcept {
            RuntimePortBindingAccess::bind_output(*static_cast<Output<Tensor>*>(declaration),
                                                  binding);
          },
      .unbind_output =
          [](void* declaration, RuntimeOutputBinding* binding) noexcept {
            RuntimePortBindingAccess::unbind_output(*static_cast<Output<Tensor>*>(declaration),
                                                    binding);
          },
      .publish_tensor_loan = [](void* declaration,
                                const void*,
                                Tensor tensor,
                                EmitOptions options) noexcept -> expected<void, Error> {
        return static_cast<Output<Tensor>*>(declaration)->emit(std::move(tensor), options);
      },
      .bind_tensor_output =
          [](void* declaration, TensorOutputBinding binding) noexcept {
            TensorOutputPortBindingAccess::bind(*static_cast<Output<Tensor>*>(declaration),
                                                binding);
          },
      .unbind_tensor_output =
          [](void* declaration) noexcept {
            TensorOutputPortBindingAccess::unbind(*static_cast<Output<Tensor>*>(declaration));
          },
  };
  return &ops;
}

}  // namespace holoscan::detail
