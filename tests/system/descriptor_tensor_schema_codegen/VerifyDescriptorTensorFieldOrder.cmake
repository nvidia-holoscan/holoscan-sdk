# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED GENERATED_TRAITS OR NOT EXISTS "${GENERATED_TRAITS}")
    message(FATAL_ERROR "GENERATED_TRAITS must name a generated descriptor traits header")
endif()
file(READ "${GENERATED_TRAITS}" generated_traits)
string(FIND "${generated_traits}"
    "kDescriptorTensorFields{\n    \"first\",\n    \"second\",\n    \"third\",\n};"
    tensor_names_position
)
if(tensor_names_position EQUAL -1)
    message(FATAL_ERROR
        "Generated descriptor Tensor names are not in canonical FlatBuffers field-id order"
    )
endif()
string(FIND "${generated_traits}"
    "kDescriptorTensorFieldIds{\n    1U,\n    3U,\n    5U,\n};"
    tensor_ids_position
)
if(tensor_ids_position EQUAL -1)
    message(FATAL_ERROR
        "Generated descriptor Tensor ids do not retain canonical reflected field ids"
    )
endif()
string(CONCAT expected_tensor_selectors
    "struct MultiDescriptorFrameTensorFields {\n"
    "  inline static constexpr holoscan::TensorFieldSelector<app3p::MultiDescriptorFrameT> "
    "first{1U, 0U};\n"
    "  inline static constexpr holoscan::TensorFieldSelector<app3p::MultiDescriptorFrameT> "
    "second{3U, 1U};\n"
    "  inline static constexpr holoscan::TensorFieldSelector<app3p::MultiDescriptorFrameT> "
    "third{5U, 2U};\n"
    "};"
)
string(FIND "${generated_traits}"
    "${expected_tensor_selectors}"
    tensor_selectors_position
)
if(tensor_selectors_position EQUAL -1)
    message(FATAL_ERROR
        "Generated descriptor Tensor selectors do not retain typed canonical field identity"
    )
endif()

foreach(non_tensor_field IN ITEMS sequence label exposure header)
    string(FIND "${generated_traits}"
        "TensorFieldSelector<app3p::MultiDescriptorFrameT> ${non_tensor_field}"
        non_tensor_selector_position
    )
    if(NOT non_tensor_selector_position EQUAL -1)
        message(FATAL_ERROR
            "Generated descriptor Tensor selectors include non-Tensor field: ${non_tensor_field}"
        )
    endif()
endforeach()
