# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS GENERATED_TRAITS_A GENERATED_TRAITS_B)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

foreach(generated_traits IN ITEMS "${GENERATED_TRAITS_A}" "${GENERATED_TRAITS_B}")
    get_filename_component(generated_directory "${generated_traits}" DIRECTORY)
    file(GLOB unexpected_intermediates
        "${generated_directory}/*.json"
        "${generated_directory}/*.reflection"
    )
    if(unexpected_intermediates)
        message(FATAL_ERROR
            "Descriptor schema generation retained undeclared reflection intermediates: "
            "${unexpected_intermediates}"
        )
    endif()
endforeach()
