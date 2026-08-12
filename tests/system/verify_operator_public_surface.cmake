# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: OPERATOR-013

if(NOT DEFINED HOLOSCAN_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR "HOLOSCAN_PUBLIC_INCLUDE_DIR is required")
endif()

if(EXISTS "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/channel/io_endpoints.hpp")
    message(FATAL_ERROR
        "removed provisional Channel Input/Output wrapper header remains installed"
    )
endif()

file(GLOB_RECURSE operator_public_headers
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/*.h"
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/*.hpp"
)
list(APPEND operator_public_headers
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/holoscan.hpp"
)

set(forbidden_identifiers
    OperatorTestHarness
    InputArray
    OutputArray
    input_array
    output_array
    run_once
    set_cuda_stream
    PublishedReadClaimAccess
    PublishedSampleAccess
    PortBindingAccess
    ValueLifecycleBinding
    ValueLifecycleAccess
    thread_role
    bind_thread_role
    thread_roles
    thread_role_bindings
    thread_role_uses
    thread_role_placements
)

foreach(header IN LISTS operator_public_headers)
    if(NOT EXISTS "${header}")
        continue()
    endif()
    file(READ "${header}" contents)
    foreach(identifier IN LISTS forbidden_identifiers)
        string(REGEX MATCH "(^|[^A-Za-z0-9_])${identifier}([^A-Za-z0-9_]|$)" match "${contents}")
        if(match)
            message(FATAL_ERROR
                "EA1 public Operator boundary exposes forbidden identifier '${identifier}' in ${header}"
            )
        endif()
    endforeach()

    # Runtime/test access helpers may be incomplete friends in installed
    # headers, but their callable definitions belong only to non-installed
    # implementation headers.
    foreach(access_type IN ITEMS PredicateBindingAccess ReadinessSourceAccess)
        string(REGEX MATCH
            "(class|struct)[ \t\r\n]+${access_type}[ \t\r\n]*\\{"
            complete_access_definition
            "${contents}"
        )
        if(complete_access_definition)
            message(FATAL_ERROR
                "EA1 public Operator boundary defines private access helper '${access_type}' in ${header}"
            )
        endif()
    endforeach()

    # These generic names are forbidden specifically as public member-call
    # verbs. Words such as payload and workload remain unaffected.
    foreach(verb IN ITEMS load retrieve run)
        string(REGEX MATCH "[.]${verb}[ \t\r\n]*\\(" match "${contents}")
        if(match)
            message(FATAL_ERROR
                "EA1 public Operator boundary exposes forbidden member verb '${verb}' in ${header}"
            )
        endif()
    endforeach()
endforeach()

# REQ: PARAM-019
# ExecutionContext exposes only the fixed compute-ingress facade. Keep its
# public include closure independent of owning control, snapshots, Graph, and
# ExecutionPlan so ordinary Operator callbacks do not pay for the cold control
# surface merely by including their callback context.
foreach(header_name IN ITEMS execution_context.hpp parameter_compute_control.hpp)
    set(header "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/${header_name}")
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR "missing installed compute-control header: ${header}")
    endif()
    file(READ "${header}" contents)
    foreach(forbidden_header IN ITEMS
            parameter_control.hpp
            parameter_snapshot.hpp
            execution_plan.hpp
            graph.hpp)
        set(forbidden_include_pattern
            "(^|[\r\n])[ \t]*#[ \t]*include[ \t]*[<\"]holoscan/core/${forbidden_header}[>\"]"
        )
        foreach(probe IN ITEMS
                "#include <holoscan/core/${forbidden_header}>"
                "  # include \"holoscan/core/${forbidden_header}\"")
            string(REGEX MATCH "${forbidden_include_pattern}" probe_match "${probe}")
            if(NOT probe_match)
                message(FATAL_ERROR
                    "forbidden-include guard does not recognize '${probe}'"
                )
            endif()
        endforeach()
        string(REGEX MATCH "${forbidden_include_pattern}" forbidden_include "${contents}")
        if(forbidden_include)
            message(FATAL_ERROR
                "${header_name} directly includes cold header ${forbidden_header}"
            )
        endif()
    endforeach()
endforeach()

# Public classes with out-of-line members must carry the generated Core export
# macro. This textual guard catches the Windows import/export failure before a
# native MSVC package consumer is available in every development environment.
set(tensor_loan_header
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/tensor_output_loan.hpp"
)
if(NOT EXISTS "${tensor_loan_header}")
    message(FATAL_ERROR "missing public tensor output loan header")
endif()
file(READ "${tensor_loan_header}" tensor_loan_contents)
foreach(type IN ITEMS TensorOutputWriteGuard TensorOutputLoan)
    string(REGEX MATCH
        "class[ \t\r\n]+HOLOSCAN_CORE_EXPORT[ \t\r\n]+${type}([^A-Za-z0-9_]|$)"
        exported_type
        "${tensor_loan_contents}"
    )
    if(NOT exported_type)
        message(FATAL_ERROR
            "public out-of-line type '${type}' lacks HOLOSCAN_CORE_EXPORT"
        )
    endif()
endforeach()

# REQ: OPERATOR-025
# OT-T28: the direct Tensor allocation facade may describe logical storage and a borrowed stream,
# but installed Core headers must not name transport, physical pool/slot identity, raw CUDA event
# authority, endpoint generations, or the source-private control implementation.
set(tensor_operator_headers
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/tensor_output_loan.hpp"
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/holoscan/core/port.hpp"
)
set(tensor_operator_forbidden_identifiers
    BufferHandle
    TensorPoolId
    TensorSlotId
    EndpointGeneration
    RuntimeTensorResources
    TensorSlotClaimLedger
    TensorOutputLoanControl
    cudaEvent_t
    CUevent
)
foreach(header IN LISTS tensor_operator_headers)
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR "missing installed Tensor Operator header: ${header}")
    endif()
    file(READ "${header}" contents)
    string(REGEX MATCH "#include[ \t]+[<\"]holomq[/\"]" holomq_include "${contents}")
    if(holomq_include)
        message(FATAL_ERROR "installed Tensor Operator header includes HoloMQ: ${header}")
    endif()
    foreach(identifier IN LISTS tensor_operator_forbidden_identifiers)
        string(REGEX MATCH "(^|[^A-Za-z0-9_])${identifier}([^A-Za-z0-9_]|$)" match "${contents}")
        if(match)
            message(FATAL_ERROR
                "installed Tensor Operator header exposes private identifier '${identifier}' in ${header}"
            )
        endif()
    endforeach()
endforeach()
