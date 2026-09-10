# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# HoloInferExample_Smoke: runs holoinfer_example against configs 01, 03, and 05 with 2 ticks each,
# asserting exit code 0 for all. CMake's add_test() only invokes a single command, so this
# wrapper script runs each invocation and fails the test if any one fails.
#
# Config 03 (multi_model_chained) is included specifically as a regression guard: InferenceOp
# only declares a port for external, non-chained tensor names (a name that is both an output_map
# value for one model and an input_map value for another is an internal handoff, not a port) --
# see InferenceOp::setup() in modules/inference/operators/inference.cpp.

if(NOT DEFINED EXAMPLE_BIN)
    message(FATAL_ERROR "EXAMPLE_BIN not set")
endif()
if(NOT DEFINED CONFIG_DIR)
    message(FATAL_ERROR "CONFIG_DIR not set")
endif()

set(configs
    "01_single_model.yaml"
    "03_multi_model_chained.yaml"
    "05_multi_model_concurrent_graphs.yaml"
)

# model_path_map entries in configs/*.yaml are relative to the example directory (the parent of
# CONFIG_DIR), e.g. "models/identity_model.onnx" -- run from there so they resolve.
get_filename_component(example_dir "${CONFIG_DIR}" DIRECTORY)

foreach(config ${configs})
    message(STATUS "HoloInferExample_Smoke: running ${config}")
    execute_process(
        COMMAND "${EXAMPLE_BIN}" --config "${CONFIG_DIR}/${config}" --ticks 2
        WORKING_DIRECTORY "${example_dir}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout_output
        ERROR_VARIABLE stderr_output
    )
    message(STATUS "${stdout_output}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "holoinfer_example --config ${config} failed with exit code "
                             "${result}\nstdout:\n${stdout_output}\nstderr:\n${stderr_output}")
    endif()
endforeach()
