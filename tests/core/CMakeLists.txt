# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_executable(diagnostic_format_test test_diagnostic.cpp)
holoscan_configure_test_clang_tidy(diagnostic_format_test)

target_link_libraries(diagnostic_format_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(diagnostic_format_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX DIAGNOSTIC_FORMAT.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(canonical_text_encoding_test test_canonical_text_encoding.cpp)
holoscan_configure_test_clang_tidy(canonical_text_encoding_test)

target_include_directories(canonical_text_encoding_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(canonical_text_encoding_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(canonical_text_encoding_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CANONICAL_TEXT_ENCODING.
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(core_hello_test test_hello.cpp)
holoscan_configure_test_clang_tidy(core_hello_test)

target_link_libraries(core_hello_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(core_hello_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CORE_HELLO.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(expected_test test_expected.cpp)
holoscan_configure_test_clang_tidy(expected_test)

target_link_libraries(expected_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(expected_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX EXPECTED.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(tag_test test_tag.cpp)
holoscan_configure_test_clang_tidy(tag_test)

target_include_directories(tag_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(tag_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(tag_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TAG.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

# HSDK5-003 starts from the installed Core contract rather than from the private
# Channel endpoint wrappers.  Keeping the public-shape tests in one target
# makes every translation unit include only the header it owns while the
# synthetic binding test can additionally consume the non-installed runtime
# seam from public/src.
add_executable(operator_test
    ../common/allocation_counter.cpp
    test_execution_context.cpp
    test_operator.cpp
    test_operator_binding.cpp
    test_operator_spec.cpp
    test_port.cpp
    test_predicate_input_view.cpp
    test_rate.cpp
    test_rate_math_internal.cpp
    test_readiness_source.cpp
    test_sample.cpp
    test_schema_identity.cpp
    test_tensor_contract.cpp
    test_pool_capacity_reducer.cpp
    test_tensor_output_loan.cpp
)
holoscan_configure_test_clang_tidy(operator_test)

target_include_directories(operator_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(operator_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(operator_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX OPERATOR.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(rate_no_exceptions_test test_rate_no_exceptions.cpp)
holoscan_configure_cxx20_target(rate_no_exceptions_test PRIVATE)
target_include_directories(rate_no_exceptions_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/include
        ${holoscan_BINARY_DIR}/include
        ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(rate_no_exceptions_test PRIVATE holoscan::core)
if(MSVC)
  target_compile_options(rate_no_exceptions_test PRIVATE /EHs-c-)
else()
  target_compile_options(rate_no_exceptions_test PRIVATE -fno-exceptions)
endif()
if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
  # clang-tidy parses every included header with the target's compile flags.  Permit exception
  # syntax during analysis so unrelated, uninstantiated throw paths remain parseable while the
  # real compiler invocation still proves this focused translation unit under -fno-exceptions.
  set_target_properties(rate_no_exceptions_test PROPERTIES
      CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--extra-arg=-fexceptions"
  )
endif()
add_test(NAME OPERATOR.RATE_NO_EXCEPTIONS COMMAND rate_no_exceptions_test)
set_tests_properties(OPERATOR.RATE_NO_EXCEPTIONS PROPERTIES
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

# HSDK5-008 keeps pointer-free lifecycle normalization and its pure bounded
# execution kernel together in one focused component target. The public-value
# and registration translation units also exercise the private source-tree
# binding seams without exposing them through the installed package.
add_executable(lifecycle_test
    ../common/allocation_counter.cpp
    $<TARGET_OBJECTS:holoscan_lifecycle_plan_private_objects>
    test_lifecycle.cpp
    test_lifecycle_executor.cpp
    test_lifecycle_graph_lowering.cpp
    test_lifecycle_plan.cpp
    test_lifecycle_registration.cpp
    test_lifecycle_transaction.cpp
)
holoscan_configure_test_clang_tidy(lifecycle_test)
holoscan_apply_logging_policy(lifecycle_test)

target_include_directories(lifecycle_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lifecycle_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(lifecycle_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX LIFECYCLE.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(graph_test
    test_graph_partition_instances.cpp
    test_graph.cpp
    test_graph_tensor_schema.cpp
)
holoscan_configure_test_clang_tidy(graph_test)

target_include_directories(graph_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(graph_test
    PRIVATE
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        GTest::gtest_main
)

gtest_discover_tests(graph_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX GRAPH.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(subgraph_test test_subgraph.cpp)
holoscan_configure_test_clang_tidy(subgraph_test)

target_include_directories(subgraph_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(subgraph_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(subgraph_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX SUBGRAPH.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_test(
    NAME GRAPH_PUBLIC_SURFACE.no_application_or_pipeline_class
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${holoscan_SOURCE_DIR}/include/holoscan
        -DHOLOSCAN_PUBLIC_SOURCE_ROOT=${holoscan_SOURCE_DIR}
        -DHOLOSCAN_BUILD_ROOT=${holoscan_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_graph_public_surface.cmake
)

add_test(
    NAME GRAPH_PUBLIC_SURFACE.ready_policy_evidence_exception_is_exact
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_GRAPH_SURFACE_SCANNER=${CMAKE_CURRENT_SOURCE_DIR}/check_graph_public_surface.cmake
        -DHOLOSCAN_GRAPH_SURFACE_FIXTURE_ROOT=${CMAKE_CURRENT_BINARY_DIR}/graph_surface_fixture
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_public_surface_scanner.cmake
)

find_program(HOLOSCAN_DUMPBIN_EXECUTABLE NAMES dumpbin)
add_test(
    NAME GRAPH_PUBLIC_SURFACE.no_forbidden_exported_symbols
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_CORE_LIBRARY=$<TARGET_FILE:core>
        -DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}
        -DHOLOSCAN_READELF_EXECUTABLE=${CMAKE_READELF}
        -DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_graph_exported_symbols.cmake
)
set_tests_properties(GRAPH_PUBLIC_SURFACE.no_forbidden_exported_symbols
    PROPERTIES SKIP_REGULAR_EXPRESSION "GRAPH_SYMBOL_SCAN_UNSUPPORTED:"
)

add_executable(temporal_contract_test
    ../common/allocation_counter.cpp
    test_predicate_evaluator.cpp
    test_temporal_contract.cpp
    test_temporal_contract_lowering.cpp
    test_temporal_input_cache.cpp
    test_temporal_preselection.cpp
)
holoscan_configure_test_clang_tidy(temporal_contract_test)

target_include_directories(temporal_contract_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(temporal_contract_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(temporal_contract_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TEMPORAL_CONTRACT.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(temporal_contract_records_test test_temporal_contract_records.cpp)
holoscan_configure_test_clang_tidy(temporal_contract_records_test)

target_link_libraries(temporal_contract_records_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(temporal_contract_records_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TEMPORAL_CONTRACT_RECORDS.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(parameter_test test_parameter.cpp test_parameter_hash.cpp test_parameter_wide.cpp)
holoscan_configure_test_clang_tidy(parameter_test)

target_include_directories(parameter_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(parameter_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(parameter_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX PARAMETER.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_library(parameter_dso_test_plugin SHARED parameter_dso_test_plugin.cpp)
holoscan_configure_test_clang_tidy(parameter_dso_test_plugin)
# The GCC-only flag below is required to keep the header-defined type tokens
# DSO-local for this conformance test. Clang-tidy parses the GCC compile command
# with Clang and rejects that flag before analyzing the source, so keep the
# small plugin shim out of compile-time linting. The same public headers remain
# covered by the host test and the other Core targets.
set_source_files_properties(parameter_dso_test_plugin.cpp PROPERTIES SKIP_LINTING ON)
set_target_properties(parameter_dso_test_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
target_compile_options(parameter_dso_test_plugin
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>
)
target_link_libraries(parameter_dso_test_plugin PRIVATE holoscan::core)

add_executable(parameter_dso_test test_parameter_dso.cpp)
holoscan_configure_test_clang_tidy(parameter_dso_test)
target_include_directories(parameter_dso_test PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(parameter_dso_test
    PRIVATE
        holoscan::core
        parameter_dso_test_plugin
        GTest::gtest_main
)
if(HOLOSCAN_BUILD_PARAMS)
    target_compile_definitions(parameter_dso_test PRIVATE HOLOSCAN_PARAMETER_DSO_PARAMS_TESTS=1)
    target_link_libraries(parameter_dso_test PRIVATE holoscan::params)
endif()

gtest_discover_tests(parameter_dso_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX PARAMETER_DSO.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(expected_helpers_test test_expected_helpers.cpp)
holoscan_configure_test_clang_tidy(expected_helpers_test)

target_link_libraries(expected_helpers_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)
target_include_directories(expected_helpers_test PRIVATE ${holoscan_SOURCE_DIR}/src)

gtest_discover_tests(expected_helpers_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX EXPECTED_HELPERS.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

function(holoscan_add_expected_compile_fail_test test_name source_name diagnostic_fragment)
    set(target_name "expected_compile_fail_${test_name}")
    set(source_path "expected_compile_fail/${source_name}")
    add_executable(${target_name} EXCLUDE_FROM_ALL "${source_path}")

    # These fixtures are intentionally ill-formed, so the configured C++
    # compiler must remain the diagnostic oracle. Static-analysis front ends
    # can otherwise fail first under a different language mode and hide the
    # exact public-API diagnostic this test is required to verify.
    set_source_files_properties("${source_path}" PROPERTIES SKIP_LINTING ON)
    target_link_libraries(${target_name} PRIVATE holoscan::core)

    add_test(NAME "EXPECTED_COMPILE_FAIL.${test_name}"
        COMMAND ${CMAKE_COMMAND}
            -DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=$<CONFIG>
            -DHOLOSCAN_TARGET=${target_name}
            -DHOLOSCAN_EXPECTED_DIAGNOSTIC=${diagnostic_fragment}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expected_compile_fail/verify_compile_failure.cmake
    )
    # These probes rebuild in the SDK tree and may relink its shared libraries.
    # Exclude running SDK tests as well as other build-mutating probes.
    set_tests_properties("EXPECTED_COMPILE_FAIL.${test_name}" PROPERTIES RUN_SERIAL TRUE)
endfunction()

holoscan_add_expected_compile_fail_test(
    non_expected
    non_expected.cpp
    "HOLOSCAN_TRY requires a holoscan::expected operand"
)
holoscan_add_expected_compile_fail_test(
    non_expected_void
    non_expected_void.cpp
    "HOLOSCAN_TRY requires a holoscan::expected operand"
)
holoscan_add_expected_compile_fail_test(
    value_in_void_form
    value_in_void_form.cpp
    "HOLOSCAN_RETURN_IF_ERROR requires holoscan::expected<void, E>"
)
holoscan_add_expected_compile_fail_test(
    parameter_duplicate_field
    parameter_duplicate_field.cpp
    "duplicate parameter field member, key, or compact id"
)
holoscan_add_expected_compile_fail_test(
    parameter_schema_requires_aggregate
    parameter_schema_requires_aggregate.cpp
    "parameter schema requires an exact default-initializable copyable aggregate"
)
if(HOLOSCAN_BUILD_PARAMS)
    holoscan_add_expected_compile_fail_test(
        parameter_schema_codec_requires_complete
        parameter_schema_codec_requires_complete.cpp
        "Schema-derived object codecs require a complete parameter schema"
    )
    holoscan_add_expected_compile_fail_test(
        parameter_schema_codec_requires_codable_fields
        parameter_schema_codec_requires_codable_fields.cpp
        "Schema-derived object codecs require ConfigCodable, writable schema fields"
    )
endif()
holoscan_add_expected_compile_fail_test(
    tensor_memory_kind_direct_requires_exact_type
    tensor_memory_kind_direct_requires_exact_type.cpp
    "direct parameter_memory_kind requires an exact MemoryKind field"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_throwing
    tensor_byte_span_projector_throwing.cpp
    "parameter byte-span projector must be declared noexcept"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_wrong_result
    tensor_byte_span_projector_wrong_result.cpp
    "parameter byte-span projector must return expected<uint64_t, Error>"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_wrong_params
    tensor_byte_span_projector_wrong_params.cpp
    "parameter byte-span projector dependencies must belong to its exact Params type"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_no_dependencies
    tensor_byte_span_projector_no_dependencies.cpp
    "parameter byte-span projector requires at least one Parameter dependency"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_nonstatic
    tensor_byte_span_projector_nonstatic.cpp
    "parameter byte-span projector must be a free or static function taking const Params&"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_wrong_arity
    tensor_byte_span_projector_wrong_arity.cpp
    "parameter byte-span projector must be a free or static function taking const Params&"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_stateful_function_object
    tensor_byte_span_projector_stateful_function_object.cpp
    "parameter byte-span projector must be a free or static function taking const Params&"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_capturing_lambda
    tensor_byte_span_projector_capturing_lambda.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_std_function
    tensor_byte_span_projector_std_function.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    tensor_byte_span_projector_null
    tensor_byte_span_projector_null.cpp
    "parameter byte-span projector must not be null"
)
holoscan_add_expected_compile_fail_test(
    parameter_throwing_field_destructor
    parameter_throwing_field_destructor.cpp
    "declared parameter fields must be nothrow-destructible for staged updates"
)
holoscan_add_expected_compile_fail_test(
    parameter_invalid_policy
    parameter_invalid_policy.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_internal_tuning
    parameter_internal_tuning.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_invalid_range
    parameter_invalid_range.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_invalid_allowed_value
    parameter_invalid_allowed_value.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_empty_named_enum_domain
    parameter_empty_named_enum_domain.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_optional_sequence_element
    parameter_optional_sequence_element.cpp
    "only top-level std::optional parameter fields are supported"
)
holoscan_add_expected_compile_fail_test(
    parameter_constraint_inapplicable
    parameter_constraint_inapplicable.cpp
    "max_utf8_bytes requires a std::string field"
)
holoscan_add_expected_compile_fail_test(
    parameter_utf8_bound_without_encoding
    parameter_utf8_bound_without_encoding.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_element_utf8_bound_without_encoding
    parameter_element_utf8_bound_without_encoding.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_allowed_values_overflow
    parameter_allowed_values_overflow.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_allowed_values_cumulative_overflow
    parameter_allowed_values_cumulative_overflow.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_nested_without_schema
    parameter_nested_without_schema.cpp
    "nested parameter field requires a declared nested schema"
)
holoscan_add_expected_compile_fail_test(
    parameter_invalid_name
    parameter_invalid_name.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_duplicate_key
    parameter_duplicate_key.cpp
    "duplicate parameter field member, key, or compact id"
)
holoscan_add_expected_compile_fail_test(
    parameter_lifecycle_stop
    parameter_lifecycle_stop.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_lifecycle_release
    parameter_lifecycle_release.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_persistence_conflict
    parameter_persistence_conflict.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_tuning_requires_runtime
    parameter_tuning_requires_runtime.cpp
    "invalid parameter field policy or constraint"
)
holoscan_add_expected_compile_fail_test(
    parameter_ref_opaque
    parameter_ref_opaque.cpp
    "param_ref requires a declared parameter schema path"
)
holoscan_add_expected_compile_fail_test(
    parameter_ref_unlisted
    parameter_ref_unlisted.cpp
    "param_ref requires a declared parameter schema path"
)
holoscan_add_expected_compile_fail_test(
    predicate_invalid_return
    predicate_invalid_return.cpp
    "predicate evaluator must return exactly bool or holoscan::expected<bool, Error>"
)
holoscan_add_expected_compile_fail_test(
    predicate_bool_convertible_return
    predicate_bool_convertible_return.cpp
    "predicate evaluator must return exactly bool or holoscan::expected<bool, Error>"
)
holoscan_add_expected_compile_fail_test(
    predicate_error_carrier
    predicate_error_carrier.cpp
    "predicate evaluator must return exactly bool or holoscan::expected<bool, Error>"
)
holoscan_add_expected_compile_fail_test(
    predicate_std_expected_return
    predicate_std_expected_return.cpp
    "predicate evaluator must return exactly bool or holoscan::expected<bool, Error>"
)
# `std::expected` is a C++23 library type. This negative consumer proves that a
# newer-language caller still cannot substitute it for the SDK's C++20 carrier;
# it does not raise the production target's language requirement.
target_compile_features(expected_compile_fail_predicate_std_expected_return PRIVATE cxx_std_23)
holoscan_add_expected_compile_fail_test(
    predicate_generic_callable
    predicate_generic_callable.cpp
    "predicate evaluator must have one fixed non-generic signature"
)
holoscan_add_expected_compile_fail_test(
    predicate_nondefault_state
    predicate_nondefault_state.cpp
    "predicate state must be default constructible"
)
holoscan_add_expected_compile_fail_test(
    predicate_unbounded_state
    predicate_unbounded_state.cpp
    "predicate state must be nothrow copy constructible"
)
holoscan_add_expected_compile_fail_test(
    void_in_value_form
    void_in_value_form.cpp
    "HOLOSCAN_ASSIGN_OR_RETURN requires holoscan::expected<T, E> with non-void T"
)
holoscan_add_expected_compile_fail_test(
    wrong_arity
    wrong_arity.cpp
    "HOLOSCAN_TRY accepts exactly one or two arguments"
)
holoscan_add_expected_compile_fail_test(
    wrong_arity_many
    wrong_arity_many.cpp
    "HOLOSCAN_TRY accepts exactly one or two arguments"
)
holoscan_add_expected_compile_fail_test(
    zero_arity
    zero_arity.cpp
    "HOLOSCAN_TRY accepts exactly one or two arguments"
)
holoscan_add_expected_compile_fail_test(
    tryx_unavailable
    tryx_unavailable.cpp
    "HOLOSCAN_TRYX is unavailable; use portable HOLOSCAN_TRY statement form"
)
holoscan_add_expected_compile_fail_test(
    unprotected_rhs_comma
    unprotected_rhs_comma.cpp
    "HOLOSCAN_TRY accepts exactly one or two arguments"
)
holoscan_add_expected_compile_fail_test(
    unbraced_value_control_flow
    unbraced_value_control_flow.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_flow_payload_mismatch
    graph_flow_payload_mismatch.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_application_type
    graph_application_type.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_pipeline_type
    graph_pipeline_type.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_member_compile
    graph_member_compile.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_member_run
    graph_member_run.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_member_run_async
    graph_member_run_async.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_partition_selector
    graph_partition_selector.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    execution_plan_partition_of
    execution_plan_partition_of.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_forbidden_capture
    graph_forbidden_capture.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_hidden_lifecycle_hook
    graph_hidden_lifecycle_hook.cpp
    "Operator lifecycle hooks must be public, unambiguous, and have the exact void() signature"
)
holoscan_add_expected_compile_fail_test(
    graph_ambiguous_lifecycle_hook
    graph_ambiguous_lifecycle_hook.cpp
    "Operator lifecycle hooks must be public, unambiguous, and have the exact void() signature"
)
holoscan_add_expected_compile_fail_test(
    graph_scalar_tensor_allocation
    graph_scalar_tensor_allocation.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    graph_carrier_tensor_allocation
    graph_carrier_tensor_allocation.cpp
    "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC"
)
holoscan_add_expected_compile_fail_test(
    published_move_only_pod
    published_move_only_pod.cpp
    "experimental raw POD payload types must be copy constructible"
)
holoscan_add_expected_compile_fail_test(
    pod_invalid_identity
    pod_invalid_identity.cpp
    "experimental raw POD identity name must be dotted"
)
holoscan_add_expected_compile_fail_test(
    pod_internal_padding
    pod_internal_padding.cpp
    "experimental raw POD fields must appear in exact byte-offset order without gaps or overlap"
)
holoscan_add_expected_compile_fail_test(
    pod_tail_padding
    pod_tail_padding.cpp
    "experimental raw POD fields must cover every object byte"
)
holoscan_add_expected_compile_fail_test(
    pod_zero_version
    pod_zero_version.cpp
    "experimental raw POD identity version must be a nonzero std::uint32_t"
)
holoscan_add_expected_compile_fail_test(
    pod_foreign_field_owner
    pod_foreign_field_owner.cpp
    "experimental raw POD layout contains a field owned by another C++ type"
)
holoscan_add_expected_compile_fail_test(
    pod_member_offset_mismatch
    pod_member_offset_mismatch.cpp
    "experimental raw POD field member does not match its declared offsetof value"
)
holoscan_add_expected_compile_fail_test(
    pod_duplicate_field_name
    pod_duplicate_field_name.cpp
    "experimental raw POD field names must be unique"
)
holoscan_add_expected_compile_fail_test(
    pod_unsupported_pointer_field
    pod_unsupported_pointer_field.cpp
    "experimental raw POD fields must be fixed-width integers"
)
holoscan_add_expected_compile_fail_test(
    descriptor_schema_missing_type_ops
    descriptor_schema_missing_type_ops.cpp
    "a descriptor-backed Tensor schema must provide a runtime_port_type_ops<T>() specialization"
)
holoscan_add_expected_compile_fail_test(
    descriptor_schema_conflicting_traits
    descriptor_schema_conflicting_traits.cpp
    "DescriptorTensorSchemaNativeTraits<T> and GeneratedSchemaNativeTraits<T> are mutually exclusive"
)

add_executable(logger_test test_logger.cpp)
holoscan_configure_test_clang_tidy(logger_test)
holoscan_apply_logging_policy(logger_test)

target_link_libraries(logger_test
    PRIVATE
        holoscan::logger
        GTest::gtest_main
)

gtest_discover_tests(logger_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX LOGGER.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(tensor_test test_tensor.cpp)
holoscan_configure_test_clang_tidy(tensor_test)
target_include_directories(tensor_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(tensor_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(tensor_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TENSOR.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(schema_test test_schema.cpp)
holoscan_configure_test_clang_tidy(schema_test)
add_dependencies(schema_test holoscan_flatbuffer_schemas_generated)

target_include_directories(schema_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)

target_link_libraries(schema_test
    PRIVATE
        holoscan::schema
        # The Image and Imu generated headers and their trait specializations come from here now.
        holoscan::sensor_io
        # holoscan::schema links FlatBuffers privately, so the test links it directly for the
        # FlatBufferBuilder / Verifier / generated-header symbols it uses itself.
        flatbuffers::flatbuffers
        GTest::gtest_main
)

gtest_discover_tests(schema_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX SCHEMA.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(tensor_device_test
    test_tensor_device.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_test_clang_tidy(tensor_device_test)
target_include_directories(tensor_device_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(tensor_device_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(tensor_device_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TENSOR_DEVICE.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
