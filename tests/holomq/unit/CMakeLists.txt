# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(holomq_cuda_unit_test_sources
    test_cuda_buffer_import_sync.cpp
    test_cuda_host_pool.cpp
    test_cuda_pool.cpp
    test_cuda_stream.cpp
    test_cuda_stream_generation.cpp
    test_cuda_vmm_import.cpp
)

set(holomq_core_unit_test_sources
    test_buffer_handle.cpp
    test_context.cpp
    test_errors.cpp
    test_fd_pass.cpp
    test_fd_pass_sockaddr.cpp
    test_host_backing_segment_cache.cpp
    test_iox2_loan_error.cpp
    test_iox2_receive_error.cpp
    test_iox2_send_error.cpp
    test_iox2_setup_error.cpp
    test_local_buffer_descriptor.cpp
    test_logger.cpp
    test_poll_set.cpp
    test_pubsub_capabilities.cpp
    test_pubsub_options.cpp
    test_publisher_sample_release_registry.cpp
    test_readiness_portability.cpp
    test_scoped_file_descriptor.cpp
    test_shm_ref_count.cpp
    test_slot_region.cpp
    test_vmm_pool_export_registry.cpp
    test_wait_set.cpp
    test_wait_trigger.cpp
)

set(core_test_sources "")
foreach(source IN LISTS holomq_core_unit_test_sources)
    list(APPEND core_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
endforeach()

set(cuda_test_sources "")
foreach(source IN LISTS holomq_cuda_unit_test_sources)
    list(APPEND cuda_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
endforeach()

if(core_test_sources STREQUAL "" AND cuda_test_sources STREQUAL "")
    message(STATUS "HoloMQ unit tests skipped (no test_*.cpp under tests/unit/)")
    return()
endif()

function(holomq_add_unittest_target target_name sources)
    add_executable(${target_name} ${sources})
    add_executable(holomq::${target_name} ALIAS ${target_name})
    set_target_properties(${target_name} PROPERTIES CXX_CLANG_TIDY "")
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../../src/holomq")
    target_link_libraries(${target_name}
        PRIVATE
            ${HOLOMQ_TEST_LIBRARY}
            GTest::gtest_main
    )
endfunction()

function(holomq_core_gtest_filter source_file out_var)
    get_filename_component(stem "${source_file}" NAME_WE)
    string(REGEX REPLACE "^test_" "" test_suffix "${stem}")
    string(REPLACE "_" ";" parts "${test_suffix}")
    set(pascal "")
    foreach(part IN LISTS parts)
        if(part STREQUAL "pubsub")
            # Preserve the acronym boundary used by the PubSub test suites.
            # `Pubsub*Test` is a valid but empty filter, so CTest would
            # otherwise report success after running zero matching tests.
            set(pascal "${pascal}PubSub")
        else()
            string(SUBSTRING "${part}" 0 1 first)
            string(SUBSTRING "${part}" 1 -1 rest)
            string(TOUPPER "${first}" first)
            set(pascal "${pascal}${first}${rest}")
        endif()
    endforeach()
    set(${out_var} "${pascal}Test.*" PARENT_SCOPE)
endfunction()

function(holomq_core_ctest_name source_file out_var)
    get_filename_component(stem "${source_file}" NAME_WE)
    string(REGEX REPLACE "^test_" "" test_suffix "${stem}")
    set(${out_var} "${test_suffix}_test" PARENT_SCOPE)
endfunction()

function(holomq_cuda_gtest_filter source_file out_var)
    get_filename_component(stem "${source_file}" NAME_WE)
    string(REGEX REPLACE "^test_cuda_" "" test_suffix "${stem}")
    string(REPLACE "_" ";" parts "${test_suffix}")
    set(pascal "")
    foreach(part IN LISTS parts)
        string(SUBSTRING "${part}" 0 1 first)
        string(SUBSTRING "${part}" 1 -1 rest)
        string(TOUPPER "${first}" first)
        set(pascal "${pascal}${first}${rest}")
    endforeach()
    set(${out_var} "Cuda${pascal}Test.*" PARENT_SCOPE)
endfunction()

function(holomq_cuda_ctest_name source_file out_var)
    get_filename_component(stem "${source_file}" NAME_WE)
    string(REGEX REPLACE "^test_cuda_" "" test_suffix "${stem}")
    set(${out_var} "cuda_${test_suffix}_test" PARENT_SCOPE)
endfunction()

set(unittest_targets "")

if(NOT core_test_sources STREQUAL "")
    set(PROJECT_NAME holomq_unittests)
    holomq_add_unittest_target(${PROJECT_NAME} "${core_test_sources}")
    list(APPEND unittest_targets ${PROJECT_NAME})

    foreach(core_source IN LISTS core_test_sources)
        holomq_core_ctest_name("${core_source}" ctest_name)
        holomq_core_gtest_filter("${core_source}" gtest_filter)
        add_test(
            NAME ${ctest_name}
            COMMAND $<TARGET_FILE:${PROJECT_NAME}> --gtest_filter=${gtest_filter}
        )
    endforeach()
endif()

if(NOT cuda_test_sources STREQUAL "")
    set(CUDA_PROJECT_NAME holomq_cuda_unittests)
    holomq_add_unittest_target(${CUDA_PROJECT_NAME} "${cuda_test_sources}")

    add_executable(holomq_cuda_host_import_child
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_host_import_child.cpp")
    set_target_properties(holomq_cuda_host_import_child PROPERTIES CXX_CLANG_TIDY "")
    target_include_directories(holomq_cuda_host_import_child PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../../src/holomq")
    target_link_libraries(holomq_cuda_host_import_child PRIVATE ${HOLOMQ_TEST_LIBRARY})
    target_compile_definitions(${CUDA_PROJECT_NAME} PRIVATE
        HOLOMQ_CUDA_HOST_IMPORT_CHILD_EXE="$<TARGET_FILE:holomq_cuda_host_import_child>")
    add_dependencies(${CUDA_PROJECT_NAME} holomq_cuda_host_import_child)

    list(APPEND unittest_targets ${CUDA_PROJECT_NAME})
    list(APPEND unittest_targets holomq_cuda_host_import_child)

    foreach(cuda_source IN LISTS cuda_test_sources)
        holomq_cuda_ctest_name("${cuda_source}" ctest_name)
        holomq_cuda_gtest_filter("${cuda_source}" gtest_filter)
        add_test(
            NAME ${ctest_name}
            COMMAND $<TARGET_FILE:${CUDA_PROJECT_NAME}> --gtest_filter=${gtest_filter}
        )
        if(ctest_name STREQUAL "cuda_vmm_import_test")
            set_tests_properties(${ctest_name} PROPERTIES TIMEOUT 30)
        endif()
    endforeach()
endif()

add_custom_target(holomq_all_unittests DEPENDS ${unittest_targets})
