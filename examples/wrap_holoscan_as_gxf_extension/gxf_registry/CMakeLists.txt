# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(BUILD_TESTING AND HOLOSCAN_REGISTER_GXF_EXTENSIONS)
    configure_file(target.yaml.in target.yaml @ONLY)
    configure_file(ping_installable.yaml.in ping_installable.yaml @ONLY)

    find_program(GXF_REGISTRY_EXECUTABLE registry REQUIRED)
    add_test(NAME gxf_native_res_registry_inspection_test
        COMMAND registry extn info -n PingVarCustomNativeResExtension
    )
    set_tests_properties(gxf_native_res_registry_inspection_test PROPERTIES
        PASS_REGULAR_EXPRESSION "name :        PingVarCustomNativeResExtension"
        PASS_REGULAR_EXPRESSION "uuid :        2b8381ed-5c27-40a1-be58-6c019eaa87be"
        PASS_REGULAR_EXPRESSION "version :     ${holoscan_VERSION}"
        PASS_REGULAR_EXPRESSION "myexts::PingVarRxNativeOpCodelet :"
    )

    add_test(NAME gxf_native_res_registry_install_graph_test
        COMMAND registry
            graph install
            -g ping_installable.yaml
            -m test_install_app/manifest.yaml
            -u test_install_app
            -i ""
            -d target.yaml
        )
    set_tests_properties(gxf_native_res_registry_install_graph_test PROPERTIES
        PASS_REGULAR_EXPRESSION "Graph installed to output directory"
    )

    find_program(GXE_EXECUTABLE gxe
        HINTS
            /usr/local/bin
            $<TARGET_FILE:GXF::gxe>
            ${HOLOSCAN_GXE_LOCATION}
        REQUIRED
    )
    add_test(NAME gxf_native_res_registry_run_graph_test
        COMMAND ${GXE_EXECUTABLE}
            -app ping_installable.yaml
            -manifest test_install_app/manifest.yaml
            -app_root ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(gxf_native_res_registry_run_graph_test PROPERTIES
        DEPENDS gxf_native_res_registry_install_graph_test
        PASS_REGULAR_EXPRESSION "Number of pings received: 10"
        FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed"
    )

endif()