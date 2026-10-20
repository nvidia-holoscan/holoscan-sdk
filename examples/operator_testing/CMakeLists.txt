# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)

if(TARGET holoscan_testing_python)
    install(FILES python/test_operator_testing.py python/test_tensor_and_schema.py
        python/test_advanced_testing.py
        DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/python"
        COMPONENT holoscan-testing-examples)
endif()
