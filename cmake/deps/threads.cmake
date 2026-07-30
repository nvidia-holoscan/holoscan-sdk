# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Threads (https://cmake.org/cmake/help/v3.18/module/FindThreads.html)
# The use of the -pthread compiler and linker flag is preferred
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
rapids_find_package(
    Threads REQUIRED
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    INSTALL_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
)
