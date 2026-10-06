#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ "$(df -k /dev/shm | grep '/dev/shm' | awk '{print $2}')" == "65536" ]]; then
  echo
  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
  echo "   insufficient for ${NVIDIA_PRODUCT_NAME:-}.  NVIDIA recommends the use of the following flags:"
  echo "   docker run --runtime=nvidia --gpus all --cap-add CAP_SYS_PTRACE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ..."
fi
