#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if [[ -z "${DISABLE_MOFED_VERSION_WARNING}" ]]; then

  _MOFED_MIN_VERSION=5.8
  _DETECTED_MOFED=$(cat /sys/module/mlx5_core/version 2>/dev/null || true)
  if [[ -n "${_DETECTED_MOFED}" ]] && \
     [[ "${_DETECTED_MOFED}" != "$(uname -r)" ]] && \
     dpkg --compare-versions "${_DETECTED_MOFED:-0}" lt ${_MOFED_MIN_VERSION}
  then
    # If the Mellanox driver is detected, appears to be from MOFED rather than
    # inbox, and is from MOFED < _MOFED_MIN_VERSION, then our container won't be compatible with
    # that old driver, and we issue a warning.

    echo
    echo "ERROR: Detected Mellanox network driver ${_DETECTED_MOFED} on the host system,"
    echo "       but version ${_MOFED_MIN_VERSION} or higher is required for multi-node"
    echo "       operation support with this container. Download the appropriate version at"
    echo "       https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed (iso or tgz)"
    echo "       and run the \`mlnxofedinstall\` installation script."
    sleep 2
  fi

  _DETECTED_NVPEERMEM=$(cat /sys/kernel/mm/memory_peers/nv_mem/version 2>/dev/null ||
                        cat /sys/kernel/mm/memory_peers/nvidia-peermem/version 2>/dev/null || true)

  if [[ -n "${_DETECTED_MOFED}" && -z "${_DETECTED_NVPEERMEM}" ]] \
    && command -v nvidia-smi &> /dev/null && ! nvidia-smi -q | grep -q "nvgpu"; then
    echo
    echo "WARNING: Mellanox network driver detected, but NVIDIA peer memory driver not"
    echo "         detected. Multi-node communication performance may be reduced. Run"
    echo "         the following commands on your host to enable GPUDirect RDMA:"
    echo
    echo "  nv_driver_version=\$(nvidia-smi -q | grep 'Driver Version' | awk '{print \$4}' | cut -d. -f1)"
    echo "  sudo dpkg-reconfigure nvidia-dkms-\$nv_driver_version && sudo modprobe nvidia-peermem"
    echo
    sleep 2
  fi

fi
