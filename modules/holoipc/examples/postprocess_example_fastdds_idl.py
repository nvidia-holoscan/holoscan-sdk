#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Post-process raw fastddsgen output for holoipc Fast DDS *examples* IDL that reference
# pointer_descriptor.idl from the SDK. Rewrites includes and C++ types to
# holoscan::ipc::transport::fastdds::gen (wire type strings included).
#
#   python3 postprocess_example_fastdds_idl.py < raw.hpp > out.hpp

from __future__ import annotations

import sys

_SDK_PTR = "holoscan::ipc::transport::fastdds::gen::PointerDescriptor"
_SDK_PTR_STR = '"holoscan::ipc::transport::fastdds::gen::PointerDescriptor"'
_SDK_REG = "::holoscan::ipc::transport::fastdds::gen::register_PointerDescriptor_type_identifier"
_SDK_INC = "<holoscan/ipc/transport/fastdds/gen/pointer_descriptor.hpp>"
_SDK_INC_PUB = "<holoscan/ipc/transport/fastdds/gen/pointer_descriptorPubSubTypes.hpp>"
_SDK_INC_TOS = "<holoscan/ipc/transport/fastdds/gen/pointer_descriptorTypeObjectSupport.hpp>"


def postprocess(text: str) -> str:
    text = text.replace(
        '#include "PointerDescriptorTypeObjectSupport.hpp"',
        f"#include {_SDK_INC_TOS}",
    )
    text = text.replace(
        '#include "pointer_descriptorTypeObjectSupport.hpp"',
        f"#include {_SDK_INC_TOS}",
    )
    text = text.replace(
        '#include "PointerDescriptorPubSubTypes.hpp"',
        f"#include {_SDK_INC_PUB}",
    )
    text = text.replace(
        '#include "pointer_descriptorPubSubTypes.hpp"',
        f"#include {_SDK_INC_PUB}",
    )
    text = text.replace('#include "PointerDescriptor.hpp"', f"#include {_SDK_INC}")
    text = text.replace('#include "pointer_descriptor.hpp"', f"#include {_SDK_INC}")
    # TypeObjectSupport.cxx wire name + register helper
    text = text.replace('"holoscan::ipc::PointerDescriptor"', _SDK_PTR_STR)
    text = text.replace(
        "::holoscan::ipc::register_PointerDescriptor_type_identifier",
        _SDK_REG,
    )
    # Remaining C++ uses of the IPC type (struct fields, parameters, casts)
    text = text.replace("holoscan::ipc::PointerDescriptor", _SDK_PTR)
    return text


def main() -> None:
    data = sys.stdin.read()
    sys.stdout.write(postprocess(data))


if __name__ == "__main__":
    main()
