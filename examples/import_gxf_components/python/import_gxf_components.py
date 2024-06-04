"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import os

from holoscan.core import Application, ComponentSpec, Operator, OperatorSpec
from holoscan.operators import GXFCodeletOp, HolovizOp, VideoStreamReplayerOp
from holoscan.resources import GXFComponentResource

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    ) from None

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class DeviceMemoryPool(GXFComponentResource):
    """Wrap an existing GXF component for use from Holoscan.

    This is illustrated here using `BlockMemoryPool` as a concrete example, but in practice
    applications would just import the existing ``BlockMemoryPool`` class from
    ``holoscan.resources``.
    ``GXFComponentResource`` would be used to wrap some GXF component not already available via
    the ``holoscan.resources`` module.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Call the base class constructor with the gxf_typename as the second argument
        super().__init__(fragment, "nvidia::gxf::BlockMemoryPool", *args, **kwargs)

    def setup(self, spec: ComponentSpec):
        # Ensure the parent class setup() is called before any additional setup code.
        super().setup(spec)

        # You can add any additional setup code here (if needed).

    def initialize(self):
        # Unlike the C++ API, this initialize() method should not call the parent class's
        # initialize() method.
        # It is a callback method invoked from the underlying C++ layer.

        # You can call any additional initialization code here (if needed).
        pass


class ToDeviceMemoryOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        # Call the base class constructor with the gxf_typename as the second argument
        super().__init__(fragment, "nvidia::gxf::TensorCopier", *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Ensure the parent class setup() is called before any additional setup code.
        super().setup(spec)

        # You can add any additional setup code here (if needed).
        # You can update conditions of the input/output ports, update the connector types, etc.
        #
        # Example:
        # - `spec.inputs["receiver"].condition(ConditionType.NONE)`
        #   to update the condition of the input port to 'NONE'.
        #   (assuming that the GXF Codelet has a Receiver component named 'receiver'.)

    def initialize(self):
        # Unlike the C++ API, this initialize() method should not call the parent class's
        # initialize() method.
        # It is a callback method invoked from the underlying C++ layer.

        # You can call any additional initialization code here (if needed).
        #
        # Example:
        # ```python
        # from holoscan.core import py_object_to_arg
        # ...
        # # add an argument to the GXF Operator
        # self.add_arg(py_object_to_arg(value, name="arg_name"))
        # ```
        pass


# Define custom Operators for use in the demo
class ImageProcessingOp(Operator):
    """Example of an operator processing input video (as a tensor).

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"

    The data from each input is processed by a CuPy gaussian filter and
    the result is sent to the output.

    In this demo, the input and output image (2D RGB) is a 3D array of shape
    (height, width, channels).
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")
        spec.param("sigma")

    def compute(self, op_input, op_output, context):
        # in_message is a dict of tensors
        in_message = op_input.receive("input_tensor")

        # smooth along first two axes, but not the color channels
        sigma = (self.sigma, self.sigma, 0)

        # out_message will be a dict of tensors
        out_message = dict()

        for key, value in in_message.items():
            print(f"message received (count: {self.count})")
            self.count += 1

            cp_array = cp.asarray(value)

            # process cp_array
            cp_array = ndi.gaussian_filter(cp_array, sigma)

            out_message[key] = cp_array

        op_output.emit(out_message, "output_tensor")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """An application that demonstrates video processing using custom and built-in operators.

    This application integrates multiple operators and resources to create a video processing
    pipeline:

    - GXFComponentResource: Creates resources from GXF Components
      (e.g., nvidia::gxf::BlockMemoryPool).
    - DeviceMemoryPool: Derived from GXFComponentResource, manages device memory.
    - GXFCodeletOp: Creates operators from GXF Codelets (e.g., nvidia::gxf::TensorCopier).
    - ToDeviceMemoryOp: Derived from GXFCodeletOp, handles tensor copying to device memory.
    - VideoStreamReplayerOp: Reads video files and outputs video frames.
    - ImageProcessingOp: Applies a Gaussian filter to video frames.
    - HolovizOp: Visualizes the processed video frames.

    Workflow:
    1. VideoStreamReplayerOp reads a video file and outputs the frames.
    2. TensorCopier (to_system_memory) copies the frames to system memory.
    3. TensorCopier (to_device_memory) copies the frames to device memory.
    4. ImageProcessingOp applies a Gaussian filter to the frames.
    5. HolovizOp displays the processed frames.
    """

    def compose(self):
        width = 960
        height = 540
        video_dir = os.path.join(sample_data_path, "racerx")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
        )

        # For both GXFComponentResource and GXFCodeletOp, the gxf_typename is positional.
        system_memory_pool = GXFComponentResource(
            self,
            "nvidia::gxf::BlockMemoryPool",
            name="system_memory_pool",
            storage_type=2,
            block_size=width * height * 3,
            num_blocks=2,
        )
        device_memory_pool = DeviceMemoryPool(
            self,
            name="device_memory_pool",
            storage_type=1,
            block_size=width * height * 3,
            num_blocks=2,
        )

        to_system_memory = GXFCodeletOp(
            self,
            gxf_typename="nvidia::gxf::TensorCopier",
            name="to_system_memory",
            allocator=system_memory_pool,
            mode=2,  # 2 is for copying tensor to system memory
        )

        to_device_memory = ToDeviceMemoryOp(
            self,
            allocator=device_memory_pool,
            mode=0,  # 0 is for copying tensor to device memory
            name="to_device_memory",
        )

        image_processing = ImageProcessingOp(
            self, name="image_processing", **self.kwargs("image_processing")
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            **self.kwargs("holoviz"),
        )

        self.add_flow(source, to_system_memory)
        self.add_flow(to_system_memory, to_device_memory)
        self.add_flow(to_device_memory, image_processing)
        self.add_flow(image_processing, visualizer, {("", "receivers")})


def main(config_file):
    app = MyVideoProcessingApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "import_gxf_components.yaml")

    main(config_file=config_file)
