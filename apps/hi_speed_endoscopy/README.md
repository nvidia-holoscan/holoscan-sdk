# Hi-Speed Endoscopy

The application showcases how high resolution cameras can be used to capture the scene, post-processed on GPU and displayed at high frame rate.

### Requirements

This application requires:
1. an Emergent Vision Technologies camera (see [setup instructions]((https://docs.nvidia.com/clara-holoscan/sdk-user-guide/emergent_setup.html)
2. a NVIDIA ConnectX SmartNIC with Rivermax SDK and drivers installed (see [prerequisites](../../README.md#prerequisites))
3. a display with high refresh rate to keep up with the camera's framerate
4. [additional setups](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/additional_setup.html) to reduce latency

### Build Instructions

The application is not currently supported in our containerized development environment (using the run script or Docker + CMake manually). Follow the advanced build instructions [here](../../README.md#advanced-local-environment-cmake) to build locally from source, and pass `-DHOLOSCAN_BUILD_HI_SPEED_ENDO_APP=ON` when configuring the project with CMake.

> ⚠️ At this time, camera controls are hardcoded within the `emergent-source` extension. To update them at the application level, the GXF extension, and the application need to be rebuilt.
For more information on the controls, refer to the [EVT Camera Attributes Manual](https://emergentvisiontec.com/resources/?tab=umg)

### Run Instructions

First, go in your `build` or `install` directory. Then, run the commands of your choice:

* RDMA disabled
    ```bash
    # C++
    sed -i -e 's#rdma:.*#rdma: false#' ./apps/hi_speed_endoscopy/cpp/app_config.yaml \
        && sudo ./apps/hi_speed_endoscopy/cpp/hi_speed_endoscopy

    # Python
    sudo -s
    export PYTHONPATH=$(pwd)/python/lib/
    sed -i -e 's#rdma:.*#rdma: false#' ./apps/hi_speed_endoscopy/python/hi_speed_endoscopy.yaml \
        && python3 ./apps/hi_speed_endoscopy/python/hi_speed_endoscopy.py
    exit
    ```

* RDMA enabled
    ```bash
    # C++
    sed -i -e 's#rdma:.*#rdma: true#' ./apps/hi_speed_endoscopy/cpp/app_config.yaml \
        && sudo MELLANOX_RINGBUFF_FACTOR=14 ./apps/hi_speed_endoscopy/cpp/hi_speed_endoscopy

    # Python
    sudo -s
    export PYTHONPATH=$(pwd)/python/lib/
    sed -i -e 's#rdma:.*#rdma: true#' ./apps/hi_speed_endoscopy/python/hi_speed_endoscopy.yaml \
        && MELLANOX_RINGBUFF_FACTOR=14 python3 ./apps/hi_speed_endoscopy/python/hi_speed_endoscopy.py
    exit
    ```

> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment


> ℹ️ The `MELLANOX_RINGBUFF_FACTOR` is used by the EVT driver to decide how much BAR1 size memory would be used on the dGPU. It can be changed to different number based on different use cases.
