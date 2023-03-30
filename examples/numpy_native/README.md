# NumPy Native

This minimal signal processing application generates a time-varying impulse, convolves it with a boxcar kernel, and prints the result to the terminal, to showcase the use of NumPy with holoscan.

## Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install numpy
  python3 <APP_DIR>/convolve.py
  ```
* **using deb package install**:
  ```bash
  python3 -m pip install numpy
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/numpy_native/convolve.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/numpy_native/convolve.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/numpy_native/convolve.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install numpy
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/numpy_native/convolve.py
  ```
