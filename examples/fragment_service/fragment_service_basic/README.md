# Fragment Service Example

This example demonstrates how to use the Fragment Service API in Holoscan SDK to register and retrieve services in your application.

## Overview

The application shows how to:
1. Create a fragment service by inheriting from `DefaultFragmentService`
2. Register the service in the fragment using `register_service()`
3. Retrieve the service from within an operator using `service<ServiceType>()` (C++) or `self.service(ServiceType)` (Python)

## Key Concepts

- **Fragment Service**: A mechanism to share resources and functionality across operators within a fragment or application
- **Service Registration**: Services must be registered with the fragment before they can be accessed by operators
- **Service Retrieval**: Operators can retrieve registered services by their type

## Implementation Details

### C++ Version
- `MyService`: A custom service class inheriting from `holoscan::DefaultFragmentService` that stores an integer value
- `MyOp`: An operator that retrieves the service and prints its value
- `FragmentServiceApp`: The application that registers the service and adds the operator

### Python Version
- `MyService`: A custom service class inheriting from `holoscan.core.DefaultFragmentService` that stores an integer value
- `MyOp`: An operator that retrieves the service and prints its value
- `FragmentServiceApp`: The application that registers the service and adds the operator

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/fragment_service/fragment_service_basic/cpp/fragment_service_basic
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/fragment_service/fragment_service_basic/cpp/fragment_service_basic
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/fragment_service/fragment_service_basic/cpp/fragment_service_basic
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/fragment_service_basic.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/fragment_service/fragment_service_basic/python/fragment_service_basic.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/fragment_service/fragment_service_basic/python/fragment_service_basic.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/fragment_service/fragment_service_basic/python/fragment_service_basic.py
  ```

## Expected Output

Both versions should output:
```
MyOp::compute() executed
MyService value: 10
```

## Note

The Fragment Service feature is marked as **experimental** in Holoscan SDK v3.4. The API is subject to change in future releases.
