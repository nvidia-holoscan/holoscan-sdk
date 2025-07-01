# Fragment Service with Resource Example

This example demonstrates how to use the Fragment Service API with Holoscan Resources in both C++ and Python.

## Overview

The application shows several patterns for using resources as fragment services:

1. **Direct Resource Registration**: Register a Resource object directly as a fragment service
2. **FragmentService Wrapper**: Create a FragmentService that wraps a Resource
3. **Self-referencing Resource**: A Resource that implements FragmentService and references itself
4. **Resource with FragmentService Interface**: A Resource that implements both Resource and FragmentService interfaces

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/fragment_service/fragment_service_with_resource/cpp/fragment_service_with_resource
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/fragment_service/fragment_service_with_resource/cpp/fragment_service_with_resource
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/fragment_service/fragment_service_with_resource/cpp/fragment_service_with_resource
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/fragment_service_with_resource.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/fragment_service/fragment_service_with_resource/python/fragment_service_with_resource.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/fragment_service/fragment_service_with_resource/python/fragment_service_with_resource.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/fragment_service/fragment_service_with_resource/python/fragment_service_with_resource.py
  ```

## Key Concepts

### Resource as Service
Resources can be registered as fragment services. When retrieved via `service<FragmentService>()`,
the resource can be accessed through the `resource()` method.

**Important**: When registering a Resource as a service, do not provide an ID parameter.
The resource's name is automatically used as the service ID. For example:
```cpp
// C++
auto resource = make_resource<MyServiceResource>("my_service", Arg("int_value") = 20);
register_service(resource);  // No ID parameter - uses resource name "my_service"
```
```python
# Python
resource = MyServiceResource(self, name="my_service", int_value=20)
self.register_service(resource)  # No ID parameter - uses resource name "my_service"
```

### Direct Resource Retrieval
Resources registered as services can also be retrieved directly by their type,
e.g., `service<MyServiceResource>()` in C++ or `service(MyServiceResource)` in Python.

### Multiple Inheritance
Resources can implement both the Resource and FragmentService interfaces, allowing them to be
used as both resources and services with custom behavior.

## Example Output

```
MyOp::compute() executed
MyService value (via FragmentService): 20
MyService value (via Resource): 20
MyResourceManager value (via MyResourceManagerSelfRef): 30
MyResourceManager value (via MyResourceManagerEnabledShared): 40
```

## C++/Python API

### Key APIs Used

- **`Fragment::register_service`** (C++) / **`Fragment.register_service`** (Python): Registers a service with the fragment. This is typically called from the `compose()` method.
  - C++: `template <typename ServiceT> bool register_service(const std::shared_ptr<ServiceT>& svc, std::string_view id = "")`
  - Python: `register_service(service: Resource | FragmentService, id: str = "") -> None`
  - When a `Resource` is registered, its name is used as the service ID and the `id` parameter is ignored.

- **`ComponentBase::service`** (C++) / **`ComponentBase.service`** (Python): Retrieves a service from any component (Operator, Resource, Condition, etc.).
  - C++: `template <typename ServiceT = FragmentService> std::shared_ptr<ServiceT> service(std::string_view id = "") const`
  - Python: `service(service_type: type, id: str = "") -> object | None`
  - Use the resource's name as the `id` when retrieving a service.

The C++ and Python examples demonstrate four approaches for exposing a resource as a service:
1. **FragmentService Wrapper**: Creating a `DefaultFragmentService` instance that wraps a `Resource` and registering it (this approach is commented out in the examples)
2. **Direct Resource Registration**: Registering a `Resource` directly (this approach is used in the examples for the basic `MyServiceResource`)
3. **Self-referencing Resource**: A `Resource` that also implements `FragmentService` and holds a weak reference to itself.
4. **Resource with FragmentService Interface**: A `Resource` that also implements `FragmentService` and can be directly used as a service.

All approaches are shown to work identically in both Python and C++.

## Note

The Fragment Service feature is marked as **experimental** in Holoscan SDK v3.4. The API is subject to change in future releases.
