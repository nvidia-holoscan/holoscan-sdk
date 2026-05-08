# Fragment Service Examples

This folder contains examples of how to use the Fragment Service API in Holoscan SDK to register and retrieve services in your applications.

## Overview

Fragment services provide a way to share resources and functionality across operators within a fragment or application. These examples demonstrate different patterns for implementing and using fragment services.

## Examples

1. [Fragment Service Basic](./fragment_service_basic)
   - [fragment_service_basic.cpp](./fragment_service_basic/cpp/fragment_service_basic.cpp)
   - [fragment_service_basic.py](./fragment_service_basic/python/fragment_service_basic.py)

   A basic example showing how to create a custom fragment service, register it, and retrieve it from within an operator.

2. [Fragment Service with Resource](./fragment_service_with_resource)
   - [fragment_service_with_resource.cpp](./fragment_service_with_resource/cpp/fragment_service_with_resource.cpp)
   - [fragment_service_with_resource.py](./fragment_service_with_resource/python/fragment_service_with_resource.py)

   An advanced example demonstrating multiple patterns for using resources as fragment services, including direct resource registration and custom resource-service combinations.

## Key Concepts

- **Fragment Service**: A mechanism to share resources and functionality across operators within a fragment or application
- **Service Registration**: Services must be registered with the fragment before they can be accessed by operators
- **Service Retrieval**: Operators can retrieve registered services by their type and optional identifier
- **Resource as Service**: Resources can be directly registered as fragment services
- **Multiple Inheritance**: Classes can implement both Resource and FragmentService interfaces for advanced use cases

## Getting Started

Start with the [Fragment Service Basic](./fragment_service_basic) example to understand the fundamental concepts, then explore the [Fragment Service with Resource](./fragment_service_with_resource) example for more advanced patterns.

## Best Practices

### Implementing Fragment Services in C++ with Python Bindings

When implementing custom fragment services that will be used in both C++ and Python applications, it is strongly recommended to implement the service in C++ and provide Python bindings using pybind11.

**Why?** When a fragment service is implemented purely in Python (by subclassing `DefaultFragmentService` or `Resource`), the service type information is not preserved when the service is registered. This causes `service<MyService>()` lookups from C++ operators to fail because the C++ runtime cannot find the service by its expected type.

For example, if you implement a service in Python:

```python
# Python-only implementation (NOT recommended for cross-language use)
class MyPythonService(DefaultFragmentService):
    def __init__(self, value):
        super().__init__()
        self._value = value
```

A C++ operator will **not** be able to retrieve this service via:

```cpp
auto my_service = service<MyPythonService>("my_service");  // Returns nullptr!
```

**Solution**: Implement your service in C++ and bind it to Python:

```cpp
// C++ implementation
class MyService : public holoscan::DefaultFragmentService {
 public:
  explicit MyService(int value) : value_(value) {}
  int value() const { return value_; }
 private:
  int value_;
};

// pybind11 binding
py::class_<MyService, DefaultFragmentService, std::shared_ptr<MyService>>(m, "MyService")
    .def(py::init<int>(), py::arg("value"))
    .def("value", &MyService::value);
```

**Important**: If your service class uses **multiple inheritance** (e.g., inherits from both `Resource` and `DistributedAppService`), you must add `py::multiple_inheritance()` to the binding:

```cpp
// For classes with multiple inheritance
py::class_<MyMultiService, Resource, DistributedAppService, std::shared_ptr<MyMultiService>>(
    m, "MyMultiService", py::multiple_inheritance())
    .def(py::init<int>(), py::arg("value"))
    .def("value", &MyMultiService::value);
```

Without `py::multiple_inheritance()`, pybind11 cannot properly handle runtime casts to non-primary base classes. This can cause silent failures where, for example, a service intended to be registered as a `FragmentService` gets registered only as a `Resource`, breaking distributed application behavior.

With this approach, the service can be:

- Instantiated and registered in Python
- Retrieved by type from both Python and C++ operators

See [PoseTreeManager](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/include/holoscan/pose_tree/pose_tree_manager.hpp) (C++ class) and its [Python binding](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/python/holoscan/pose_tree/pose_tree.cpp) for an example of a production fragment service with multiple inheritance implemented in C++ with Python bindings.

### When Pure Python Services Are Acceptable

Pure Python fragment services are acceptable when:

1. Your application is Python-only (no C++ operators need to access the service)
2. The service is only accessed from Python operators using `self.service(MyService, "id")`

## Note

The Fragment Service feature is marked as **experimental** in Holoscan SDK v3.4. The API is subject to change in future releases.
