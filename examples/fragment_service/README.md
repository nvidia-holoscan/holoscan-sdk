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

## Note

The Fragment Service feature is marked as **experimental** in Holoscan SDK v3.4. The API is subject to change in future releases.
