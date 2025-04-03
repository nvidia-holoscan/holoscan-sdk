# Execution Control Examples

This folder contains examples of how to control the execution of an application using the execution-related APIs.

1. [operator_status_tracking](./operator_status_tracking)
  - This example demonstrates how to track the status of an operator using `ExecutionContext::get_operator_status()` and `ExecutionContext::find_operator()`, and stop the operator using `Operator::stop_execution()`.
2. [async_operator_execution_control](./async_operator_execution_control)
  - This example demonstrates how to control operator execution from outside the Holoscan runtime using `Operator::async_condition()`. It shows how to execute operators in a specific sequence, coordinate between operators using notification callbacks, and gracefully shut down an application with asynchronous operators. The example includes practical use cases, comparisons with alternative approaches, and implements thread-safe notification mechanisms between operators.
