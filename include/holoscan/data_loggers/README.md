# Data Loggers

These are the data loggers included as part of the Holoscan SDK:

- **basic_console_logger**: example of a simple data logger that prints basic information about each message sent
  - This console logger inherits from `DataLoggerResource` which performs the logging synchronously on the same thread
  that is being used for the `Operator::compute` call.
- **async_console_logger**: example of a simple data logger that prints basic information about each message sent
  - This console logger inherits from `AsyncDataLoggerResource` which offloads logging to separate threads. The emit/receive calls from `Operator::compute` push the items onto a queue, where the logger's own threads will process the logging.
