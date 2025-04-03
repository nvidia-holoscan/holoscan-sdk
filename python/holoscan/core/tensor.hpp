/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_CORE_TENSOR_HPP
#define PYHOLOSCAN_CORE_TENSOR_HPP

#include <pybind11/pybind11.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "holoscan/core/domain/tensor.hpp"

namespace py = pybind11;

namespace holoscan {

void init_tensor(py::module_&);

static const std::unordered_map<DLDataTypeCode, const char*> dldatatypecode_namemap{
    {kDLInt, "DLINT"},
    {kDLUInt, "DLUINT"},
    {kDLFloat, "DLFLOAT"},
    {kDLOpaqueHandle, "DLOPAQUEHANDLE"},
    {kDLBfloat, "DLBFLOAT"},
    {kDLComplex, "DLCOMPLEX"},
};

/**
 * @brief Class to wrap the deleter of a DLManagedTensor in Python.
 *
 * This class is used with DLManagedTensorContext class to wrap the DLManagedTensor.
 *
 * A shared pointer to this class in DLManagedTensorContext class is used as the deleter of the
 * DLManagedTensorContext::memory_ref
 *
 * When the last reference to the DLManagedTensorContext object is released,
 * DLManagedTensorContext::memory_ref will also be destroyed, which will call the deleter function
 * of the DLManagedTensor object.
 *
 * Compared to the C++ version (DLManagedMemoryBuffer), this class is used to acquire the GIL
 * before calling the deleter function.
 *
 */
class PyDLManagedMemoryBuffer {
 public:
  explicit PyDLManagedMemoryBuffer(DLManagedTensor* self);
  ~PyDLManagedMemoryBuffer();

 private:
  DLManagedTensor* self_ = nullptr;
};

/**
 * @brief Class to wrap the deleter of a DLManagedTensorVersioned in Python.
 *
 * This class is used with DLManagedTensorContext class to wrap the DLManagedTensorVersioned.
 *
 * A shared pointer to this class in DLManagedTensorContext class is used as the deleter of the
 * DLManagedTensorContext::memory_ref
 *
 * When the last reference to the DLManagedTensorContext object is released,
 * DLManagedTensorContext::memory_ref will also be destroyed, which will call the deleter function
 * of the DLManagedTensorVersioned object.
 *
 * Compared to the C++ version (DLManagedMemoryBufferVersioned), this class is used to acquire the
 * GIL before calling the deleter function.
 */
class PyDLManagedMemoryBufferVersioned {
 public:
  explicit PyDLManagedMemoryBufferVersioned(DLManagedTensorVersioned* self);
  ~PyDLManagedMemoryBufferVersioned();

 private:
  DLManagedTensorVersioned* self_ = nullptr;
};

/**
 * @brief A class facilitating lazy, asynchronous deletion of DLManagedTensor objects.
 *
 * This class allows DLManagedTensor objects to be enqueued for deferred deletion, which is carried
 * out in a distinct thread to evade the obstruction of the main execution thread.
 *
 * Instances of LazyDLManagedTensorDeleter are reference-counted. The thread responsible for
 * deletion is initiated upon the creation of the first instance and is ceased upon the destruction
 * of the last existing instance. The add() method can be employed to insert DLManagedTensor objects
 * into the deletion queue. The class destructor ensures the completion of all pending deletions
 * before finalizing.
 */
class LazyDLManagedTensorDeleter {
 public:
  /**
   * @brief Default constructor that initializes the LazyDLManagedTensorDeleter instance.
   *
   * Increment the reference count and start the deletion thread if it hasn't already started.
   * Register the pthread_atfork() and atexit() handlers if they aren't already registered.
   */
  LazyDLManagedTensorDeleter();

  /**
   * @brief Destructor that decrements the reference count and stops the deletion thread if the
   * count reaches zero.
   */
  ~LazyDLManagedTensorDeleter();

  /**
   * @brief Adds a DLManagedTensor pointer to the queue for deletion.
   * @param dl_managed_tensor_ptr The pointer to the DLManagedTensor to be deleted.
   */
  static void add(DLManagedTensor* dl_managed_tensor_ptr);

  /**
   * @brief Adds a DLManagedTensorVersioned pointer to the queue for deletion.
   * @param dl_managed_tensor_ver_ptr The pointer to the DLManagedTensorVersioned to be deleted.
   */
  static void add(DLManagedTensorVersioned* dl_managed_tensor_ver_ptr);

 private:
  /**
   * @brief The main function for the deletion thread, which waits for tensors to be available in
   * the queue and deletes them.
   */
  static void run();

  /**
   * @brief Decrements the reference count and stops the deletion thread if the count reaches zero.
   */
  static void release();

  /// Callback function for the atexit() function.
  static void on_exit();
  /// Callback function for the pthread_atfork() function's prepare handler.
  static void on_fork_prepare();
  /// Callback function for the pthread_atfork() function's parent handler.
  static void on_fork_parent();
  /// Callback function for the pthread_atfork() function's child handler.
  static void on_fork_child();

  /// Type alias for the variant that can hold either DLManagedTensor* or DLManagedTensorVersioned*
  using TensorPtr = std::variant<DLManagedTensor*, DLManagedTensorVersioned*>;

  /// The queue of tensors to be deleted.
  static inline std::queue<TensorPtr> s_dlmanaged_tensors_queue;
  ///< Mutex to protect the shared resources (queue, condition variable, etc.)
  static inline std::mutex s_mutex;
  ///< Condition variable to synchronize the deletion thread.
  static inline std::condition_variable s_cv;
  ///< A flag indicating whether the atfork handlers have been registered.
  static inline bool s_pthread_atfork_registered = false;
  ///< A flag indicating whether s_cv should not wait for the deletion thread so that fork() can
  ///< work.
  static inline bool s_cv_do_not_wait_thread = false;
  ///< The deletion thread.
  static inline std::thread s_thread;
  ///< The reference count of LazyDLManagedTensorDeleter instances.
  static inline std::atomic<int64_t> s_instance_count{0};
  ///< A flag indicating whether the deletion thread should stop.
  static inline bool s_stop = false;
  ///< A flag indicating whether the deletion thread is running.
  static inline bool s_is_running = false;
};

class PyTensor : public Tensor {
 public:
  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorContext.
   *
   * @param ctx A shared pointer to the DLManagedTensorContext to be used in Tensor construction.
   */
  explicit PyTensor(std::shared_ptr<DLManagedTensorContext>& ctx);

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensor pointer.
   *
   * @param ctx A pointer to the DLManagedTensor to be used in Tensor construction.
   */
  explicit PyTensor(DLManagedTensor* dl_managed_tensor_ptr);

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorVersioned pointer.
   *
   * @param dl_managed_tensor_ver_ptr A pointer to the DLManagedTensorVersioned to be used in Tensor
   * construction.
   */
  explicit PyTensor(DLManagedTensorVersioned* dl_managed_tensor_ver_ptr);

  PyTensor() = default;

  /**
   * @brief Create a new Tensor object from a py::object
   *
   * The given py::object must support the array interface protocol or dlpack protocol.
   *
   * @param obj A py::object that can be converted to a Tensor
   * @return A new Tensor object
   */
  static py::object as_tensor(const py::object& obj);
  static std::shared_ptr<PyTensor> from_array_interface(const py::object& obj, bool cuda = false);
  static std::shared_ptr<PyTensor> from_cuda_array_interface(const py::object& obj) {
    return from_array_interface(obj, true);
  }
  static std::shared_ptr<PyTensor> from_dlpack(const py::object& obj,
                                               py::object device = py::none(),
                                               py::object copy = py::none());
  static py::object from_dlpack_pyobj(const py::object& obj, py::object device = py::none(),
                                      py::object copy = py::none());
  static py::capsule dlpack(const py::object& obj, py::object stream = py::none(),
                            std::optional<std::tuple<int, int>> max_version = std::nullopt,
                            std::optional<std::tuple<DLDeviceType, int>> dl_device = std::nullopt,
                            std::optional<bool> copy = std::nullopt);
  static py::tuple dlpack_device(const py::object& obj);
};

bool is_tensor_like(const py::object& value);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_TENSOR_HPP */
