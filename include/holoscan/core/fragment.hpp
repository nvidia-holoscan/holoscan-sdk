/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_FRAGMENT_HPP
#define HOLOSCAN_CORE_FRAGMENT_HPP

#include <fmt/format.h>

#include <future>        // for std::future
#include <iostream>      // for std::cout
#include <memory>        // for std::shared_ptr
#include <set>           // for std::set
#include <shared_mutex>  // for std::shared_mutex
#include <string>        // for std::string
#include <string_view>   // for std::string_view
#include <tuple>
#include <type_traits>  // for std::enable_if_t, std::is_constructible
#include <typeinfo>     // for std::type_info
#include <unordered_map>
#include <unordered_set>
#include <utility>  // for std::pair
#include <vector>

#include "common.hpp"
#include "config.hpp"
#include "data_logger.hpp"
#include "dataflow_tracker.hpp"
#include "executor.hpp"
#include "fragment_service_provider.hpp"
#include "graph.hpp"
#include "io_spec.hpp"
#include "network_context.hpp"
#include "resources/data_logger.hpp"
#include "scheduler.hpp"
#include "subgraph.hpp"

namespace holoscan {

namespace gxf {
// Forward declarations
class GXFExecutor;
}  // namespace gxf

class ThreadPool;

// Forward declare ComponentBase for the friend declaration or internal setter
class ComponentBase;

/// The name of the start operator. Used to identify the start operator in the graph.
constexpr static const char* kStartOperatorName = "<|start|>";

// NOLINTBEGIN(whitespace/indent_namespace)

// key = operator name,  value = (input port names, output port names, multi-receiver names)
using FragmentPortMap =
    std::unordered_map<std::string,
                       std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>,
                                  std::unordered_set<std::string>>>;
// NOLINTEND(whitespace/indent_namespace)

// Data structure containing port information for multiple fragments. Fragments are composed by
// the workers and port information is sent back to the driver for addition to this map.
// The keys are the fragment names.
using MultipleFragmentsPortMap = std::unordered_map<std::string, FragmentPortMap>;

constexpr MetadataPolicy kDefaultMetadataPolicy = MetadataPolicy::kRaise;
constexpr bool kDefaultMetadataEnabled = true;

/**
 * @brief The fragment of the application.
 *
 * A fragment is a building block of the Application. It is a directed graph of
 * operators. A fragment can be assigned to a physical node of a Holoscan cluster during execution.
 * The run-time execution manages communication across fragments. In a Fragment, Operators (Graph
 * Nodes) are connected to each other by flows (Graph Edges).
 */
class Fragment : public FragmentServiceProvider {
 public:
  /**
   * @brief Accessor class for GPU-resident specific functions of a Fragment.
   *
   * This class provides a convenient interface for accessing GPU-resident specific
   * functionality of a Fragment. It acts as a mediator to expose GPU-resident operations
   * through a cleaner API pattern: `fragment->gpu_resident().function()`.
   *
   * This is a lightweight accessor class that maintains a reference to the parent Fragment.
   */
  class GPUResidentAccessor {
   public:
    // Delete default constructor
    GPUResidentAccessor() = delete;
    /**
     * @brief Construct a new GPUResidentAccessor object
     *
     * @param fragment Pointer to the parent Fragment
     */
    explicit GPUResidentAccessor(Fragment* fragment) : fragment_(fragment) {}

    /**
     * @brief Set the timeout for GPU-resident execution.
     *
     * GPU-resident execution occurs asynchronously. This sets the timeout so that
     * execution is stopped after it exceeds the specified duration.
     *
     * @param timeout_ms The timeout in milliseconds.
     */
    void timeout_ms(unsigned long long timeout_ms);

    /**
     * @brief Send a tear down signal to the GPU-resident CUDA graph.
     *
     * The timeout has to be set to zero for this to work for now.
     */
    void tear_down();

    /**
     * @brief Check if the result of a single iteration of the GPU-resident CUDA graph is ready.
     *
     * @return true if the result is ready, false otherwise.
     */
    bool result_ready();

    /**
     * @brief Inform the GPU-resident CUDA graph that the data is ready for the main workload.
     */
    void data_ready();

    /**
     * @brief Check if the GPU-resident CUDA graph has been launched.
     *
     * @return true if the CUDA graph has been launched, false otherwise.
     */
    bool is_launched();

    /**
     * @brief Get the CUDA graph of the main workload in this fragment. This
     * returns a clone of the main workload graph, and certain CUDA graph nodes
     * (e.g. memory allocation, memory free, conditional nodes) are not
     * supported for cloning.
     *
     * @return A clone of the CUDA graph of the main workload in this fragment.
     */
    cudaGraph_t workload_graph();

    /**
     * @brief Get the CUDA device pointer for the data_ready signal.
     *
     * This returns the actual device memory address that the GPU-resident CUDA graph
     * uses to check if data is ready for processing. Can be used for advanced
     * GPU-resident applications that need direct access to these control signals.
     *
     * @return Pointer to the device memory location for data_ready signal.
     */
    void* data_ready_device_address();

    /**
     * @brief Get the CUDA device pointer for the result_ready signal.
     *
     * Similar to data_ready_device_address(), but for the result_ready signal.
     *
     * @return Pointer to the device memory location for result_ready signal.
     */
    void* result_ready_device_address();

    /**
     * @brief Get the CUDA device pointer for the tear_down signal.
     *
     * Similar to data_ready_device_address(), but for the tear_down signal.
     *
     * @return Pointer to the device memory location for tear_down signal.
     */
    void* tear_down_device_address();

    /**
     * @brief Register a data ready handler to this fragment. The data ready handler
     * will be executed at the beginning of every iteration of the GPU-resident CUDA
     * Graph. The data ready handler will usually indicate whether input data is ready
     * for processing. If the data ready handler marks data to be ready, then main
     * workload CUDA graph will be executed on this iteration, otherwise main workload
     * processing will be skipped for this iteration.
     *
     * @param data_ready_handler_fragment Shared pointer to a fragment that
     * will be added as the data ready handler to this fragmemt.
     */
    void register_data_ready_handler(std::shared_ptr<Fragment> data_ready_handler_fragment);

    /**
     * @brief Get the registered data ready handler fragment.
     *
     * @return The data ready handler fragment, or nullptr if none is registered.
     */
    std::shared_ptr<Fragment> data_ready_handler_fragment();

   private:
    Fragment* fragment_;  ///< Pointer to the parent Fragment
  };

  Fragment() = default;
  ~Fragment() override;

  // Delete copy and move operations due to std::shared_mutex member
  Fragment(const Fragment&) = delete;
  Fragment& operator=(const Fragment&) = delete;
  Fragment(Fragment&&) = delete;
  Fragment& operator=(Fragment&&) = delete;

  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this fragment (for chaining).
   */
  Fragment& name(const std::string& name) &;

  /**
   * @brief Set the name of the operator.
   *
   * @param name The name of the operator.
   * @return The reference to this fragment (for chaining).
   */
  Fragment&& name(const std::string& name) &&;

  /**
   * @brief Get the name of the fragment.
   *
   * @return The name of the fragment.
   */
  const std::string& name() const;

  /**
   * @brief Set the application of the fragment.
   *
   * @param app The pointer to the application of the fragment.
   * @return The reference to this fragment (for chaining).
   */
  Fragment& application(Application* app);

  /**
   * @brief Get the application of the fragment.
   *
   * @return The pointer to the application of the fragment.
   */
  Application* application() const;

  /**
   * @brief Set the configuration of the fragment.
   *
   * The configuration file is a YAML file that has the information of GXF extension paths and some
   * parameter values for operators.
   *
   * The `extensions` field in the YAML configuration file is a list of GXF extension paths.
   * The paths can be absolute or relative to the current working directory, considering paths in
   * `LD_LIBRARY_PATH` environment variable.
   *
   * The paths can consist of the following parts:
   *
   * - GXF core extensions
   *   - built-in extensions such as `libgxf_std.so` and `libgxf_cuda.so`.
   *   - `libgxf_std.so`, `libgxf_cuda.so`, `libgxf_multimedia.so`, `libgxf_serialization.so` are
   *     always loaded by default.
   *   - GXF core extensions are copied to the `lib` directory of the build/installation directory.
   * - Other GXF extensions
   *   - GXF extensions that are required for operators that this fragment uses.
   *   - some core GXF extensions such as `libgxf_stream_playback.so` are always loaded by default.
   *   - these paths are usually relative to the build/installation directory.
   *
   * The extension paths are used to load dependent GXF extensions at runtime when
   * `::run()` method is called.
   *
   * For other fields in the YAML file, you can freely define the parameter values for
   * operators/fragments.
   *
   * For example:
   *
   * ```yaml
   * extensions:
   *   - libmy_recorder.so
   *
   * replayer:
   *   directory: "../data/racerx"
   *   basename: "racerx"
   *   frame_rate: 0   # as specified in timestamps
   *   repeat: false   # default: false
   *   realtime: true  # default: true
   *   count: 0        # default: 0 (no frame count restriction)
   *
   * recorder:
   *   out_directory: "/tmp"
   *   basename: "tensor_out"
   * ```
   *
   * You can get the value of this configuration file by calling `::from_config()` method.
   *
   * If the application is executed with `--config` option or HOLOSCAN_CONFIG_PATH environment,
   * the configuration file is overridden by the configuration file specified by the option or
   * environment variable.
   *
   * @param config_file The path to the configuration file.
   * @param prefix The prefix string that is prepended to the key of the configuration. (not
   * implemented yet)
   * @throws RuntimeError if the config_file is non-empty and the file doesn't exist.
   */
  void config(const std::string& config_file, [[maybe_unused]] const std::string& prefix = "");

  /**
   * @brief Set the configuration of the fragment.
   *
   * If you want to set the configuration of the fragment manually, you can use this method.
   * However, it is recommended to use `::config(const std::string&, const std::string&)` method
   * because once you set the configuration manually, you cannot get the configuration from the
   * override file (through `--config` option or HOLOSCAN_CONFIG_PATH environment variable).
   *
   * @param config The shared pointer to the configuration of the fragment (`Config` object).
   */
  void config(std::shared_ptr<Config>& config);

  /**
   * @brief Get the configuration of the fragment.
   *
   * @return The reference to the configuration of the fragment (`Config` object.)
   */
  Config& config();

  /**
   * @brief Get the shared pointer to the configuration of the fragment.
   *
   * @return The shared pointer to the configuration of the fragment.
   */
  std::shared_ptr<Config> config_shared();

  /**
   * @brief Get the graph of the fragment.
   *
   * @return The reference to the graph of the fragment (`Graph` object.)
   */
  OperatorGraph& graph();

  /**
   * @brief Get the shared pointer to the graph of the fragment.
   *
   * @return The shared pointer to the graph of the fragment.
   */
  std::shared_ptr<OperatorGraph> graph_shared();

  /**
   * @brief Set the executor of the fragment.
   *
   * @param executor The executor to be added.
   */
  void executor(const std::shared_ptr<Executor>& executor);

  /**
   * @brief Get the executor of the fragment.
   *
   * @return The reference to the executor of the fragment (`Executor` object.)
   */
  Executor& executor();

  /**
   * @brief Get the shared pointer to the executor of the fragment.
   *
   * @return The shared pointer to the executor of the fragment.
   */
  std::shared_ptr<Executor> executor_shared();

  /**
   * @brief Get the scheduler used by the executor
   *
   * @return The reference to the scheduler of the fragment's executor (`Scheduler` object.)
   */
  std::shared_ptr<Scheduler> scheduler();

  /**
   * @brief Get the scheduler used by the executor
   *
   * @return The reference to the scheduler of the fragment's executor (`Scheduler` object.)
   */
  std::shared_ptr<Scheduler> scheduler() const;

  // /**
  //  * @brief Set the scheduler used by the executor
  //  *
  //  * @param scheduler The scheduler to be added.
  //  */
  void scheduler(const std::shared_ptr<Scheduler>& scheduler);

  /**
   * @brief Get the network context used by the executor
   *
   * @return The reference to the network context of the fragment's executor (`NetworkContext`
   * object.)
   */
  std::shared_ptr<NetworkContext> network_context();

  /**
   * @brief Set the network context used by the executor
   *
   * @param network_context The network context to be added.
   */
  void network_context(const std::shared_ptr<NetworkContext>& network_context);

  /**
   * @brief Get the Argument(s) from the configuration file.
   *
   * For the given key, this method returns the value of the configuration file.
   *
   * For example:
   *
   * ```yaml
   * source: "replayer"
   * do_record: false   # or 'true' if you want to record input video stream.
   *
   * capture_card:
   *   width: 1920
   *   height: 1080
   *   rdma: true
   * ```
   *
   * `from_config("capture_card")` returns an ArgList (vector-like) object that contains the
   * following items:
   *
   * - `Arg("width") = 1920`
   * - `Arg("height") = 1080`
   * - `Arg("rdma") = true`
   *
   * You can use '.' (dot) to access nested fields.
   *
   * `from_config("capture_card.rdma")` returns an ArgList object that contains only one item and it
   * can be converted to `bool` through `ArgList::as()` method:
   *
   * ```cpp
   * auto is_rdma = from_config("capture_card.rdma").as<bool>();
   * ```
   *
   * @param key The key of the configuration.
   * @return The argument list of the configuration for the key.
   */
  ArgList from_config(const std::string& key);

  /**
   * @brief Determine the set of keys present in a Fragment's config.
   *
   * @return The set of valid keys.
   */
  std::unordered_set<std::string> config_keys();

  /**
   * @brief Create a new operator.
   *
   * @tparam OperatorT The type of the operator.
   * @param name The name of the operator.
   * @param args The arguments for the operator.
   * @return The shared pointer to the operator.
   */
  template <typename OperatorT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<OperatorT> make_operator(StringT name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating operator '{}'", name);
    auto op = std::make_shared<OperatorT>(std::forward<ArgsT>(args)...);
    op->name(name);

    setup_component_internals(op.get());

    auto spec = std::make_shared<OperatorSpec>(this);
    op->setup(*spec.get());
    op->spec(spec);

    // We used to initialize operator here, but now it is initialized in initialize_fragment
    // function after a graph of a fragment has been composed.

    return op;
  }
  /**
   * @brief Create a new operator.
   *
   * @tparam OperatorT The type of the operator.
   * @param args The arguments for the operator.
   * @return The shared pointer to the operator.
   */
  template <typename OperatorT, typename... ArgsT>
  std::shared_ptr<OperatorT> make_operator(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating operator");
    auto op = make_operator<OperatorT>("noname_operator", std::forward<ArgsT>(args)...);
    return op;
  }

  /**
   * @brief Create a new (operator) resource.
   *
   * @tparam ResourceT The type of the resource.
   * @param name The name of the resource.
   * @param args The arguments for the resource.
   * @return The shared pointer to the resource.
   */
  template <typename ResourceT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ResourceT> make_resource(StringT name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating resource '{}'", name);
    auto resource = std::make_shared<ResourceT>(std::forward<ArgsT>(args)...);
    resource->name(name);

    setup_component_internals(resource.get());

    auto spec = std::make_shared<ComponentSpec>(this);
    resource->setup(*spec.get());
    resource->spec(spec);

    // Skip initialization. `resource->initialize()` is done in GXFOperator::initialize()

    return resource;
  }
  /**
   * @brief Create a new (operator) resource.
   *
   * @tparam ResourceT The type of the resource.
   * @param args The arguments for the resource.
   * @return The shared pointer to the resource.
   */
  template <typename ResourceT, typename... ArgsT>
  std::shared_ptr<ResourceT> make_resource(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating resource");
    auto resource = make_resource<ResourceT>("noname_resource", std::forward<ArgsT>(args)...);
    return resource;
  }

  /**
   * @brief Create a new condition.
   *
   * @tparam ConditionT The type of the condition.
   * @param name The name of the condition.
   * @param args The arguments for the condition.
   * @return The shared pointer to the condition.
   */
  template <typename ConditionT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<ConditionT> make_condition(StringT name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating condition '{}'", name);
    auto condition = std::make_shared<ConditionT>(std::forward<ArgsT>(args)...);
    condition->name(name);

    setup_component_internals(condition.get());

    auto spec = std::make_shared<ComponentSpec>(this);
    condition->setup(*spec.get());
    condition->spec(spec);

    // Skip initialization. `condition->initialize()` is done in GXFOperator::initialize()

    return condition;
  }

  /**
   * @brief Create a new condition.
   *
   * @tparam ConditionT The type of the condition.
   * @param args The arguments for the condition.
   * @return The shared pointer to the condition.
   */
  template <typename ConditionT, typename... ArgsT>
  std::shared_ptr<ConditionT> make_condition(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating condition");
    auto condition = make_condition<ConditionT>("noname_condition", std::forward<ArgsT>(args)...);
    return condition;
  }

  /**
   * @brief Create a new scheduler.
   *
   * @tparam SchedulerT The type of the scheduler.
   * @param name The name of the scheduler.
   * @param args The arguments for the scheduler.
   * @return The shared pointer to the scheduler.
   */
  template <typename SchedulerT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<SchedulerT> make_scheduler(StringT name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating scheduler '{}'", name);
    auto scheduler = std::make_shared<SchedulerT>(std::forward<ArgsT>(args)...);
    scheduler->name(name);
    scheduler->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    scheduler->setup(*spec.get());
    scheduler->spec(spec);

    // Skip initialization. `scheduler->initialize()` is done in GXFExecutor::run()

    return scheduler;
  }

  /**
   * @brief Create a new scheduler.
   *
   * @tparam SchedulerT The type of the scheduler.
   * @param args The arguments for the scheduler.
   * @return The shared pointer to the scheduler.
   */
  template <typename SchedulerT, typename... ArgsT>
  std::shared_ptr<SchedulerT> make_scheduler(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating scheduler");
    auto scheduler = make_scheduler<SchedulerT>("", std::forward<ArgsT>(args)...);
    return scheduler;
  }

  /**
   * @brief Create a new network context.
   *
   * @tparam NetworkContextT The type of the network context.
   * @param name The name of the network context.
   * @param args The arguments for the network context.
   * @return The shared pointer to the network context.
   */
  template <typename NetworkContextT, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<NetworkContextT> make_network_context(StringT name, ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating network context '{}'", name);
    auto network_context = std::make_shared<NetworkContextT>(std::forward<ArgsT>(args)...);
    network_context->name(name);
    network_context->fragment(this);
    auto spec = std::make_shared<ComponentSpec>(this);
    network_context->setup(*spec.get());
    network_context->spec(spec);

    // Skip initialization. `network_context->initialize()` is done in GXFExecutor::run()

    return network_context;
  }

  /**
   * @brief Create a new network context.
   *
   * @tparam NetworkContextT The type of the network context.
   * @param args The arguments for the network context.
   * @return The shared pointer to the network context.
   */
  template <typename NetworkContextT, typename... ArgsT>
  std::shared_ptr<NetworkContextT> make_network_context(ArgsT&&... args) {
    HOLOSCAN_LOG_DEBUG("Creating network_context");
    auto network_context = make_network_context<NetworkContextT>("", std::forward<ArgsT>(args)...);
    return network_context;
  }

  /**
   * @brief Create a new thread pool resource.
   *
   * @param name The name of the thread pool.
   * @param initial_size The initial number of threads in the thread pool.
   * @return The shared pointer to the thread pool resource.
   */
  std::shared_ptr<ThreadPool> make_thread_pool(const std::string& name, int64_t initial_size = 1);

  /**
   * @brief Add default green context pool.
   *
   * @param dev_id The device id.
   * @param sms_per_partition The number of SMs per partition.
   * @param default_context_index The index of the default green context.
   * @param min_sm_size The minimum number of SMs per partition.
   * @return The shared pointer to the green context pool resource.
   */
  std::shared_ptr<CudaGreenContextPool> add_default_green_context_pool(
      int32_t dev_id, std::vector<uint32_t> sms_per_partition = {},
      int32_t default_context_index = -1, uint32_t min_sm_size = 2);

  /**
   * @brief Get the default green context pool.
   *
   * @return The shared pointer to the default green context pool.
   */
  std::shared_ptr<CudaGreenContextPool> get_default_green_context_pool();

  /**
   * @brief Get a fragment service by type information and identifier.
   *
   * Implementation of the FragmentServiceProvider interface method for retrieving
   * registered fragment services using runtime type information. This method provides
   * type-erased access to services and is thread-safe.
   *
   * @param service_type The type information of the service to retrieve.
   * @param id The identifier of the service. If empty, retrieves by type only.
   * @return The shared pointer to the fragment service, or nullptr if not found.
   */
  std::shared_ptr<FragmentService> get_service_erased(const std::type_info& service_type,
                                                      std::string_view id) const override;

  /**
   * @brief Register an existing fragment service instance.
   *
   * Registers an already created fragment service instance with the specified identifier.
   * This allows the fragment service to be retrieved later using the service() method.
   *
   * @tparam ServiceT The type of the fragment service.
   * @param svc The shared pointer to the fragment service instance to register.
   * @param id The identifier for the fragment service registration. If empty, uses the fragment
   * service type as identifier.
   * @return true if the service was successfully registered, false otherwise.
   */
  template <typename ServiceT>
  bool register_service(const std::shared_ptr<ServiceT>& svc, std::string_view id = "") {
    static_assert(holoscan::is_one_of_derived_v<ServiceT, Resource, FragmentService>,
                  "ServiceT must inherit from Resource or FragmentService");

    if (!svc) {
      HOLOSCAN_LOG_ERROR("Cannot register null pointer to fragment service");
      return false;
    }

    bool is_service = true;
    std::shared_ptr<FragmentService> svc_to_register;
    std::shared_ptr<Resource> resource;
    if constexpr (std::is_base_of_v<Resource, ServiceT>) {
      resource = svc;
      is_service = false;
    }
    if constexpr (std::is_base_of_v<FragmentService, ServiceT>) {
      svc_to_register = svc;
      resource = svc->resource();
      if constexpr (holoscan::is_one_of_derived_v<ServiceT, Resource>) {
        // For classes that inherit from both FragmentService and Resource, use the object itself
        // as the resource if no other resource has been specified.
        if (!resource) {
          resource = std::const_pointer_cast<ServiceT>(svc);
          svc->resource(resource);
        }
      }
      // When a class inherits from both Resource and FragmentService, prioritize treating it as a
      // service
      is_service = true;
    }

    // If the resource is available, we use resource's name for id and the id should be empty
    if (resource) {
      if (!id.empty()) {
        HOLOSCAN_LOG_ERROR(
            "If the Holoscan Resource is registered as a service, the id should be empty");
        return false;
      }
      id = resource->name();

      if (fragment_resource_services_by_name_.find(std::string(id)) !=
          fragment_resource_services_by_name_.end()) {
        HOLOSCAN_LOG_ERROR(
            "Resource service '{}' already exists in the fragment. Please specify a unique "
            "name when creating a Resource instance.",
            id);
        return false;
      }
    }

    // If the service is a resource, we need to create a new DefaultFragmentService object with the
    // resource
    if (!is_service) {
      auto fragment_service = std::make_shared<DefaultFragmentService>(resource);
      svc_to_register = fragment_service;
    }

    std::unique_lock<std::shared_mutex> lock(fragment_service_registry_mutex_);

    ServiceKey key{is_service ? typeid(*svc_to_register) : typeid(DefaultFragmentService),
                   std::string(id)};

    if (resource) {
      fragment_resource_services_by_name_[std::string(id)] = resource;
      // We use 'insert_or_assign' here since ServiceKey contains a std::type_index member which
      // cannot be default-constructed
      fragment_resource_to_service_key_map_.insert_or_assign(resource, key);

      // Also register the service with its resource type
      ServiceKey resource_key{typeid(*resource), std::string(id)};
      fragment_services_by_key_[resource_key] = svc_to_register;
    }

    fragment_services_by_key_[key] = std::move(svc_to_register);

    HOLOSCAN_LOG_DEBUG("Registered service '{}' with id '{}'", typeid(ServiceT).name(), id);
    return true;
  }

  virtual bool register_service_from(Fragment* fragment, std::string_view id);

  /**
   * @brief Retrieve a registered fragment service or resource
   *
   * Retrieves a previously registered fragment service or resource by its type and optional
   * identifier. Returns nullptr if no service/resource is found with the specified type and
   * identifier.
   *
   * Note that any changes to the service retrieval logic in this method should be synchronized with
   * the implementation in `ComponentBase::service()` method to maintain consistency.
   *
   * @tparam ServiceT The type of the service/resource to retrieve. Must inherit from either
   * Resource or FragmentService. Defaults to DefaultFragmentService if
   * not specified.
   * @param id The identifier of the service/resource. If empty, retrieves by type only.
   * @return The shared pointer to the service/resource, or nullptr if not found or if type casting
   * fails.
   */
  template <typename ServiceT = DefaultFragmentService>
  std::shared_ptr<ServiceT> service(std::string_view id = "") const {
    static_assert(holoscan::is_one_of_derived_v<ServiceT, Resource, FragmentService>,
                  "ServiceT must inherit from Resource or FragmentService");

    // Get the base service from the service registry
    auto base_service = get_service_erased(typeid(ServiceT), id);
    if (!base_service) {
      HOLOSCAN_LOG_DEBUG("Fragment '{}': Service of type {} with id '{}' not found.",
                         name(),
                         typeid(ServiceT).name(),
                         std::string(id));
      return nullptr;
    }

    // Handle Resource-derived services
    if constexpr (std::is_base_of_v<Resource, ServiceT>) {
      auto resource_ptr = base_service->resource();
      if (!resource_ptr) {
        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': No service resource is available for service with id '{}'.",
            name(),
            std::string(id));
        return nullptr;
      }

      // Attempt to cast the resource to the requested type
      auto typed_resource = std::dynamic_pointer_cast<ServiceT>(resource_ptr);
      if (!typed_resource) {
        HOLOSCAN_LOG_DEBUG(
            "Fragment '{}': Service resource with id '{}' is not type-castable to type '{}'.",
            name(),
            std::string(id),
            typeid(ServiceT).name());
      }
      return typed_resource;
    } else {
      // Handle FragmentService-derived services
      // Since DefaultFragmentService implements FragmentService, we can safely cast
      auto typed_service = std::dynamic_pointer_cast<ServiceT>(base_service);
      if (!typed_service) {
        HOLOSCAN_LOG_DEBUG("Fragment '{}': Service with id '{}' is not type-castable to type '{}'.",
                           name(),
                           std::string(id),
                           typeid(ServiceT).name());
      }
      return typed_service;
    }
  }

  /**
   * @brief Retrieve a registered fragment service or resource for Python bindings.
   *
   * This is a helper method for Python bindings to retrieve a service by its C++ type info.
   *
   * @param service_type The type info of the service/resource to retrieve.
   * @param id The identifier of the service/resource. If empty, retrieves by type only.
   * @return The shared pointer to the base service, or nullptr if not found.
   */
  std::shared_ptr<FragmentService> get_service_by_type_info(const std::type_info& service_type,
                                                            std::string_view id = "") const {
    return get_service_erased(service_type, id);
  }

  /**
   * @brief Get the fragment services by key.
   *
   * @return The fragment services by key.
   */
  const std::unordered_map<ServiceKey, std::shared_ptr<FragmentService>, ServiceKeyHash>&
  fragment_services_by_key() const {
    return fragment_services_by_key_;
  }

  /**
   * @brief Get or create the start operator for this fragment.
   *
   * This operator is nothing but the first operator that was added to the fragment.
   * It has the name of `<|start|>` and has a condition of `CountCondition(1)`.
   * This Operator is used to start the execution of the fragment.
   * Entry operators who want to start the execution of the fragment should connect to this
   * operator.
   *
   * If this method is not called, no start operator is created.
   * Otherwise, the start operator is created if it does not exist, and the shared pointer to the
   * start operator is returned.
   *
   * @return The shared pointer to the start operator.
   */
  virtual const std::shared_ptr<Operator>& start_op();

  /**
   * @brief Add an operator to the graph.
   *
   * The information of the operator is stored in the Graph object.
   * If the operator is already added, this method does nothing.
   *
   * @param op The operator to be added.
   */
  virtual void add_operator(const std::shared_ptr<Operator>& op);

  /**
   * @brief Add a flow between two operators.
   *
   * An output port of the upstream operator is connected to an input port of the
   * downstream operator.
   * The information about the flow (edge) is stored in the Graph object.
   *
   * If the upstream operator or the downstream operator is not in the graph, it will be added to
   * the graph.
   *
   * If there are multiple output ports in the upstream operator or multiple input ports in the
   * downstream operator, it shows an error message.
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op);

  /**
   * @brief Add a flow between two operators.
   *
   * An output port of the upstream operator is connected to an input port of the
   * downstream operator.
   * The information about the flow (edge) is stored in the Graph object.
   *
   * If the upstream operator or the downstream operator is not in the graph, it will be added to
   * the graph.
   *
   * In `port_pairs`, an empty port name ("") can be used for specifying a port name if the operator
   * has only one input/output port.
   *
   * If a non-existent port name is specified in `port_pairs`, it first checks if there is a
   * parameter with the same name but with a type of `std::vector<holoscan::IOSpec*>` in the
   * downstream operator.
   * If there is such a parameter (e.g., `receivers`), it creates a new input port with a specific
   * label (`<parameter name>:<index>`. e.g., `receivers:0`), otherwise it shows an error message.
   *
   * For example, if a parameter `receivers` want to have an arbitrary number of receivers,
   *
   *     class HolovizOp : public holoscan::ops::GXFOperator {
   *         ...
   *         private:
   *           Parameter<std::vector<holoscan::IOSpec*>> receivers_;
   *         ...
   *
   * Instead of creating a fixed number of input ports (e.g., `source_video` and `tensor`) and
   * assigning them to the parameter (`receivers`):
   *
   *     void HolovizOp::setup(OperatorSpec& spec) {
   *       ...
   *
   *       auto& in_source_video = spec.input<holoscan::gxf::Entity>("source_video");
   *       auto& in_tensor = spec.input<holoscan::gxf::Entity>("tensor");
   *
   *       spec.param(receivers_,
   *                  "receivers",
   *                  "Input Receivers",
   *                  "List of input receivers.",
   *                  {&in_source_video, &in_tensor});
   *       ...
   *
   * You can skip the creation of input ports and assign them to the parameter (`receivers`) as
   * follows:
   *
   *     void HolovizOp::setup(OperatorSpec& spec) {
   *       ...
   *       spec.param(receivers_,
   *                  "receivers",
   *                  "Input Receivers",
   *                  "List of input receivers.",
   *                  {&in_source_video, &in_tensor});
   *       ...
   *
   * This makes the following code possible in the Application's `compose()` method:
   *
   *     add_flow(source, visualizer_format_converter);
   *     add_flow(visualizer_format_converter, visualizer, {{"", "receivers"}});
   *
   *     add_flow(source, format_converter);
   *     add_flow(format_converter, inference);
   *     add_flow(inference, visualizer, {{"", "receivers"}});
   *
   * Instead of:
   *
   *     add_flow(source, visualizer_format_converter);
   *     add_flow(visualizer_format_converter, visualizer, {{"", "source_video"}});
   *
   *     add_flow(source, format_converter);
   *     add_flow(format_converter, inference);
   *     add_flow(inference, visualizer, {{"", "tensor"}});
   *
   * By using the parameter (`receivers`) with `std::vector<holoscan::IOSpec*>` type, the framework
   * creates input ports (`receivers:0` and `receivers:1`) implicitly and connects them (and adds
   * the references of the input ports to the `receivers` vector).
   *
   * Since Holoscan SDK v2.3, users can define a multi-receiver input port using `spec.input()`
   * with `IOSpec::kAnySize` instead of using `spec.param()`
   * with `Parameter<std::vector<IOSpec*>> receivers_;`. It is now recommended to use this
   * new `spec.input`-based approach and the old "receivers" parameter approach should be
   * considered deprecated.
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   * @param port_pairs The port pairs. The first element of the pair is the port of the upstream
   * operator and the second element is the port of the downstream operator.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs);
  /**
   * @brief Add a flow between two operators with a connector type.
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   * @param connector_type The connector type.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Add a flow between two operators with specified port pairs and a connector type.
   *
   * @param upstream_op The upstream operator.
   * @param downstream_op The downstream operator.
   * @param port_pairs The port pairs. The first element of the pair is the port of the upstream
   * operator and the second element is the port of the downstream operator.
   * @param connector_type The connector type.
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Set a callback function to define dynamic flows for an operator at runtime.
   *
   * This method allows operators to modify their connections with other operators during execution.
   * The callback function is called after the operator executes and can add dynamic flows using
   * the operator's `add_dynamic_flow()` methods.
   *
   * @param op The operator to set dynamic flows for
   * @param dynamic_flow_func The callback function that defines the dynamic flows. Takes a shared
   *                         pointer to the operator as input and returns void.
   */
  virtual void set_dynamic_flows(
      const std::shared_ptr<Operator>& op,
      const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func);

  /**
   * @brief Create and compose a Subgraph
   *
   * Creates a Subgraph that directly populates this Fragment's operator graph.
   * The Subgraph is composed immediately and its operators are added with
   * qualified names to the Fragment's main graph.
   *
   * @tparam SubgraphT The subgraph class type (must inherit from Subgraph)
   * @param instance_name Unique name for this instance (used for operator qualification)
   * @param args Additional arguments to pass to the subgraph constructor
   * @return Shared pointer to the composed Subgraph
   */
  template <typename SubgraphT, typename... ArgsT>
  std::shared_ptr<SubgraphT> make_subgraph(const std::string& instance_name, ArgsT&&... args) {
    // Check for duplicate subgraph instance names
    if (subgraph_instance_names_.find(instance_name) != subgraph_instance_names_.end()) {
      throw std::runtime_error(
          fmt::format("Fragment::make_subgraph: Duplicate subgraph instance name '{}'. "
                      "Each subgraph instance must have a unique name within the same fragment.",
                      instance_name));
    }

    // Register the instance name
    subgraph_instance_names_.insert(instance_name);

    // Create Subgraph with Fragment* and instance_name, plus any additional args
    auto subgraph = std::make_shared<SubgraphT>(this, instance_name, std::forward<ArgsT>(args)...);

    // Compose immediately - operators added directly to Fragment's main graph
    if (!subgraph->is_composed()) {
      subgraph->compose();
      subgraph->set_composed(true);
    }

    return subgraph;
  }

  /**
   * @brief Connect Operator to Subgraph
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_port, subgraph_interface_port}
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Subgraph to Operator
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param port_pairs Port connections: {subgraph_interface_port, downstream_port}
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Subgraph to Subgraph
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_interface_port, downstream_interface_port}
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs = {});

  /**
   * @brief Connect Operator to Subgraph with connector type
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Operator to Subgraph with port pairs and connector type
   *
   * @param upstream_op The upstream operator
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_port, subgraph_interface_port}
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Operator with connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Operator with port pairs and connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_op The downstream operator
   * @param port_pairs Port connections: {subgraph_interface_port, downstream_port}
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Subgraph with connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Connect Subgraph to Subgraph with port pairs and connector type
   *
   * @param upstream_subgraph The upstream subgraph
   * @param downstream_subgraph The downstream subgraph
   * @param port_pairs Port connections: {upstream_interface_port, downstream_interface_port}
   * @param connector_type The connector type
   */
  virtual void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type);

  /**
   * @brief Compose a graph.
   *
   * The graph is composed by adding operators and flows in this method.
   */
  virtual void compose();

  /**
   * @brief Initialize the graph and run the graph.
   *
   * This method calls `compose()` to compose the graph, and runs the graph.
   */
  virtual void run();

  /**
   * @brief Initialize the graph and run the graph asynchronously.
   *
   * This method calls `compose()` to compose the graph, and runs the graph asynchronously.
   *
   * @return The future object.
   */
  virtual std::future<void> run_async();

  /**
   * @brief Turn on data frame flow tracking.
   *
   * A reference to a DataFlowTracker object is returned rather than a pointer so that the
   * developers can use it as an object without unnecessary pointer dereferencing.
   *
   * @param num_start_messages_to_skip The number of messages to skip at the beginning.
   * @param num_last_messages_to_discard The number of messages to discard at the end.
   * @param latency_threshold The minimum end-to-end latency in milliseconds to account for
   * in the end-to-end latency metric calculations.
   * @param is_limited_tracking If true, the tracking is limited to root and leaf nodes, minimizing
   * the timestamps by avoiding intermediate operators.
   * @return A reference to the DataFlowTracker object in which results will be
   * stored.
   */
  DataFlowTracker& track(uint64_t num_start_messages_to_skip = kDefaultNumStartMessagesToSkip,
                         uint64_t num_last_messages_to_discard = kDefaultNumLastMessagesToDiscard,
                         int latency_threshold = kDefaultLatencyThreshold,
                         bool is_limited_tracking = false);

  /**
   * @brief Get the DataFlowTracker object for this fragment.
   *
   * @return The pointer to the DataFlowTracker object.
   */
  DataFlowTracker* data_flow_tracker() { return data_flow_tracker_.get(); }

  /**
   * @brief Calls compose() if the graph is not composed yet.
   */
  virtual void compose_graph();

  /**
   * @brief Get an easily serializable summary of port information.
   *
   * The FragmentPortMap class is used by distributed applications to send port information
   * between application workers and the driver.
   *
   * @return An unordered_map of the fragment's port information where the keys are operator names
   * and the values are a 3-tuple. The first two elements of the tuple are the set of input and
   * output port names, respectively. The third element of the tuple is the set of "receiver"
   * parameters (those with type std::vector<IOSpec*>).
   */
  FragmentPortMap port_info() const;

  /**
   * @brief Determine whether metadata is enabled by default for operators in this fragment.
   *
   * Note that individual operators may still have been configured to override this default
   * via Operator::enable_metadata.
   *
   * @return Boolean indicating whether metadata is enabled.
   */
  virtual bool is_metadata_enabled() const;

  /**
   * @brief Deprecated method for controlling whether metadata is enabled for the fragment.
   *
   * Please use `enable_metadata` instead.
   *
   * @param enabled Boolean indicating whether metadata should be enabled.
   */
  virtual void is_metadata_enabled(bool enabled);

  /**
   * @brief Enable or disable metadata for the fragment.
   *
   * Controls whether metadata is enabled or disabled by default for operators within this fragment.
   * If this method is not called, and this fragment is part of a distributed application, then the
   * the parent application's metadata policy will be used. Otherwise metadata is enabled by
   * default. Individual operators can override this setting using the Operator::enable_metadata()
   * method.
   *
   * @param enable Boolean indicating whether metadata should be enabled.
   */
  virtual void enable_metadata(bool enable);

  /**
   * @brief Get the default metadata update policy used for operators within this fragment.
   *
   * If a value was set for a specific operator via `Operator::metadata_policy` that value will
   * take precedence over this fragment default. If no policy was set for the fragment and this
   * fragment is part of a distributed application, the default metadata policy of the application
   * will be used.
   *
   * @returns The default metadata update policy used by operators in this fragment.
   */
  virtual MetadataPolicy metadata_policy() const;

  /**
   * @brief Set the default metadata update policy to be used for operators within this fragment.
   *
   * The metadata policy determines how metadata is merged across multiple receive calls:
   *    - `MetadataPolicy::kUpdate`: Update the existing value when a key already exists.
   *    - `MetadataPolicy::kInplaceUpdate`: Update the existing MetadataObject's value in-place
   *    when a key already exists.
   *    - `MetadataPolicy::kReject`: Do not modify the existing value if a key already exists.
   *    - `MetadataPolicy::kRaise`: Raise an exception if a key already exists (default).
   *
   * @param policy The metadata update policy to be used by this operator.
   */
  virtual void metadata_policy(MetadataPolicy policy);

  /**
   * @brief Stop the execution of all operators in the fragment.
   *
   * This method is used to stop the execution of all operators in the fragment by setting the
   * internal async condition of each operator to EVENT_NEVER state, which sets the scheduling
   * condition to NEVER.
   * Once stopped, the operators will not be scheduled for execution
   * (the `compute()` method will not be called), which may lead to application termination
   * depending on the application's design.
   *
   * Note that executing this method does not trigger the operators' `stop()` method.
   * The `stop()` method is called only when the scheduler deactivates all operators together.
   *
   * @param op_name The name of the operator to stop. If empty, all operators will be stopped.
   */
  virtual void stop_execution(const std::string& op_name = "");

  /**
   * @brief Add a data logger to the fragment.
   *
   * @param logger The shared pointer to the data logger to add.
   */
  void add_data_logger(const std::shared_ptr<DataLogger>& logger);

  /**
   * @brief Get the data loggers associated with this fragment.
   *
   * @return A const reference to the vector of data loggers.
   */
  const std::vector<std::shared_ptr<DataLogger>>& data_loggers() const { return data_loggers_; }

  /**
   * @brief Check if the fragment has GPU-resident operators.
   *
   * @return True if the fragment has GPU-resident operators, false otherwise.
   */
  bool is_gpu_resident() const { return is_gpu_resident_; }

  /**
   * @brief Get an accessor for GPU-resident specific functions.
   *
   * This method returns a GPUResidentAccessor object that provides convenient access to
   * GPU-resident specific functionality. It allows for a cleaner API pattern:
   *
   * ```cpp
   * fragment->gpu_resident().timeout_ms(1000);
   * fragment->gpu_resident().data_ready();
   * fragment->gpu_resident().result_ready();
   * fragment->gpu_resident().is_launched();
   * fragment->gpu_resident().tear_down();
   * ```
   *
   * @return A GPUResidentAccessor object for accessing GPU-resident functions.
   * @throws RuntimeError if the fragment does not have GPU-resident operators.
   */
  GPUResidentAccessor gpu_resident();

 protected:
  friend class Application;  // to access 'scheduler_' in Application
  friend class AppDriver;
  friend class gxf::GXFExecutor;
  friend class holoscan::ComponentBase;  // Allow ComponentBase to access internal setup
  friend class GPUResidentAccessor;      // Allow GPUResidentAccessor to access
                                         // get_gpu_resident_executor

  template <typename ConfigT, typename... ArgsT>
  std::shared_ptr<Config> make_config(ArgsT&&... args) {
    return std::make_shared<ConfigT>(std::forward<ArgsT>(args)...);
  }

  template <typename GraphT>
  std::shared_ptr<GraphT> make_graph() {
    return std::make_shared<GraphT>();
  }

  /**
   * @brief Create and assign an Executor to the fragment
   */
  template <typename ExecutorT, typename... ArgsT>
  std::shared_ptr<Executor> make_executor(ArgsT&&... args) {
    executor_ = std::make_shared<ExecutorT>(this, std::forward<ArgsT>(args)...);
    return executor_;
  }

  /// Cleanup helper that will be called by the executor prior to destroying any backend context
  void reset_backend_objects();

  /// Shutdown data loggers to ensure async loggers complete before GXF context destruction.
  void shutdown_data_loggers();

  /**
   * @brief Reset internal fragment state to allow for multiple run calls
   *
   * This method resets the necessary internal state to allow multiple consecutive
   * calls to run() or run_async() without requiring manual cleanup.
   */
  virtual void reset_state();

  /// Load the GXF extensions specified in the configuration.
  void load_extensions_from_config();

  std::vector<std::shared_ptr<ThreadPool>>& thread_pools() { return thread_pools_; }

  /**
   * @brief Set up internal state for a component.
   *
   * Configures the component's internal references to this fragment and its service provider.
   * This method is called internally when creating operators, resources, conditions, and other
   * components to ensure they have proper access to fragment services.
   *
   * @param component Pointer to the ComponentBase instance to configure. Must not be nullptr.
   */
  void setup_component_internals(ComponentBase* component);

  /**
   * @brief Resolve Subgraph interface port to actual operator and port
   *
   * @param subgraph The Subgraph to resolve the port in
   * @param interface_port The interface port name
   * @return Pair of (operator, actual_port_name) or (nullptr, "") if not found
   */
  std::pair<std::shared_ptr<Operator>, std::string> resolve_subgraph_port(
      const std::shared_ptr<Subgraph>& subgraph, const std::string& interface_port);

  // ========== Helper functions for port auto-resolution ==========

  /**
   * @brief Get output port names from an operator
   */
  std::vector<std::string> get_operator_output_ports(const std::shared_ptr<Operator>& op);

  /**
   * @brief Get input port names from an operator
   */
  std::vector<std::string> get_operator_input_ports(const std::shared_ptr<Operator>& op);

  /**
   * @brief Get output interface port names from a subgraph
   */
  std::vector<std::string> get_subgraph_output_ports(const std::shared_ptr<Subgraph>& subgraph);

  /**
   * @brief Get input interface port names from a subgraph
   */
  std::vector<std::string> get_subgraph_input_ports(const std::shared_ptr<Subgraph>& subgraph);

  /**
   * @brief Attempt auto-resolution of port pairs between two entities
   *
   * @param upstream_ports Output ports from upstream entity
   * @param downstream_ports Input ports from downstream entity
   * @param upstream_name Name of upstream entity (for error messages)
   * @param downstream_name Name of downstream entity (for error messages)
   * @param port_pairs Output parameter: will contain the resolved port pair if successful
   * @throws std::runtime_error if auto-resolution fails
   */
  void try_auto_resolve_ports(const std::vector<std::string>& upstream_ports,
                              const std::vector<std::string>& downstream_ports,
                              const std::string& upstream_name, const std::string& downstream_name,
                              std::set<std::pair<std::string, std::string>>& port_pairs);

  /**
   * @brief Resolve and create flows for operator-to-subgraph connections
   */
  void resolve_and_create_op_to_subgraph_flows(
      const std::shared_ptr<Operator>& upstream_op,
      const std::shared_ptr<Subgraph>& downstream_subgraph,
      const std::set<std::pair<std::string, std::string>>& port_pairs,
      const IOSpec::ConnectorType connector_type);

  /**
   * @brief Resolve and create flows for subgraph-to-operator connections
   */
  void resolve_and_create_subgraph_to_op_flows(
      const std::shared_ptr<Subgraph>& upstream_subgraph,
      const std::shared_ptr<Operator>& downstream_op,
      const std::set<std::pair<std::string, std::string>>& port_pairs,
      const IOSpec::ConnectorType connector_type);

  /**
   * @brief Resolve and create flows for subgraph-to-subgraph connections
   */
  void resolve_and_create_subgraph_to_subgraph_flows(
      const std::shared_ptr<Subgraph>& upstream_subgraph,
      const std::shared_ptr<Subgraph>& downstream_subgraph,
      const std::set<std::pair<std::string, std::string>>& port_pairs,
      const IOSpec::ConnectorType connector_type);

  // ========== Helper functions for control flow connections ==========

  /**
   * @brief Validate prerequisites for establishing a control flow connection
   *
   * @param upstream_op The upstream operator
   * @param downstream_op The downstream operator
   * @param connector_type The connector type (cannot be kAsyncBuffer for control flow)
   * @return true if validation passes, false otherwise (error message will be logged)
   */
  bool validate_control_flow_prerequisites(const std::shared_ptr<Operator>& upstream_op,
                                           const std::shared_ptr<Operator>& downstream_op,
                                           const IOSpec::ConnectorType connector_type);

  /**
   * @brief Create and register a control flow connection between two operators
   *
   * This helper creates the port map, sets self_shared on both operators,
   * adds the flow to the graph, and registers it with the executor.
   *
   * @param upstream_op The upstream operator
   * @param downstream_op The downstream operator
   */
  void create_control_flow_connection(const std::shared_ptr<Operator>& upstream_op,
                                      const std::shared_ptr<Operator>& downstream_op);

  // Note: Maintain the order of declarations (executor_ and graph_) to ensure proper destruction
  //       of the executor's context.
  std::string name_;                      ///< The name of the fragment.
  Application* app_ = nullptr;            ///< The application that this fragment belongs to.
  std::shared_ptr<Config> config_;        ///< The configuration of the fragment.
  std::shared_ptr<Executor> executor_;    ///< The executor for the fragment.
  std::shared_ptr<OperatorGraph> graph_;  ///< The graph of the fragment.
  mutable std::shared_ptr<Scheduler>
      scheduler_;  ///< Lazily initialized scheduler (mutable for const access).
  std::shared_ptr<NetworkContext> network_context_;  ///< The network_context used by the executor
  std::shared_ptr<DataFlowTracker> data_flow_tracker_;  ///< The DataFlowTracker for the fragment
  std::vector<std::shared_ptr<ThreadPool>>
      thread_pools_;            ///< Any thread pools used by the fragment
  bool is_composed_ = false;    ///< Whether the graph is composed or not.
  bool is_run_called_ = false;  ///< Whether run() or run_async() has been called.
  std::optional<bool> is_metadata_enabled_ =
      std::nullopt;  ///< Whether metadata is enabled or not. If nullopt, value from Application()
                     ///< is used if it has been set. Otherwise defaults to true.
  std::optional<MetadataPolicy> metadata_policy_ = std::nullopt;
  std::shared_ptr<Operator> start_op_;  ///< The start operator of the fragment (optional).
  std::vector<std::shared_ptr<DataLogger>> data_loggers_;  ///< Data loggers (optional)

  // Service registry members
  mutable std::shared_mutex
      fragment_service_registry_mutex_;  ///< Mutex for thread-safe service registry access
  std::unordered_map<ServiceKey, std::shared_ptr<FragmentService>, ServiceKeyHash>
      fragment_services_by_key_;  ///< service registry map
  std::unordered_map<std::string, std::shared_ptr<Resource>>
      fragment_resource_services_by_name_;  ///< service resource registry map
  std::unordered_map<std::shared_ptr<Resource>, ServiceKey>
      fragment_resource_to_service_key_map_;  ///< service resource registry map

  // The default green context pool in the fragment.
  std::vector<std::shared_ptr<CudaGreenContextPool>> green_context_pools_;

  // Track subgraph instance names to detect duplicates
  std::unordered_set<std::string> subgraph_instance_names_;

 private:
  bool verify_gpu_resident_connections(const std::shared_ptr<Operator>& upstream_op,
                                       const std::shared_ptr<Operator>& downstream_op,
                                       const std::shared_ptr<OperatorEdgeDataElementType> port_map);

  /**
   * @brief Helper function to get GPU resident executor with error handling
   *
   * @param func_name The name of the calling function (typically __func__)
   * @return std::shared_ptr<GPUResidentExecutor> The GPU resident executor
   * @throws RuntimeError if casting fails
   */
  std::shared_ptr<GPUResidentExecutor> get_gpu_resident_executor(const char* func_name);

  bool is_gpu_resident_ = false;  ///< Whether the fragment is a GPU resident fragment.
};

// Subgraph template method implementations - placed here to resolve circular dependency
// These methods depend on the full Fragment definition being available

template <typename OperatorT, typename StringT, typename... ArgsT, typename>
std::shared_ptr<OperatorT> Subgraph::make_operator(StringT name, ArgsT&&... args) {
  if (!fragment_) {
    throw std::runtime_error(
        "Subgraph::make_operator called but fragment_ is nullptr. "
        "Subgraph must be created via Fragment::make_subgraph to set the target fragment.");
  }
  auto qualified_name = get_qualified_name(std::string(name), "operator");
  return fragment_->make_operator<OperatorT>(qualified_name, std::forward<ArgsT>(args)...);
}

template <typename OperatorT, typename... ArgsT>
std::shared_ptr<OperatorT> Subgraph::make_operator(ArgsT&&... args) {
  auto qualified_name = get_qualified_name("noname_operator", "operator");
  return make_operator<OperatorT>("noname_operator", std::forward<ArgsT>(args)...);
}

template <typename ConditionT, typename StringT, typename... ArgsT, typename>
std::shared_ptr<ConditionT> Subgraph::make_condition(StringT name, ArgsT&&... args) {
  if (!fragment_) {
    throw std::runtime_error(
        "Subgraph::make_condition called but fragment_ is nullptr. "
        "Subgraph must be created via Fragment::make_subgraph to set the target fragment.");
  }

  // Use qualified name to avoid conflicts between Subgraph instances
  auto qualified_name = get_qualified_name(std::string(name), "condition");
  return fragment_->make_condition<ConditionT>(qualified_name, std::forward<ArgsT>(args)...);
}

template <typename ConditionT, typename... ArgsT>
std::shared_ptr<ConditionT> Subgraph::make_condition(ArgsT&&... args) {
  return make_condition<ConditionT>("noname_condition", std::forward<ArgsT>(args)...);
}

template <typename ResourceT, typename StringT, typename... ArgsT, typename>
std::shared_ptr<ResourceT> Subgraph::make_resource(StringT name, ArgsT&&... args) {
  if (!fragment_) {
    throw std::runtime_error(
        "Subgraph::make_resource called but fragment_ is nullptr. "
        "Subgraph must be created via Fragment::make_subgraph to set the target fragment.");
  }

  // Use qualified name to avoid conflicts between Subgraph instances
  auto qualified_name = get_qualified_name(std::string(name), "resource");
  return fragment_->make_resource<ResourceT>(qualified_name, std::forward<ArgsT>(args)...);
}

template <typename ResourceT, typename... ArgsT>
std::shared_ptr<ResourceT> Subgraph::make_resource(ArgsT&&... args) {
  return make_resource<ResourceT>("noname_resource", std::forward<ArgsT>(args)...);
}

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FRAGMENT_HPP */
