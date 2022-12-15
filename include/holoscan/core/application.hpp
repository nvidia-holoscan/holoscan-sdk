/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_APPLICATION_HPP
#define HOLOSCAN_CORE_APPLICATION_HPP

#include <iostream>     // for std::cout
#include <memory>       // for std::shared_ptr
#include <set>          // for std::set
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if_t, std::is_constructible
#include <utility>      // for std::pair

#include "./fragment.hpp"

namespace holoscan {

/**
 * @brief Utility function to create an application.
 *
 * @tparam AppT The type of the application to create.
 * @param args The arguments to pass to the application constructor.
 * @return The shared pointer to the created application.
 */
template <typename AppT, typename... ArgsT>
std::shared_ptr<AppT> make_application(ArgsT&&... args) {
  return std::make_shared<AppT>(std::forward<ArgsT>(args)...);
}

/**
 * @brief Application class.
 *
 * An application acquires and processes streaming data. An application is a collection of fragments
 * where each fragment can be allocated to execute on a physical node of a Holoscan cluster.
 */
class Application : public Fragment {
 public:
  Application() = default;
  ~Application() override = default;

  /**
   * @brief Create a new fragment.
   *
   * @tparam FragmentT The type of the fragment to create.
   * @param name The name of the fragment.
   * @param args The arguments to pass to the fragment constructor.
   * @return The shared pointer to the created fragment.
   */
  template <typename FragmentT = Fragment, typename StringT, typename... ArgsT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  std::shared_ptr<Fragment> make_fragment(const StringT& name, ArgsT&&... args) {
    auto fragment = std::make_shared<FragmentT>(std::forward<ArgsT>(args)...);
    fragment->name(name);
    fragment->application(this);
    return fragment;
  }

  /**
   * @brief Create a new fragment.
   *
   * @tparam FragmentT The type of the fragment to create.
   * @param args The arguments to pass to the fragment constructor.
   * @return The shared pointer to the created fragment.
   */
  template <typename FragmentT, typename... ArgsT>
  std::shared_ptr<FragmentT> make_fragment(ArgsT&&... args) {
    auto fragment = std::make_shared<FragmentT>(std::forward<ArgsT>(args)...);
    fragment->application(this);
    return fragment;
  }

  // Inherit Fragment's add_flow methods (for Operator) in addition to the overloads below
  using Fragment::add_flow;

  virtual void add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                        const std::shared_ptr<Fragment>& downstream_frag) {
    (void)upstream_frag;
    (void)downstream_frag;
    HOLOSCAN_LOG_ERROR("Application::add_flow() for Fragment is not implemented yet");
  }

  virtual void add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                        const std::shared_ptr<Fragment>& downstream_frag,
                        const std::set<std::pair<std::string, std::string>>& port_pairs) {
    (void)upstream_frag;
    (void)downstream_frag;
    (void)port_pairs;
    HOLOSCAN_LOG_ERROR("Application::add_flow() for Fragment is not implemented yet");
  }
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_APPLICATION_HPP */
