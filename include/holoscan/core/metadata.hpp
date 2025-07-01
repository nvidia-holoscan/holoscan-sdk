/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_METADATA_HPP
#define HOLOSCAN_CORE_METADATA_HPP

#include <memory>
#include <shared_mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./common.hpp"
#include "./message.hpp"

using std::string_literals::operator""s;

namespace holoscan {

// reuse holoscan::Message so same serialization codecs can be reused for metadata
using MetadataObject = holoscan::Message;  ///< The metadata object type used in MetadataDictionary

/**
 * @brief Enum to define the policy for handling behavior of MetadataDictionary::set<ValueT>
 */
enum class MetadataPolicy {
  kReject,         ///< Reject the new value if the key already exists
  kInplaceUpdate,  ///< Replace the `std::any` value within the existing
                   ///< `std::shared_ptr<MetadataObject>` if the key already exists
  kUpdate,   ///< Replace the `std::shared_ptr<MetadataObject>` with a newly constructed one if the
             ///< key already exists
  kRaise,    ///< Raise an exception if the key already exists
  kDefault,  ///< Default value indicating that the user did not explicitly set a policy via
             ///< `Operator::metadata_policy`. The policy used will be that set via
             ///< `Fragment::metadata_policy` if that was set. Otherwise the default of kRaise is
             ///< used.
};

/**
 * @brief Class to define a metadata dictionary object.
 */
class MetadataDictionary {
 public:
  using MapType = std::unordered_map<std::string, std::shared_ptr<MetadataObject>>;
  using Iterator = MapType::iterator;
  using ConstIterator = MapType::const_iterator;

  // Constructors
  explicit MetadataDictionary(const MetadataPolicy& policy = MetadataPolicy::kDefault)
      : dictionary_(std::make_shared<MapType>()), policy_(policy) {}
  MetadataDictionary(const MetadataDictionary&) = default;
  MetadataDictionary(MetadataDictionary&&) = default;

  // assignment operators
  MetadataDictionary& operator=(const MetadataDictionary&);
  MetadataDictionary& operator=(MetadataDictionary&&) = default;

  // Destructor
  virtual ~MetadataDictionary() = default;

  /// Get a vector of the keys in the dictionary
  std::vector<std::string> keys() const;

  /// Provide indexing into the underlying dictionary
  std::shared_ptr<MetadataObject>& operator[](const std::string&);

  /// Provide indexing into the underlying dictionary
  const MetadataObject* operator[](const std::string&) const;

  /**
   * @brief Get a shared pointer to the MetadataObject corresponding to the provided key
   *
   * See the templated variant of get for a version of this method that extracts the std::any value
   * from the MetadataObject and performs the any_cast<ValueT> operation on it.
   *
   * @param key The key for the value to retrieve.
   * @return The value corresponding to the key
   */
  std::shared_ptr<MetadataObject> get(const std::string& key) const;

  /**
   * @brief Get the value contained in the MetadataObject corresponding to the provided key
   *
   * Calling `dict->get<T>(key)` is a convenience method for the following:
   *
   * ```cpp
   * auto metadata_obj = dict->get(key);
   * auto value = std::any_cast<T>(metadata_obj->value());
   * ```
   * This functions raises a std::runtime_error if the key does not exist.
   *
   * @tparam ValueT The type of data stored in the std::any returned by MetadataObject.value()
   * @param key The key for the value to retrieve.
   * @return The value contained in the metadata object corresponding to the key
   */
  template <typename ValueT>
  ValueT get(const std::string& key) const {
    auto it = dictionary_->find(key);
    if (it == dictionary_->end()) {
      throw std::runtime_error(fmt::format("Key '{}' does not exist", key));
    }
    const auto& shared_obj = it->second;
    return std::any_cast<ValueT>(shared_obj->value());
  }

  /**
   * @brief Get the value contained in the MetadataObject corresponding to the provided key
   *
   * Calling `dict->get<T>(key)` is a convenience method for the following:
   *
   * ```cpp
   * auto metadata_obj = dict->get(key);
   * auto value = std::any_cast<T>(metadata_obj->value());
   * ```
   * This function returns the provided default_value if the key does not exist
   *
   * @tparam ValueT The type of data stored in the std::any returned by MetadataObject.value()
   * @param key The key for the value to insert.
   * @param default_value The value to return if the key does not exist in the dictionary
   * @return The value contained in the metadata object corresponding to the key
   */
  template <typename ValueT>
  ValueT get(const std::string& key, const ValueT& default_value) const {
    auto it = dictionary_->find(key);
    if (it == dictionary_->end()) { return default_value; }
    const auto& shared_obj = it->second;
    return std::any_cast<ValueT>(shared_obj->value());
  }

  /**
   * @brief Insert a new value at the specified key (or update an existing one)
   *
   * This method inserts a new item. If the key already exists, this method will have a behavior
   * that depends on the policy (`MetadataPolicy`) set for this dictionary.
   *
   * For this method, kInplaceUpdate behaves the same as kUpdate.
   *
   * @param key The key for the value to insert (or update).
   * @param value The value to store.
   */
  void set(const std::string& key, std::shared_ptr<MetadataObject> value);

  /**
   * @brief Insert or update a value at the specified key.
   *
   * This method inserts a new item. If the key already exists, this method will have a behavior
   * that depends on the policy (`MetadataPolicy`) set for this dictionary.
   *
   * @tparam ValueT The type of metadata to insert
   * @param key The key for the value to insert (or update).
   * @param value The value to store.
   */
  template <typename ValueT, typename = std::enable_if_t<!std::is_same_v<
                                 std::decay_t<ValueT>, std::shared_ptr<MetadataObject>>>>
  void set(const std::string& key, ValueT value) {
    ensure_unique();
    auto it = dictionary_->find(key);
    if (it == dictionary_->end()) {
      (*dictionary_)[key] = std::make_shared<MetadataObject>(value);
    } else {
      switch (policy_) {
        case MetadataPolicy::kReject:
          // keep the old value
          break;
        case MetadataPolicy::kInplaceUpdate:
          // set the std::any member within the existing MetadataObject
          (it->second)->set_value(value);
          break;
        case MetadataPolicy::kUpdate:
          // replace the MetadataObject with a newly constructed one
          (*dictionary_)[key] = std::make_shared<MetadataObject>(value);
          break;
        case MetadataPolicy::kDefault:
        case MetadataPolicy::kRaise:
          throw std::runtime_error(
              fmt::format("Key '{}' already exists. The application should be updated to avoid "
                          "duplicate metadata keys or a different holoscan::MetadataPolicy should "
                          "be set",
                          key));
        default:
          // Handle unknown policy
          throw std::runtime_error("Unknown metadata policy");
      }
      return;
    }
  }
  /// Get the metadata policy used by this MetadataDictionary
  MetadataPolicy policy() const { return policy_; }

  /// Set the metadata policy used by this MetadataDictionary
  void policy(const MetadataPolicy& metadata_policy) { policy_ = metadata_policy; }

  /**
   * @brief Determine if an item already exists in the dictionary.
   *
   * @param key The key for the value to insert (or update).
   * @return Returns true if the key already exists in the dictionary, false otherwise.
   */
  bool has_key(const std::string& key) const;

  /**
   * @brief Erase an item from the dictionary
   *
   * @param key The key of the item to erase.
   * @return Returns true if the key was erased and false if the key was not found.
   */
  bool erase(const std::string& key);

  /// begin() iterator of the underlying dictionary
  Iterator begin();

  /// begin() const iterator of the underlying dictionary
  ConstIterator begin() const;

  /// end() iterator of the underlying dictionary
  Iterator end();

  /// end() const iterator of the underlying dictionary
  ConstIterator end() const;

  /// find() on the underlying dictionary. Returns an iterator to the element if found, else end()
  Iterator find(const std::string& key);

  /// find() on the underlying dictionary. Returns an iterator to the element if found, else end()
  ConstIterator find(const std::string& key) const;

  /// clear all values from the dictionary
  void clear();

  /// return the number of items in the dictionary
  std::size_t size() const;

  /// swap the contents of this dictionary and other
  void swap(MetadataDictionary& other);

  /// merge (move) the contents of other dictionary into this dictionary
  void merge(MetadataDictionary& other);

  /// Insert items the other dictionary into this dictionary. Pre-existing values are not
  /// updated.
  void insert(MetadataDictionary& other);

  /**
   * @brief Update the dictionary with items present in other.
   *
   * This method will update the dictionary with values from other. If a key already exists in this
   * dictionary, the behavior is determined by the policy (MetadataPolicy) set for this dictionary.
   *
   * For this method, kInplaceUpdate behaves the same as kUpdate.
   *
   * @param other metadata dictionary to copy items from
   */
  void update(MetadataDictionary& other);

 private:
  /// If the dictionary is shared, make a copy of it so that it is unique
  bool ensure_unique();

  std::shared_ptr<MapType> dictionary_{};            ///< The underlying dictionary object
  MetadataPolicy policy_{MetadataPolicy::kDefault};  ///< The policy for handling metadata
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_METADATA_HPP */
