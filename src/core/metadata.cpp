/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/metadata.hpp"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace holoscan {

inline void swap(MetadataDictionary& a, MetadataDictionary& b) {
  a.swap(b);
}

MetadataDictionary& MetadataDictionary::operator=(const MetadataDictionary& old) {
  if (this != &old) {
    // perform shallow copy, so dictionary_ is shared
    dictionary_ = old.dictionary_;
  }
  return *this;
}

std::shared_ptr<MetadataObject>& MetadataDictionary::operator[](const std::string& key) {
  ensure_unique();
  return (*dictionary_)[key];
}

const MetadataObject* MetadataDictionary::operator[](const std::string& key) const {
  auto iter = dictionary_->find(key);
  if (iter == dictionary_->end()) { return nullptr; }

  const MetadataObject* constentry = iter->second.get();
  return constentry;
}

std::shared_ptr<MetadataObject> MetadataDictionary::get(const std::string& key) const {
  auto it = dictionary_->find(key);
  if (it == dictionary_->end()) {
    throw std::runtime_error(fmt::format("Key '{}' does not exist", key));
  }
  const auto& shared_obj = it->second;
  return shared_obj;
}

void MetadataDictionary::set(const std::string& key, std::shared_ptr<MetadataObject> object) {
  ensure_unique();
  if (has_key(key)) {
    if (policy_ == MetadataPolicy::kRaise) {
      throw std::runtime_error(
          fmt::format("Key '{}' already exists. The application should be updated to avoid "
                      "duplicate metadata keys or a different holoscan::MetadataPolicy should "
                      "be set",
                      key));
    } else if (policy_ == MetadataPolicy::kReject) {
      // don't replace existing value
      return;
    }
  }
  (*dictionary_)[key] = object;
}

bool MetadataDictionary::has_key(const std::string& key) const {
  return dictionary_->find(key) != dictionary_->end();
}

std::vector<std::string> MetadataDictionary::keys() const {
  using VectorType = std::vector<std::string>;
  VectorType keys;

  for (MapType::const_iterator it = dictionary_->begin(); it != dictionary_->end(); ++it) {
    keys.push_back(it->first);
  }

  return keys;
}

size_t MetadataDictionary::size() const {
  return dictionary_->size();
}

MetadataDictionary::Iterator MetadataDictionary::begin() {
  ensure_unique();
  return dictionary_->begin();
}

MetadataDictionary::ConstIterator MetadataDictionary::begin() const {
  return dictionary_->begin();
}

MetadataDictionary::Iterator MetadataDictionary::end() {
  ensure_unique();
  return dictionary_->end();
}

MetadataDictionary::ConstIterator MetadataDictionary::end() const {
  return dictionary_->end();
}

MetadataDictionary::Iterator MetadataDictionary::find(const std::string& key) {
  ensure_unique();
  return dictionary_->find(key);
}

MetadataDictionary::ConstIterator MetadataDictionary::find(const std::string& key) const {
  return dictionary_->find(key);
}

void MetadataDictionary::clear() {
  // Construct a new one instead of enforcing uniqueness then clearing
  this->dictionary_ = std::make_shared<MapType>();
}

void MetadataDictionary::swap(MetadataDictionary& other) {
  using std::swap;
  swap(dictionary_, other.dictionary_);
}

void MetadataDictionary::merge(MetadataDictionary& other) {
  dictionary_->merge(*(other.dictionary_));
}

void MetadataDictionary::insert(MetadataDictionary& other) {
  dictionary_->insert(other.dictionary_->begin(), other.dictionary_->end());
}

void MetadataDictionary::update(MetadataDictionary& other) {
  switch (policy_) {
    case MetadataPolicy::kReject:
      insert(other);
      break;
    case MetadataPolicy::kInplaceUpdate:
    case MetadataPolicy::kUpdate:
      for (const auto& [key, value] : *other.dictionary_) {
        dictionary_->insert_or_assign(key, value);
      }
      break;
    case MetadataPolicy::kRaise:
      // rely on inplace implementation in set
      for (const auto& [key, value] : *other.dictionary_) { set(key, value); }
      break;
    default:
      // Handle unknown policy
      throw std::runtime_error("Unknown metadata policy");
  }
}

bool MetadataDictionary::ensure_unique() {
  if (dictionary_.use_count() > 1) {
    // copy the shared dictionary.
    dictionary_ = std::make_shared<MapType>(*dictionary_);
    return true;
  }
  return false;
}

bool MetadataDictionary::erase(const std::string& key) {
  auto it = dictionary_->find(key);
  const MapType::iterator end = dictionary_->end();

  if (it != end) {
    if (ensure_unique()) {
      // Need to find the correct iterator, in the new copy
      it = dictionary_->find(key);
    }
    dictionary_->erase(it);
    return true;
  }
  return false;
}

}  // namespace holoscan
