# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create a CMake cache variable named ${lib}_DIR and ensure it is set
# so it can be added to `CMAKE_PREFIX_PATH` to search for the package
macro(define_search_dir_for lib description)
  set(${lib}_DIR CACHE PATH ${description})
  if(EXISTS ${${lib}_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${${lib}_DIR})
  else()
    message(FATAL_ERROR "${description} (${lib}_DIR) is not defined")
  endif()
endmacro()