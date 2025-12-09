// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ADAPTER_H
#define ADAPTER_H

#include <iostream>
#include "openxr/openxr.h"
#include "session.h"

namespace SecureMR {

/**
 * Adapter for any OpenXR handle, to support copy, auto boxing/unboxing
 * and auto destruct
 *
 * @tparam XR_HANDLE The XR handle to be wrapped
 */
template <typename XR_HANDLE>
class XrHandleAdapter {
 protected:
  XR_HANDLE m_handle;

 public:
  explicit operator XR_HANDLE() const { return m_handle; }
  explicit XrHandleAdapter(XR_HANDLE rawHandle) : m_handle(rawHandle) {}
  XrHandleAdapter() : m_handle(XR_NULL_HANDLE) {}
  XrHandleAdapter(const XrHandleAdapter& other) = default;
  XrHandleAdapter(XrHandleAdapter&& other) = default;
  XrHandleAdapter& operator=(const XrHandleAdapter& other) = default;
  XrHandleAdapter& operator=(XrHandleAdapter&& other) = default;

  virtual ~XrHandleAdapter() = default;
};
}  // namespace SecureMR

#endif
