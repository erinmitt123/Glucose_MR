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

#include "session.h"
#include "check.h"

namespace SecureMR {

PFN_xrCreateSecureMrFrameworkPICO FrameworkSession::xrCreateSecureMrFrameworkPICO = nullptr;
PFN_xrDestroySecureMrFrameworkPICO FrameworkSession::xrDestroySecureMrFrameworkPICO = nullptr;
PFN_xrDestroySecureMrTensorPICO FrameworkSession::xrDestroySecureMrTensorPICO = nullptr;

FrameworkSession::FrameworkSession(const XrInstance& instance, const XrSession& rootSession, int width, int height)
    : m_instance(instance), m_session(rootSession) {
  try {
    if (xrCreateSecureMrFrameworkPICO == nullptr) {
      xrCreateSecureMrFrameworkPICO =
          getAPIFromXrInstance<PFN_xrCreateSecureMrFrameworkPICO>("xrCreateSecureMrFrameworkPICO");
      if (xrCreateSecureMrFrameworkPICO == nullptr) {
        throw std::runtime_error("Failed to get xrCreateSecureMrFrameworkPICO");
      }
    }
    if (xrDestroySecureMrFrameworkPICO == nullptr) {
      xrDestroySecureMrFrameworkPICO =
          getAPIFromXrInstance<PFN_xrDestroySecureMrFrameworkPICO>("xrDestroySecureMrFrameworkPICO");
      if (xrDestroySecureMrFrameworkPICO == nullptr) {
        throw std::runtime_error("Failed to get xrDestroySecureMrFrameworkPICO");
      }
    }
    if (xrDestroySecureMrTensorPICO == nullptr) {
      xrDestroySecureMrTensorPICO =
          getAPIFromXrInstance<PFN_xrDestroySecureMrTensorPICO>("xrDestroySecureMrTensorPICO");
      if (xrDestroySecureMrTensorPICO == nullptr) {
        throw std::runtime_error("Failed to get xrDestroySecureMrTensorPICO");
      }
    }

    XrSecureMrFrameworkCreateInfoPICO createInfo{
        .type = XR_TYPE_SECURE_MR_FRAMEWORK_CREATE_INFO_PICO,
        .width = width,
        .height = height,
    };
    auto result = xrCreateSecureMrFrameworkPICO(m_session, &createInfo, &m_frameworkSession);
    CHECK_XRRESULT(result, "xrCreateSecureMrFrameworkPICO(...)");
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("Exception during FrameworkSession construction: %s", e.what()));
    throw;  // Re-throw the exception to indicate failure
  }
}

FrameworkSession::~FrameworkSession() {
  xrDestroySecureMrFrameworkPICO(m_frameworkSession);
  m_frameworkSession = XR_NULL_HANDLE;
}
}  // namespace SecureMR
