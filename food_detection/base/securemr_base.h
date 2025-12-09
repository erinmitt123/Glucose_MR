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

#ifndef SECURE_MR_DEMOS_SECUREMR_BASE_H
#define SECURE_MR_DEMOS_SECUREMR_BASE_H

#include "pch.h"
#include <fstream>
#include <random>
#include "logger.h"
#include "common.h"

namespace SecureMR {

/**
 * Interface for SecureMR logic in each demo app. Each app <b>must</b> implement
 * this interface, which will be called from <code>base/openxr_program.cpp</code>.
 */
class ISecureMR {
 public:
  virtual void UpdateHandPose(const XrVector3f* leftHandDelta, const XrVector3f* rightHandDelta) {}

  virtual ~ISecureMR() = default;

  /**
   * This method will be called first, after the OpenXR instance and session are ready.
   *
   * The method's implementation is expected to complete the following tasks:
   * <ol>
   * <li>A Secure MR Framework handle which holds the MR resources and serves as a camera provider will be created
   * and</li> <li>The camera resolution shall be determined</li>
   * </ol>
   *
   * The calling of this method starts the lifecycle of a camera client.
   */
  virtual void CreateFramework() = 0;

  /**
   * This method will be called after <code>CreateFramework</code>.
   *
   * The method's implementation is expected to complete the following tasks:
   * <ol>
   * <li>Load assets and contents</li>
   * <li>Create global tensors and set content for them</li>
   * <li>Create pipelines where you may: </li>
   * <ol>
   * <li>Declare pipeline local tensors or placeholders inside pipelines</li>
   * <li>Add operators to pipelines to assemble MR logics</li>
   * </ol>
   * </ol>
   *
   * NOTE: As the creation of tensors/pipelines may be time consuming, you are
   * suggested to launch a secondary thread in this method to complete the
   * setup, avoiding blocking the caller thread (the OpenXR program's main thread)
   *
   * If you choose to hand over the burden of setting up to other threads, you
   * <b>must</b> maintain a cross-thread signal mechanism, so that any pipeline shall
   * not be executed unless its creation is finished by another thread.
   */
  virtual void CreatePipelines() = 0;

  /**
   * This method will be called <i>before</i> the OpenXR app's main loop, which starts
   * the submission of Secure MR pipelines created in <code>CreatePipelines</code>
   *
   * Note this method will only be called once. Hence, if you are considering running
   * some MR pipelines continuously, you <i>may</i> consider to start a new thread
   * in this method's implementation, which submits the desired pipelines timely.
   */
  virtual void RunPipelines() = 0;

  /**
   * This method is designed to indicate the OpenXR app's main loop whether the Secure MR
   * resources (framework, tensors, pipelines) are all ready. In our demos, a loading animation
   * is displayed until this method returns <code>true</code>.
   *
   * The method can be useful if you create a secondary thread in <code>CreatePipeline</code>
   * to take over the initialization task.
   *
   * @return True if all Secure MR resources are ready.
   */
  [[nodiscard]] virtual bool LoadingFinished() const = 0;
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session);

}  // namespace SecureMR

#endif  // SECURE_MR_DEMOS_SECUREMR_BASE_H
