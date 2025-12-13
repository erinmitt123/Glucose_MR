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

#include "rendercommand.h"

namespace SecureMR {

XrSecureMrOperatorPICO RenderCommand::createOperator(const XrSecureMrOperatorCreateInfoPICO* config) const {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  auto xrResult =
      pipeline->xrCreateSecureMrOperatorPICO(static_cast<XrSecureMrPipelinePICO>(*pipeline), config, &opHandle);
  CHECK_XRRESULT(xrResult, "xrCreateSecureMrOperatorPICO")
  setOperandByName(opHandle, gltfTensor, "gltf");
  return opHandle;
}

void RenderCommand::setOperandByName(XrSecureMrOperatorPICO opHandle, const std::shared_ptr<PipelineTensor>& tensor,
                                     const std::string& opName) const {
  if (tensor != nullptr) {
    CHECK_MSG(gltfTensor->getPipeline()->verifyPipelineTensor(tensor),
              "operand tensors for render command are not associated with the same pipeline of the target glTF "
              "placeholder tensor");
    auto result = gltfTensor->getPipeline()->xrSetSecureMrOperatorOperandByNamePICO(
        static_cast<XrSecureMrPipelinePICO>(*gltfTensor->getPipeline()), opHandle,
        static_cast<XrSecureMrPipelineTensorPICO>(*tensor), opName.c_str());
    CHECK_XRRESULT(result, Fmt("xrSetSecureMrOperatorOperandByNamePICO(..., %s)", opName.c_str()).c_str())
  }
}

void RenderCommand_Render::execute() {
  auto pipeline = gltfTensor->getPipeline();

  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, pose, "world pose");
  setOperandByName(op, viewLocked, "view locked");
  setOperandByName(op, visible, "visible");
}

void RenderCommand_UpdateTextures::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorUpdateGltfPICO updateConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO,
                                                .attribute = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_TEXTURE_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&updateConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, textureNewContents, "rgb image");
  setOperandByName(op, textureIds, "texture ID");
}

void RenderCommand_UpdateAnimation::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorUpdateGltfPICO updateConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO,
                                                .attribute = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_ANIMATION_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&updateConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, animationId, "animation ID");
  setOperandByName(op, animationTimer, "animation timer");
}

void RenderCommand_UpdatePose::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorUpdateGltfPICO updateConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO,
                                                .attribute = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_WORLD_POSE_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&updateConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);

  setOperandByName(op, newPose, "world pose");
}

void RenderCommand_UpdateNodesLocalPoses::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorUpdateGltfPICO updateConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO,
                                                .attribute = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_LOCAL_TRANSFORM_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&updateConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, nodeNewLocalPoses, "transform");
  setOperandByName(op, nodeIds, "node ID");
}

void RenderCommand_UpdateMaterial::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorUpdateGltfPICO updateConfig{
      .type = XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO,
      .attribute = static_cast<XrSecureMrGltfOperatorAttributePICO>(attribute)};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&updateConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, materialIds, "material ID");
  setOperandByName(op, materialValues, "value");
}

void RenderCommand_DrawText::execute() {
  auto pipeline = gltfTensor->getPipeline();
  XrSecureMrOperatorRenderTextPICO textConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_RENDER_TEXT_PICO,
                                              .typeface = static_cast<XrSecureMrFontTypefacePICO>(typeFace),
                                              .languageAndLocale = languageAndLocale.c_str(),
                                              .width = canvasWidth,
                                              .height = canvasHeight};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&textConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO,
  };
  auto op = createOperator(&operatorCreateInfo);
  setOperandByName(op, text, "text");
  setOperandByName(op, startPosition, "start");
  setOperandByName(op, colors, "colors");
  setOperandByName(op, textureId, "texture ID");
  setOperandByName(op, fontSize, "font size");
}
}  // namespace SecureMR
