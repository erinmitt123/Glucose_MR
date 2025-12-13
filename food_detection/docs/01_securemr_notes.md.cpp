/*
 * This file is a collection of securemr code snippets.
 *
 * How to setup a pipeline? There are three steps:
 *
 * 	1. Define operator (both for operator and input/output tensor)
 *  2. If you have multiple operators, link them correctly by setting input/output tensor.
 *  3. Setup placeholder for pipeline, and feed placeholder with actual tensor..
 *
 * That's all!
 */

/* Create a uint8_ch3_mat.*/
// [OpenMR]
std::vector<int> shape{1024, 1024};
constexpr int uint8_ch3_mat =
    int(Engine::EDataType::UINT8) | Engine::BaseType::MAT | (Engine::BaseType::CHANNEL_MASK & 3);
std::shared_ptr<Engine::Tensor> tensor = Engine::TensorFactory::create(shape.begin(), shape.end(), uint8_ch3_mat);
// [Native Api]
XrSecureMrPipelineTensorPICO tensor;
int32_t shape[2] = {1024, 1024};
XrSecureMrTensorFormatPICO tensorFormat{XR_SECURE_MR_TENSOR_BIT_TYPE_MAT_PICO, 3,
                                        XR_SECURE_MR_TENSOR_BIT_TYPE_UINT8_PICO};
XrSecureMrShapeTensorCreateInfoPICO createInfo{
    XR_TYPE_SECURE_MR_SHAPE_TENSOR_CREATE_INFO_PICO, nullptr, false, 2, shape, &tensorFormat};
CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
    m_secureMrPipeline, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tensor));

/* Create a int32_vec4 */
// [OpenMR]
std::vector<int> shape{1};
constexpr int int32_vec4 = int(Engine::EDataType::INT32) | Engine::BaseType::VEC_4;
std::shared_ptr<Engine::Tensor> timestamp = Engine::TensorFactory::create(shape.begin(), shape.end(), int32_vec4);
// [Native Api]
XrSecureMrPipelineTensorPICO tsTensor;
int32_t shape[1] = {1};
XrSecureMrTensorFormatPICO tensorFormat{XR_SECURE_MR_TENSOR_BIT_TYPE_VECTOR_PICO, 4,
                                        XR_SECURE_MR_TENSOR_BIT_TYPE_INT32_PICO};
XrSecureMrShapeTensorCreateInfoPICO createInfo{
    XR_TYPE_SECURE_MR_SHAPE_TENSOR_CREATE_INFO_PICO, nullptr, false, 1, shape, &tensorFormat};
CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
    m_secureMrPipeline, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&tensorCreateInfo), &tsTensor));

/* Bind tensor to operator input or output. */
// [OpenMR]
op->dataAsOperand(inputTensor, 0);
op->connectResultToDataArray(0, outputTensor);
// [Native Api]
CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrPipeline, op, inputTensor, "input_rgb"));
CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrPipeline, op, outputTensor, "left image"));
CHECK_XRCMD(xrSetSecureMrOperatorOperandByIndexPICO(m_secureMrPipeline, op, inputTensor, 0));
CHECK_XRCMD(xrSetSecureMrOperatorResultByIndexPICO(m_secureMrPipeline, op, outputTensor, 0));
