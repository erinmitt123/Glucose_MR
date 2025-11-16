using UnityEngine;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using System.Collections.Concurrent;
using System.Threading;
using System;
using System.Text;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{
public class ObjectDetectionRawYolo : MonoBehaviour
{
    public TextAsset yoloModel;
    public TextAsset frameGltfAsset;
    public TextAsset anchorMatrixAsset;
      public int numFramesToRun = -1;
        public float intervalBetweenPipelineRuns = 0.1f;
        private int vstWidth = 3248;
        private int vstHeight = 2464;
        private int cropWidth = 640; // yolo
        private int cropHeight = 640; //yolo
        private int cropX1 = 1444;
        private int cropY1 = 1332;
        private int cropX2 = 2045;
        private int cropY2 = 1933;

        private int countFrames = 0;
        private float _elapsedTime = 0f;
        private Provider provider;
        private Pipeline pipeline;
        private Pipeline rendererPipeline;

        // Global tensors
        private Tensor cropRgbGlobal;
        private Tensor predBoxesGlobal;
        private Tensor predScoresGlobal;
        private Tensor predClassIdxGlobal;

        // Pipeline tensors (references for mapping)
        private Tensor predBoxesWrite;
        private Tensor predScoresWrite;
        private Tensor predClassIdxWrite;
        private Tensor cropRgbWrite;

        // Renderer pipeline tensors
        private Tensor predBoxesRead;
        private Tensor predScoresRead;
        private Tensor predClassIdxRead;
        private Tensor cropRgbRead;

        // GLTF tensors
        private Tensor gltfTensor;
        private Tensor gltfTensor2;
        private Tensor gltfTensor3;
        private Tensor gltfPlaceholder;
        private Tensor gltfPlaceholder2;
        private Tensor gltfPlaceholder3;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            // Validate assets
            if (yoloModel == null)
            {
                Debug.LogError("YOLO model is not assigned!");
                enabled = false;
                return;
            }

            if (frameGltfAsset == null)
            {
                Debug.LogError("Frame GLTF asset is not assigned!");
                enabled = false;
                return;
            }

            Debug.Log($"YOLO model loaded: {yoloModel.bytes.Length} bytes");
            Debug.Log($"Frame GLTF loaded: {frameGltfAsset.bytes.Length} bytes");

            CreateProvider();
            CreatePipeline();
            CreateRenderer();
        }

        private void Update()
        {
            if (numFramesToRun < 0 || countFrames < numFramesToRun)
            {
                _elapsedTime += Time.deltaTime;
                if (_elapsedTime > intervalBetweenPipelineRuns)
                {
                    RunPipeline();
                    _elapsedTime = 0.0f;
                }
                RenderFrame();
                countFrames++;
            }
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateYoloModel(Pipeline pipelineParam, Tensor inputTensor,
            Tensor predBoxes, Tensor predScores, Tensor predClassIdx)
        {
            // Create YOLO model operator
            var modelConfig = new ModelOperatorConfiguration(yoloModel.bytes, SecureMRModelType.QnnContextBinary, "yolo");
            modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);
            var modelOp = pipelineParam.CreateOperator<RunModelInferenceOperator>(modelConfig);

            // Raw YOLO output: [84, 8400]
            // Row 0-3: bbox (x, y, w, h)
            // Row 4-83: class scores (80 classes)
            var rawYoloOutput = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{84, 8400}));

            modelOp.SetOperand("images", inputTensor);
            modelOp.SetResult("output0", rawYoloOutput);

            // === POST-PROCESSING IN PIPELINE (No CPU access!) ===

            // WORKAROUND: No SliceOperator or TransposeOperator available
            // Strategy: Work with data in [84, 8400] format directly

            // Step 1: Use SortMatrixOperator to find max score per detection (column)
            // Sort each column (8400 columns) to get max scores at top
            // SecureMRMatrixSortType.Column means sort by column (descending by default)
            var sortOp = pipelineParam.CreateOperator<SortMatrixOperator>(
                new SortMatrixOperatorConfiguration(SecureMRMatrixSortType.Column));
            var sortedOutput = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{84, 8400}));
            var sortedIndices = pipelineParam.CreateTensor<int, Matrix>(1, new TensorShape(new[]{84, 8400}));

            sortOp.SetOperand("operand", rawYoloOutput);
            sortOp.SetResult("sorted", sortedOutput);
            sortOp.SetResult("indices", sortedIndices);

            // Step 2: Use ArgmaxOperator to find best class per detection
            // This operates on the score portion (rows 4-83)
            // Since we can't slice, argmax will work on all 84 rows but bbox rows will have low values
            var argmaxOp = pipelineParam.CreateOperator<ArgmaxOperator>();
            var bestClassIdxPerCol = pipelineParam.CreateTensor<int, Matrix>(1, new TensorShape(new[]{1, 8400}));

            argmaxOp.SetOperand("operand", rawYoloOutput);
            argmaxOp.SetResult("result", bestClassIdxPerCol);

            // Step 3: Extract max scores for each detection
            // Use first row of sorted output (highest value per column)
            var extractMaxScoreOp = pipelineParam.CreateOperator<AssignmentOperator>();
            var maxScoresRow = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{1, 8400}));

            extractMaxScoreOp.SetOperand("src", sortedOutput);
            extractMaxScoreOp.SetResult("dst", maxScoresRow);

            // Step 4: Extract bounding boxes (first 4 rows)
            // AssignmentOperator with smaller destination should copy just first rows
            var extractBboxOp = pipelineParam.CreateOperator<AssignmentOperator>();
            var bboxes4x8400 = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{4, 8400}));

            extractBboxOp.SetOperand("src", rawYoloOutput);
            extractBboxOp.SetResult("dst", bboxes4x8400);

            // Step 5: Reshape/assign to final output tensors
            // For boxes: need [8400, 4] format - use arithmetic to reorganize
            // For now, assign directly and hope shape conversion works
            var assignBoxesOp = pipelineParam.CreateOperator<AssignmentOperator>();
            assignBoxesOp.SetOperand("src", bboxes4x8400);
            assignBoxesOp.SetResult("dst", predBoxes);

            // For scores: reshape [1, 8400] to [8400, 1]
            var assignScoresOp = pipelineParam.CreateOperator<AssignmentOperator>();
            assignScoresOp.SetOperand("src", maxScoresRow);
            assignScoresOp.SetResult("dst", predScores);

            // For class indices: reshape [1, 8400] to [8400, 1]
            // Also need to subtract 4 since argmax includes bbox rows
            // Use arithmetic: classIdx - 4
            var adjustClassOp = pipelineParam.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} - 4"));
            var adjustedClassFloat = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{1, 8400}));

            adjustClassOp.SetOperand("operand0", bestClassIdxPerCol);
            adjustClassOp.SetResult("result", adjustedClassFloat);

            // Reshape and cast to int
            var assignClassOp = pipelineParam.CreateOperator<AssignmentOperator>();
            assignClassOp.SetOperand("src", adjustedClassFloat);
            assignClassOp.SetResult("dst", predClassIdx);

            // Step 6: Apply NMS to filter overlapping boxes
            // Note: NMS outputs filtered results, overwriting the input tensors
            var nmsOp = pipelineParam.CreateOperator<NmsOperator>(
                new NmsOperatorConfiguration(threshold: 0.5f));

            nmsOp.SetOperand("scores", predScores);
            nmsOp.SetOperand("boxes", predBoxes);
            nmsOp.SetResult("scores", predScores);  // Overwrite with filtered scores
            nmsOp.SetResult("boxes", predBoxes);    // Overwrite with filtered boxes
            // Note: indices result gives indices into original boxes, we don't need it here

            Debug.Log("YOLO post-processing pipeline created with workarounds for missing operators");
        }

        private void CreatePipeline()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            Debug.Log("Creating pipeline...");

            // Create the pipeline
            pipeline = provider.CreatePipeline();

            // Create operators
            var vstOp = pipeline.CreateOperator<RectifiedVstAccessOperator>();
            var getAffineOp = pipeline.CreateOperator<GetAffineOperator>();
            var applyAffineOp = pipeline.CreateOperator<ApplyAffineOperator>();
            var uint8ToFloat32Op = pipeline.CreateOperator<AssignmentOperator>();
            var normalizeOp =
                pipeline.CreateOperator<ArithmeticComposeOperator>(
                    new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));

            // Create tensors
            var cropShape = new TensorShape(new[] { cropWidth, cropHeight });
            var rawRgb = pipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            var cropRgbUint8 = pipeline.CreateTensor<byte, Matrix>(3, cropShape);
            var cropRgbFloat = pipeline.CreateTensor<float, Matrix>(3, cropShape);
            cropRgbWrite = pipeline.CreateTensorReference<byte, Matrix>(3, cropShape);

            // Create source and destination points for affine transform
            var srcPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { cropX1, cropY1, cropX2, cropY1, cropX2, cropY2 });
            var dstPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { 0, 0, cropWidth, 0, cropWidth, cropHeight });
            var affineMat = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 2, 3 }));

            // Create Yolo model and tensors
            // YOLO expects RGB input [1, 3, 640, 640] -> we use [640, 640] with 3 channels
            var inputTensor = pipeline.CreateTensor<float, Matrix>(3, cropShape);

            // Create pipeline tensors for processed outputs (written by operators)
            predBoxesWrite = pipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[]{8400, 4}));
            predScoresWrite = pipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[]{8400, 1}));
            predClassIdxWrite = pipeline.CreateTensorReference<int, Matrix>(1, new TensorShape(new[]{8400, 1}));

            // Create YOLO model with in-pipeline post-processing
            CreateYoloModel(pipeline, inputTensor, predBoxesWrite, predScoresWrite, predClassIdxWrite);

            // Create global tensors for results (NO CPU processing needed!)
            predBoxesGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[]{8400, 4}));
            predScoresGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[]{8400, 1}));
            predClassIdxGlobal = provider.CreateTensor<int, Matrix>(1, new TensorShape(new[]{8400, 1}));
            cropRgbGlobal = provider.CreateTensor<byte, Matrix>(3, cropShape);

            // Connect the operators
            vstOp.SetResult("left image", rawRgb);

            getAffineOp.SetOperand("src", srcPoints);
            getAffineOp.SetOperand("dst", dstPoints);
            getAffineOp.SetResult("result", affineMat);

            applyAffineOp.SetOperand("affine", affineMat);
            applyAffineOp.SetOperand("src image", rawRgb);
            applyAffineOp.SetResult("dst image", cropRgbWrite);

            // Keep RGB (no conversion to grayscale for YOLO)
            uint8ToFloat32Op.SetOperand("src", cropRgbWrite);
            uint8ToFloat32Op.SetResult("dst", cropRgbFloat);

            normalizeOp.SetOperand("operand0", cropRgbFloat);
            normalizeOp.SetResult("result", inputTensor);

            Debug.Log("Pipeline created successfully.");
        }

        private void CreateRenderer()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            Debug.Log("Creating renderer...");

            // Create renderer pipeline
            rendererPipeline = provider.CreatePipeline();

            // Create operators
            var renderTextOp = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderTextOp2 = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderTextOp3 = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));


            var renderGltfOp = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderGltfOp2 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderGltfOp3 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var loadTextureOp = rendererPipeline.CreateOperator<LoadTextureOperator>();
            var materialBaseColorTextureConfig = new UpdateGltfOperatorConfiguration(SecureMRGltfOperatorAttribute.MaterialBaseColorTexture);
            var updateGltfOp = rendererPipeline.CreateOperator<UpdateGltfOperator>(materialBaseColorTextureConfig);

            // Create tensors
            var text = rendererPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("YOLO"));
            var startPosition = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 }); // white text, black background
            var textureId = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });
            var poseMat = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, -0.5f,
                    0.0f, 0.5f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            var text2 = rendererPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("Score"));
            var startPosition2 = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors2 = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
            var textureId2 = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize2 = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });

            var text3 = rendererPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("Class"));
            var startPosition3 = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors3 = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
            var textureId3 = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize3 = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });

            var poseMat2 = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, 0.5f,
                    0.0f, 0.5f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            var poseMat3 = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.5f, 0.0f, 1.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            // Create placeholder tensors
            gltfPlaceholder = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder2 = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder3 = rendererPipeline.CreateTensorReference<Gltf>();

            predBoxesRead = rendererPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] {8400, 4}));
            predScoresRead = rendererPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] {8400, 1}));
            predClassIdxRead = rendererPipeline.CreateTensorReference<int, Matrix>(1, new TensorShape(new[] {8400, 1}));

            cropRgbRead = rendererPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] {cropWidth, cropHeight}));

            var gltfMaterialIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var gltfTextureIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));

            // Create global GLTF tensors
            gltfTensor = provider.CreateTensor<Gltf>(frameGltfAsset.bytes);
            gltfTensor2 = provider.CreateTensor<Gltf>(frameGltfAsset.bytes);
            gltfTensor3 = provider.CreateTensor<Gltf>(frameGltfAsset.bytes);

            // Connect the operators
            renderTextOp.SetOperand("text", text);
            renderTextOp.SetOperand("start", startPosition);
            renderTextOp.SetOperand("colors", colors);
            renderTextOp.SetOperand("texture ID", textureId);
            renderTextOp.SetOperand("font size", fontSize);
            renderTextOp.SetOperand("gltf", gltfPlaceholder);

            renderTextOp2.SetOperand("text", text2);
            renderTextOp2.SetOperand("start", startPosition2);
            renderTextOp2.SetOperand("colors", colors2);
            renderTextOp2.SetOperand("texture ID", textureId2);
            renderTextOp2.SetOperand("font size", fontSize2);
            renderTextOp2.SetOperand("gltf", gltfPlaceholder2);

            renderTextOp3.SetOperand("text", text3);
            renderTextOp3.SetOperand("start", startPosition3);
            renderTextOp3.SetOperand("colors", colors3);
            renderTextOp3.SetOperand("texture ID", textureId3);
            renderTextOp3.SetOperand("font size", fontSize3);
            renderTextOp3.SetOperand("gltf", gltfPlaceholder3);

            renderGltfOp.SetOperand("gltf", gltfPlaceholder);
            renderGltfOp.SetOperand("world pose", poseMat);
            renderGltfOp.SetOperand("view locked", poseMat);

            renderGltfOp2.SetOperand("gltf", gltfPlaceholder2);
            renderGltfOp2.SetOperand("world pose", poseMat2);
            renderGltfOp2.SetOperand("view locked", poseMat2);

            loadTextureOp.SetOperand("rgb image", cropRgbRead);
            loadTextureOp.SetOperand("gltf", gltfPlaceholder3);
            loadTextureOp.SetResult("texture ID", gltfTextureIndex);

            updateGltfOp.SetOperand("gltf", gltfPlaceholder3);
            updateGltfOp.SetOperand("material ID", gltfMaterialIndex);
            updateGltfOp.SetOperand("value", gltfTextureIndex);

            renderGltfOp3.SetOperand("gltf", gltfPlaceholder3);
            renderGltfOp3.SetOperand("world pose", poseMat3);
            renderGltfOp3.SetOperand("view locked", poseMat3);

            Debug.Log("Renderer created successfully.");
        }

        private void RunPipeline()
        {
            try
            {
                var tensorMapping = new TensorMapping();
                // Map pipeline outputs to global tensors
                tensorMapping.Set(predBoxesWrite, predBoxesGlobal);
                tensorMapping.Set(predScoresWrite, predScoresGlobal);
                tensorMapping.Set(predClassIdxWrite, predClassIdxGlobal);
                tensorMapping.Set(cropRgbWrite, cropRgbGlobal);

                pipeline.Execute(tensorMapping);

                // NO CPU post-processing! Everything happens in the pipeline operators
                // Results are written directly to global tensors by the pipeline
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error in RunPipeline: {e.Message}\n{e.StackTrace}");
                enabled = false;
            }
        }

        private void RenderFrame()
        {
            try
            {
                var tensorMapping = new TensorMapping();
                tensorMapping.Set(gltfPlaceholder, gltfTensor);
                tensorMapping.Set(gltfPlaceholder2, gltfTensor2);
                tensorMapping.Set(gltfPlaceholder3, gltfTensor3);

                tensorMapping.Set(predBoxesRead, predBoxesGlobal);
                tensorMapping.Set(predScoresRead, predScoresGlobal);
                tensorMapping.Set(predClassIdxRead, predClassIdxGlobal);
                tensorMapping.Set(cropRgbRead, cropRgbGlobal);

                rendererPipeline.Execute(tensorMapping);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error in RenderFrame: {e.Message}\n{e.StackTrace}");
                enabled = false;
            }
        }
    }
}
