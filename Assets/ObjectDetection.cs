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
public class ObjectDetection : MonoBehaviour
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
        private Tensor rawYoloOutputGlobal;
        private Tensor cropRgbGlobal;

        // Processed result tensors (for rendering)
        private Tensor predBoxesGlobal;
        private Tensor predScoresGlobal;
        private Tensor predClassIdxGlobal;

        // Pipeline tensors
        private Tensor rawYoloOutputWrite;
        private Tensor cropRgbWrite;

        // Renderer pipeline tensors
        private Tensor predBoxesRead;
        private Tensor predScoresRead;
        private Tensor predClassIdxRead;
        private Tensor cropRgbRead;

        // Post-processed results (stored for rendering)
        private float[,] predBoxesArray;
        private float[] predScoresArray;
        private int[] predClassIdxArray;

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

        private void CreateYoloModel(Pipeline pipelineParam, Tensor inputTensor, Tensor rawYoloOutput)
        {
            // Create YOLO model operator
            var modelConfig = new ModelOperatorConfiguration(yoloModel.bytes, SecureMRModelType.QnnContextBinary, "yolo");
            modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);
            var modelOp = pipelineParam.CreateOperator<RunModelInferenceOperator>(modelConfig);

            modelOp.SetOperand("images", inputTensor);
            modelOp.SetResult("output0", rawYoloOutput);
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

            // Raw YOLO output: [1, 84, 8400] - but drop batch dimension -> [84, 8400]
            rawYoloOutputWrite = pipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[]{84, 8400}));

            CreateYoloModel(pipeline, inputTensor, rawYoloOutputWrite);

            // Create global tensor for raw output
            rawYoloOutputGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[]{84, 8400}));
            cropRgbGlobal = provider.CreateTensor<byte, Matrix>(3, cropShape);

            // Create global tensors for processed results (after CPU post-processing)
            predBoxesGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[]{8400, 4}));
            predScoresGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[]{8400, 1}));
            predClassIdxGlobal = provider.CreateTensor<int, Matrix>(1, new TensorShape(new[]{8400, 1}));

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

            normalizeOp.SetOperand("{0}", cropRgbFloat);
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
                tensorMapping.Set(rawYoloOutputWrite, rawYoloOutputGlobal);
                tensorMapping.Set(cropRgbWrite, cropRgbGlobal);

                pipeline.Execute(tensorMapping);

                // Post-process the raw YOLO output on CPU
                PostProcessYoloOutput();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error in RunPipeline: {e.Message}\n{e.StackTrace}");
                enabled = false;
            }
        }

        private void PostProcessYoloOutput()
        {
            // TODO: Read raw YOLO output from rawYoloOutputGlobal tensor
            // The tensor shape is [84, 8400] where:
            // - First 4 rows: bbox coordinates (x, y, w, h)
            // - Last 80 rows: class scores for COCO classes

            Debug.Log("PostProcessYoloOutput: Reading raw YOLO tensor data...");

            // For now, just log that we got the output
            // In a real implementation, you would:
            // 1. Read the tensor data using GetData() or similar
            // 2. Extract bbox and scores
            // 3. Find argmax for each detection
            // 4. Apply confidence threshold
            // 5. Apply NMS (Non-Maximum Suppression)

            // Initialize arrays if needed
            if (predBoxesArray == null)
            {
                predBoxesArray = new float[8400, 4];
                predScoresArray = new float[8400];
                predClassIdxArray = new int[8400];
            }

            Debug.Log("PostProcessYoloOutput: Post-processing complete (placeholder)");
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
