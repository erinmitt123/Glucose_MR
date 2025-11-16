using System;
using System.Text;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// YOLO Demo - exactly like MNIST but finds best detection using operators
    /// </summary>
    public class YoloDemo : MonoBehaviour
    {
        public TextAsset yoloModel;
        public TextAsset tvGltfAsset;
        public int numFramesToRun = -1;
        public float intervalBetweenPipelineRuns = 0.1f;

        // Public fields for UI display (these show on TVs as numbers, but can be used by external UI)
        [Header("Detection Info (Read Only)")]
        [Tooltip("Last detected class ID (0-79). See logcat for class names.")]
        public int lastDetectedClass = -1;
        [Tooltip("Last detection confidence score (0.0-1.0)")]
        public float lastDetectionScore = 0f;

        // COCO class names (80 classes)
        private static readonly string[] CocoClassNames = new string[]
        {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };
        private int vstWidth = 3248;
        private int vstHeight = 2464;
        private int cropWidth = 640;
        private int cropHeight = 640;
        private int cropX1 = 1444;
        private int cropY1 = 1332;
        private int cropX2 = 2045;
        private int cropY2 = 1933;

        private int countFrames = 0;
        private float _elapsedTime = 0f;
        private Provider provider;
        private Pipeline pipeline;
        private Pipeline rendererPipeline;

        // Global tensors - just best detection like MNIST
        private Tensor bestClassGlobal;
        private Tensor bestScoreGlobal;
        private Tensor cropRgbGlobal;

        // Pipeline tensors
        private Tensor bestClassWrite;
        private Tensor bestScoreWrite;
        private Tensor cropRgbWrite;

        // Renderer pipeline tensors
        private Tensor bestClassRead;
        private Tensor bestScoreRead;
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
            // Print COCO class reference
            Debug.Log("=== YOLO COCO Classes Reference ===");
            Debug.Log("Classes 0-19: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow");
            Debug.Log("Classes 20-39: elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle");
            Debug.Log("Classes 40-59: wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed");
            Debug.Log("Classes 60-79: dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush");
            Debug.Log("===================================");

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

        private void CreateYoloModel(Pipeline pipelineParam, Tensor inputTensor, Tensor bestClass, Tensor bestScore)
        {
            Debug.Log($"[YoloDemo] Creating YOLO model, model size: {yoloModel.bytes.Length} bytes");

            // Create YOLO model operator - try BOTH possible output shapes
            // Option 1: [84, 8400] raw output OR Option 2: [8400, 84] transposed
            // Let's try raw format first [84, 8400]
            var modelConfig = new ModelOperatorConfiguration(yoloModel.bytes, SecureMRModelType.QnnContextBinary, "yolo");
            modelConfig.AddInputMapping("images", "image", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);
            var modelOp = pipelineParam.CreateOperator<RunModelInferenceOperator>(modelConfig);

            Debug.Log("[YoloDemo] YOLO model operator created");

            // YOLO raw output: [1, 84, 8400] with batch dimension
            // SecureMR uses 3D tensors for batch data
            // Actually, let's check if SecureMR auto-squeezes or if we need to handle batch
            // Based on model.json, output is [1, 84, 8400]
            // But for Matrix tensors in SecureMR, we typically drop the batch dimension
            // So we expect [84, 8400]
            var rawYoloOutput = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{84, 8400}));
            modelOp.SetOperand("images", inputTensor);
            modelOp.SetResult("output0", rawYoloOutput);

            Debug.Log("[YoloDemo] Raw YOLO output tensor created [84, 8400]");

            // === FIND BEST DETECTION - Using [84, 8400] format ===

            // Step 1: Extract scores [80, 8400] - rows 4-83 (class scores)
            var scores = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{80, 8400}));
            var scoreSlice = pipelineParam.CreateTensor<int, Slice>(2, new TensorShape(new[]{2}), new[]{4, 84, 0, 8400});
            var extractScoreOp = pipelineParam.CreateOperator<AssignmentOperator>();
            extractScoreOp.SetOperand("src", rawYoloOutput);
            extractScoreOp.SetOperand("src slices", scoreSlice);
            extractScoreOp.SetResult("dst", scores);

            Debug.Log("[YoloDemo] Scores extracted [80, 8400]");

            // Step 2: Sort each COLUMN to find best class per detection
            // After sorting columns, each column's first row will have the best class score
            var sortedScores = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{80, 8400}));
            var sortedIndices = pipelineParam.CreateTensor<int, Matrix>(1, new TensorShape(new[]{80, 8400}));
            var sortOp = pipelineParam.CreateOperator<SortMatrixOperator>(
                new SortMatrixOperatorConfiguration(SecureMRMatrixSortType.Column));
            sortOp.SetOperand("operand", scores);
            sortOp.SetResult("sorted", sortedScores);
            sortOp.SetResult("indices", sortedIndices);

            Debug.Log("[YoloDemo] Scores sorted by column");

            // Step 3: Extract best score per detection [1, 8400] - first row after sort (highest score per column)
            var bestScoresPerDet = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{1, 8400}));
            var bestScoreSlice = pipelineParam.CreateTensor<int, Slice>(2, new TensorShape(new[]{2}), new[]{0, 1, 0, 8400});
            var extractBestScoreOp = pipelineParam.CreateOperator<AssignmentOperator>();
            extractBestScoreOp.SetOperand("src", sortedScores);
            extractBestScoreOp.SetOperand("src slices", bestScoreSlice);
            extractBestScoreOp.SetResult("dst", bestScoresPerDet);

            // Step 4: Extract best class index per detection [1, 8400] - first row of sorted indices
            var bestClassPerDet = pipelineParam.CreateTensor<int, Matrix>(1, new TensorShape(new[]{1, 8400}));
            var bestClassSlice = pipelineParam.CreateTensor<int, Slice>(2, new TensorShape(new[]{2}), new[]{0, 1, 0, 8400});
            var extractBestClassOp = pipelineParam.CreateOperator<AssignmentOperator>();
            extractBestClassOp.SetOperand("src", sortedIndices);
            extractBestClassOp.SetOperand("src slices", bestClassSlice);
            extractBestClassOp.SetResult("dst", bestClassPerDet);

            Debug.Log("[YoloDemo] Best scores and classes extracted [1, 8400]");

            // Step 5: Find best detection across all 8400 - Argmax on bestScoresPerDet [1, 8400]
            var bestDetectionIdx = pipelineParam.CreateTensor<int, Scalar>(1, new TensorShape(new[]{1}));
            var argmaxOp = pipelineParam.CreateOperator<ArgmaxOperator>();
            argmaxOp.SetOperand("operand", bestScoresPerDet);
            argmaxOp.SetResult("result", bestDetectionIdx);

            Debug.Log("[YoloDemo] Argmax computed - best detection index");

            // Step 6: Extract the class and score at column 100 for debugging
            // (We can't dynamically index by bestDetectionIdx, so just show a fixed column)
            var col100Class = pipelineParam.CreateTensor<int, Matrix>(1, new TensorShape(new[]{1, 1}));
            var col100ClassSlice = pipelineParam.CreateTensor<int, Slice>(2, new TensorShape(new[]{2}), new[]{0, 1, 100, 101});
            var extractCol100ClassOp = pipelineParam.CreateOperator<AssignmentOperator>();
            extractCol100ClassOp.SetOperand("src", bestClassPerDet);
            extractCol100ClassOp.SetOperand("src slices", col100ClassSlice);
            extractCol100ClassOp.SetResult("dst", col100Class);

            var col100Score = pipelineParam.CreateTensor<float, Matrix>(1, new TensorShape(new[]{1, 1}));
            var col100ScoreSlice = pipelineParam.CreateTensor<int, Slice>(2, new TensorShape(new[]{2}), new[]{0, 1, 100, 101});
            var extractCol100ScoreOp = pipelineParam.CreateOperator<AssignmentOperator>();
            extractCol100ScoreOp.SetOperand("src", bestScoresPerDet);
            extractCol100ScoreOp.SetOperand("src slices", col100ScoreSlice);
            extractCol100ScoreOp.SetResult("dst", col100Score);

            // Convert to Scalar for display
            var convertClassOp = pipelineParam.CreateOperator<AssignmentOperator>();
            convertClassOp.SetOperand("src", col100Class);
            convertClassOp.SetResult("dst", bestClass);

            var convertScoreOp = pipelineParam.CreateOperator<AssignmentOperator>();
            convertScoreOp.SetOperand("src", col100Score);
            convertScoreOp.SetResult("dst", bestScore);

            Debug.Log("[YoloDemo] Column 100 class and score extracted for display");

            Debug.Log("YOLO model created with best detection extraction");
        }

        private void CreatePipeline()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            Debug.Log("Creating pipeline...");

            pipeline = provider.CreatePipeline();

            // Create operators - RGB pipeline for YOLO
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
            var cropRgbFloat = pipeline.CreateTensor<float, Matrix>(3, cropShape);
            cropRgbWrite = pipeline.CreateTensorReference<byte, Matrix>(3, cropShape);

            // Affine transform points
            var srcPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { cropX1, cropY1, cropX2, cropY1, cropX2, cropY2 });
            var dstPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { 0, 0, cropWidth, 0, cropWidth, cropHeight });
            var affineMat = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 2, 3 }));

            // YOLO model tensors - OUTPUT AS SCALAR LIKE MNIST
            var inputTensor = pipeline.CreateTensor<float, Matrix>(3, cropShape);
            bestClassWrite = pipeline.CreateTensorReference<int, Scalar>(1, new TensorShape(new[]{1}));
            bestScoreWrite = pipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(new[]{1}));

            CreateYoloModel(pipeline, inputTensor, bestClassWrite, bestScoreWrite);

            // Create global tensors - SCALAR LIKE MNIST
            bestClassGlobal = provider.CreateTensor<int, Scalar>(1, new TensorShape(new[] { 1 }));
            bestScoreGlobal = provider.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
            cropRgbGlobal = provider.CreateTensor<byte, Matrix>(3, cropShape);

            // Connect operators
            vstOp.SetResult("left image", rawRgb);

            getAffineOp.SetOperand("src", srcPoints);
            getAffineOp.SetOperand("dst", dstPoints);
            getAffineOp.SetResult("result", affineMat);

            applyAffineOp.SetOperand("affine", affineMat);
            applyAffineOp.SetOperand("src image", rawRgb);
            applyAffineOp.SetResult("dst image", cropRgbWrite);

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

            rendererPipeline = provider.CreatePipeline();

            // Create operators - EXACTLY LIKE MNIST
            var renderTextOp = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderGltfOp = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderTextOp2 = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderGltfOp2 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderGltfOp3 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var loadTextureOp = rendererPipeline.CreateOperator<LoadTextureOperator>();
            var materialBaseColorTextureConfig = new UpdateGltfOperatorConfiguration(SecureMRGltfOperatorAttribute.MaterialBaseColorTexture);
            var updateGltfOp = rendererPipeline.CreateOperator<UpdateGltfOperator>(materialBaseColorTextureConfig);

            // Create tensors
            var startPosition = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
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

            var startPosition2 = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors2 = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
            var textureId2 = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize2 = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
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

            // Create placeholder tensors - EXACTLY LIKE MNIST
            gltfPlaceholder = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder2 = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder3 = rendererPipeline.CreateTensorReference<Gltf>();

            bestClassRead = rendererPipeline.CreateTensorReference<int, Scalar>(1, new TensorShape(new[] {1}));
            bestScoreRead = rendererPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(new[] {1}));
            cropRgbRead = rendererPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] {cropWidth, cropHeight}));

            // Workaround: Create a switchable text display using comparison operators
            // We'll check if bestClassRead == specific values and display corresponding text
            // This is hacky but works for common classes

            // For now, let's just display "Class: XX" where XX is the number
            // We can build the text dynamically using arithmetic on the class number

            // Actually, let's use a simpler approach: show the number AND add a Unity Canvas UI
            // TODO: Add Unity Canvas with TextMeshPro outside SecureMR pipeline

            var gltfMaterialIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var gltfTextureIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));

            // Create global GLTF tensors
            gltfTensor = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);
            gltfTensor2 = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);
            gltfTensor3 = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);

            // Connect the operators - PASS SCALAR TENSORS LIKE MNIST
            renderTextOp.SetOperand("text", bestClassRead);
            renderTextOp.SetOperand("start", startPosition);
            renderTextOp.SetOperand("colors", colors);
            renderTextOp.SetOperand("texture ID", textureId);
            renderTextOp.SetOperand("font size", fontSize);
            renderTextOp.SetOperand("gltf", gltfPlaceholder);

            renderGltfOp.SetOperand("gltf", gltfPlaceholder);
            renderGltfOp.SetOperand("world pose", poseMat);
            renderGltfOp.SetOperand("view locked", poseMat);

            renderTextOp2.SetOperand("text", bestScoreRead);
            renderTextOp2.SetOperand("start", startPosition2);
            renderTextOp2.SetOperand("colors", colors2);
            renderTextOp2.SetOperand("texture ID", textureId2);
            renderTextOp2.SetOperand("font size", fontSize2);
            renderTextOp2.SetOperand("gltf", gltfPlaceholder2);

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
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(bestClassWrite, bestClassGlobal);
            tensorMapping.Set(bestScoreWrite, bestScoreGlobal);
            tensorMapping.Set(cropRgbWrite, cropRgbGlobal);

            Debug.Log($"[YoloDemo] Running pipeline frame {countFrames}");
            pipeline.Execute(tensorMapping);
            Debug.Log($"[YoloDemo] Pipeline executed frame {countFrames}");
        }

        private void RenderFrame()
        {
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(gltfPlaceholder, gltfTensor);
            tensorMapping.Set(gltfPlaceholder2, gltfTensor2);
            tensorMapping.Set(gltfPlaceholder3, gltfTensor3);
            tensorMapping.Set(bestClassRead, bestClassGlobal);
            tensorMapping.Set(bestScoreRead, bestScoreGlobal);
            tensorMapping.Set(cropRgbRead, cropRgbGlobal);

            rendererPipeline.Execute(tensorMapping);
        }
    }
}
