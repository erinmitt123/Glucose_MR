using UnityEngine;
using System.IO;

/// <summary>
/// Local testing version of ObjectDetection that works without Pico device.
/// Loads images from disk and runs YOLO inference locally.
/// </summary>
public class ObjectDetectionLocal : MonoBehaviour
{
    [Header("Model Configuration")]
    public TextAsset yoloModel; // QNN model file

    [Header("Input Image")]
    public string testImagePath = "Assets/TestImages/test_image.jpg"; // Path to test image
    public Texture2D inputTexture; // Or assign directly in inspector

    [Header("Model Parameters")]
    private int inputWidth = 640;
    private int inputHeight = 640;

    [Header("Output Predictions")]
    public float[,] predBoxes; // [8400, 4] - bounding boxes (x, y, w, h)
    public float[,] predScores; // [8400, 1] - confidence scores
    public int[,] predClassIdx; // [8400, 1] - class indices

    [Header("Post-processing")]
    public float confidenceThreshold = 0.5f;
    public float nmsThreshold = 0.45f;

    private bool inferenceComplete = false;

    void Start()
    {
        LoadInputImage();
        RunInference();
    }

    /// <summary>
    /// Load image from file or use assigned texture
    /// </summary>
    private void LoadInputImage()
    {
        if (inputTexture == null && !string.IsNullOrEmpty(testImagePath))
        {
            // Load image from file
            if (File.Exists(testImagePath))
            {
                byte[] imageData = File.ReadAllBytes(testImagePath);
                inputTexture = new Texture2D(2, 2);
                inputTexture.LoadImage(imageData);
                Debug.Log($"Loaded image: {testImagePath} - Size: {inputTexture.width}x{inputTexture.height}");
            }
            else
            {
                Debug.LogError($"Image not found at: {testImagePath}");
                return;
            }
        }

        if (inputTexture != null)
        {
            Debug.Log($"Input texture loaded: {inputTexture.width}x{inputTexture.height}");
        }
    }

    /// <summary>
    /// Preprocess image for YOLO model
    /// </summary>
    private float[] PreprocessImage(Texture2D texture)
    {
        // Resize to 640x640
        Texture2D resized = ResizeTexture(texture, inputWidth, inputHeight);

        // Convert to grayscale and normalize
        Color[] pixels = resized.GetPixels();
        float[] processedImage = new float[inputWidth * inputHeight];

        for (int i = 0; i < pixels.Length; i++)
        {
            // Convert to grayscale (matching the pipeline: RGB to Gray)
            float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
            // Normalize to [0, 1]
            processedImage[i] = gray; // Already in [0,1] from Unity Color
        }

        Debug.Log($"Preprocessed image to {inputWidth}x{inputHeight} grayscale, normalized [0-1]");
        return processedImage;
    }

    /// <summary>
    /// Resize texture to target dimensions
    /// </summary>
    private Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
        RenderTexture.active = rt;

        Graphics.Blit(source, rt);
        Texture2D result = new Texture2D(targetWidth, targetHeight);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }

    /// <summary>
    /// Run YOLO inference (placeholder - needs actual model integration)
    /// </summary>
    private void RunInference()
    {
        if (inputTexture == null)
        {
            Debug.LogError("No input texture available for inference!");
            return;
        }

        Debug.Log("=== Starting YOLO Inference ===");

        // Preprocess image
        float[] preprocessedImage = PreprocessImage(inputTexture);

        // NOTE: This is where you'd integrate with actual YOLO model
        // For Pico's QNN model, you'd need to:
        // 1. Load the QNN context binary (yoloModel.bytes)
        // 2. Create inference session
        // 3. Feed preprocessedImage as input
        // 4. Get outputs: _571 (boxes), _530 (scores), _532 (class_idx)

        if (yoloModel != null)
        {
            Debug.Log($"Model loaded: {yoloModel.bytes.Length} bytes");
            Debug.Log("Model type: QNN Context Binary");
            Debug.Log("Input: 'image' (Float32, 640x640)");
            Debug.Log("Outputs:");
            Debug.Log("  - '_571' (Float32, [8400, 4]) -> Bounding boxes");
            Debug.Log("  - '_530' (Float32, [8400, 1]) -> Scores");
            Debug.Log("  - '_532' (Int8, [8400, 1]) -> Class indices");

            // TODO: Replace with actual inference
            RunMockInference(preprocessedImage);
        }
        else
        {
            Debug.LogWarning("YOLO model not assigned! Running mock inference...");
            RunMockInference(preprocessedImage);
        }
    }

    /// <summary>
    /// Mock inference for testing without actual model
    /// Replace this with real model inference
    /// </summary>
    private void RunMockInference(float[] inputData)
    {
        Debug.Log("Running MOCK inference (replace with actual model)");

        // Initialize output arrays
        predBoxes = new float[8400, 4];
        predScores = new float[8400, 1];
        predClassIdx = new int[8400, 1];

        // Generate some mock detections for testing
        int numMockDetections = 5;
        System.Random rand = new System.Random(42);

        for (int i = 0; i < numMockDetections; i++)
        {
            // Mock bounding boxes (x, y, w, h) in normalized coordinates
            predBoxes[i, 0] = (float)rand.NextDouble() * 640; // x
            predBoxes[i, 1] = (float)rand.NextDouble() * 640; // y
            predBoxes[i, 2] = 50 + (float)rand.NextDouble() * 100; // w
            predBoxes[i, 3] = 50 + (float)rand.NextDouble() * 100; // h

            // Mock scores
            predScores[i, 0] = 0.5f + (float)rand.NextDouble() * 0.5f; // 0.5 - 1.0

            // Mock class indices
            predClassIdx[i, 0] = rand.Next(0, 80); // COCO has 80 classes
        }

        // Rest are zeros/low confidence
        for (int i = numMockDetections; i < 8400; i++)
        {
            predScores[i, 0] = (float)rand.NextDouble() * 0.1f; // Low confidence
        }

        inferenceComplete = true;
        Debug.Log($"Mock inference complete! Generated {numMockDetections} detections");

        // Post-process and display results
        PostProcess();
    }

    /// <summary>
    /// Post-process predictions: filter by confidence and apply NMS
    /// </summary>
    private void PostProcess()
    {
        Debug.Log("=== Post-processing Results ===");

        int validDetections = 0;

        for (int i = 0; i < 8400; i++)
        {
            float score = predScores[i, 0];

            if (score > confidenceThreshold)
            {
                int classId = predClassIdx[i, 0];
                float x = predBoxes[i, 0];
                float y = predBoxes[i, 1];
                float w = predBoxes[i, 2];
                float h = predBoxes[i, 3];

                Debug.Log($"Detection {validDetections}: Class={classId}, Score={score:F3}, " +
                         $"BBox=({x:F1}, {y:F1}, {w:F1}, {h:F1})");

                validDetections++;
            }
        }

        Debug.Log($"Total valid detections (score > {confidenceThreshold}): {validDetections}");
    }

    /// <summary>
    /// Visualize detections on the image
    /// </summary>
    private void OnGUI()
    {
        if (inputTexture == null) return;

        GUILayout.BeginArea(new Rect(10, 10, 400, 600));
        GUILayout.Label($"Object Detection - Local Testing");
        GUILayout.Label($"Input: {inputTexture.width}x{inputTexture.height}");
        GUILayout.Label($"Model: {(yoloModel != null ? "Loaded" : "Not loaded")}");
        GUILayout.Label($"Inference: {(inferenceComplete ? "Complete" : "Pending")}");

        if (inferenceComplete)
        {
            GUILayout.Label($"\nPredictions Shape:");
            GUILayout.Label($"  Boxes: [{predBoxes.GetLength(0)}, {predBoxes.GetLength(1)}]");
            GUILayout.Label($"  Scores: [{predScores.GetLength(0)}, {predScores.GetLength(1)}]");
            GUILayout.Label($"  Classes: [{predClassIdx.GetLength(0)}, {predClassIdx.GetLength(1)}]");

            if (GUILayout.Button("Re-run Inference"))
            {
                inferenceComplete = false;
                RunInference();
            }
        }

        GUILayout.EndArea();

        // Draw input image
        if (inputTexture != null)
        {
            GUI.DrawTexture(new Rect(420, 10, 300, 300), inputTexture);
        }
    }

    /// <summary>
    /// Public method to run inference with a new image
    /// </summary>
    public void RunInferenceOnImage(Texture2D newImage)
    {
        inputTexture = newImage;
        inferenceComplete = false;
        RunInference();
    }

    /// <summary>
    /// Get predictions as structured data
    /// </summary>
    public (float[,] boxes, float[,] scores, int[,] classes) GetPredictions()
    {
        return (predBoxes, predScores, predClassIdx);
    }
}
