## Sample: Food Detection

This sample builds an application for PICO using the SecureMR
APIs. The application demonstrates real-time food detection
using a YOLOX model converted from ONNX format.

This app detects only 10 food items: banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, and cake.

![Demo of YOLO making detections on objects on the desk](../../docs/Demo-YOLO.gif)

### Code Walk-through

The implementation follows a pipeline-based architecture with four main stages:

1. **VST Image Pipeline**
   - Handles camera feed processing
   - Converts raw camera input to appropriate formats (uint8 and fp32)
   - Manages camera timestamps and matrices

2. **Model Inference Pipeline**
   - Processes the prepared images through the YOLO model
   - Performs class selection and NMS (Non-Maximum Suppression)
   - Outputs bounding boxes and confidence scores

3. **2D to 3D Mapping Pipeline**
   - Maps 2D detection results to 3D space
   - Uses stereo camera information for depth estimation
   - Generates 3D coordinates for detected objects

4. **Rendering Pipeline**
   - Visualizes detection results in the XR environment
   - Renders bounding boxes and labels using GLTF assets
   - Handles scale and positioning of visual elements

### Key Components

The <mcsymbol name="YoloDetector" filename="yolo_object_detection.h" path="/Users/bytedance/Projects/SecureMR_Samples/samples/yolo_det/cpp/yolo_object_detection.h" startline="29" type="class"></mcsymbol> class manages:

- **Framework Setup**: Initializes SecureMR runtime and pipeline components
- **Pipeline Management**: Creates and coordinates four main processing pipelines
- **Data Flow**: Uses GlobalTensors for inter-pipeline communication
- **Resource Management**: Handles GLTF assets and tensor memory
- **Threading**: Implements multi-threaded pipeline execution

### Configuration

- Model path: `yolom.serialized.bin`
- GLTF asset path: `frame2.gltf`
- Detection threshold: `0.2` (lowered for better food detection sensitivity)
- Food classes filtered: Only 10 specific food items detected
- Supports both left and right camera inputs for stereo processing

### Build and Install

To build the project:
```bash
./gradlew :samples:food_detection:assembleDebug
```

To install on Pico device:
```bash
adb install -r samples/food_detection/build/outputs/apk/debug/food_detection-debug.apk
```

### Technical Details

The implementation uses SecureMR's pipeline architecture for efficient:
- Tensor operations
- Multi-threaded execution
- Hardware-accelerated processing
- XR session management
- Real-time visualization

Each pipeline can be monitored and controlled independently, with synchronized data flow managed through global tensors.
