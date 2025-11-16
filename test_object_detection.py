"""
Local Object Detection Testing Script
Mimics the ObjectDetection.cs pipeline for testing without Pico device
"""

import numpy as np
import cv2
from PIL import Image
import os


class ObjectDetectionLocal:
    """
    Local testing version of ObjectDetection that works without Pico device.
    Matches the preprocessing pipeline from ObjectDetection.cs
    """

    def __init__(self, model_path=None):
        """
        Initialize object detection

        Args:
            model_path: Path to YOLO model file (QNN context binary)
        """
        self.model_path = model_path
        self.input_width = 640
        self.input_height = 640

        # VST camera dimensions (from ObjectDetection.cs)
        self.vst_width = 3248
        self.vst_height = 2464

        # Crop region (from ObjectDetection.cs lines 23-26)
        self.crop_x1 = 1444
        self.crop_y1 = 1332
        self.crop_x2 = 2045
        self.crop_y2 = 1933

        # Output tensors (matching ObjectDetection.cs)
        self.pred_boxes = None    # [8400, 4] - Float32
        self.pred_scores = None   # [8400, 1] - Float32
        self.pred_class_idx = None # [8400, 1] - Int8

        print("ObjectDetectionLocal initialized")
        print(f"Input size: {self.input_width}x{self.input_height}")
        print(f"Expected outputs: boxes[8400,4], scores[8400,1], classes[8400,1]")

    def load_image(self, image_path):
        """
        Load image from file

        Args:
            image_path: Path to input image

        Returns:
            numpy array: Loaded image in RGB format
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load with PIL and convert to RGB
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        print(f"Loaded image: {image_path}")
        print(f"Image shape: {image_np.shape}")

        return image_np

    def preprocess_image(self, image):
        """
        Preprocess image matching the ObjectDetection.cs pipeline

        Pipeline steps (from ObjectDetection.cs lines 123-188):
        1. Affine transform & crop (simulated - just resize here)
        2. RGB to Grayscale
        3. uint8 to float32
        4. Normalize (divide by 255.0)

        Args:
            image: Input image as numpy array (H, W, 3)

        Returns:
            numpy array: Preprocessed image (640, 640) in range [0, 1]
        """
        print("\n=== Preprocessing Image ===")

        # Step 1: Resize to 640x640 (simulating affine transform + crop)
        # In real pipeline, this crops from VST image using affine transform
        resized = cv2.resize(image, (self.input_width, self.input_height))
        print(f"1. Resized to: {resized.shape}")

        # Step 2: RGB to Grayscale (ConvertColorOperator, line 126-128)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        print(f"2. Converted to grayscale: {gray.shape}, dtype={gray.dtype}")

        # Step 3: uint8 to float32 (AssignmentOperator, line 129)
        gray_float = gray.astype(np.float32)
        print(f"3. Converted to float32: dtype={gray_float.dtype}")

        # Step 4: Normalize to [0, 1] (ArithmeticComposeOperator, line 131-132)
        normalized = gray_float / 255.0
        print(f"4. Normalized to [0,1]: min={normalized.min():.3f}, max={normalized.max():.3f}")

        # Add channel dimension to match expected input shape [1, 640, 640]
        # or [640, 640, 1] depending on model format
        preprocessed = normalized

        print(f"Final preprocessed shape: {preprocessed.shape}")
        print(f"Final range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

        return preprocessed

    def run_inference(self, preprocessed_image):
        """
        Run YOLO inference

        Model details (from ObjectDetection.cs lines 96-107):
        - Input: "image" (Float32, 640x640)
        - Output 1: "_571" (Float32, [8400, 4]) -> Bounding boxes
        - Output 2: "_530" (Float32, [8400, 1]) -> Scores
        - Output 3: "_532" (Int8, [8400, 1]) -> Class indices

        Args:
            preprocessed_image: Preprocessed image array

        Returns:
            tuple: (boxes, scores, class_idx)
        """
        print("\n=== Running Inference ===")

        if self.model_path and os.path.exists(self.model_path):
            print(f"Model path: {self.model_path}")
            print("Model type: QNN Context Binary")
            print("NOTE: QNN model inference requires Qualcomm runtime")
            print("      Running MOCK inference instead...")

            # TODO: Integrate actual QNN model inference
            # This would require Qualcomm's QNN SDK
            return self._run_mock_inference(preprocessed_image)
        else:
            print("No model provided - running MOCK inference")
            return self._run_mock_inference(preprocessed_image)

    def _run_mock_inference(self, preprocessed_image):
        """
        Mock inference for testing without actual model
        Generates random detections matching expected output format

        Returns:
            tuple: (boxes, scores, class_idx) matching YOLO output format
        """
        print("Running MOCK inference...")

        # Initialize outputs matching ObjectDetection.cs (lines 156-158)
        self.pred_boxes = np.zeros((8400, 4), dtype=np.float32)      # [8400, 4]
        self.pred_scores = np.zeros((8400, 1), dtype=np.float32)     # [8400, 1]
        self.pred_class_idx = np.zeros((8400, 1), dtype=np.int8)     # [8400, 1]

        # Generate some mock detections
        num_mock_detections = 5
        np.random.seed(42)

        for i in range(num_mock_detections):
            # Mock bounding boxes (x, y, w, h) in pixel coordinates
            self.pred_boxes[i] = [
                np.random.uniform(0, 640),      # x
                np.random.uniform(0, 640),      # y
                np.random.uniform(50, 150),     # w
                np.random.uniform(50, 150)      # h
            ]

            # Mock confidence scores [0.5, 1.0]
            self.pred_scores[i, 0] = np.random.uniform(0.5, 1.0)

            # Mock class indices (COCO has 80 classes)
            self.pred_class_idx[i, 0] = np.random.randint(0, 80)

        # Rest have low confidence
        self.pred_scores[num_mock_detections:] = np.random.uniform(0, 0.1,
                                                                   (8400 - num_mock_detections, 1))

        print(f"Generated {num_mock_detections} mock detections")
        print(f"Output shapes:")
        print(f"  boxes: {self.pred_boxes.shape}")
        print(f"  scores: {self.pred_scores.shape}")
        print(f"  class_idx: {self.pred_class_idx.shape}")

        return self.pred_boxes, self.pred_scores, self.pred_class_idx

    def post_process(self, confidence_threshold=0.5, nms_threshold=0.45):
        """
        Post-process predictions: filter by confidence

        Args:
            confidence_threshold: Minimum confidence score
            nms_threshold: NMS IoU threshold (not implemented in mock)

        Returns:
            list: Filtered detections
        """
        print(f"\n=== Post-processing (threshold={confidence_threshold}) ===")

        valid_detections = []

        for i in range(8400):
            score = self.pred_scores[i, 0]

            if score > confidence_threshold:
                detection = {
                    'index': i,
                    'class_id': int(self.pred_class_idx[i, 0]),
                    'score': float(score),
                    'bbox': {
                        'x': float(self.pred_boxes[i, 0]),
                        'y': float(self.pred_boxes[i, 1]),
                        'w': float(self.pred_boxes[i, 2]),
                        'h': float(self.pred_boxes[i, 3])
                    }
                }
                valid_detections.append(detection)

        print(f"Found {len(valid_detections)} valid detections")

        for det in valid_detections:
            print(f"  Detection {det['index']}: "
                  f"Class={det['class_id']}, "
                  f"Score={det['score']:.3f}, "
                  f"BBox=({det['bbox']['x']:.1f}, {det['bbox']['y']:.1f}, "
                  f"{det['bbox']['w']:.1f}, {det['bbox']['h']:.1f})")

        return valid_detections

    def visualize_detections(self, image, detections, output_path='output.jpg'):
        """
        Visualize detections on image

        Args:
            image: Original input image
            detections: List of detections from post_process()
            output_path: Path to save visualization
        """
        vis_image = image.copy()

        # Resize to 640x640 for visualization
        vis_image = cv2.resize(vis_image, (640, 640))

        for det in detections:
            bbox = det['bbox']
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"Class {det['class_id']}: {det['score']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"\nVisualization saved to: {output_path}")

    def run_full_pipeline(self, image_path, output_path='output.jpg',
                         confidence_threshold=0.5):
        """
        Run full detection pipeline on an image

        Args:
            image_path: Path to input image
            output_path: Path to save output visualization
            confidence_threshold: Minimum confidence for detections

        Returns:
            list: Detected objects
        """
        print("="*60)
        print("OBJECT DETECTION - LOCAL TESTING")
        print("="*60)

        # Load image
        image = self.load_image(image_path)

        # Preprocess
        preprocessed = self.preprocess_image(image)

        # Run inference
        self.run_inference(preprocessed)

        # Post-process
        detections = self.post_process(confidence_threshold)

        # Visualize
        if len(detections) > 0:
            self.visualize_detections(image, detections, output_path)

        print("="*60)
        return detections


# COCO class names (for reference)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


if __name__ == '__main__':
    # Example usage
    detector = ObjectDetectionLocal(model_path=None)  # Set to your QNN model path

    # Test with an image
    test_image_path = 'test_image.jpg'  # Replace with your image path

    if os.path.exists(test_image_path):
        detections = detector.run_full_pipeline(
            test_image_path,
            output_path='detection_output.jpg',
            confidence_threshold=0.5
        )

        print(f"\nTotal detections: {len(detections)}")
    else:
        print(f"\nTest image not found: {test_image_path}")
        print("Please provide a test image path")
        print("\nYou can also use the detector programmatically:")
        print("  detector = ObjectDetectionLocal()")
        print("  detections = detector.run_full_pipeline('your_image.jpg')")
