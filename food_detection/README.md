# Food Detection for SecureMR

## Real-time Food Detection Application

![Food Detection Demo](https://raw.githubusercontent.com/erinmitt123/Glucose_MR/c96a3421def9a5cd7d26e429465f777858789faf/image-assets/banana-object.gif)

This project demonstrates real-time food detection capabilities using SecureMR on PICO 4 Ultra devices. The application uses a YOLOX model to detect and identify 10 common food items in mixed reality.

## Detected Food Items

The application can detect the following food items:
- Banana
- Apple
- Sandwich
- Orange
- Broccoli
- Carrot
- Hot dog
- Pizza
- Donut
- Cake

## Model Used

| Application | Model | Source |
|------------|-------|--------|
| Food Detection | YOLOv11 Detection (YOLOX variant) | [Qualcomm AI Hub - YOLOv11 Detection](https://aihub.qualcomm.com/models/yolov11_det?searchTerm=yolo) |

## Project Overview

This project demonstrates the functionalities and usage of the SecureMR interfaces through a food detection application. The application achieves customized MR-based effects with deployment of an optimized YOLO-based machine learning model.

The project provides utility classes located under [`./base/securemr_utils`](base/securemr_utils/README.md) to simplify development of SecureMR-enabled applications.

A Docker configuration is contained under the `Docker/` directory for deploying custom algorithm packages.

## Repository Structure

```
.
├── Docker
|                Docker files and resources to convert ML algorithm packages
├── assets
│   │            Assets required for the food detection application
│   │
│   └── common
│                Assets shared by all components
│
├── base
│   │            Base source codes including fundamental OpenXR code
│   │
│   ├── oxr_utils
│   │            Utility for fundamental OpenXR APIs, such as
│   │            XR API result verification and Vulkan renderer
│   │
│   ├── securemr_utils
│   │            Utility classes to simplify SecureMR development
│   │
│   └── vulkan_shaders
|                Vulkan shaders for rendering
|
├── docs
│                Documentation files
|
├── external
|                External dependencies
|
├── samples
│   │            Sample application directory
│   │
│   └── food_detection
│                Real-time food detection application using YOLO model
|
└── ...
```

## Prerequisite

#### (A) To run the application, you will need

1. A PICO 4 Ultra device with the latest system update (OS version >= 5.14.0U)
2. Android Studio, with the Android SDK and NDK installed.
   - You can install the Android SDK via the SDK Manager in Android Studio (`Tools` > `SDK Manager`).
   - You will need Android SDK Platform 34. You can install it from the `SDK Platforms` tab in the SDK Manager.
   - The suggested NDK version is 25. You can install a specific NDK version from the `SDK Tools` tab. Check "Show Package Details", find `NDK (Side by side)` and select a `25.x.x` version to install.
3. Gradle and Android Gradle plugin (usually bundled with Android Studio install),
   suggested Gradle version = 8.7, Android Gradle Plugin version = 8.3.2
4. Java version at least 17 (required by the Android Gradle Plugin), recommended to be 21

#### (B) To run Docker for model conversion, you will need

1. Docker Desktop installed

## Deployment and Test

1. Install and configure according to the [prerequisite](#prerequisite).
1. Open the repository root in Android Studio as an Android project
1. After project sync, you will find the `food_detection` module under the `samples` folder
1. Connect to a PICO 4 Ultra device with the latest OS update installed
1. Select the `food_detection` module and click the launch button

Alternatively, build and install via command line:
```bash
./gradlew :samples:food_detection:assembleDebug
adb install -r samples/food_detection/build/outputs/apk/debug/food_detection-debug.apk
```

## PICO Developer Reference

You can view the full SecureMR documentation via
[this link to PICO Developer website](https://developer-cn.picoxr.com/document/native/securemr-overview/).
