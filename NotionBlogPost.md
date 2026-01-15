# Privacy-First Spatial ML: How a StanfordXR Team Built “Glucose MR” for Real-Time Nutrition Awareness

**_By [Livia Ellen](https://www.linkedin.com/in/liviaellen/), [Cyril Medabalimi](https://www.linkedin.com/in/cyril-medabalimi-64b74215a/), [Erin Mittmann](https://www.linkedin.com/in/erin-mittmann-a22450172/), [Manuel Rebol](https://devpost.com/manuel-rebol)_**

[https://www.youtube.com/watch?v=i1UGXdcjhVw](https://www.youtube.com/watch?v=i1UGXdcjhVw)

**Developer Spotlight Series:**

Mixed reality introduces new opportunities for on-device perception and real-time context. But it also raises a core technical question:

**How can developers build ML-powered mixed-reality experiences without exposing camera data or relying on the cloud?**

At StanfordXR’s _Immerse the Bay_ hackathon, one team explored that question head-on. Their project, **Glucose MR**, is a diabetes-assistance prototype that analyzes food in real time and provides personalized nutritional guidance — all running entirely on-device using **SecureMR**, PICO’s privacy-focused machine-learning runtime.

The team built a system to support a vision selection, a nutrition pipeline, decision logic, and MR UX, in 36 hours.

This post walks through what they built, how it works, and the design patterns developers can reuse when building privacy-preserving spatial applications.

As the Glucose MR team explained during the hackathon:

> “Managing diabetes is deeply personal, and we knew from the start that any MR assistant dealing with glucose data had to be fully private by design. SecureMR let us run all of our food detection and nutrition analysis directly on the PICO headset, so that no sensitive health information ever leaves the device. This was a non-negotiable requirement for us. We built Glucose MR to show how spatial computing can help people make real-time food decisions—without sacrificing security or trust.”

---

## **Problem Context: Context-Aware Health Data Without Cloud Dependence**

For people with Type 1 or Type 2 diabetes, food decisions happen many times per day:

- How many carbs are in this meal?
- Is this safe to eat right now?
- Will this cause a spike?

These decisions normally require manual lookup or past experience. The team’s goal was to use MR to provide just-in-time, visual guidance — **without sending images or sensor data to external servers**.

This constraint shaped the entire system design.

---

# **System Overview**

Glucose MR is built around three core components:

1. **On-device food detection** using an open-source YOLO model
2. **Local nutrition lookup** using merged SR Legacy + FNDDS datasets
3. **Mixed reality UI** that adapts recommendations based on user-provided glucose values

All of this runs inside **SecureMR**, which provides a sandboxed ML runtime:

- No camera frames leave the device
- Developers cannot access raw sensor data
- All inference returns structured results only
- No network connectivity required

This mirrors the privacy model seen in:

- **Google’s Private Compute Core**
- **Apple’s CoreML local inference model**
- **Android ML Kit on-device pipelines**

SecureMR applies that same philosophy to spatial computing.

---

# **Implementation Details**

## **Object Detection Pipeline**

The team used a standard YOLO model for food identification.

**Architecture Pattern: Local Detection → Structured Output**

SecureMR returns detection results in a structured format:

No image buffers are exposed.

[https://drive.google.com/file/d/1vt5i7D2s4hmh68M1PN5TgzgGHkBk37Yw/view?usp=sharing](https://drive.google.com/file/d/1vt5i7D2s4hmh68M1PN5TgzgGHkBk37Yw/view?usp=sharing)

This matters because developers get ML-powered detection without handling camera permissions, user data, or image sanitization.

## **Nutrition Data Integration**

Nutrition lookup uses a merged dataset (SR Legacy + FNDDS), similar to how Google Health or Fitbit combine heterogeneous datasets for richer profile inference.

**Pipeline:**

1. YOLO → label
2. Label → nearest nutrition entry
3. Extract serving-level metrics (carbs, sugars, fiber, protein, fats)
4. Compute per-unit conversions
5. Apply diabetes-specific decision logic

This is a clean pattern for spatial ML:

**vision output → domain dataset → local inference → MR output**

## **Decision Logic**

The team implemented separate workflows for:

- **Type 1 diabetes** (carb counting, fast vs. complex carbs, serving estimation)
- **Type 2 diabetes** (nutrient density, sugar quality, overall health score)

All logic runs in ~100–200 ms on mid-range hardware.

**Takeaway:**

Real-world ML applications often require domain-specific inference separate from the vision model. Here, the model identifies food, but **all meaningful value comes from domain logic**.

## **Speech Input Under Platform Constraints**

The SecureMR environment intentionally blocks access to traditional speech APIs.

The team implemented a lightweight fallback:

- Speech → Text
- Text → Numeric Glucose Parsing
- Parsed Values injected into the decision pipeline

This is a typical constraint in sandboxed environments (similar to Chrome extensions, Android Work Profile, or iOS Data Protection classes).

The parser was originally built in a performant C# structure, but after observing unexpected spikes and frame drops in the profiler, the structure was ported into a DLL for faster and less resource-intensive data extraction.

The workaround demonstrates how to combine **secure environments + flexible UX**.

# **Key Engineering Lessons**

From reviewing the implementation, several reusable patterns emerge:

### **On-Device ML Enables Real-World Use Cases**

Low-latency, private pipelines unlock health, education, and enterprise applications where cloud models are not permissible.

### **Vision Models Are Only Step 1**

The value often comes from:

- domain logic
- data integration
- local state
- context adaptation

### **Mixed Reality UX Must Be Extremely Lightweight**

High-density UI fails when layered on top of real-world scenes.

Simple indicators outperform complex overlays.

### **Privacy by Design Accelerates Prototyping**

Developers never handle raw camera buffers, reducing required compliance work and lowering cognitive overhead.

### **Datasets + ML + MR = New Interaction Patterns**

This “ML → dataset → inference → MR” loop is a useful mental model for future spatial ML projects.

# **What’s Coming Next**

The team plans to extend the project with:

- ONNX-optimized models for lower latency
- Expanded food-class recognition
- Multi-food segmentation
- Glycemic index integration
- Optional CGM connectivity
- Conversational agent mode
- Cross-platform builds (AVP, Quest, Android XR)

When the project is fully cleaned up, we will publish it as a **SecureMR sample app**, including:

- documented architecture
- starter scripts
- reproducible build instructions
- reusable UI patterns
- nutrition pipeline examples

This will help developers learn Spatial ML patterns using real code, not theoretical samples.

# [**Try GlucoseMR Yourself**](https://github.com/erinmitt123/Glucose_MR)

# **Final Thoughts**

Glucose MR demonstrates a growing trend in spatial computing: **using local ML to provide real-time, privacy-preserving awareness of the world.**

This is exactly the kind of developer pattern we expect to see more of as on-device ML and mixed reality converge.

If you or your team are exploring similar use cases, follow our upcoming posts showcasing all nine StanfordXR projects, each illustrating a different pattern for building practical, real-world spatial applications.
