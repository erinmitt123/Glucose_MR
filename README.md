# 🩺 Glucose MR — Pico SecureMR Diabetes Assistance

### *Turn every glance into guidance for managing your glucose.*
<img src="https://raw.githubusercontent.com/erinmitt123/Glucose_MR/c96a3421def9a5cd7d26e429465f777858789faf/image-assets/full%20flow.gif" />

---

## Inspiration

Managing diabetes often requires constant attention to diet, portion sizes, and nutritional balance, which can feel overwhelming in daily life. The inspiration for developing Glucose MR came from the desire to make this process simpler and more intuitive.

People with Type 1 and Type 2 Diabetes shared that they constantly estimate carbs, read labels, and second-guess whether a food is safe for their current glucose level. Mixed Reality gives us a chance to make this effortless—just look at food and get instant, personalized feedback.

Our goal is to turn diabetes management into a proactive, empowering experience, all while keeping users' data completely private with SecureMR.

---

## What it does

Glucose MR is a privacy-first MR assistant that provides real-time, glucose-aware feedback on the foods around you.

* The PICO 4 Ultra headset uses an **open-source YOLO food detection model** to recognize foods in the environment.
* Users speak or enter their **current glucose reading**.
* The system pulls nutrition details (carbs, sugars, fats, protein, fiber, serving size, etc.) from the **MyFoodData Nutrition Facts Spreadsheet — a combined dataset of SR Legacy + FNDDS entries**.
* Based on diabetes type + glucose reading, Glucose MR tells users:

  * Whether the food is safe to eat right now
  * How many carbs per serving (ideal for Type 1 users)
  * If sugars are fast-acting or complex
  * How nutrient-dense the food is
  * A health/safety rating tailored to their current glucose

The feedback appears as **friendly MR overlays** with emojis and color indicators—fast, intuitive, and easy to understand at a glance.

Because of **SecureMR**, all camera processing happens locally, ensuring complete privacy.

---

## How we built it

### Food Detection
* Integrated an **open-source YOLO model** to identify common foods in real time.
* For the MVP we used the standard YOLO implementation.
* Future improvements include **deploying an optimized ONNX version** for higher FPS and lower latency on the PICO headset.

![Banana Object Detection](https://raw.githubusercontent.com/erinmitt123/Glucose_MR/c96a3421def9a5cd7d26e429465f777858789faf/image-assets/banana-object.gif)

![Banana Nutrition](https://raw.githubusercontent.com/erinmitt123/Glucose_MR/c96a3421def9a5cd7d26e429465f777858789faf/image-assets/banana-nutrition.gif)


### Privacy via SecureMR
* SecureMR ensures **no raw camera data leaves the device**.
* Developers cannot access user camera frames.
* All inference and processing happens fully on-device.

### Nutrition Data Pipeline

Our nutrition lookup engine is powered by the **MyFoodData Nutrition Facts Spreadsheet**, which merges:

* **SR Legacy** (USDA Standard Reference)
* **FNDDS** (Food and Nutrient Database for Dietary Studies)

The dataset provides detailed fields such as:
* Carbohydrates
* Sugars & added sugars
* Fat (including healthy fats)
* Fiber
* Protein
* Serving weight
* Nutrient density

Pipeline:
1. YOLO detects a food.
2. We match the detected class to the closest entry in the SR Legacy + FNDDS spreadsheet.
3. We pull nutrient metrics for the typical serving size.
4. We calculate carbs per:
   * cup
   * piece
   * tablespoon
   * gram
   * serving weight (CU)
5. We apply **Type 1–specific** and **Type 2–specific** decision logic:

   * Type 1 → carb counting, fast vs complex carbs, "15g for lows", extended bolus considerations
   * Type 2 → overall health score, sugar quality, nutrient density, additives, order of eating
6. The MR UI displays a **simple rating + emoji + recommendation text**.

### Speech Input
* Built speech input for glucose readings.
* Pico's security originally blocked our initial speech API attempts, so we created a custom workaround to enable reliable recognition.

### User Experience
* Uses emojis (super happy → warning → critical)
* RGB health ratings
* Minimal text, highly readable in MR
* Adjusts ratings dynamically based on whether a user is low, normal, or high.

---

## Challenges we ran into
* Pico's speech recognition security sandbox blocked standard pipelines.
* Mapping YOLO food labels to SR Legacy/FNDDS categories required custom logic.
* Creating **two separate medical logic flows** (Type 1 and Type 2) that remain simple and safe.
* Ensuring YOLO ran smoothly on MR without dropping frame rates.

---

## Accomplishments we're proud of
* An end-to-end system: **User Input → Nutrition DB → Diabetes Logic → MR UI**
* Running Object Detection SecureMR on PICO
* Enabling Speech Input Interface to interact with assets in Unity Pico projects
* A glucose-aware adaptive rating system that feels alive and reactive.
* Getting speech recognition working securely.
* Building an experience that remains simple despite complex calculations under the hood.

---

## What we learned
* How **SecureMR** enables strongly private, on-device ML for health-based XR experiences.
* Combining medical nutritional datasets with computer vision requires careful mapping and unit handling.
* Mixed Reality UX must be both simple and dynamic to be useful.
* Real-time health decisions need to be communicated in < 1 second for actual usefulness.

---

## What's next for Glucose MR

* **Deploying an ONNX-optimized food detection model**
  Not just for faster inference, but for more stable, low-latency MR performance on PICO devices.

* **Expanding object recognition beyond YOLO's default classes**
  COCO-style YOLO only recognizes ~15–20 food items. We plan to introduce a **custom-trained, broad food-group model** that can recognize:

  * More diverse fruits & vegetables
  * Grains, proteins, dairy
  * Prepared meals (pasta, curries, soups)
  * Snacks & packaged foods
  * Mixed plates (multi-food segmentation)

* **Support for mixed and complex meals**
  Multi-food bounding boxes and per-item nutrition estimation.

* **Integration of glycemic index / glycemic load data**
  To give more medically relevant guidance on sugar absorption rates.

* **Continuous glucose awareness**
  Future opt-in integration with CGMs to automatically adjust recommendations.

* **Richer voice-first interaction**
  A conversational mode where users can ask, "Is this good for me right now?" and hear personalized guidance.

* **Platform expansion**
  Bringing Glucose MR to Vision Pro, Quest 3, and other XR headsets.

---

## Research Citations & Technical References

**Reynolds A, Mitri J. Dietary Advice For Individuals with Diabetes.** Updated 2024 Apr 28. In: Feingold KR, Ahmed SF, Anawalt B, et al., editors. Endotext [Internet]. South Dartmouth (MA): MDText.com, Inc.; 2000–.
🔗 https://www.ncbi.nlm.nih.gov/books/NBK279012/

This reference outlines evidence-based dietary guidance for individuals with Type 1 and Type 2 Diabetes. It informed our recommendations related to carbohydrate timing, nutrient balance, fiber, fat intake, and metabolic response.

**Blood Sugar Level Chart: What's Normal, Low, and High?**
A patient-friendly medical resource defining blood glucose thresholds and how food choices should change depending on whether a user is hypoglycemic, euglycemic, or hyperglycemic. It shaped our logic for glucose-aware food ratings.

**Nutrition Facts Database Spreadsheet — MyFoodData (SR Legacy + FNDDS)**
The primary nutrition dataset powering Glucose MR. Includes detailed nutrition facts such as carbs, sugars, added sugars, fats, fiber, protein, and serving sizes. Used for all nutrient calculations and food analysis.

**Device:**
PICO 4 Ultra
Used for all Mixed Reality testing, object overlay logic, and SecureMR integration.

**Food Object Detection Model:**
YOLOv8 for Nutrition-Focused Food Detection
🔗 https://arxiv.org/html/2408.10532v1
