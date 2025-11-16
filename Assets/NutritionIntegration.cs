using UnityEngine;
using Unity.XR.PXR.SecureMR;
using System.Collections.Generic;

namespace PicoXR.SecureMR.Demo
{
    /// <summary>
    /// Simple nutrition integration - NO QNN MODEL NEEDED!
    /// Maps YOLO class index to food_id and looks up nutrition data
    /// </summary>
    public class NutritionIntegration : MonoBehaviour
    {
        // Mapping: YOLO class index → food_id (46-55)
        // TODO: Update these based on your YOLO model's actual class indices
        private Dictionary<int, int> yoloClassToFoodId = new Dictionary<int, int>()
        {
            // Example mapping - adjust based on your YOLO model
            {46, 46},  // YOLO class 46 → banana (food_id 46)
            {47, 47},  // YOLO class 47 → apple (food_id 47)
            {48, 48},  // YOLO class 48 → sandwich (food_id 48)
            {49, 49},  // YOLO class 49 → orange (food_id 49)
            {50, 50},  // YOLO class 50 → brocolli (food_id 50)
            {51, 51},  // YOLO class 51 → carrot (food_id 51)
            {52, 52},  // YOLO class 52 → hot dog (food_id 52)
            {53, 53},  // YOLO class 53 → pizza (food_id 53)
            {54, 54},  // YOLO class 54 → donut (food_id 54)
            {55, 55},  // YOLO class 55 → cake (food_id 55)
        };

        /// <summary>
        /// Get nutrition data for a YOLO detection
        /// Call this after YOLO model outputs class index
        /// </summary>
        /// <param name="yoloClassIdx">Class index from YOLO model</param>
        /// <returns>Nutrition info or null if not found</returns>
        public NutritionDatabase.NutritionData GetNutritionForYoloClass(int yoloClassIdx)
        {
            // Map YOLO class to food_id
            if (yoloClassToFoodId.TryGetValue(yoloClassIdx, out int foodId))
            {
                // Look up nutrition data
                if (NutritionDatabase.Instance != null)
                {
                    return NutritionDatabase.Instance.GetNutritionData(foodId);
                }
                else
                {
                    Debug.LogError("NutritionDatabase not initialized!");
                }
            }
            else
            {
                Debug.LogWarning($"YOLO class {yoloClassIdx} is not a food item");
            }

            return null;
        }

        /// <summary>
        /// Get formatted nutrition text for display
        /// </summary>
        public string GetNutritionTextForYoloClass(int yoloClassIdx)
        {
            var nutrition = GetNutritionForYoloClass(yoloClassIdx);
            if (nutrition != null)
            {
                return $"{nutrition.foodName}: {nutrition.carbohydrate}g carbs, {nutrition.protein}g protein, {nutrition.fat}g fat";
            }
            return $"Unknown (class {yoloClassIdx})";
        }

        /// <summary>
        /// Check if a YOLO class is a food item
        /// </summary>
        public bool IsFoodItem(int yoloClassIdx)
        {
            return yoloClassToFoodId.ContainsKey(yoloClassIdx);
        }
    }
}
