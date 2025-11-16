using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Database class that stores and retrieves nutritional information based on food ID
/// Maps food IDs to their nutritional values
/// Loads data from CSV file
/// </summary>
public class NutritionDatabase : MonoBehaviour
{
    // Singleton instance for easy access
    public static NutritionDatabase Instance { get; private set; }

    // Path to CSV file (can be set in Inspector or use TextAsset)
    [Header("CSV Configuration")]
    [Tooltip("TextAsset containing the nutrition CSV data")]
    public TextAsset nutritionCSV;

    [Tooltip("Alternative: Path to CSV file relative to Assets folder")]
    public string csvFilePath = "Resources/nutrition_data.csv";

    // Dictionary to store nutrition data indexed by food ID
    private Dictionary<int, NutritionData> nutritionLookup;

    [System.Serializable]
    public class NutritionData
    {
        public int id;                      // matched_nutrition_ID (46-55)
        public string foodName;             // Name of the food
        public float carbohydrate;          // Carbohydrate (g)
        public float fat;                   // Fat (g)
        public float fiber;                 // Fiber (g)
        public float protein;               // Protein (g)
        public float sugars;                // Sugars (g)
        public float addedSugar;            // Added Sugar (g) - (-1 means no data)
        public float servingWeight;         // Serving Weight 1 (g)

        public NutritionData(int id, string name, float carb, float fat, float fiber,
                           float protein, float sugar, float addedSugar, float servingWeight)
        {
            this.id = id;
            this.foodName = name;
            this.carbohydrate = carb;
            this.fat = fat;
            this.fiber = fiber;
            this.protein = protein;
            this.sugars = sugar;
            this.addedSugar = addedSugar;
            this.servingWeight = servingWeight;
        }

        /// <summary>
        /// Converts nutrition data to array format compatible with FoodInfo.secureMLValues
        /// </summary>
        public double[] ToSecureMLArray()
        {
            return new double[]
            {
                id,                 // [0] - Food ID
                carbohydrate,       // [1] - Carbs
                fat,                // [2] - Fats
                fiber,              // [3] - Fiber
                protein,            // [4] - Protein
                sugars,             // [5] - Sugar
                addedSugar,         // [6] - Added Sugars
                servingWeight       // [7] - Serving weight
            };
        }
    }

    void Awake()
    {
        // Singleton pattern
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            InitializeDatabase();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    /// <summary>
    /// Initialize the nutrition database by loading from CSV file
    /// CSV Format: matched_nutrition_ID,food_id,food_name,Carbohydrate (g),Fat (g),Fiber (g),Protein (g),Sugars (g),Added Sugar (g),Serving Weight 1 (g)
    /// </summary>
    private void InitializeDatabase()
    {
        nutritionLookup = new Dictionary<int, NutritionData>();

        // Try to load from TextAsset first (preferred method in Unity)
        if (nutritionCSV != null)
        {
            LoadFromTextAsset(nutritionCSV);
        }
        // Fallback to file path
        else if (!string.IsNullOrEmpty(csvFilePath))
        {
            LoadFromFilePath(csvFilePath);
        }
        else
        {
            Debug.LogError("No CSV source configured! Please assign a TextAsset or set csvFilePath.");
        }

        Debug.Log($"Nutrition database initialized with {nutritionLookup.Count} items");
    }

    /// <summary>
    /// Load nutrition data from a TextAsset (recommended for Unity)
    /// </summary>
    /// <param name="csvAsset">TextAsset containing CSV data</param>
    private void LoadFromTextAsset(TextAsset csvAsset)
    {
        if (csvAsset == null)
        {
            Debug.LogError("CSV TextAsset is null!");
            return;
        }

        ParseCSVData(csvAsset.text);
    }

    /// <summary>
    /// Load nutrition data from file path (for runtime loading)
    /// </summary>
    /// <param name="filePath">Path to CSV file</param>
    private void LoadFromFilePath(string filePath)
    {
        string fullPath = Path.Combine(Application.dataPath, filePath);

        if (!File.Exists(fullPath))
        {
            Debug.LogError($"CSV file not found at: {fullPath}");
            return;
        }

        try
        {
            string csvData = File.ReadAllText(fullPath);
            ParseCSVData(csvData);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error reading CSV file: {e.Message}");
        }
    }

    /// <summary>
    /// Parse CSV data and populate the nutrition lookup dictionary
    /// CSV Format: matched_nutrition_ID,food_id,food_name,Carbohydrate (g),Fat (g),Fiber (g),Protein (g),Sugars (g),Added Sugar (g),Serving Weight 1 (g)
    /// </summary>
    /// <param name="csvData">Raw CSV text data</param>
    private void ParseCSVData(string csvData)
    {
        if (string.IsNullOrEmpty(csvData))
        {
            Debug.LogError("CSV data is empty!");
            return;
        }

        string[] lines = csvData.Split('\n');
        int loadedCount = 0;
        int skippedCount = 0;

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i].Trim();

            // Skip empty lines
            if (string.IsNullOrEmpty(line))
                continue;

            // Skip header line (if it contains column names)
            if (i == 0 && line.Contains("matched_nutrition_ID"))
            {
                Debug.Log("Skipping CSV header row");
                continue;
            }

            // Parse the line
            string[] values = line.Split(',');

            // Validate that we have enough columns
            // Format: matched_nutrition_ID, food_id, food_name, Carbohydrate, Fat, Fiber, Protein, Sugars, Added Sugar, Serving Weight
            if (values.Length < 10)
            {
                Debug.LogWarning($"Line {i + 1} has insufficient columns ({values.Length}): {line}");
                skippedCount++;
                continue;
            }

            try
            {
                // Parse values from CSV
                // Note: We use food_id (column index 1) as the key, not matched_nutrition_ID
                int foodId = int.Parse(values[1].Trim());
                string foodName = values[2].Trim();
                float carbohydrate = float.Parse(values[3].Trim());
                float fat = float.Parse(values[4].Trim());
                float fiber = float.Parse(values[5].Trim());
                float protein = float.Parse(values[6].Trim());
                float sugars = float.Parse(values[7].Trim());
                float addedSugar = float.Parse(values[8].Trim());
                float servingWeight = float.Parse(values[9].Trim());

                // Create nutrition data object
                NutritionData data = new NutritionData(
                    foodId,
                    foodName,
                    carbohydrate,
                    fat,
                    fiber,
                    protein,
                    sugars,
                    addedSugar,
                    servingWeight
                );

                // Add to lookup dictionary
                nutritionLookup[foodId] = data;
                loadedCount++;

                Debug.Log($"Loaded: ID={foodId}, {foodName}, Carbs={carbohydrate}g");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Error parsing line {i + 1}: {line}\nError: {e.Message}");
                skippedCount++;
            }
        }

        Debug.Log($"CSV parsing complete: {loadedCount} items loaded, {skippedCount} skipped");
    }

    /// <summary>
    /// Get nutrition data as a double array (compatible with FoodInfo.secureMLValues)
    /// </summary>
    /// <param name="foodId">The matched_nutrition_ID (46-55)</param>
    /// <returns>Array of nutrition values, or null if not found</returns>
    public double[] GetNutritionArray(int foodId)
    {
        if (nutritionLookup.TryGetValue(foodId, out NutritionData data))
        {
            return data.ToSecureMLArray();
        }

        Debug.LogWarning($"No nutrition data found for food ID: {foodId}");
        return null;
    }

    /// <summary>
    /// Get full nutrition data object
    /// </summary>
    /// <param name="foodId">The matched_nutrition_ID (46-55)</param>
    /// <returns>NutritionData object, or null if not found</returns>
    public NutritionData GetNutritionData(int foodId)
    {
        if (nutritionLookup.TryGetValue(foodId, out NutritionData data))
        {
            return data;
        }

        Debug.LogWarning($"No nutrition data found for food ID: {foodId}");
        return null;
    }

    /// <summary>
    /// Check if a food ID exists in the database
    /// </summary>
    public bool HasFoodId(int foodId)
    {
        return nutritionLookup.ContainsKey(foodId);
    }

    /// <summary>
    /// Get the food name for a given ID
    /// </summary>
    public string GetFoodName(int foodId)
    {
        if (nutritionLookup.TryGetValue(foodId, out NutritionData data))
        {
            return data.foodName;
        }
        return "Unknown";
    }
}
