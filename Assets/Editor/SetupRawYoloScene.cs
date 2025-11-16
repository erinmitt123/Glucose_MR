using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

/// <summary>
/// Automated setup script for ObjectDetectionRawYolo in FoodDetection scene
/// </summary>
public class SetupRawYoloScene : EditorWindow
{
    [MenuItem("Tools/Setup Raw YOLO Scene")]
    public static void ShowWindow()
    {
        GetWindow<SetupRawYoloScene>("Setup Raw YOLO");
    }

    private void OnGUI()
    {
        GUILayout.Label("Setup ObjectDetectionRawYolo", EditorStyles.boldLabel);
        GUILayout.Space(10);

        GUILayout.Label("This will:");
        GUILayout.Label("1. Open FoodDetection scene");
        GUILayout.Label("2. Find existing ObjectDetection GameObject");
        GUILayout.Label("3. Create new ObjectDetectionRawYolo GameObject");
        GUILayout.Label("4. Assign yolo_raw.serialized.bytes model");
        GUILayout.Label("5. Disable old ObjectDetection");
        GUILayout.Space(10);

        if (GUILayout.Button("Setup Now", GUILayout.Height(40)))
        {
            SetupScene();
        }

        GUILayout.Space(10);
        if (GUILayout.Button("Setup in ObjectDetection Scene", GUILayout.Height(40)))
        {
            SetupSceneAlt();
        }
    }

    private static void SetupScene()
    {
        // Open FoodDetection scene
        Scene scene = EditorSceneManager.OpenScene("Assets/Scenes/FoodDetection.unity");
        Debug.Log("Opened FoodDetection scene");

        SetupRawYoloInScene(scene);
    }

    private static void SetupSceneAlt()
    {
        // Open ObjectDetection scene
        Scene scene = EditorSceneManager.OpenScene("Assets/Scenes/ObjectDetection.unity");
        Debug.Log("Opened ObjectDetection scene");

        SetupRawYoloInScene(scene);
    }

    private static void SetupRawYoloInScene(Scene scene)
    {
        // Find existing ObjectDetection GameObject
        GameObject existingObj = null;
        foreach (GameObject rootObj in scene.GetRootGameObjects())
        {
            var odComponent = rootObj.GetComponent<PicoXR.SecureMR.Demo.ObjectDetection>();
            if (odComponent != null)
            {
                existingObj = rootObj;
                Debug.Log($"Found existing ObjectDetection on GameObject: {rootObj.name}");
                break;
            }

            // Check children too
            var odInChildren = rootObj.GetComponentInChildren<PicoXR.SecureMR.Demo.ObjectDetection>(true);
            if (odInChildren != null)
            {
                existingObj = odInChildren.gameObject;
                Debug.Log($"Found existing ObjectDetection in children: {existingObj.name}");
                break;
            }
        }

        if (existingObj == null)
        {
            Debug.LogError("Could not find existing ObjectDetection GameObject!");
            EditorUtility.DisplayDialog("Error", "Could not find existing ObjectDetection GameObject in scene!", "OK");
            return;
        }

        // Duplicate the GameObject
        GameObject newObj = Object.Instantiate(existingObj);
        newObj.name = "ObjectDetectionRawYolo";
        newObj.transform.SetParent(existingObj.transform.parent);
        newObj.transform.SetSiblingIndex(existingObj.transform.GetSiblingIndex() + 1);
        Debug.Log("Created duplicate GameObject: ObjectDetectionRawYolo");

        // Remove old ObjectDetection component from new object
        var oldComponent = newObj.GetComponent<PicoXR.SecureMR.Demo.ObjectDetection>();
        if (oldComponent != null)
        {
            Object.DestroyImmediate(oldComponent);
            Debug.Log("Removed old ObjectDetection component");
        }

        // Add ObjectDetectionRawYolo component
        var newComponent = newObj.AddComponent<PicoXR.SecureMR.Demo.ObjectDetectionRawYolo>();
        Debug.Log("Added ObjectDetectionRawYolo component");

        // Load the new YOLO model
        string modelPath = "Assets/MLModels/yolo_raw.serialized.bytes";
        var modelAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(modelPath);

        if (modelAsset != null)
        {
            // Use reflection to set the yoloModel field
            var field = typeof(PicoXR.SecureMR.Demo.ObjectDetectionRawYolo).GetField("yoloModel",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            if (field != null)
            {
                field.SetValue(newComponent, modelAsset);
                Debug.Log($"Assigned yolo_raw.serialized.bytes model");
            }

            // Copy other fields from old component
            var oldComp = existingObj.GetComponent<PicoXR.SecureMR.Demo.ObjectDetection>();
            if (oldComp != null)
            {
                // Copy frameGltfAsset
                var gltfField = typeof(PicoXR.SecureMR.Demo.ObjectDetectionRawYolo).GetField("frameGltfAsset",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                var oldGltfField = typeof(PicoXR.SecureMR.Demo.ObjectDetection).GetField("frameGltfAsset",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (gltfField != null && oldGltfField != null)
                {
                    var gltfValue = oldGltfField.GetValue(oldComp);
                    gltfField.SetValue(newComponent, gltfValue);
                    Debug.Log("Copied frameGltfAsset");
                }

                // Copy anchorMatrixAsset if it exists
                var anchorField = typeof(PicoXR.SecureMR.Demo.ObjectDetectionRawYolo).GetField("anchorMatrixAsset",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                var oldAnchorField = typeof(PicoXR.SecureMR.Demo.ObjectDetection).GetField("anchorMatrixAsset",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (anchorField != null && oldAnchorField != null)
                {
                    var anchorValue = oldAnchorField.GetValue(oldComp);
                    anchorField.SetValue(newComponent, anchorValue);
                    Debug.Log("Copied anchorMatrixAsset");
                }
            }
        }
        else
        {
            Debug.LogWarning($"Could not find model at {modelPath}. Please assign manually.");
        }

        // Disable the old ObjectDetection GameObject
        existingObj.SetActive(false);
        Debug.Log("Disabled old ObjectDetection GameObject");

        // Mark scene as dirty and save
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Scene saved!");

        EditorUtility.DisplayDialog("Success!",
            "ObjectDetectionRawYolo has been set up!\n\n" +
            "✅ New GameObject created\n" +
            "✅ ObjectDetectionRawYolo component added\n" +
            "✅ yolo_raw.serialized.bytes assigned\n" +
            "✅ Old ObjectDetection disabled\n" +
            "✅ Scene saved\n\n" +
            "You can now build and test!",
            "OK");
    }
}
