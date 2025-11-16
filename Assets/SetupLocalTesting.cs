using UnityEngine;
using UnityEditor;

/// <summary>
/// Helper script to setup local testing environment in Unity Editor
/// Usage: Attach to GameObject or use menu: Tools > Setup Object Detection Testing
/// </summary>
public class SetupLocalTesting : MonoBehaviour
{
#if UNITY_EDITOR
    [MenuItem("Tools/Setup Object Detection Testing")]
    static void SetupTesting()
    {
        // Create GameObject with ObjectDetectionLocal
        GameObject testObj = new GameObject("ObjectDetection_LocalTest");
        ObjectDetectionLocal detector = testObj.AddComponent<ObjectDetectionLocal>();

        Debug.Log("✓ Created GameObject with ObjectDetectionLocal component");
        Debug.Log("Next steps:");
        Debug.Log("1. Select 'ObjectDetection_LocalTest' in Hierarchy");
        Debug.Log("2. In Inspector, assign 'Yolo Model' (optional for testing)");
        Debug.Log("3. Set 'Test Image Path' or assign 'Input Texture'");
        Debug.Log("4. Press Play to run inference");

        Selection.activeGameObject = testObj;
    }
#endif
}
