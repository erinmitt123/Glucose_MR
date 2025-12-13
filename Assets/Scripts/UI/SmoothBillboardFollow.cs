using UnityEngine;

public class SmoothBillboardFollow : MonoBehaviour
{
    [Header("Rotation Settings")]
    [Tooltip("How fast the sprite rotates to face the user.")]
    public float rotationSpeed = 5f;

    // Cached reference to the main camera's transform (the user's view or central eye anchor)
    private Transform mainCameraTransform;

    private void Start()
    {
        if (Camera.main != null)
        {
            mainCameraTransform = Camera.main.transform;
        }
        else
        {
            Debug.LogError("SmoothBillboardFollow requires a main camera in the scene!");
            enabled = false;
        }
    }

    private void Update()
    {
        // Doesn't need a check since the behavior is disabled in Start if values aren't found or assigned
        HandleSmoothBillboarding();
    }

    private void HandleSmoothBillboarding()
    {
        // Direction from the sprite to the camera
        Vector3 lookDirection = mainCameraTransform.position - transform.position;

        // Ensure the rotation only happens around the world Y-axis (prevents tilting up/down)
        lookDirection.y = 0;

        // Calculate the rotation needed to look at that direction
        Quaternion targetRotation = Quaternion.LookRotation(lookDirection);

        // Smoothly rotate the sprite towards the target rotation
        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            targetRotation,
            rotationSpeed * Time.deltaTime
        );
    }

}
