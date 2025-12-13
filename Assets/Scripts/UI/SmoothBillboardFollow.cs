using UnityEngine;

public class SmoothBillboardFollow : MonoBehaviour
{
    [Header("Positioning")]
    [Tooltip("How far the sprite should hover above the controller in world space.")]
    public float hoverHeight = 0.1f;
    [Tooltip("How far the sprite should be offset forward/away from the user (World Z-plane).")]
    public float forwardOffset = 0.05f;

    [Header("Rotation Settings")]
    [Tooltip("How fast the sprite rotates to face the user.")]
    public float rotationSpeed = 5f;

    [HideInInspector] public Transform targetController;

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
        HandlePositioning();
        HandleSmoothBillboarding();
    }

    // Public function to set before the game object is set active, to prevent jumping
    public void SetInitialPositionAndRotation()
    {
        HandlePositioning();
        HandleSmoothBillboarding();
    }


    private void HandlePositioning()
    {
        // Get the directional vector the controller is being viewed from
        Vector3 cameraToControllerDir = targetController.position - mainCameraTransform.position;
        cameraToControllerDir.y = 0; // Flatten the vector
        cameraToControllerDir.Normalize();

        // Calculate the target position: the controller's position plus the hover height on its local Y-axis.
        Vector3 targetPosition = targetController.position + Vector3.up * hoverHeight;

        // Pushes the icon AWAY from the camera along the flattened Camera->Controller direction.
        targetPosition += cameraToControllerDir * forwardOffset;

        // Set the sprite's position instantly to follow the controller
        transform.position = targetPosition;
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
