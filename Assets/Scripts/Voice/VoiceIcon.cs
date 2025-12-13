using UnityEngine;

public class VoiceIcon : MonoBehaviour
{
    [Header("Controller Reference Points")]
    [Tooltip("The target transforms of the left and right controllers anchor points")]
    [SerializeField] private Transform leftController;
    [SerializeField] private Transform rightController;

    [Header("Positioning")]
    [Tooltip("How far the sprite should hover above the controller in world space.")]
    public float hoverHeight = 0.15f;

    private void Start()
    {
        DeactivateAndUnparent();
    }

    #region -----PUBLIC METHODS FOR CONTROLLER INPUT-----

    /// <summary>
    /// Called when the user presses the input (e.g., the Grip button) on an XR Controller.
    /// </summary>
    /// <param name="controller">The Transform of the XR Controller that pressed the button.</param>
    public void ActivateAndParentToController(bool isLeft)
    {
        // Determines which controller transform to parent to
        Transform controller = isLeft ? leftController : rightController;

        // Parent the sprite to the controller's transform
        transform.SetParent(controller);

        // Calculate the target position in WORLD SPACE: Controller Position + World UP (Vector3.up) offset.
        Vector3 worldTargetPosition = controller.position + Vector3.up * hoverHeight;

        // Convert that World Space position into Local Space relative to the new parent.
        transform.localPosition = controller.InverseTransformPoint(worldTargetPosition);

        // Reset local rotation to prevent strange inherited rotations
        transform.localRotation = Quaternion.identity;

        // Activate the sprite GameObject
        gameObject.SetActive(true);
    }

    /// <summary>
    /// Called when the user releases the input (e.g., the Trigger button).
    /// </summary>
    public void DeactivateAndUnparent()
    {
        gameObject.SetActive(false);
        transform.SetParent(null);
    }

    #endregion
}
