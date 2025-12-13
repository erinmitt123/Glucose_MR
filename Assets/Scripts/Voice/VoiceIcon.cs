using UnityEngine;

[RequireComponent (typeof(SmoothBillboardFollow))]
public class VoiceIcon : MonoBehaviour
{
    [Header("Controller Reference Points")]
    [Tooltip("The target transforms of the left and right controllers anchor points")]
    [SerializeField] private Transform leftController;
    [SerializeField] private Transform rightController;

    private SmoothBillboardFollow follow;

    private void Start()
    {
        follow = GetComponent<SmoothBillboardFollow>();
        DeactivateAndUnsetController();
    }

    #region -----PUBLIC METHODS FOR CONTROLLER INPUT-----

    /// <summary>
    /// Called when the user presses the input (e.g., the Grip button) on an XR Controller.
    /// </summary>
    /// <param name="controller">The Transform of the XR Controller that pressed the button.</param>
    public void ActivateAndSetController(bool isLeft)
    {
        // Determines which controller transform to parent to
        follow.targetController = isLeft ? leftController : rightController;
        follow.SetInitialPositionAndRotation();

        gameObject.SetActive(true);
    }

    /// <summary>
    /// Called when the user releases the input (e.g., the Trigger button).
    /// </summary>
    public void DeactivateAndUnsetController()
    {
        gameObject.SetActive(false);
        follow.targetController = null;
    }

    #endregion
}
