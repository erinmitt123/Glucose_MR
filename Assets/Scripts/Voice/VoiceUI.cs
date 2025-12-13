using UnityEngine;

[RequireComponent (typeof(SmoothBillboardFollow), typeof(AudioSource))]
public class VoiceUI : MonoBehaviour
{
    [Header("Controller Reference Points")]
    [Tooltip("The target transforms of the left and right controllers anchor points")]
    [SerializeField] private Transform leftController;
    [SerializeField] private Transform rightController;

    [Header("Audio Clips")]
    [SerializeField] private AudioClip startSound;
    [SerializeField] private AudioClip endSound;

    private SmoothBillboardFollow follow;
    private AudioSource audioSource;
    private SpriteRenderer spriteRenderer;

    private void Start()
    {
        follow = GetComponent<SmoothBillboardFollow>();
        audioSource = GetComponent<AudioSource>();
        spriteRenderer = GetComponent<SpriteRenderer>();

        DeactivateVisuals();
    }

    #region -----PUBLIC METHODS FOR CONTROLLER INPUT-----

    /// <summary>
    /// Called when the user presses the input (e.g., the Grip button) on an XR Controller.
    /// </summary>
    /// <param name="controller">The Transform of the XR Controller that pressed the button.</param>
    public void Activate(bool isLeft)
    {
        // Determines which controller transform to parent to
        follow.targetController = isLeft ? leftController : rightController;

        ActivateSound();
        ActivateVisuals();
    }

    /// <summary>
    /// Called when the user releases the input (e.g., the Trigger button).
    /// </summary>
    public void Deactivate()
    {
        DeactivateSound();
        DeactivateVisuals();
    }

    #endregion

    #region -----PRIVATE METHODS FOR MODULAR UI-----

    private void ActivateVisuals()
    {
        follow.SetInitialPositionAndRotation();
        follow.enabled = true;
        spriteRenderer.enabled = true;
    }

    private void ActivateSound()
    {
        audioSource.clip = startSound;
        audioSource.Play();
    }

    private void DeactivateVisuals()
    {
        follow.enabled = false;
        spriteRenderer.enabled = false;
    }

    private void DeactivateSound()
    {
        audioSource.clip = endSound;
        audioSource.Play();
    }

    #endregion
}
