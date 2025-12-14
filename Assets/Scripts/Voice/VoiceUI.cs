using System.Xml.Serialization;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Inputs;
using static UnityEngine.XR.Interaction.Toolkit.Inputs.XRInputModalityManager;

[RequireComponent (typeof(SmoothBillboardFollow), typeof(AudioSource))]
public class VoiceUI : MonoBehaviour
{
    [Header("XR Anchor Provider")]
    [Tooltip("Typically attached to the XR Rig, containing references to the controllers and hands")]
    [SerializeField] private XRAnchorProvider anchorProvider;

    [Header("Audio Clips")]
    [SerializeField] private AudioClip startSound;
    [SerializeField] private AudioClip endSound;

    private SmoothBillboardFollow follow;
    private AudioSource audioSource;
    private SpriteFader spriteFader;
    private PulseScale pulseScaler;

    private void Start()
    {
        follow = GetComponent<SmoothBillboardFollow>();
        audioSource = GetComponent<AudioSource>();
        spriteFader = GetComponent<SpriteFader>();
        pulseScaler = GetComponent<PulseScale>();

        DeactivateVisuals();
    }

    #region -----PUBLIC METHODS FOR PLAYER INPUT-----

    /// <summary>
    /// Called when the user presses the input (e.g., the Grip button) on an XR Controller.
    /// </summary>
    /// <param name="controller">The Transform of the XR Controller that pressed the button.</param>
    public void Activate(bool isLeft)
    {
        // Determines which controller transform to parent to, and exits if nothing is being tracked
        Transform anchor = isLeft ? anchorProvider.Left : anchorProvider.Right;
        if (!anchor) return;

        // Determines which controller transform to parent to
        follow.targetController = anchor;

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

        pulseScaler.ResetScale();
        pulseScaler.enabled = true;

        spriteFader.FadeIn();
    }

    private void ActivateSound()
    {
        audioSource.clip = startSound;
        audioSource.Play();
    }

    private void DeactivateVisuals()
    {
        follow.enabled = false;
        pulseScaler.enabled = false;
        spriteFader.FadeOut();
    }

    private void DeactivateSound()
    {
        audioSource.clip = endSound;
        audioSource.Play();
    }

    #endregion
}
