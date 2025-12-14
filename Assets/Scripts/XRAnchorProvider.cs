using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Inputs;
using static UnityEngine.XR.Interaction.Toolkit.Inputs.XRInputModalityManager;

public class XRAnchorProvider : MonoBehaviour
{
    [SerializeField] private XRInputModalityManager modalityManager;

    public Transform Left { get; private set; }
    public Transform Right { get; private set; }

    [Header("Optional Override Transforms")]
    [SerializeField] private Transform leftHand;
    [SerializeField] private Transform rightHand;
    [SerializeField] private Transform leftController;
    [SerializeField] private Transform rightController;

    private void Start()
    {
        if (modalityManager == null && !TryGetComponent<XRInputModalityManager>(out modalityManager))
            Debug.LogError("XR Anchor Provider was unable to locate an XR Input Modality Manager to pull references from");           
    }

    private void OnEnable()
    {
        CacheRefs();
        OnInputModeChanged(XRInputModalityManager.currentInputMode.Value);
        XRInputModalityManager.currentInputMode.Subscribe(OnInputModeChanged);
    }

    private void OnDisable() => XRInputModalityManager.currentInputMode.Unsubscribe(OnInputModeChanged);

    private void CacheRefs()
    {
        leftHand = leftHand != null ? leftHand : modalityManager.leftHand.transform;
        rightHand = rightHand != null ? rightHand : modalityManager.rightHand.transform;
        leftController = leftController != null ? leftController : modalityManager.leftController.transform;
        rightController = rightController != null ? rightController : modalityManager.rightController.transform;
    }

    private void OnInputModeChanged(InputMode mode)
    {
        switch (mode)
        {
            case InputMode.TrackedHand:
                Left = leftHand;
                Right = rightHand;
                break;

            case InputMode.MotionController:
                Left = leftController;
                Right = rightController;
                break;

            default:
                Left = null;
                Right = null;
                break;
        }
    }
}
