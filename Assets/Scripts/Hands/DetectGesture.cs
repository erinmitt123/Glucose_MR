using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Hands.Gestures;
using UnityEngine.XR.Hands.Samples.GestureSample;

[System.Serializable]
public class HandednessEvent : UnityEvent<Handedness> { }

public class DetectGesture : MonoBehaviour
{
    [Header("Hand Data")]
    [SerializeField] private XRHandTrackingEvents handTrackingEvents;
    [SerializeField] private XRHandShape handShape;
    [SerializeField] private HandShapeCompletenessCalculator handShapeCompletenessCalculator;

    [Header("Detection Parameters")]
    [SerializeField] private float gestureDetectionInterval = 0.1f;
    [SerializeField] private float minimumDetectionThreshold = 0.9f;
    [Tooltip("Time in seconds to wait before the event can be triggered again.")]
    [SerializeField] private float cooldownDuration = 2.0f;


    [Header("Events")]
    public HandednessEvent onGestureDetected;

    private float timeOfLastConditionCheck;
    private float lastTriggerTime = -999f;
    private bool isGestureActive = false;

    private void OnEnable() => handTrackingEvents.jointsUpdated.AddListener(OnJointsUpdated);

    private void OnDisable() => handTrackingEvents.jointsUpdated.RemoveListener(OnJointsUpdated);

    private void OnJointsUpdated(XRHandJointsUpdatedEventArgs args)
    {
        // Skips if not enough time has passed since the last score check
        if (Time.time - timeOfLastConditionCheck < gestureDetectionInterval)
            return;

        // Determines if it's been too soon since the last positive check
        bool isInCoolDown = Time.time < lastTriggerTime + cooldownDuration;

        handShapeCompletenessCalculator.TryCalculateHandShapeCompletenessScore(args.hand, handShape, out float completenessScore);
        bool poseIsDetected = handTrackingEvents.handIsTracked && completenessScore >= minimumDetectionThreshold;

        if (poseIsDetected && !isGestureActive && !isInCoolDown)
        {
            Debug.Log($"Hand Gesture Performed: {handShape.name} | Score: {completenessScore}");
            onGestureDetected.Invoke(handTrackingEvents.handedness);

            lastTriggerTime = Time.time;
            isGestureActive = true;
        }
        else if (!poseIsDetected)
            isGestureActive = false;
        
        timeOfLastConditionCheck = Time.time;
    }
}


