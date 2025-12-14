using UnityEngine;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Hands.Gestures;
using UnityEngine.XR.Hands.Samples.GestureSample;

public class DetectGesture : MonoBehaviour
{
    [Header("Hand Data")]
    [SerializeField] private XRHandTrackingEvents handTrackingEvents;
    [SerializeField] private XRHandShape[] handShapes;
    [SerializeField] private HandShapeCompletenessCalculator handShapeCompletenessCalculator;

    [Header("Detection Parameters")]
    [SerializeField] private float gestureDetectionInterval = 0.1f;
    [SerializeField] private float minimumDetectionThreshold = 0.9f;

    private float timeOfLastConditionCheck;

    private void OnEnable() => handTrackingEvents.jointsUpdated.AddListener(OnJointsUpdated);

    private void OnDisable() => handTrackingEvents.jointsUpdated.RemoveListener(OnJointsUpdated);

    private void OnJointsUpdated(XRHandJointsUpdatedEventArgs args)
    {
        if (Time.time - timeOfLastConditionCheck < gestureDetectionInterval)
            return;

        foreach (var handShape in handShapes)
        {
            handShapeCompletenessCalculator.TryCalculateHandShapeCompletenessScore(args.hand, handShape, out float completenessScore);

            if (handTrackingEvents.handIsTracked && completenessScore >= minimumDetectionThreshold)
                Debug.Log($"Hand Gesture Detected: {handShape.name} | Score: {completenessScore}");
        }

        timeOfLastConditionCheck = Time.time;
    }
}


