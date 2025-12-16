using UnityEngine;
using System;
using System.Collections;
using UnityEngine.XR.Interaction.Toolkit.UI.BodyUI;
using UnityEngine.XR.Interaction.Toolkit;
using System.Runtime.CompilerServices;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using UnityEngine.UIElements;

public class HandMenuInteractionDisabler : MonoBehaviour
{
    [SerializeField] private HandMenu handMenu;

    [Header("Interaction Groups")]
    [SerializeField] private XRInteractionGroup leftHandGroup;
    [SerializeField] private XRInteractionGroup leftControllerGroup;
    [SerializeField] private XRInteractionGroup rightHandGroup;
    [SerializeField] private XRInteractionGroup rightControllerGroup;

    private HandMenu.MenuHandedness lastTargetHand = HandMenu.MenuHandedness.None;
    private HandMenu.MenuHandedness lastMenuHandedness = HandMenu.MenuHandedness.None;

    private void Awake()
    {
        if (handMenu == null)
            Debug.LogError("Hand Menu Interaction Disabler cannot detect a hand menu to keep track of");
    }

    private void OnDisable()
    {
        // Re-enable all interactors on both hands
        SetHandGroupsEnabled(HandMenu.MenuHandedness.Left, true);
        SetHandGroupsEnabled(HandMenu.MenuHandedness.Right, true);

        // Reset tracking
        lastTargetHand = HandMenu.MenuHandedness.None;
    }

    private void Update()
    {
        if (handMenu == null) return;

        // Determine which hand is currently holding the menu
        HandMenu.MenuHandedness targetHand = handMenu.menuHandedness;

        // Changes targetHand so that it is not either and will go with the most recently detected hand
        if (targetHand == HandMenu.MenuHandedness.Either)
        {
            // lastMenuHandedness should not be able to be equal to None here
            if (lastMenuHandedness != HandMenu.MenuHandedness.Either)
                targetHand = lastMenuHandedness == HandMenu.MenuHandedness.Left ? HandMenu.MenuHandedness.Right : HandMenu.MenuHandedness.Left;
            else
                targetHand = lastTargetHand;
        }

        if (lastTargetHand != targetHand)
        {
            if (lastTargetHand != HandMenu.MenuHandedness.None)
                SetHandGroupsEnabled(lastTargetHand, true);

            if (targetHand != HandMenu.MenuHandedness.None)
                SetHandGroupsEnabled(targetHand, false);

            lastTargetHand = targetHand;
        }

        lastMenuHandedness = handMenu.menuHandedness;
    }

    private void SetHandGroupsEnabled(HandMenu.MenuHandedness hand, bool enabled)
    {
        XRInteractionGroup[] groups = hand == HandMenu.MenuHandedness.Left
            ? new[] { leftHandGroup, leftControllerGroup }
            : new[] { rightHandGroup, rightControllerGroup };

        foreach (var group in groups)
        {
            if (group == null) continue;
            foreach (var interactor in group.startingGroupMembers)
            {
                if (interactor is XRBaseInteractor xrInteractor)
                    xrInteractor.enabled = enabled;
            }
        }
    }
}
