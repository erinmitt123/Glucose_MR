using System;
using TMPro;
using UnityEngine;

//This script switches between which ui is active

public class UIControllerScript : MonoBehaviour
{
    public GameObject diabetesCanvas;
    public GameObject foodCanvas;
    public bool isTypeOne = true;

    [SerializeField] private TMP_InputField inputField;
    public event Action OnNonEmpty;
    public event Action OnEmpty;

    private bool wasEmpty = true;
    private bool isStartup = true;

    public void OnTypeTwoClicked(bool isOn = true)
    {
        if (!isOn) { return; }
        isTypeOne = false;
        setupGlucoseCanvas();
    }
    public void OnTypeOneClicked(bool isOn = true)
    {
        if (!isOn) { return; }
        isTypeOne = true;
        setupGlucoseCanvas();
    }
    public void setupGlucoseCanvas()
    {
        if (isStartup)
        {
            diabetesCanvas.SetActive(false);
            foodCanvas.SetActive(true);
            isStartup = false;
        }
        else { FoodInfo.Instance.UpdateValuesAndDisplay(); }
    }


}
