using System;
using TMPro;
using UnityEngine;

public class UIControllerScript : MonoBehaviour
{
    public GameObject diabetesCanvas;
    public GameObject foodCanvas;
    public bool isTypeOne = true;

    [SerializeField] private TMP_InputField inputField;
    public event Action OnNonEmpty;
    public event Action OnEmpty;

    private bool wasEmpty = true;


    public void OnTypeTwoClicked()
    {
        isTypeOne = false;
        setupGlucoseCanvas();
    }
    public void OnTypeOneClicked()
    {
        isTypeOne = true;
        setupGlucoseCanvas();
    }
    public void setupGlucoseCanvas()
    {
        diabetesCanvas.SetActive(false);
        foodCanvas.SetActive(true);
    }

}
