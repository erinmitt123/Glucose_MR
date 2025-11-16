using System;
using TMPro;
using UnityEngine;

public class UIControllerScript : MonoBehaviour
{
    public GameObject diabetesCanvas;
    public GameObject glucoseCanvas;
    public GameObject foodCanvas;
    public bool isTypeOne = true;

    [SerializeField] private TMP_InputField inputField;
    public event Action OnNonEmpty;
    public event Action OnEmpty;

    private bool wasEmpty = true;

    private void Start()
    {
        inputField.onSubmit.AddListener(OnSubmit);
    }

    private void Awake()
    {
        if (inputField == null)
            inputField = GetComponent<TMP_InputField>();

        // Subscribe to value changes
        inputField.onValueChanged.AddListener(HandleTextChange);
    }

    private void HandleTextChange(string newText)
    {
        bool isEmpty = string.IsNullOrWhiteSpace(newText);

        if (isEmpty && !wasEmpty)
        {
            // Just became empty
            OnEmpty?.Invoke();
        }
        else if (!isEmpty && wasEmpty)
        {
            // Just became non-empty
            OnNonEmpty?.Invoke();
        }

        wasEmpty = isEmpty;
    }
    private void OnSubmit(string text)
    {
        Debug.Log("Enter pressed! Submitted: " + text);
        setupfoodCanvas();
    }

    private void DoSomething()
    {
        // Your action here
        Debug.Log("Action triggered from XR keyboard Enter!");
    }
    public void OnTypeTwoClicked()
    {
        isTypeOne = true;
        setupGlucoseCanvas();
    }
    public void OnTypeOneClicked()
    {
        isTypeOne = false;
        setupGlucoseCanvas();
    }
    public void setupGlucoseCanvas()
    {
        diabetesCanvas.SetActive(false);
        glucoseCanvas.SetActive(true);
    }
    public void setupfoodCanvas()
    {
        glucoseCanvas.SetActive(false);
        foodCanvas.SetActive(true);
    }

}
