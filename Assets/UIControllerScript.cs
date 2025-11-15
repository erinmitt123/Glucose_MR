using UnityEngine;

public class UIControllerScript : MonoBehaviour
{
    bool isTypeOne = false;
    void Start()
    {
        
    }

    // Update is called once per frame
    public void OnTypeTwoClicked()
    {
        isTypeOne = true;
    }
    public void OnTypeOneClicked()
    {
        isTypeOne = false;
    }
}
