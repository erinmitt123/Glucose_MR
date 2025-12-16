using UnityEngine;
using Unity.XR.PXR;
using System.Collections;

public class ApplicationManager : MonoBehaviour
{
    public static ApplicationManager Instance { get; private set; }

    private void Awake()
    {
        // Enforce singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

        // Unparents the game object so that it can be protected on load
        gameObject.transform.parent = null;
        DontDestroyOnLoad(gameObject);

        // Enables seethrough mode to start mixed reality scene
        PXR_Manager.EnableVideoSeeThrough = true;
    }

}
