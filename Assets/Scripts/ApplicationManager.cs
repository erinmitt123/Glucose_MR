using UnityEngine;
using Unity.XR.PXR;

public class ApplicationManager : MonoBehaviour
{
    private void Awake()
    {
        PXR_Manager.EnableVideoSeeThrough = true;
    }
}
