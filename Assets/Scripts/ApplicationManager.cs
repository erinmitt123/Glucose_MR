using UnityEngine;
using Unity.XR.PXR;
using System.Collections;

public class ApplicationManager : MonoBehaviour
{
    private void Awake()
    {
        PXR_Manager.EnableVideoSeeThrough = true;
    }

}
