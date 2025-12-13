using UnityEngine;

public class PulseScale : MonoBehaviour
{
    [SerializeField] private float pulseSpeed = 2f;
    [SerializeField] private float pulseAmplitude = 0.001f; // world-relative delta

    private Vector3 baseScale;

    void OnEnable()
    {
        baseScale = transform.localScale;
    }

    void Update()
    {
        float offset = Mathf.Sin(Time.time * pulseSpeed) * pulseAmplitude;
        transform.localScale = baseScale + Vector3.one * offset;
    }

    public void ResetScale() => transform.localScale = baseScale;
    
}