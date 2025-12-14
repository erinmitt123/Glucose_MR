using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class TransformDebugger : MonoBehaviour
{
    [SerializeField] private bool logContinuously = false;
    [SerializeField] private float logTime = 2f;
    [SerializeField] private List<Transform> targets;

    private void Start()
    {
        if (logContinuously) StartCoroutine(LogRepeatedly());
    }

    private IEnumerator LogRepeatedly()
    {
        while (logContinuously)
        {
            LogPositions();
            yield return new WaitForSeconds(logTime);
        }
    }

    [ContextMenu("Log Transform Positions")]
    private void LogPositions()
    {
        foreach (var t in targets)
        {
            if (!t) continue;
            Debug.Log($"[TransformDebugger] [{t.name}] Position: {t.position}");
        }
    }
}
