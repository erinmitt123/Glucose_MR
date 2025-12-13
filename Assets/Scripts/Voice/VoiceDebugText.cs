using TMPro;
using UnityEngine;

public class VoiceDebugText : MonoBehaviour
{
    private TMP_Text text;
    private VoiceManager voiceManager;

    private void Start()
    {
        text = GetComponent<TMP_Text>();
        voiceManager = ApplicationManager.Instance.voiceManager;
    }

    private void Update()
    {
        text.text = voiceManager.storedAsrResult;
    }
}
