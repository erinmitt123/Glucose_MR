using UnityEngine;
using System.Text.RegularExpressions;
using System.Globalization;

public class ApplicationManager : MonoBehaviour
{
    public static ApplicationManager Instance { get; private set; }
    public FoodInfo foodInfo;

    [SerializeField] VoiceManager voiceManager;

    public double glucoseLevel = 100.00;

    private string regexPattern = @"\d+(\.\d+)?";

    private void Awake()
    {
        // Enforce singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);

        if (voiceManager == null)
        {
            voiceManager = GetComponentInChildren<VoiceManager>();
            if (voiceManager == null)
            {
                Debug.Log("Voice Manager does not exist or is not a child of Application Manager.");
            }
        }
    }

    public void ParseGlucoseLevelFromVoice()
    {
        if (voiceManager.storedAsrResult == null)
        {
            Debug.Log("The glucose hasn't been shared yet. Please use the mic button to share your reading");
            return;
        }

        var number = Utils.GetDoubleFromNonnumericalSentence(voiceManager.storedAsrResult);

        if (number != null)
        {
            glucoseLevel = (double)number;
            foodInfo.UpdateValues();

            Debug.Log($"Voice Parsing Success: the glucose level is {glucoseLevel}");
        }
        else
        {
            Debug.Log("No float found. Please repeat and state your glucose level.");
        }
    }

}
