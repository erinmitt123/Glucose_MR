using UnityEngine;
using System.Text.RegularExpressions;
using System.Globalization;
using System.Threading.Tasks;

public class ApplicationManager : MonoBehaviour
{
    public static ApplicationManager Instance { get; private set; }

    [SerializeField] VoiceManager voiceManager;

    [Header("Runtime Values")]
    public double glucoseLevel = 100.00;
    public string identifiedObject = "banana";
    public int identifiedObjectIndex = 0;

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

    public async void ParseVoice()
    {
        Task glucoseTask = ParseGlucoseLevel();
        Task identifiedObjectTask = ParseKeyword();

        await Task.WhenAll(glucoseTask, identifiedObjectTask);

        FoodInfo.Instance.UpdateValuesAndDisplay();
    }

    private async Task ParseKeyword()
    {
        var keyword = await Task.Run(() =>
            KeywordParser.FindKeyword(voiceManager.storedAsrResult)
        );

        if (keyword != null)
        {
            identifiedObject = keyword;
            identifiedObjectIndex = FoodInfo.Instance.dictFoods[keyword] - 46;
            Debug.Log($"Object Identified: {identifiedObject}");
        }
        else
        {
            Debug.Log("There is no new identified object");
        }
    }

    private async Task ParseGlucoseLevel()
    {
        if (voiceManager.storedAsrResult == null)
        {
            Debug.Log("The glucose hasn't been shared yet. Please use the mic button to share your reading");
            return;
        }

        var number = await Task.Run(() => 
            Utils.GetDoubleFromNonnumericalSentence(voiceManager.storedAsrResult)
        );

        if (number != null)
        {
            glucoseLevel = (double)number;
            Debug.Log($"Voice Parsing Success: the glucose level is {glucoseLevel}");
        }
        else
        {
            Debug.Log("No float found. Please repeat and state your glucose level.");
        }
    }

}
