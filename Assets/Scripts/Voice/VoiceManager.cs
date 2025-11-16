using Pico.Platform;
using TMPro;
using UnityEngine.Android;
using UnityEngine;
using UnityEngine.InputSystem;


public class VoiceManager : MonoBehaviour
{
    public TMP_Text displayAsrResult;
    public string storedAsrResult;

    [Header("Parameters")]
    [SerializeField] private int maxDuration = 8;
    [SerializeField] private bool autoStop = false;
    [SerializeField] private bool showPunctuation = true;

    [Header("Controller References")]
    [SerializeField] private InputActionAsset inputActions;
    [SerializeField] private InputActionReference micAction;

    private bool _inited = false;
    private bool _isMicOn = false;


    private void Awake()
    {

        CoreService.Initialize();

        RequestPermissions();

        SetCallbacks();

        InitializeAsrEngine();
    }


    // Checks if necessary permissions have been granted for using the speech service
    private void RequestPermissions()
    {
        // Utilities to access Java objects
        //AndroidJavaObject javaObj = new AndroidJavaObject("applicationpermissions.AndroidPermissions");
        //AndroidJavaClass jc = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
        //AndroidJavaObject jo = jc.GetStatic<AndroidJavaObject>("currentActivity");
        //javaObj.Call("setUnityActivity", jo);
        //javaObj.Call("requestExternalStorage");
        
        // Unity permissions requests
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Debug.Log("Permission wasn't given for mic");
            Permission.RequestUserPermission(Permission.Microphone);
        }

        if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageWrite))
        {
            Debug.Log("Permission wasn't given for write");
            Permission.RequestUserPermission(Permission.ExternalStorageWrite);
        }
    }

    // Sets callback messages and result logging for Speech Service
    private void SetCallbacks()
    {
        SpeechService.SetOnAsrResultCallback(msg =>
        {
            Debug.Log("ASR Result Callback done successfully");
            var m = msg.Data;
            storedAsrResult = m.Text;

            Debug.Log($"text={m.Text} isFinal={m.IsFinalResult}");
            displayAsrResult.SetText($"[{m.IsFinalResult}]{m.Text}");
        });
        SpeechService.SetOnSpeechErrorCallback(msg =>
        {
            var m = msg.Data;
            Debug.Log($"SpeechError :{JsonUtility.ToJson(m)}");
        });
    }

    private void InitializeAsrEngine()
    {
        var res = SpeechService.InitAsrEngine();
        if (res != AsrEngineInitResult.Success)
        {
            Debug.Log($"Init ASR Engine failed :{res}");
        }
        else
        {
            _inited = true;
            Debug.Log("Init engine successfully.");
        }
    }


    private void Start()
    {

        //// Start Button
        //buttonStart.onClick.AddListener(() =>
        //{
        //    if (!_inited)
        //    {
        //        Debug.Log($"Please init before start ASR");
        //        textAsrResult.SetText("Please init engine first");
        //        return;
        //    }

        //    //bool autoStop = toggleAutoStop.isOn;
        //    //bool showPunctual = toggleShowPunctual.isOn;
        //    //int.TryParse(inputMaxDuration.text, out var maxDuration);

        //    SpeechService.StartAsr(autoStop, showPunctuation, maxDuration);
        //    Debug.Log($"engine started, {autoStop}, {showPunctuation}, {maxDuration}");
        //});

        //// Stop Button 
        //buttonStop.onClick.AddListener(() =>
        //{
        //    if (!_inited)
        //    {
        //        Debug.Log($"Please init before start ASR");
        //        textAsrResult.SetText("Please init engine first");
        //        return;
        //    }

        //    SpeechService.StopAsr();
        //    Debug.Log("engine stopped");
        //});


    }

    public void OnMicButtonPressed(InputAction.CallbackContext context)
    {
        if (context.started)
        {
            Debug.Log("Grip Button Pressed");

            if (!_inited)
            {
                Debug.Log($"Please init before start ASR");
                return;
            }

            if (!_isMicOn)
            {
                SpeechService.StartAsr(autoStop, showPunctuation, maxDuration);
                Debug.Log($"engine started, {autoStop}, {showPunctuation}, {maxDuration}");
                _isMicOn = true;
            }
            else
            {
                SpeechService.StopAsr();
                Debug.Log("engine stopped");
                _isMicOn = false;
            }
            
        }
    }

    private void OnEnable()
    {
        if (micAction != null)
        {
            micAction.action.started += OnMicButtonPressed;
            micAction.action.Enable();
        }
    }

    private void OnDisable()
    {
        if (micAction != null)
        {
            micAction.action.started -= OnMicButtonPressed;
            micAction.action.Disable();
        }       
    }

}