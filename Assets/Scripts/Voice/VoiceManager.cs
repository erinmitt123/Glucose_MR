using Pico.Platform;
using TMPro;
using UnityEngine.Android;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR;
using System.Linq;
using CommonUsages = UnityEngine.InputSystem.CommonUsages;
using UnityEngine.XR.Hands;


public class VoiceManager : MonoBehaviour
{
    // Publicly accessible storage of transcribed text from the user
    public string storedAsrResult;

    // Speech Service settings
    [Header("Parameters")]
    [SerializeField] private int maxDuration = 8; // longest time it runs
    [SerializeField] private bool autoStop = false; // whether manually stopped or autostopped based on voice detection
    [SerializeField] private bool showPunctuation = true; 

    // How the Speech Service can be triggered by the user
    [Header("Input Action References")]
    [SerializeField] private InputActionReference micAction;

    [Header("Optional UI")]
    public bool useUI;
    [SerializeField] VoiceUI voiceIcon;

    private bool _inited = false;
    private bool _isMicOn = false;
    private bool _isLeft = false;


    private void Awake()
    {
        CoreService.Initialize();

        RequestPermissions();

        SetCallbacks();

        InitializeAsrEngine();
    }

    #region Android-Version-Dependent Permission Setup

    private bool IsAndroidAbove12()
    {
        using (var version = new AndroidJavaClass("android.os.Build$VERSION"))
        {
            int sdkInt = version.GetStatic<int>("SDK_INT");
            return sdkInt > 32;
        }
    }

    private void ForceRequestStoragePermission()
    {
        AndroidJavaObject javaObj = new AndroidJavaObject("applicationPermissions.AndroidPermissions");
        AndroidJavaClass jc = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
        AndroidJavaObject jo = jc.GetStatic<AndroidJavaObject>("currentActivity");
        javaObj.Call("setUnityActivity", jo);
        javaObj.Call("requestExternalStorage");
    }

    #endregion

    #region Setup

    // Checks if necessary permissions have been granted for using the speech service
    private void RequestPermissions()
    {
        // Microphone Permission
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Debug.Log("Permission wasn't given for mic");
            Permission.RequestUserPermission(Permission.Microphone);
        }

#if UNITY_ANDROID && !UNITY_EDITOR

        if (IsAndroidAbove12()) 
        {
            ForceRequestStoragePermission();
        }
        else
        {
            if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageRead))
            {
                Debug.Log("Permission wasn't given for EXTERNAL_READ");
                Permission.RequestUserPermission(Permission.ExternalStorageRead);
            }

            if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageWrite))
            {
                Debug.Log("Permission wasn't given for EXTERNAL_WRITE");
                Permission.RequestUserPermission(Permission.ExternalStorageWrite);
            }
        }
        
#endif

    }

    // Mandatory code to initialize the speech engine so it's accessible to the user on command
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

    // Sets callback messages and result logging for Speech Service
    private void SetCallbacks()
    {
        SpeechService.SetOnAsrResultCallback(msg =>
        {
            Debug.Log("ASR Result Callback done successfully");
            var m = msg.Data;

            storedAsrResult = m.Text;
            Debug.Log($"text={m.Text} isFinal={m.IsFinalResult}");

            if (m.IsFinalResult)
                StopAsrEngine();
        });

        SpeechService.SetOnSpeechErrorCallback(msg =>
        {
            var m = msg.Data;
            Debug.Log($"SpeechError :{JsonUtility.ToJson(m)}");
            StopAsrEngine();
        });
    }

#endregion

    // Turns on the speech engine with preset parameters
    private void StartAsrEngine()
    {
        SpeechService.StartAsr(autoStop, showPunctuation, maxDuration);
        Debug.Log($"engine started, {autoStop}, {showPunctuation}, {maxDuration}");

        if (useUI) voiceIcon.Activate(_isLeft);
        _isMicOn = true;
    }

    // Manually force-stops the Speech Engine. Note: This does not stop SpeechService callbacks and can therefore finish before the last callback
    private void StopAsrEngine()
    {
        SpeechService.StopAsr();
        Debug.Log("engine stopped");

        if (useUI) voiceIcon.Deactivate();
        _isMicOn = false;

        ParseManager.Instance.ParseVoice();
    }

    #region Mic Input Controls

    // Event subcriber to verify the user is trying to trigger the mic and to perform the appropriate action based on if autostop is on
    public void OnMicControllerInput(InputAction.CallbackContext context)
    {
        // Verifies the button was pressed and the engine is ready, else exits
        if (!context.started) return;
        Debug.Log("Grip Button Pressed");

        // Determine which hand controller triggered the mic for hand-specific UI interactions
        if (useUI)
            _isLeft = context.control.device.usages.Contains(CommonUsages.LeftHand);

        TriggerSpeechService();
    }

    public void MicHandGestureInput(Handedness hand)
    {
        Debug.Log("Mic Hand Gesture Started");

        if (useUI)
            _isLeft = hand == Handedness.Left;

        TriggerSpeechService();
    }

    private void TriggerSpeechService()
    {
        if (!_inited)
        {
            Debug.Log($"Please init before start ASR");
            return;
        }

        // Triggers the appropriate speech service controls
        if (autoStop) StartAsrEngine();
        else ManualMic();
    }

    // Turns on or off the speech engine using the same input trigger
    // TODO: Buggy and untested on headset
    private void ManualMic()
    {
        if (!_isMicOn) StartAsrEngine();
        else StopAsrEngine();           
    }

    private void OnEnable()
    {
        // Subscribes the appropriate event to the mic controls
        if (micAction != null)
        {
            micAction.action.started += OnMicControllerInput;
            micAction.action.Enable();
        }
    }

    private void OnDisable()
    {
        // Unsubcribes the appropriate event from the mic controls
        if (micAction != null)
        {
            micAction.action.started -= OnMicControllerInput;
            micAction.action.Disable();
        }       
    }

    #endregion

}