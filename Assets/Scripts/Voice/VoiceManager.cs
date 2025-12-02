using Pico.Platform;
using TMPro;
using UnityEngine.Android;
using UnityEngine;
using UnityEngine.InputSystem;


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

        // Storage-Write Permission
        if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageWrite))
        {
            Debug.Log("Permission wasn't given for write");
            Permission.RequestUserPermission(Permission.ExternalStorageWrite);
        }
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
        });
    }

    #endregion

    // Turns on the speech engine with preset parameters
    private void StartAsrEngine()
    {
        SpeechService.StartAsr(autoStop, showPunctuation, maxDuration);
        Debug.Log($"engine started, {autoStop}, {showPunctuation}, {maxDuration}");
        _isMicOn = true;
    }

    // Turns on the speech engine with custom parameters
    private void StartAsrEngine(bool isAutoStop, bool isShowPunctuation, int setMaxDuration)
    {
        SpeechService.StartAsr(isAutoStop, isShowPunctuation, setMaxDuration);
        Debug.Log($"engine started, {isAutoStop}, {isShowPunctuation}, {setMaxDuration}");
        _isMicOn = true;
    }


    // Manually force-stops the Speech Engine. Note: This does not stop SpeechService callbacks and can therefore finish before the last callback
    private void StopAsrEngine()
    {
        SpeechService.StopAsr();
        _isMicOn = false;
        Debug.Log("engine stopped");

        ApplicationManager.Instance.ParseVoice();
    }


    #region Mic Input Controls

    // Event subcriber to verify the user is trying to trigger the mic and to perform the appropriate action based on if autostop is on
    public void OnMicControlInput(InputAction.CallbackContext context)
    {
        // Verifies the button was pressed and the engine is ready, else exits
        if (!context.started) return;
        Debug.Log("Grip Button Pressed");

        if (!_inited)
        {
            Debug.Log($"Please init before start ASR");
            return;
        }

        // Triggers the appropriate speech service controls
        if (autoStop) AutostopMic();
        else ManualMic();
    }

    // Turns on the speech engine without a manual option to stop it
    private void AutostopMic()
    {
        SpeechService.StartAsr(autoStop, showPunctuation, maxDuration);
        Debug.Log($"engine started, {autoStop}, {showPunctuation}, {maxDuration}");
        _isMicOn = true;
    }

    // Turns on or off the speech engine using the same input trigger
    // TODO: Buggy and untested on headset
    private void ManualMic()
    {
       
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

            ApplicationManager.Instance.ParseVoice();
        }
           
    }

    private void OnEnable()
    {
        // Subscribes the appropriate event to the mic controls
        if (micAction != null)
        {
            micAction.action.started += OnMicControlInput;
            micAction.action.Enable();
        }
    }

    private void OnDisable()
    {
        // Unsubcribes the appropriate event from the mic controls
        if (micAction != null)
        {
            micAction.action.started -= OnMicControlInput;
            micAction.action.Disable();
        }       
    }

    #endregion

}