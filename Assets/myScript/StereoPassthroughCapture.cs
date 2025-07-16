using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.UI;

public class PassthroughStereoCapture : MonoBehaviour
{
    public RawImage leftEyeRawImage;   // opzionale, serve per debug o mostrare feed
    public RawImage rightEyeRawImage;  // opzionale

    public int captureWidth = 1024;
    public int captureHeight = 1024;

    public client imageFilterClient;   // assegna in Inspector

    private WebCamTexture leftEyeCamTex;
    private WebCamTexture rightEyeCamTex;

    private bool isCapturing = false;
    public XRNode controllerHand = XRNode.RightHand;
    private bool triggerPressed = false;

    void Start()
    {
        StartCoroutine(InitPassthroughCameras());
    }

    IEnumerator InitPassthroughCameras()
    {
        yield return new WaitForSeconds(1f);

        var devices = WebCamTexture.devices;
        string leftCamName = null;
        string rightCamName = null;

        foreach (var device in devices)
        {
            Debug.Log("WebCam device found: " + device.name);
            if (device.name.ToLower().Contains("left"))
                leftCamName = device.name;
            if (device.name.ToLower().Contains("right"))
                rightCamName = device.name;
        }

        if (leftCamName == null || rightCamName == null)
        {
            Debug.LogError("WebCam left or right eye not found!");
            yield break;
        }

        leftEyeCamTex = new WebCamTexture(leftCamName, captureWidth, captureHeight, 30);
        leftEyeRawImage.texture = leftEyeCamTex;
        leftEyeCamTex.Play();

        rightEyeCamTex = new WebCamTexture(rightCamName, captureWidth, captureHeight, 30);
        rightEyeRawImage.texture = rightEyeCamTex;
        rightEyeCamTex.Play();
    }

    void Update()
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(controllerHand);
        if (!device.isValid)
            return;

        if (device.TryGetFeatureValue(CommonUsages.triggerButton, out bool pressed))
        {
            if (pressed && !triggerPressed && !isCapturing)
            {
                StartCoroutine(CaptureAndSend());
            }
            triggerPressed = pressed;
        }
    }

    IEnumerator CaptureAndSend()
    {
        isCapturing = true;

        yield return new WaitForEndOfFrame();

        if (leftEyeCamTex == null || rightEyeCamTex == null)
        {
            Debug.LogError("Camere passthrough non inizializzate");
            isCapturing = false;
            yield break;
        }

        Texture2D leftTex = new Texture2D(leftEyeCamTex.width, leftEyeCamTex.height, TextureFormat.RGB24, false);
        leftTex.SetPixels(leftEyeCamTex.GetPixels());
        leftTex.Apply();

        Texture2D rightTex = new Texture2D(rightEyeCamTex.width, rightEyeCamTex.height, TextureFormat.RGB24, false);
        rightTex.SetPixels(rightEyeCamTex.GetPixels());
        rightTex.Apply();

        if (imageFilterClient != null)
        {
            yield return StartCoroutine(imageFilterClient.RequestLableForImage(rightTex, leftTex));
        }

        Destroy(leftTex);
        Destroy(rightTex);

        isCapturing = false;
    }
}
