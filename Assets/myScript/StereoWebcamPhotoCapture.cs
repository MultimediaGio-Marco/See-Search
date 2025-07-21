using UnityEngine;
using UnityEngine.UI; // <--- aggiunto
using UnityEngine.XR;
using System.Collections;
using PassthroughCameraSamples;
using System;

public class StereoWebcamPhotoCapture : MonoBehaviour
{
    [Header("WebCamTexture Manager")]
    public WebCamTextureManager webCamTextureManager;

    [Header("RawImage UI")]
    public RawImage leftRawImage;
    public RawImage rightRawImage;

    [Header("Controller Settings")]
    public XRNode controllerHand = XRNode.RightHand;

    [Header("Image Filter Client")]
    public client imageFilterClient;

    private Texture2D leftPhoto;
    private Texture2D rightPhoto;

    private bool triggerPressed = false;
    private bool isCapturing = false;

    void Update()
    {
        if (!isCapturing)
            CheckTriggerInput();
    }

    void CheckTriggerInput()
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(controllerHand);

        if (device.isValid && device.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerValue))
        {
            if (triggerValue && !triggerPressed)
            {
                Debug.Log("Grilletto premuto - Avvio cattura stereo");
                StartCoroutine(CaptureStereoAndSend());
            }
            triggerPressed = triggerValue;
        }
    }

    IEnumerator CaptureStereoAndSend()
    {
        isCapturing = true;

        // LEFT
        webCamTextureManager.Eye = PassthroughCameraEye.Left;
        yield return new WaitForEndOfFrame();
        yield return new WaitForSeconds(0.1f);
        CapturePhoto(ref leftPhoto);
        leftRawImage.texture = leftPhoto;
        Debug.Log("📷 Foto occhio sinistro acquisita");

        // RIGHT
        webCamTextureManager.Eye = PassthroughCameraEye.Right;
        yield return new WaitForEndOfFrame();
        yield return new WaitForSeconds(0.1f);
        CapturePhoto(ref rightPhoto);
        rightRawImage.texture = rightPhoto;
        Debug.Log("📷 Foto occhio destro acquisita");

        // INVIO
        if (imageFilterClient != null)
        {
            Debug.Log("🌐 Invio immagini al server...");
            yield return StartCoroutine(imageFilterClient.RequestLableForImage(rightPhoto, leftPhoto));
        }
        else
        {
            Debug.LogWarning("⚠️ imageFilterClient non assegnato!");
        }

        isCapturing = false;
    }

    void CapturePhoto(ref Texture2D photo)
    {
        int width = webCamTextureManager.WebCamTexture.width;
        int height = webCamTextureManager.WebCamTexture.height;

        if (photo == null || photo.width != width || photo.height != height)
        {
            photo = new Texture2D(width, height, TextureFormat.RGB24, false);
        }

        Color32[] pixels = new Color32[width * height];
        webCamTextureManager.WebCamTexture.GetPixels32(pixels);
        photo.SetPixels32(pixels);
        photo.Apply();
    }
}
