using UnityEngine;
using UnityEngine.XR;
using System.Collections;
using PassthroughCameraSamples;
using System;

public class StereoPhotoCapture : MonoBehaviour
{
    [Header("WebCamTexture Manager")]
    public WebCamTextureManager webCamTextureManager;

    [Header("Quad Renderers")]
    public Renderer leftQuadRenderer;
    public Renderer rightQuadRenderer;

    [Header("Controller Settings")]
    public XRNode controllerHand = XRNode.RightHand;
    public string TextureName = "_MainTex";

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

        if (device.isValid)
        {
            bool triggerValue;
            if (device.TryGetFeatureValue(CommonUsages.triggerButton, out triggerValue))
            {
                if (triggerValue && !triggerPressed)
                {
                    Debug.Log("Grilletto premuto - Avvio cattura stereo");
                    StartCoroutine(CaptureStereo());
                }
                triggerPressed = triggerValue;
            }
        }
    }

    IEnumerator CaptureStereo()
    {
        isCapturing = true;

        // --- LEFT EYE ---
        webCamTextureManager.Eye = PassthroughCameraEye.Left;
        yield return new WaitForEndOfFrame(); // attende aggiornamento texture
        yield return new WaitForSeconds(0.1f); // opzionale: attende che la webcam aggiorni davvero
        CapturePhoto(ref leftPhoto);
        leftQuadRenderer.material.SetTexture(TextureName, leftPhoto);
        Debug.Log("Foto occhio sinistro acquisita");

        // --- RIGHT EYE ---
        webCamTextureManager.Eye = PassthroughCameraEye.Right;
        yield return new WaitForEndOfFrame();
        yield return new WaitForSeconds(0.1f);
        CapturePhoto(ref rightPhoto);
        rightQuadRenderer.material.SetTexture(TextureName, rightPhoto);
        Debug.Log("Foto occhio destro acquisita");

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
