using UnityEngine;
using UnityEngine.XR;
using System.Collections;
using PassthroughCameraSamples;
using System;

public class SingleEyePhotoCapture : MonoBehaviour
{
    [Header("WebCamTexture Manager")]
    public WebCamTextureManager webCamTextureManager;
    

    [Header("Quad Renderer")]
    public Renderer quadRenderer;

    [Header("Controller Settings")]
    public XRNode controllerHand = XRNode.RightHand;
    public String TextureName;

    private Texture2D photo;
 

    private bool triggerPressed = false;

    void Update()
    {
        CheckTriggerInput();
         // Imposta l'occhio da catturare
    }

    public void TakeAPhoto()
    {
        int width = webCamTextureManager.WebCamTexture.width;
        int height = webCamTextureManager.WebCamTexture.height;
        if (photo == null)
        {
            photo = new Texture2D(width, height, TextureFormat.RGB24, false);
        }
        Color32[] pixels = new Color32[width * height];
        webCamTextureManager.WebCamTexture.GetPixels32(pixels);
        photo.SetPixels32(pixels);
        photo.Apply();
        quadRenderer.material.SetTexture(TextureName, photo);
    }
    void CheckTriggerInput()
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(controllerHand);
        
        if (device.isValid)
        {
            bool triggerValue;
            if (device.TryGetFeatureValue(CommonUsages.triggerButton, out triggerValue))
            {
                // Rileva il momento del click (quando il grilletto passa da non premuto a premuto)
                if (triggerValue && !triggerPressed)
                {
                    Debug.Log("Grilletto premuto - Avvio cattura stereo");
                    TakeAPhoto();
                }
                triggerPressed = triggerValue;
            }
        }
    }
}