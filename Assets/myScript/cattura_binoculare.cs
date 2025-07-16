using UnityEngine;
using System.IO;
using System.Collections;
using UnityEngine.XR;

public class StereoCapture : MonoBehaviour
{
    public Camera leftEyeCamera;
    public Camera rightEyeCamera;
    public int imageWidth = 1024;
    public int imageHeight = 1024;
    
    [Header("VR Image Filter Client")]
    public client imageFilterClient;

    [Header("Input Settings")]
    public XRNode controllerHand = XRNode.RightHand; // Quale controller usare

    private bool triggerPressed = false;

    void Start()
    {
        // Verifica se XR è inizializzato
        if (!XRSettings.enabled)
        {
            Debug.LogWarning("XR non è abilitato!");
        }
    }

    void Update()
    {
        // Controlla input del grilletto
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
                // Rileva il momento del click (quando il grilletto passa da non premuto a premuto)
                if (triggerValue && !triggerPressed)
                {
                    Debug.Log("Grilletto premuto - Avvio cattura stereo");
                    CaptureStereoImages();
                }
                triggerPressed = triggerValue;
            }
        }
    }

    public void CaptureStereoImages()
    {
        StartCoroutine(CaptureCoroutine());
    }

    IEnumerator CaptureCoroutine()
    {
        yield return new WaitForEndOfFrame();

        // Cattura sinistro
        Texture2D leftImage = CaptureFromCamera(leftEyeCamera);
        
        // Cattura destro
        Texture2D rightImage = CaptureFromCamera(rightEyeCamera);

        // Invoca RequestLableForImage direttamente con le texture
        if (imageFilterClient != null)
        {
            yield return StartCoroutine(imageFilterClient.RequestLableForImage(rightImage, leftImage));
        }
        else
        {
            Debug.LogWarning("VRImageFilterClient non è assegnato!");
        }

        // Pulisci le texture dalla memoria
        Destroy(leftImage);
        Destroy(rightImage);

        Debug.Log("Stereo images processed and cleaned from memory");
    }

    Texture2D CaptureFromCamera(Camera cam)
    {
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        image.Apply();

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        return image;
    }
}