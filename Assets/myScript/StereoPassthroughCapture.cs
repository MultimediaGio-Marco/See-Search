using UnityEngine;
using System.Collections;
using UnityEngine.XR;

public class StereoPassthroughCapture : MonoBehaviour
{
    public int imageWidth = 1024;  // larghezza per occhio
    public int imageHeight = 1024; // altezza per occhio

    [Header("VR Image Filter Client")]
    public client imageFilterClient;

    [Header("Input Settings")]
    public XRNode controllerHand = XRNode.RightHand;

    private bool triggerPressed = false;

    void Update()
    {
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
                    Debug.Log("Trigger pressed - Starting stereo passthrough capture");
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

        // Lo schermo composito stereo è tipicamente due immagini affiancate:
        // Sinistro sulla metà sinistra, destro sulla metà destra dello schermo

        int totalWidth = imageWidth * 2;  // 2 occhi affiancati

        // Cattura occhio sinistro (metà sinistra)
        Texture2D leftEyeImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        leftEyeImage.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        leftEyeImage.Apply();

        // Cattura occhio destro (metà destra)
        Texture2D rightEyeImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        rightEyeImage.ReadPixels(new Rect(imageWidth, 0, imageWidth, imageHeight), 0, 0);
        rightEyeImage.Apply();

        // Passa le texture al client (in ordine destro, sinistro come nel tuo esempio)
        if (imageFilterClient != null)
        {
            yield return StartCoroutine(imageFilterClient.RequestLableForImage(rightEyeImage, leftEyeImage));
        }
        else
        {
            Debug.LogWarning("VRImageFilterClient non assegnato");
        }

        Destroy(leftEyeImage);
        Destroy(rightEyeImage);

        Debug.Log("Stereo passthrough images processed and cleaned");
    }
}
