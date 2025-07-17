using UnityEngine;
using UnityEngine.XR;
using System.Collections;
using PassthroughCameraSamples;

public class StereoPhotoCapture : MonoBehaviour
{
    [Header("WebCamTexture Managers")]
    public WebCamTextureManager leftWebCamTextureManager;
    public WebCamTextureManager rightWebCamTextureManager;

    [Header("Quad Renderers")]
    public Renderer leftQuadRenderer;
    public Renderer rightQuadRenderer;

    [Header("Controller Settings")]
    public XRNode controllerHand = XRNode.RightHand;
    private bool triggerPressed = false;

    void Update()
    {
        // Controlla il grilletto
        InputDevice device = InputDevices.GetDeviceAtXRNode(controllerHand);
        if (!device.isValid)
            return;

        bool triggerValue;
        if (device.TryGetFeatureValue(CommonUsages.triggerButton, out triggerValue))
        {
            if (triggerValue && !triggerPressed)
            {
                Debug.Log("[StereoPhotoCapture] Trigger pressed, starting stereo capture.");
                StartCoroutine(TakeStereoPhoto());
            }
            triggerPressed = triggerValue;
        }
    }

    private IEnumerator TakeStereoPhoto()
    {
        // Cattura occhio sinistro
        yield return StartCoroutine(CaptureEye(
            leftWebCamTextureManager,
            leftQuadRenderer,
            PassthroughCameraEye.Left));

        // Piccola pausa per stabilizzare
        yield return new WaitForEndOfFrame();

        // Cattura occhio destro
        yield return StartCoroutine(CaptureEye(
            rightWebCamTextureManager,
            rightQuadRenderer,
            PassthroughCameraEye.Right));
    }

    private IEnumerator CaptureEye(
        WebCamTextureManager manager,
        Renderer targetQuad,
        PassthroughCameraEye eye)
    {
        if (manager == null || targetQuad == null)
        {
            Debug.LogError("[StereoPhotoCapture] Manager o Quad non assegnato.");
            yield break;
        }

        // 1) Spegni il manager per permettere il cambio
        manager.enabled = false;
        yield return null; // aspetta un frame

        // 2) Cambia l'occhio e riaccendi
        manager.Eye = eye;
        manager.enabled = true;

        // 3) Aspetta che la WebCamTexture sia in play
        //    e attendi un frame per sicurezza
        while (manager.WebCamTexture == null || !manager.WebCamTexture.isPlaying)
            yield return null;
        yield return new WaitForEndOfFrame();

        // 4) Crea la Texture2D e copia i pixel
        var webcamTex = manager.WebCamTexture;
        Texture2D photo = new Texture2D(
            webcamTex.width,
            webcamTex.height,
            TextureFormat.RGB24,
            false);

        Color32[] pixels = webcamTex.GetPixels32();
        photo.SetPixels32(pixels);
        photo.Apply();

        // 5) Assegna la foto al quad
        targetQuad.material.mainTexture = photo;
        Debug.Log($"[StereoPhotoCapture] Captured {eye} eye: {webcamTex.width}x{webcamTex.height}");
    }
}
