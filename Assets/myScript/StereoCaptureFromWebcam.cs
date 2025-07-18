using UnityEngine;
using System.Collections;
using UnityEngine.XR;
using PassthroughCameraSamples; // Assicurati che WebCamTextureManager venga da lì

public class StereoCaptureFromWebcam : MonoBehaviour
{
    [Header("WebCamTexture Managers")]
    public WebCamTextureManager leftWebCamTextureManager;
    public WebCamTextureManager rightWebCamTextureManager;

    [Header("Image Settings")]
    public int imageWidth = 1024;
    public int imageHeight = 1024;

    [Header("VR Image Filter Client")]
    public client imageFilterClient;

    [Header("Input Settings")]
    public XRNode controllerHand = XRNode.RightHand;

    [Header("Preview Quads")]
    public Renderer leftPreviewQuad;
    public Renderer rightPreviewQuad;

    // Texture2D per memorizzare le foto catturate (come nel TakeAPhoto originale)
    private Texture2D leftPhoto;
    private Texture2D rightPhoto;

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
                    Debug.Log("Grilletto premuto - Avvio cattura stereo da WebCamTexture");
                    TakeAStereoPhoto();
                }
                triggerPressed = triggerValue;
            }
        }
    }

    // Metodo adattato dal TakeAPhoto originale per funzionare con due webcam
    public void TakeAStereoPhoto()
    {
        if (leftWebCamTextureManager?.WebCamTexture == null || rightWebCamTextureManager?.WebCamTexture == null)
        {
            Debug.LogWarning("Una delle WebCamTexture non è inizializzata!");
            return;
        }

        // Cattura l'immagine sinistra
        int leftWidth = leftWebCamTextureManager.WebCamTexture.width;
        int leftHeight = leftWebCamTextureManager.WebCamTexture.height;
        
        if (leftPhoto == null)
        {
            leftPhoto = new Texture2D(leftWidth, leftHeight, TextureFormat.RGB24, false);
        }
        
        Color32[] leftPixels = new Color32[leftWidth * leftHeight];
        leftWebCamTextureManager.WebCamTexture.GetPixels32(leftPixels);
        leftPhoto.SetPixels32(leftPixels);
        leftPhoto.Apply();

        // Cattura l'immagine destra
        int rightWidth = rightWebCamTextureManager.WebCamTexture.width;
        int rightHeight = rightWebCamTextureManager.WebCamTexture.height;
        
        if (rightPhoto == null)
        {
            rightPhoto = new Texture2D(rightWidth, rightHeight, TextureFormat.RGB24, false);
        }
        
        Color32[] rightPixels = new Color32[rightWidth * rightHeight];
        rightWebCamTextureManager.WebCamTexture.GetPixels32(rightPixels);
        rightPhoto.SetPixels32(rightPixels);
        rightPhoto.Apply();

        // Mostra le immagini sui quad (come PhotoQuadRenderer.material.mainTexture = photo)
        if (leftPreviewQuad != null)
        {
            leftPreviewQuad.material.mainTexture = leftPhoto;
        }
        if (rightPreviewQuad != null)
        {
            rightPreviewQuad.material.mainTexture = rightPhoto;
        }

        // Invia le immagini al client per il processing
        if (imageFilterClient != null)
        {
            StartCoroutine(imageFilterClient.RequestLableForImage(rightPhoto, leftPhoto));
        }
        else
        {
            Debug.LogWarning("imageFilterClient non è assegnato!");
        }

        Debug.Log("Stereo photos captured and displayed on quads.");
    }

    // Metodo pubblico per catturare manualmente le immagini stereo
    public void CaptureStereoImages()
    {
        TakeAStereoPhoto();
    }

    // Metodo legacy per compatibilità con il vecchio sistema
    IEnumerator CaptureFromWebcamsCoroutine()
    {
        yield return new WaitForEndOfFrame();
        TakeAStereoPhoto();
    }

    // Metodo di utility per ottenere le foto catturate
    public Texture2D GetLeftPhoto()
    {
        return leftPhoto;
    }

    public Texture2D GetRightPhoto()
    {
        return rightPhoto;
    }

    // Cleanup
    void OnDestroy()
    {
        if (leftPhoto != null)
        {
            Destroy(leftPhoto);
        }
        if (rightPhoto != null)
        {
            Destroy(rightPhoto);
        }
    }
}