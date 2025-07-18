using UnityEngine;
using PassthroughCameraSamples;

public class PassthroughCameraDisplayStereo : MonoBehaviour
{
    [Header("WebCamTexture Managers")]
    public WebCamTextureManager leftWebCamTextureManager;
    public WebCamTextureManager rightWebCamTextureManager;

    [Header("Quad Renderers")]
    public Renderer leftQuadRenderer;
    public Renderer rightQuadRenderer;

    void Update()
    {
        // Occhio sinistro
        if (leftWebCamTextureManager != null)
        {
            if (leftWebCamTextureManager.WebCamTexture != null)
            {
                leftQuadRenderer.material.mainTexture = leftWebCamTextureManager.WebCamTexture;
            }
            else
            {
                Debug.LogWarning("[MYError] Left WebCamTexture is null.");
            }
        }
        else
        {
            Debug.LogError("[MYError] Left WebCamTextureManager not assigned.");
        }

        // Occhio destro
        if (rightWebCamTextureManager != null)
        {
            if (rightWebCamTextureManager.WebCamTexture != null)
            {
                rightQuadRenderer.material.mainTexture = rightWebCamTextureManager.WebCamTexture;
            }
            else
            {
                Debug.LogWarning("[MYError] Right WebCamTexture is null.");
            }
        }
        else
        {
            Debug.LogError("[MYError] Right WebCamTextureManager not assigned.");
        }
    }
}
