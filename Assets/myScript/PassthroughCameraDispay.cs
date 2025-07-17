using UnityEngine;
using PassthroughCameraSamples;
public class PassthroughCameraDispay : MonoBehaviour
{
    public WebCamTextureManager webCamTextureManager;
    public Renderer quadRenderer;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
{
    if (webCamTextureManager != null)
    {
        if (webCamTextureManager.WebCamTexture != null)
        {
            quadRenderer.material.mainTexture = webCamTextureManager.WebCamTexture;
        }
        else
        {
            Debug.LogWarning("[MYError] WebCamTexture is null.");
        }
    }
    else
    {
        Debug.LogError("[MYError] webCamTextureManager not assigned.");
    }
}

}
