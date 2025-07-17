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
            // Update the texture of the quad with the webcam texture
            quadRenderer.material.mainTexture = webCamTextureManager.WebCamTexture;
        }
        
    }
}
