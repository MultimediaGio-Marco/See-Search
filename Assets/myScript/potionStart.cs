using UnityEngine;

public class CanvasAtCenterEyeAnchor : MonoBehaviour
{
    [Tooltip("Assegna il Transform di CenterEyeAnchor")]
    public Transform centerEyeAnchor;

    void Start()
    {
        if (centerEyeAnchor == null)
        {
            Debug.LogWarning("CenterEyeAnchor non assegnato!");
            return;
        }

        // Posiziona il Canvas nella stessa posizione e rotazione del CenterEyeAnchor
        transform.position = centerEyeAnchor.position;
        transform.rotation = centerEyeAnchor.rotation;
    }
}
