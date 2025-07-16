using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using TMPro;
using UnityEngine.UI;


[Serializable]
public class LableRequest
{
    public string right_image;
    public string left_image;
}

[Serializable]
public class LableResponse
{
    public string label;
    public string description;
    public string image;
}

public class client : MonoBehaviour
{
    public Image image;  // Da assegnare via editor

    public TMP_Text lable;
    public TMP_Text desc;
    [Header("Server Config")]
    public string serverUrl = "http://192.168.1.100:5000/api/process";

    [Header("Immagine da inviare")]
    public string basePath = "Images/";


    public void Start()
    {

    }

    public IEnumerator RequestLableForImage(Texture2D mr,Texture2D ml)
    {
        // Converti la texture in bytes
        
        byte[] imageBytesL = ml.EncodeToPNG();
        string mimeType = "image/png";
        string base64ImageL = $"data:{mimeType};base64,{Convert.ToBase64String(imageBytesL)}";
        byte[] imageBytesR = mr.EncodeToPNG();
        string base64ImageR = $"data:{mimeType};base64,{Convert.ToBase64String(imageBytesR)}";

        // 2. Costruisci JSON
        LableRequest request = new LableRequest
        {
            right_image = base64ImageR,  // Ora è Filter[] invece di string[]
            left_image = base64ImageL
        };

        string jsonBody = JsonUtility.ToJson(request);
        
        // 3. Invia la richiesta al server
        UnityWebRequest www = new UnityWebRequest(serverUrl, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");

        Debug.Log("[GameLM] 📤 Inviando immagine originale senza alterazioni...");

        yield return www.SendWebRequest();

        // 4. Gestione risposta
        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("[GameLM] ❌ Errore nella richiesta: " + www.error);
        }
        else
        {
            try
            {
                string jsonResponse = www.downloadHandler.text;
                Debug.Log("[GameLM] 📥 Risposta ricevuta: " + jsonResponse);

                LableResponse response = JsonUtility.FromJson<LableResponse>(jsonResponse);

                if (string.IsNullOrEmpty(response.label) || string.IsNullOrEmpty(response.description))
                {
                    Debug.LogError("[GameLM] ❌ Nessuno oggetto riconosciuto");
                    yield break;
                }
                lable.text = response.label;
                desc.text = response.description;
                string base64Image = response.image;
                if (base64Image.StartsWith("data:image"))
                {
                    base64Image = base64Image.Substring(base64Image.IndexOf(",") + 1);
                }

                byte[] imageBytes = Convert.FromBase64String(base64Image);
                Texture2D tex = new Texture2D(2, 2); // placeholder size
                if (tex.LoadImage(imageBytes))
                {
                    Sprite newSprite = Sprite.Create(
                        tex,
                        new Rect(0, 0, tex.width, tex.height),
                        new Vector2(0.5f, 0.5f) // pivot al centro
                    );

                    image.sprite = newSprite;
                    image.enabled = true;
                    Debug.Log("[GameLM] 🖼️ Immagine caricata correttamente");
                }

                else
                {
                    Debug.LogError("[GameLM] ❌ Errore nel caricamento della texture");
                }

                Debug.Log("[GameLM] ✔ Immagine filtrata aggiornata in: ");
               
            }
            catch (Exception e)
            {
                Debug.LogError("[GameLM] ❌ Errore parsing risposta: " + e.Message);
            }
        }
    }
}