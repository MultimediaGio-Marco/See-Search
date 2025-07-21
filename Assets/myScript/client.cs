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
    public string serverUrl = "http://192.168.1.100:5000";

    [Header("Immagine da inviare")]
    public string basePath = "Images/";
    void Start()
    {
        StartCoroutine(PingServer());
    }

    IEnumerator PingServer()
    {
        string pingUrl = serverUrl + "/ping"; // es: http://192.168.1.100:5000/ping
        UnityWebRequest www = UnityWebRequest.Get(pingUrl);
        www.timeout = 3;

        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
            Debug.Log($"[MyDEBUG] [PING] ✅ Ping riuscito: {www.downloadHandler.text}");
        else
            Debug.LogError($"[MyDEBUG] [PING] ❌ Ping fallito: {www.error}");
    }


    public IEnumerator RequestLableForImage(Texture2D mr, Texture2D ml)
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
        string posturl = serverUrl + "/api/process"; // Assicurati che questo sia l'endpoint corretto
        Debug.Log($"[MyDEBUG] [GameLM] 📤 Inviando richiesta al server URL:{posturl}");
        UnityWebRequest www = new UnityWebRequest(posturl, "POST");
        www.timeout = 300; // Imposta un timeout di 5 minuti
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");

        Debug.Log("[MyDEBUG] [GameLM] 📤 Inviando immagine originale senza alterazioni...");

        yield return www.SendWebRequest();

        // 4. Gestione risposta
        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("[MyDEBUG] [GameLM] ❌ Errore nella richiesta: " + www.error);
        }
        else
        {
            try
            {
                string jsonResponse = www.downloadHandler.text;
                Debug.Log("[MyDEBUG] [GameLM] 📥 Risposta ricevuta: " + jsonResponse);

                LableResponse response = JsonUtility.FromJson<LableResponse>(jsonResponse);

                if (string.IsNullOrEmpty(response.label) || string.IsNullOrEmpty(response.description))
                {
                    Debug.LogError("[MyDEBUG] [GameLM] ❌ Nessuno oggetto riconosciuto");
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
                    Debug.Log("[MyDEBUG] [GameLM] 🖼️ Immagine caricata correttamente");
                }

                else
                {
                    Debug.LogError("[MyDEBUG] [GameLM] ❌ Errore nel caricamento della texture");
                }

                Debug.Log("[MyDEBUG] [GameLM] ✔ Immagine filtrata aggiornata in: ");

            }
            catch (Exception e)
            {
                Debug.LogError("[MyDEBUG] [GameLM] ❌ Errore parsing risposta: " + e.Message);
            }
        }
    }
}