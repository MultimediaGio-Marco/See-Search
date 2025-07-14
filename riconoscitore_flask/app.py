from flask import Flask, render_template, request, jsonify, url_for
from objectRecinozer import ObjectRecognizer
from scraper import ScraperWiki
import os
import base64
import time
from PIL import Image
import io

app = Flask(__name__)
recognizer = ObjectRecognizer()
scraper = ScraperWiki()

# Assicura che la cartella static esista
os.makedirs('static', exist_ok=True)

def process_and_save_image(image_file, filename='upload.jpg'):
    """
    Processa e salva l'immagine in più formati per garantire compatibilità
    """
    try:
        # Salva l'immagine originale
        original_path = os.path.join('static', filename)
        image_file.save(original_path)
        
        # Apri l'immagine con PIL per processarla
        pil_image = Image.open(original_path)
        
        # Converti in RGB se necessario (per evitare problemi con PNG/RGBA)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            pil_image = pil_image.convert('RGB')
        
        # Ridimensiona se l'immagine è troppo grande
        max_size = (800, 600)
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Salva l'immagine processata
        processed_path = os.path.join('static', 'processed_' + filename)
        pil_image.save(processed_path, 'JPEG', quality=85)
        
        # Crea anche una versione base64 come fallback
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return {
            'original_path': original_path,
            'processed_path': processed_path,
            'base64_data': image_base64,
            'success': True
        }
    
    except Exception as e:
        print(f"Errore nel processing dell'immagine: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_enhanced_description(label):
    """
    Ottiene una descrizione potenziata usando il nuovo scraper
    """
    try:
        # Cerca informazioni dettagliate
        description = scraper.search(label)
        
        # Se la descrizione è troppo corta, prova con una query più specifica
        if len(description) < 100:
            enhanced_description = scraper.search(f"what is {label} definition")
            if len(enhanced_description) > len(description):
                description = enhanced_description
        
        return description
    
    except Exception as e:
        print(f"Errore nella ricerca: {e}")
        return f"Errore nel recupero delle informazioni per {label}: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    description = None
    image_info = None
    error = None
    
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        if image.filename:
            try:
                # Processa e salva l'immagine
                image_info = process_and_save_image(image)
                
                if image_info['success']:
                    # Riconoscimento oggetto usando il percorso processato
                    label = recognizer.recognize(image_info['processed_path'])
                    
                    if label:
                        # Ottieni descrizione potenziata
                        description = get_enhanced_description(label)
                        print(f"Riconosciuto: {label}")
                        print(f"Descrizione: {description[:200]}...")
                    else:
                        description = "Nessuna etichetta riconosciuta. Prova con un'immagine più chiara o con un oggetto diverso."
                        error = "Riconoscimento fallito"
                else:
                    error = f"Errore nel processing dell'immagine: {image_info.get('error', 'Errore sconosciuto')}"
            
            except Exception as e:
                error = f"Errore durante l'elaborazione: {str(e)}"
                print(f"Errore nell'elaborazione: {e}")
    
    return render_template('index.html', 
                         label=label, 
                         description=description,
                         image_info=image_info,
                         error=error,
                         timestamp=int(time.time()))

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint per predizioni (mantiene compatibilità con eventuali client esterni)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Nessuna immagine trovata nella richiesta'}), 400

    image = request.files['image']
    if not image.filename:
        return jsonify({'error': 'Nessun file selezionato'}), 400

    try:
        # Processa l'immagine
        image_info = process_and_save_image(image, 'api_upload.jpg')
        
        if not image_info['success']:
            return jsonify({'error': f'Errore nel processing: {image_info.get("error")}'}), 500
        
        # Riconoscimento
        label = recognizer.recognize(image_info['processed_path'])
        
        if label:
            # Ottieni descrizione potenziata
            description = get_enhanced_description(label)
            print(f"API - Riconosciuto: {label}")
            print(f"API - Descrizione: {description[:200]}...")
        else:
            description = "Nessuna etichetta riconosciuta."
        
        return jsonify({
            'label': label,
            'description': description,
            'image_processed': True,
            'timestamp': int(time.time())
        })
    
    except Exception as e:
        print(f"Errore nell'API predict: {e}")
        return jsonify({'error': f'Errore durante l\'elaborazione: {str(e)}'}), 500

@app.route("/health")
def health_check():
    """
    Endpoint per verificare lo stato dell'applicazione
    """
    return jsonify({
        'status': 'healthy',
        'recognizer_loaded': recognizer is not None,
        'scraper_loaded': scraper is not None,
        'timestamp': int(time.time())
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error='Pagina non trovata'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error='Errore interno del server'), 500

if __name__ == '__main__':
    print("🚀 Avvio del server Flask...")
    print("📱 Recognizer caricato:", recognizer is not None)
    print("🔍 Scraper caricato:", scraper is not None)
    print("📁 Cartella static:", os.path.exists('static'))
    app.run(debug=True, host='0.0.0.0', port=5000)