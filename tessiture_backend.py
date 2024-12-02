
from flask import Flask, request, jsonify
import librosa
import numpy as np
import os

app = Flask(__name__)

def detect_tessiture(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)  # Charge le fichier audio
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    tessiture = classify_tessiture(pitches)
    return tessiture

def classify_tessiture(pitches):
    pitches = pitches[pitches > 0]  # Filtrer les fréquences valides
    if len(pitches) == 0:
        return "Unknown"
    min_pitch = np.min(pitches)
    max_pitch = np.max(pitches)

    # Plages de fréquences des tessitures vocales
    if min_pitch >= 261 and max_pitch <= 1046:
        return "Soprano"
    elif min_pitch >= 196 and max_pitch <= 784:
        return "Alto"
    elif min_pitch >= 130 and max_pitch <= 523:
        return "Tenor"
    elif min_pitch >= 82 and max_pitch <= 392:
        return "Bass"
    else:
        return "Unknown"

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files['audio']
    file_path = f"./{audio.filename}"
    audio.save(file_path)

    try:
        tessiture = detect_tessiture(file_path)
    except Exception as e:
        os.remove(file_path)  # Nettoyage en cas d'erreur
        return jsonify({"error": str(e)}), 500

    os.remove(file_path)  # Nettoyage après traitement
    return jsonify({"tessiture": tessiture})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
