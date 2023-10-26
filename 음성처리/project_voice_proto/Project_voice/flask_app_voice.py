from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import tensorflow as tf
import soundfile as sf
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

app = Flask(__name__)
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/synthesize', methods=['POST'])
def synthesize():
    # Get input text from the request
    input_text = request.json['text']
    
    # Convert input text to sequence
    input_ids = processor.text_to_sequence(input_text)
    
    # Perform TTS inference (text-to-mel)
    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
    
    # Perform mel-to-wav synthesis using MelGAN
    audio = mb_melgan.inference(mel_after)[0, :, 0]
    
    # Write the audio to a file
    sf.write('./static/audio.wav', audio, 22050, "PCM_16")
    
    # Return the path to the saved audio file
    return jsonify({'audio_path': '/static/audio.wav'})

# play
@app.route('/play', methods=['GET'])
def play():
    return send_file('static/audio.wav')

# host="0.0.0.0", port="8080"
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")
