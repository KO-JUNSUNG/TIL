import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

# from 텍스트가 나올 py 파일 import text


# processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
# tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-kss-ko")
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-kss-ko")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-kss-ko")

text = "550호 아저씨 95000원 내세요."

input_ids = processor.text_to_sequence(text)

# tacotron2 inference (text-to-mel)
mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),)
# melgan inference (mel-to-wav)
audio = mb_melgan.inference(mel_after)[0, :, 0]

# save to file
sf.write('./audio.wav', audio, 22050, "PCM_16")