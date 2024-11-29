import os
import uuid
import torch
from speechbrain.pretrained import Tacotron2, HIFIGAN

class TextToSpeechSynthesizer:
    def __init__(self, audio_folder=None):
        if audio_folder is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.audio_folder = os.path.join(current_dir, '..', 'api_gateway', 'audio')
        else:
            self.audio_folder = audio_folder
        os.makedirs(self.audio_folder, exist_ok=True)

        # Initialize the TTS model and vocoder
        self.tts_model = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tts_model")
        self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="vocoder_model")

    def synthesize_speech(self, text):
        # Generate a unique filename
        audio_filename = str(uuid.uuid4()) + ".wav"
        audio_path = os.path.join(self.audio_folder, audio_filename)

        # Generate mel-spectrogram from text
        with torch.no_grad():
            mel_outputs, mel_lengths, alignments = self.tts_model.encode_text(text)

            # Generate waveform from mel-spectrogram using vocoder
            waveforms = self.vocoder.decode_batch(mel_outputs)

            # Save the waveform to a file
            waveform = waveforms.squeeze(1).cpu().numpy()[0]
            sample_rate = 22050  # Sample rate used by the model
            from scipy.io.wavfile import write
            write(audio_path, sample_rate, waveform)

        return audio_filename