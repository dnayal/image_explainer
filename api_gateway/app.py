from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from image_captioner import ImageCaptioner
from tts_synthesizer import TextToSpeechSynthesizer

app = Flask(__name__)

audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
ttssynthesizer = TextToSpeechSynthesizer(audio_folder = audio_folder)
image_captioner = ImageCaptioner()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if "image" not in request.files:
        return jsonify({'error':'No image uploaded'}), 400
    
    image_file = request.files["image"]
    image_bytes = image_file.read()

    caption = image_captioner.generate_caption(image_bytes)

    audio_filename = ttssynthesizer.synthesize_speech(caption)
    audio_url = f'/api/get_audio/{audio_filename}'

    return jsonify({'caption':caption, 'audio_url':audio_url})


@app.route("/api/get_audio/<filename>", methods=['GET'])
def get_audio(filename):
    return send_from_directory(ttssynthesizer.audio_folder, filename)


if __name__ == "__main__":
    app.run(port=5000) 