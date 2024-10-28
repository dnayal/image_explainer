import torch
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self):
        # Load the BLIP model and processor
        model_name = "Salesforce/blip-image-captioning-large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_caption(self, image_bytes):
        # Open the image and convert it to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

        return caption 