from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def recognize_handwriting(image_path):
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)

    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return extracted_text

def recognize_handwriting_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)

    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return extracted_text

text_output = recognize_handwriting_from_url("https://raw.githubusercontent.com/Nava-s/Handwritten-text/refs/heads/main/images/handwritten/Figure_1.png")
print("Extracted Text:", text_output)
