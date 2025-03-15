from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import openai
import os

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

def summarize_text(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that summarizes notes."},
            {"role": "user", "content": f"Summarize the following handwritten note: {text}"}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    text_output = recognize_handwriting_from_url("https://miro.medium.com/v2/resize:fit:900/format:webp/1*-1mnBe7XQytSGa2YYZnWLQ.png")
    print("Extracted Text:", text_output)
    
    summary = summarize_text(text_output)
    print("Summarized Text:", summary)
