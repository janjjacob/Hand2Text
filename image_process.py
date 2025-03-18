import boto3
from PIL import Image
import requests
import openai
import os
import io

textract = boto3.client('textract')

def recognize_handwriting(image_path):
    with open(image_path, 'rb') as document:
        image_bytes = document.read()

    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    extracted_text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + '\n'

    return extracted_text.strip()

def recognize_handwriting_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    with io.BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()

    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    extracted_text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + '\n'

    return extracted_text.strip()

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
    text_output = recognize_handwriting_from_url("https://images.squarespace-cdn.com/content/v1/58764bfdb3db2b3e1ed14061/d081d72e-8acb-4682-b5ed-097692369efa/IMG_4072.jpeg")
    print("Extracted Text:", text_output)
    
    summary = summarize_text(text_output)
    print("Summarized Text:", summary)
