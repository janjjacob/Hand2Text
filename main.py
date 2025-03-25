from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
from PIL import Image
import requests
import openai
import os
import io
from botocore.config import Config

app = FastAPI()

textract_config = Config(
    connect_timeout=10,
    read_timeout=20
)
textract = boto3.client('textract', config=textract_config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/upload/path")
async def recognize_handwriting_from_upload(image_path):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="File not found.")
    
    try:
        with open(image_path, 'rb') as document:
            image_bytes = document.read()

        response = textract.detect_document_text(Document={'Bytes': image_bytes})

        extracted_text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + '\n'

        text = extracted_text.strip()
        summary = await _summarize_text(text)

        return {"text": text, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload/url")
async def recognize_handwriting_from_url(url):
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL.")
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()

        response = textract.detect_document_text(Document={'Bytes': image_bytes})

        extracted_text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + '\n'

        text = extracted_text.strip()
        summary = await _summarize_text(text)

        return {"text": text, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _summarize_text(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that summarizes notes."},
            {"role": "user", "content": f"Summarize the following handwritten note: {text}"}
        ],
        timeout=15
    )
    return response["choices"][0]["message"]["content"].strip()

@app.get("/")
async def health_check():
    return {"message": "Server is running!"}

# if __name__ == "__main__":
#     text_output = recognize_handwriting_from_url("https://images.squarespace-cdn.com/content/v1/58764bfdb3db2b3e1ed14061/d081d72e-8acb-4682-b5ed-097692369efa/IMG_4072.jpeg")
#     print("Extracted Text:", text_output)
    
#     summary = _summarize_text(text_output)
#     print("Summarized Text:", summary)
