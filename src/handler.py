import io
import os
import uuid
import soundfile as sf
import runpod
import torch
from transformers import VitsModel, AutoTokenizer
import boto3
from dotenv import load_dotenv

load_dotenv()


HF_MODEL_DICT = {
    "kn": "facebook/mms-tts-kan",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "hi": "facebook/mms-tts-hin",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    access_key_id=os.environ["RUNPOD_SECRET_AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["RUNPOD_SECRET_AWS_REGION"],
)

# Load model and tokenizer outside the handler
LANG_MODELS = {}
TOKENIZERS = {}

for lang, model_name in HF_MODEL_DICT.items():
    LANG_MODELS[lang] = VitsModel.from_pretrained(model_name).to(device)
    TOKENIZERS[lang] = AutoTokenizer.from_pretrained(HF_MODEL_DICT[lang])
    print(f"Model and tokenizer loaded for {lang}")


def handler(event):
    input_data = event.get("input", {})

    # Extract sentence and language
    sentence = input_data["sentence"]
    language = input_data["language"]

    # Error handling in case of invalid input
    if not sentence:
        return {"error": "sentence is required"}
    if not language:
        return {"error": "language is required"}
    if language not in LANG_MODELS:
        return {"error": "language not supported"}

    # Get model and tokenizer for the language
    model = LANG_MODELS[language]
    tokenizer = TOKENIZERS[language]

    # Tokenize and prepare input
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    waveform = outputs.waveform[0].cpu().numpy()

    # Save the waveform as a wav file
    output = io.BytesIO()
    sf.write(output, waveform, 16000, format="wav")
    output.seek(0)

    # Upload the file to S3
    key = f"{uuid.uuid4()}.wav"
    s3.upload_fileobj(
        output,
        os.environ["RUNPOD_SECRET_S3_BUCKET_NAME"],
        key,
        ExtraArgs={"ContentType": "audio/wav"},
    )
    cdn_url = f"{os.environ['RUNPOD_SECRET_CDN_URL']}/{key}"

    return {"url": cdn_url}


runpod.serverless.start({"handler": handler})
