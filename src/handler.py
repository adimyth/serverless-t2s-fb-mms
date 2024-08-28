import io
import base64
import soundfile as sf
import runpod
import torch
from transformers import VitsModel, AutoTokenizer


HF_MODEL_DICT = {
    "kn": "facebook/mms-tts-kan",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "hi": "facebook/mms-tts-hin",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Return the audio as base64 encoded string
    buffer = io.BytesIO()
    sf.write(buffer, waveform, 16000, format="wav")
    buffer.seek(0)
    audio_bytes = buffer.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio": audio_base64}


runpod.serverless.start({"handler": handler})
