import joblib
from sklearn_crfsuite import CRF
import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI(
    title="NER Intelligence Studio",
    description="Named Entity Recognition powered by BERT"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500"
],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = "./bert_ner_model"
device = torch.device("cpu")

print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
# Load the trained CRF model
print("Loading CRF model...")

crf_model = joblib.load("crf_model.pkl")

print("CRF model loaded successfully!")

print("BERT model loaded successfully!")


class TextRequest(BaseModel):
    text: str


# CRF feature extraction — same features used during training
def get_crf_features(tokens, i):
    word = tokens[i]

    features = {
        "word.lower": word.lower(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "prefix2": word[:2],
        "suffix2": word[-2:],
        "bias": 1.0,
    }

    if i > 0:
        previous = tokens[i - 1]
        features["previous.lower"] = previous.lower()
        features["previous.istitle"] = previous.istitle()
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
        features["next.lower"] = next_word.lower()
        features["next.istitle"] = next_word.istitle()
    else:
        features["EOS"] = True

    return features


def predict_with_crf(text):
    words = re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", text)

    if not words:
        return {"total_entities": 0, "entities": []}

    features = [
        get_crf_features(words, i)
        for i in range(len(words))
    ]

    labels = crf_model.predict_single(features)
    entities = []
    current = None

    for i, (word, label) in enumerate(zip(words, labels)):
        if label == "O" or "-" not in label:
            if current:
                entities.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B" or (
            current and current["type"] != entity_type
        ):
            if current:
                entities.append(current)

            current = {
                "text": word,
                "type": entity_type,
                "start_word": i,
                "end_word": i + 1
            }

        elif prefix == "I" and current:
            current["text"] += " " + word
            current["end_word"] = i + 1
        else:
            current = {
                "text": word,
                "type": entity_type,
                "start_word": i,
                "end_word": i + 1
            }

    if current:
        entities.append(current)

    return {
        "total_entities": len(entities),
        "entities": entities
    }

def analyze_text(text):
    # Keep original word positions for readable results
    words = re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", text)
    entities = []

    for start in range(0, len(words), 100):
        chunk = words[start:start + 100]

        inputs = tokenizer(
            chunk,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        word_ids = inputs.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(**inputs.to(device)).logits[0]

        probabilities = torch.softmax(logits, dim=-1)
        predicted_ids = logits.argmax(dim=-1).tolist()
        seen = set()
        current = None

        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue

            seen.add(word_id)
            label = model.config.id2label[predicted_ids[token_index]]
            confidence = float(
                probabilities[token_index, predicted_ids[token_index]].item()
            ) * 100

            if label == "O" or "-" not in label:
                if current:
                    entities.append(current)
                    current = None
                continue

            prefix, entity_type = label.split("-", 1)
            word = chunk[word_id]

            if prefix == "B" or (
                current and current["type"] != entity_type
            ):
                if current:
                    entities.append(current)

                current = {
                    "text": word,
                    "type": entity_type,
                    "confidence": round(confidence, 2),
                    "start_word": start + word_id,
                    "end_word": start + word_id + 1
                }

            elif prefix == "I" and current:
                current["text"] += " " + word
                current["end_word"] = start + word_id + 1
                current["confidence"] = round(
                    (current["confidence"] + confidence) / 2, 2
                )

        if current:
            entities.append(current)

    return {
        "total_entities": len(entities),
        "entities": entities
    }


@app.get("/")
def home():
    return {
        "message": "NER Intelligence Studio API is running!",
        "model": "BERT",
        "status": "ready"
    }


@app.post("/analyze")
def analyze(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please enter some text.")

    if len(text) > 100000:
        raise HTTPException(
            status_code=400,
            detail="Text must be under 100,000 characters."
        )

    return analyze_text(text)

@app.post("/compare")
def compare_models(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please enter some text."
        )

    if len(text) > 100000:
        raise HTTPException(
            status_code=400,
            detail="Text must be under 100,000 characters."
        )

    bert_result = analyze_text(text)
    crf_result = predict_with_crf(text)

    return {
        "text": text,
        "bert": bert_result,
        "crf": crf_result,
        "comparison": {
            "bert_entities": bert_result["total_entities"],
            "crf_entities": crf_result["total_entities"]
        }
    }

