
import joblib
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score
from seqeval.metrics import precision_score, recall_score, f1_score

# Labels used by the CoNLL-2003 dataset
label_names = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]

print("Loading test dataset...")
test_data = load_dataset("lhoestq/conll2003")["test"]

print("Loading CRF model...")
crf = joblib.load("crf_model.pkl")


# Same features as train_crf.py
def get_features(tokens, i):
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


print("Loading BERT model...")
model_path = "./bert_ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
bert = AutoModelForTokenClassification.from_pretrained(model_path)
bert.eval()

device = torch.device("cpu")
bert.to(device)

# Verify BERT label mapping before evaluation
print("\nBERT label mapping:")
print(bert.config.id2label)

expected_labels = set(label_names)
configured_labels = set(bert.config.id2label.values())

if configured_labels != expected_labels:
    raise ValueError(
        "BERT label mapping does not match the dataset labels. "
        "Check the model config before evaluating."
    )

true_labels = []
crf_predictions = []
bert_predictions = []
truncation_warnings = 0

print("\nEvaluating models on the same test dataset...")

for example in test_data:
    words = example["tokens"]
    gold_tags = [label_names[x] for x in example["ner_tags"]]

    # CRF prediction
    features = [get_features(words, i) for i in range(len(words))]
    crf_tags = crf.predict_single(features)

    # BERT prediction
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    word_ids = inputs.word_ids(batch_index=0)

    with torch.no_grad():
        logits = bert(**inputs.to(device)).logits[0]

    predicted_ids = logits.argmax(dim=-1).tolist()

    bert_tags = []
    seen_words = set()

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue

        seen_words.add(word_id)
        label_id = predicted_ids[token_index]
        bert_tags.append(bert.config.id2label[label_id])

    # Keep only complete words represented in BERT's tokenized input
    length = len(bert_tags)

    if length != len(words):
        truncation_warnings += 1
        print(
            f"Warning: sentence has {len(words)} words, "
            f"but BERT processed {length} words."
        )

    if len(crf_tags) != len(words) or len(gold_tags) != len(words):
        raise ValueError("CRF or gold labels do not match the original words.")

    # Align gold, CRF, and BERT sequences
    true_labels.append(gold_tags[:length])
    crf_predictions.append(crf_tags[:length])
    bert_predictions.append(bert_tags)


print("\n===== MODEL COMPARISON =====")

for name, predictions in [
    ("CRF", crf_predictions),
    ("BERT", bert_predictions)
]:
    # Check sentence-by-sentence alignment
    for gold, pred in zip(true_labels, predictions):
        if len(gold) != len(pred):
            raise ValueError(f"{name}: prediction length mismatch.")

    actual_flat = [
        tag for sentence in true_labels for tag in sentence
    ]
    predicted_flat = [
        tag for sentence in predictions for tag in sentence
    ]

    print(f"\n{name} Results:")
    print(f"Token Accuracy: {accuracy_score(actual_flat, predicted_flat):.4f}")
    print(f"Entity Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Entity Recall:    {recall_score(true_labels, predictions):.4f}")
    print(f"Entity F1-score:  {f1_score(true_labels, predictions):.4f}")

print(f"\nSentences with truncation warnings: {truncation_warnings}")
print("\nComparison completed!")
