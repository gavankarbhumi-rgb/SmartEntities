
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load our trained BERT model
model_path = "./bert_ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()


def predict_entities(sentence):
    words = re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", sentence)

    if not words:
        print("Please enter a sentence.")
        return

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    word_ids = inputs.word_ids(batch_index=0)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predictions = torch.argmax(probabilities, dim=-1)

    print("\nIdentified entities:")

    seen_words = set()
    found = False

    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue

        seen_words.add(word_id)

        label_id = predictions[i].item()
        label = model.config.id2label[label_id]
        confidence = probabilities[i][label_id].item() * 100

        if label != "O":
            entity_type = label.split("-", 1)[-1]
            print(f"{words[word_id]} -> {entity_type} ({confidence:.2f}%)")
            found = True

    if not found:
        print("No entities identified.")


print("===== BERT NAMED ENTITY RECOGNITION =====")

while True:
    sentence = input("\nEnter a sentence (or type 'exit' to stop): ")

    if sentence.strip().lower() == "exit":
        print("Testing completed!")
        break

    predict_entities(sentence)
