from transformers import pipeline

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

print("===== BERT NAMED ENTITY RECOGNITION =====")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("Program stopped.")
        break

    results = ner(text)

    if not results:
        print("No entities found.\n")
        continue

    print("\nEntities found:")

    for entity in results:
        word = entity["word"]
        label = entity["entity_group"]
        score = entity["score"]

        # Convert BERT labels into readable labels
        label_names = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
            "MISC": "MISCELLANEOUS"
        }

        label = label_names.get(label, label)

        print(f"{word} → {label} ({score:.2%})")

    print()