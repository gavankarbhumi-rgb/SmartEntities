import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# 1. Select the model
MODEL_NAME = "bert-base-cased"

# 2. Load dataset
print("Loading CoNLL-2003 dataset...")
dataset = load_dataset("lhoestq/conll2003")

label_names = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

# 3. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Convert word labels to BERT token labels
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128
    )

    all_labels = []

    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

print("Preparing dataset...")
train_data = dataset["train"].select(range(3000))
validation_data = dataset["validation"].select(range(1000))

tokenized_train = train_data.map(
    tokenize_and_align_labels,
    batched=True
)

tokenized_validation = validation_data.map(
    tokenize_and_align_labels,
    batched=True
)

# 5. Load BERT model
print("Loading BERT model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# 6. Evaluation metric
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        current_predictions = []
        current_labels = []

        for pred, actual in zip(prediction, label):
            if actual != -100:
                current_predictions.append(label_names[pred])
                current_labels.append(label_names[actual])

        true_predictions.append(current_predictions)
        true_labels.append(current_labels)

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

# 7. Configure training
training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",
    use_cpu=True
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Train and evaluate
print("\nStarting BERT training...")
trainer.train()

print("\nEvaluating BERT...")
results = trainer.evaluate()
print("\nValidation results:")
print(results)

# 9. Save trained model
trainer.save_model("./bert_ner_model")
tokenizer.save_pretrained("./bert_ner_model")

print("\nBERT model saved in the bert_ner_model folder!")
print("Training completed successfully!")