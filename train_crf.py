
from datasets import load_dataset
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score

print("Loading CoNLL-2003 dataset...")

dataset = load_dataset("lhoestq/conll2003", trust_remote_code=False)

label_names = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]
print("Dataset loaded successfully!")
print("Labels:", label_names)


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


def prepare_data(split):
    X, y = [], []

    for example in dataset[split]:
        tokens = example["tokens"]
        tag_ids = example["ner_tags"]

        X.append([
            get_features(tokens, i)
            for i in range(len(tokens))
        ])

        y.append([
            label_names[tag_id]
            for tag_id in tag_ids
        ])

    return X, y


print("Preparing training data...")
X_train, y_train = prepare_data("train")

print("Preparing validation data...")
X_valid, y_valid = prepare_data("validation")

print("Training CRF model...")

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)

crf.fit(X_train, y_train)

print("CRF training completed!")

print("Evaluating on validation data...")
y_pred = crf.predict(X_valid)

print("\nValidation F1-score:", round(f1_score(y_valid, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_valid, y_pred, digits=4))

import joblib

joblib.dump(crf, "crf_model.pkl")
print("\nModel saved as crf_model.pkl")
