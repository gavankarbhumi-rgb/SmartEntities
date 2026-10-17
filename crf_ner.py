
import sklearn_crfsuite

# Training sentences
X_train = [
    [
        {"word": "Elon", "is_title": True, "is_upper": False},
        {"word": "Musk", "is_title": True, "is_upper": False},
        {"word": "founded", "is_title": False, "is_upper": False},
        {"word": "SpaceX", "is_title": True, "is_upper": False},
    ],
    [
        {"word": "Sundar", "is_title": True, "is_upper": False},
        {"word": "Pichai", "is_title": True, "is_upper": False},
        {"word": "works", "is_title": False, "is_upper": False},
        {"word": "at", "is_title": False, "is_upper": False},
        {"word": "Google", "is_title": True, "is_upper": False},
    ],
    [
        {"word": "Alice", "is_title": True, "is_upper": False},
        {"word": "lives", "is_title": False, "is_upper": False},
        {"word": "in", "is_title": False, "is_upper": False},
        {"word": "London", "is_title": True, "is_upper": False},
    ],
]

y_train = [
    ["B-PER", "I-PER", "O", "B-ORG"],
    ["B-PER", "I-PER", "O", "O", "B-ORG"],
    ["B-PER", "O", "O", "B-LOC"],
]


def word_features(sentence, index):
    word = sentence[index]["word"]

    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "prefix-2": word[:2],
        "suffix-2": word[-2:],
    }

    if index > 0:
        previous = sentence[index - 1]["word"]
        features["previous.lower()"] = previous.lower()
        features["previous.istitle()"] = previous.istitle()
    else:
        features["BOS"] = True

    if index < len(sentence) - 1:
        next_word = sentence[index + 1]["word"]
        features["next.lower()"] = next_word.lower()
        features["next.istitle()"] = next_word.istitle()
    else:
        features["EOS"] = True

    return features


def sentence_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]


# Train the CRF model
X = [sentence_features(s) for s in X_train]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)

crf.fit(X, y_train)

print("CRF model trained successfully!")
print("Enter a sentence, or type 'exit' to stop.")

while True:
    text = input("\nEnter sentence: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        print("Please enter a sentence.")
        continue

    words = text.split()

    sentence = [
        {"word": word.strip(".,!?;:\"'()")}
        for word in words
    ]

    sentence = [item for item in sentence if item["word"]]

    if not sentence:
        print("Please enter valid words.")
        continue

    features = sentence_features(sentence)
    predictions = crf.predict_single(features)

    print("\nDetected entities:")
    for item, label in zip(sentence, predictions):
        print(f"{item['word']} -> {label}")
