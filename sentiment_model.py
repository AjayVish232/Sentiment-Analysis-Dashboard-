from __future__ import annotations

from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAINING_DATA = [
    ("I love this product", "positive"),
    ("This is amazing and wonderful", "positive"),
    ("I absolutely loved this experience", "positive"),
    ("Great service and excellent quality", "positive"),
    ("The staff were friendly and the outcome was fantastic", "positive"),
    ("I hate this", "negative"),
    ("This is terrible and disappointing", "negative"),
    ("This was a terrible and disappointing product", "negative"),
    ("Awful experience and poor quality", "negative"),
    ("I would not recommend this to anyone", "negative"),
]


def train_model() -> Pipeline:
    texts, labels = zip(*TRAINING_DATA)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(texts, labels)
    return model


def predict_sentiment(model: Pipeline, text: str) -> Dict[str, float | str]:
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    return {
        "label": prediction,
        "confidence": round(confidence, 3),
    }
