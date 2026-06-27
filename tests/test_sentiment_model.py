import pytest

from sentiment_model import predict_sentiment, train_model


def test_train_model_and_predict_positive():
    model = train_model()
    prediction = predict_sentiment(model, "I absolutely loved this experience")

    assert prediction["label"] == "positive"
    assert prediction["confidence"] >= 0.5


def test_predict_negative_review():
    model = train_model()
    prediction = predict_sentiment(model, "This was a terrible and disappointing product")

    assert prediction["label"] == "negative"
    assert prediction["confidence"] >= 0.5
