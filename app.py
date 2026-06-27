import streamlit as st

from sentiment_model import predict_sentiment, train_model

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊")


@st.cache_resource
def get_model():
    return train_model()


model = get_model()

st.title("Sentiment Analysis Dashboard")
st.caption("Natural language processing with a logistic regression model")

review = st.text_area(
    "Enter a review",
    value="I absolutely loved this experience",
    height=140,
)

if st.button("Analyze sentiment"):
    result = predict_sentiment(model, review)
    if result["label"] == "positive":
        st.success(f"Positive sentiment • {result['confidence'] * 100:.1f}% confidence")
    else:
        st.error(f"Negative sentiment • {result['confidence'] * 100:.1f}% confidence")

    st.markdown("### Model details")
    st.write("Model: Logistic Regression with TF-IDF features")
    st.write("Use case: Quick sentiment analysis for short text reviews")

st.markdown("### Sample reviews")
for sample in [
    "This product is fantastic and worth every penny",
    "I would not buy this again",
    "The service was okay but the product was disappointing",
]:
    st.write(f"- {sample}")
