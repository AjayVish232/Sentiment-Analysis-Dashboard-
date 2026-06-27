# Sentiment Analysis Dashboard

A simple NLP dashboard built with Streamlit and a logistic regression sentiment classifier.

## Features
- Enter a review and classify it as positive or negative
- Display confidence for each prediction
- Use a lightweight TF-IDF + logistic regression model

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Testing
Run the regression tests with:
```bash
python -m pytest -q
```
