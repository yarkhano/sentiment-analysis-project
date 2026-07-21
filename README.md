# Sentiment Analysis of Product Reviews

Classifies customer product reviews as **Negative**, **Neutral**, or **Positive** using TF-IDF feature extraction and classical ML models, with an interactive Streamlit UI for live demo predictions.

![CI](https://github.com/<your-username>/<your-repo-name>/actions/workflows/ci.yml/badge.svg)

## Overview

- **Input:** `product_reviews_mock_data.csv` (review text + star rating)
- **Label:** derived from rating — 1-2 → Negative, 3 → Neutral, 4-5 → Positive
- **Pipeline:** text cleaning → stopword removal (negations kept) → lemmatization → TF-IDF → model training
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Final model:** Logistic Regression (`class_weight="balanced"`) — see `report.md` for full reasoning and results

## Project Structure

```
.
├── .github/workflows/ci.yml       # CI pipeline
├── data/
│   └── product_reviews_mock_data.csv
├── sentiment_analysis.py          # full pipeline: load → preprocess → train → evaluate → serialize
├── app.py                         # Streamlit UI for live predictions
├── sentiment_model.pkl            # trained Logistic Regression model
├── tfidf_vectorizer.pkl           # fitted TF-IDF vectorizer
├── report.md                      # project report
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run the training pipeline

```bash
python sentiment_analysis.py
```
This loads the data, preprocesses it, trains all three models, prints evaluation metrics, saves confusion matrix plots, and serializes the final model as `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.

## Run the interactive UI

```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`. Type a review and get a live sentiment prediction with confidence scores.

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.81 | 0.66 |
| Logistic Regression | 0.99 | 0.99 |
| Linear SVM | 1.00 | 1.00 |

> Near-perfect scores reflect the templated, synthetic nature of the mock dataset rather than production-level generalization. See `report.md` for full discussion.

## License

MIT