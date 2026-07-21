# Sentiment Analysis of Product Reviews — Project Report

## 1. Approach

**Data.** `product_reviews_mock_data.csv` — synthetic reviews with `ReviewID`, `ProductID`, `UserID`, `Rating` (1-5), `ReviewText`, `ReviewDate`. `ReviewID`, `ProductID`, `UserID`, and `ReviewDate` were dropped as irrelevant to text sentiment. `Rating` was mapped to a 3-class label: 1-2 → Negative, 3 → Neutral, 4-5 → Positive.

**Preprocessing.** Lowercasing, punctuation removal, stopword removal (with negation/intensifier words `not`, `no`, `very` explicitly retained since they carry sentiment signal), and lemmatization. Exact duplicate reviews were dropped.

**Class balance.** Negative: 255, Positive: 250, Neutral: 136 — a moderate imbalance, with Neutral under-represented at roughly half the size of the other two classes.

**Feature extraction.** TF-IDF (`max_features=5000`), fit on the training split only and applied to the test split, to avoid data leakage.

**Split.** 80/20 train/test, stratified on the label to preserve class ratios in both sets.

**Models trained.** Multinomial Naive Bayes, Logistic Regression (`class_weight="balanced"`), and Linear SVM (`class_weight="balanced"`). Class weighting was used to counter the Neutral-class imbalance; Naive Bayes does not support this parameter, which is a known limitation of the algorithm.

## 2. Results

| Model | Accuracy | Macro F1 | Neutral Recall |
|---|---|---|---|
| Naive Bayes | 0.81 | 0.66 | 0.11 |
| Logistic Regression | 0.99 | 0.99 | 1.00 |
| Linear SVM | 1.00 | 1.00 | 1.00 |

Naive Bayes performed worst on the minority Neutral class (0.11 recall, 0.19 F1) — without class weighting, its predictions were dominated by the majority classes, correctly illustrating a real limitation of the algorithm on imbalanced data.

Logistic Regression and SVM both scored 99-100% accuracy. **These numbers are not treated at face value.** This dataset is synthetic and template-generated (e.g. "fantastic. wonderful experience.", "worst purchase. one star."), producing a small, low-diversity vocabulary with strong, near-linear separability between classes. On genuine, naturalistic customer reviews — with sarcasm, mixed sentiment, longer text, and greater vocabulary overlap between classes — this level of performance would not be expected. The result is reported as a property of the synthetic dataset, not evidence of a production-ready model.

## 3. Model Selected

**Logistic Regression** was selected as the final model over SVM. SVM's 100% accuracy is the more likely candidate for overfitting to the templated data; Logistic Regression's 99% still demonstrates strong class separation, while remaining more interpretable (inspectable per-word coefficients), computationally cheaper, and easier to calibrate for probability outputs — a meaningful advantage for a production-facing sentiment tool.

## 4. Challenges

- **Class imbalance.** Neutral (136 examples) is under-represented relative to Negative/Positive (~250 each), which risked poor recall on that class — confirmed in the Naive Bayes baseline and addressed via `class_weight="balanced"` in the other two models.
- **Synthetic data ceiling.** Near-perfect scores are an artifact of templated, low-vocabulary mock data rather than genuine linguistic generalization. This limits how much the reported metrics say about real-world performance.
- **Ambiguity of the Neutral class.** Even in human-labeled data, "neutral" sentiment is the hardest boundary to draw consistently, and this is expected to be more pronounced on real reviews than in this dataset.

## 5. Potential Improvements

- Validate the trained model on a small set of real, naturalistically-written reviews (not templated) to get an honest read on generalization.
- Compare TF-IDF against pre-trained embeddings (GloVe/Word2Vec) or a transformer-based model (e.g. DistilBERT) for a stronger baseline on real-world text.
- Expand and diversify the Neutral class specifically, since it is both the smallest and most linguistically ambiguous category.
- Add explicit handling for negation scope (e.g. "not good" vs "good") beyond simply retaining the negation token, since TF-IDF alone does not capture word order.