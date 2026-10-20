# AI-Powered Phishing Email Detection ( With the help of ML)

## 1. Project Overview

This project is a demonstration of how AI can be used for the purpose of cybersecurity via analysing email content

The application uses supervised machine learning and Natural Language Processing (NLP) to classify emails as phishing or legitimate.

## 2. Technologies Used

- **Python:** Main programming language
- **Pandas:** Dataset preparation
- **Scikit-learn:** Machine learning implementation
- **TF-IDF:** Email text feature extraction
- **Logistic Regression:** Binary classification
- **Streamlit:** Interactive web application
- **Joblib:** Model storage and loading

## 3. Methodology

The system follows five stages:

1. Collect and label email examples as phishing (1) or legitimate (0).
2. Convert email text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier on the labelled examples.
4. Predict whether new emails are phishing or legitimate.
5. Display the model's classification, phishing score, and influential text features.

## 4. Training Dataset

The demonstration model was trained on 24 synthetic email examples:

- 12 phishing emails
- 12 legitimate emails

The TF-IDF vectorizer uses English stop-word removal and supports unigrams and bigrams.

## 5. Model Evaluation

The model was evaluated on a separate set of 10 manually labelled synthetic emails.

| Metric | Result |
|---|---|
| Accuracy | 70.00% |
| Precision (phishing) | 66.67% |
| Recall (phishing) | 80.00% |
| F1-score (phishing) | 72.73% |

The evaluation produced four true positives, three true negatives, two false positives, and one false negative.

These results are illustrative and are not sufficient to estimate real-world phishing detection performance reliably.

## 6. Strengths

- Simple and computationally inexpensive
- Does not require a GPU or external API
- Produces phishing classifications quickly
- Supports interpretation through individual text-feature contributions
- Can be retrained using additional labelled emails

## 7. Limitations

- Very small synthetic training dataset
- May incorrectly flag legitimate emails containing security terminology
- May miss unfamiliar phishing patterns
- Cannot verify sender identities or investigate URLs
- Vulnerable to changes in phishing language and attacker techniques

## 8. Conclusion

The demonstration shows that supervised machine learning can identify suspicious language patterns in email messages.

However, the evaluation reveals false positives and false negatives. Consequently, the model should not be used as a standalone security control.

Future improvements could include larger real-world datasets, URL analysis, sender authentication checks, and comparisons with more advanced AI approaches.