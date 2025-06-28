# Fake News Detection (Colab)

This project builds a simple fake news classifier using Python, scikit-learn, and TF-IDF in Google Colab. The goal is to classify news articles as **Real** or **Fake** based on their text content.

---

## 📌 Project Overview

We created two small CSV files — one with real news headlines and one with fake news headlines — then built a machine learning pipeline to train and evaluate a logistic regression classifier.

---

## 🛠️ Steps Followed

1. **Dataset Preparation**
   - Created two CSV files: `FakeNews_True.csv` (20 real news samples) and `FakeNews_Fake.csv` (20 fake news samples).
   - Uploaded both CSV files to Colab's `/content/` directory.

2. **Environment Setup**
   - Installed and downloaded NLTK’s `stopwords` corpus (skipped `punkt` tokenizer since we used Python’s `split()`).

3. **Data Loading**
   - Loaded both CSVs using `pandas.read_csv()`.
   - Added a new column `label` with 1 for real news and 0 for fake news.

4. **Dataset Combination**
   - Combined true and fake datasets into a single DataFrame.
   - Shuffled the combined data to avoid training bias.

5. **Data Preprocessing**
   - Created a `clean_text()` function:
     - Lowercased text.
     - Tokenized using `.split()`.
     - Removed punctuation.
     - Filtered out stopwords.
   - Applied `clean_text()` on each article to produce a `clean_text` column.

6. **Train/Test Split**
   - Split the cleaned dataset into training and testing sets using `train_test_split()`.

7. **Feature Engineering**
   - Used `TfidfVectorizer` to convert text data into TF-IDF vectors.

8. **Model Training**
   - Trained a `LogisticRegression` model on the training vectors.

9. **Evaluation**
   - Predicted labels on the test set.
   - Calculated accuracy, confusion matrix, and a classification report.
   - Observed accuracy and class-wise precision/recall metrics.

10. **Prediction Function**
    - Wrote `predict_news()` function to classify new user-input text.

11. **Example Prediction**
    - Ran an example sentence through the model and printed whether it’s predicted as Real or Fake.

---

## 🗂️ Files

- `fake_news_detection.ipynb`: Jupyter notebook with all code and explanation.
- `FakeNews_True.csv`: Sample real news dataset (20 examples).
- `FakeNews_Fake.csv`: Sample fake news dataset (20 examples).

---

## 🚀 How to Run

1. Clone the repo or download the notebook and CSV files.
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Upload both CSV files to Colab’s file explorer.
4. Run the notebook cells sequentially.

---

## 🔎 Example Usage

```python
test_news = "The president announced a new health policy today."
print(predict_news(test_news))  # Output: 'Real' or 'Fake'
