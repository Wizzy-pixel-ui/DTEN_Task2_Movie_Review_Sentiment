# DTEN Task 2: Movie Review Sentiment Analysis

This project fulfills **Task 2: Sentiment Analysis on Text Data** from the Daryl Tech & Educational Network AI and Machine Learning internship.

## Project objective

The project classifies movie reviews into three categories:

- Negative
- Neutral
- Positive

The workflow covers text cleaning, tokenization, stopword removal, negation-aware preprocessing, TF-IDF vectorization, model training, evaluation, visualization, and limitation analysis.

## Dataset sources

The primary dataset source is the [Kaggle Sentiment Analysis on Movie Reviews competition](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews). Its original labels range from 0 to 4:

| Original labels | Meaning | Project label |
|---|---|---|
| 0, 1 | Negative and somewhat negative | Negative |
| 2 | Neutral | Neutral |
| 3, 4 | Somewhat positive and positive | Positive |

For reproducible execution when Kaggle authentication is unavailable, the notebook also supports the public [SST-5 dataset on Hugging Face](https://huggingface.co/datasets/SetFit/sst5). SST-5 is a movie-review sentiment corpus with the same five-level sentiment structure.

## Project-specific algorithm

The main model is called **Negation-Aware Contrastive Centroid Classification**. It is a transparent, project-specific algorithm rather than a black-box neural network.

The algorithm operates in five stages:

1. It represents each sentiment class with a TF-IDF centroid.
2. It normalizes the centroids for cosine-style similarity comparison.
3. It calculates contrastive feature weights from the differences between class centroids.
4. It compares each new review with the negative, neutral, and positive centroids.
5. It assigns ambiguous reviews to the neutral class when the top two class scores are too close.

The preprocessing also preserves negation. Words appearing near `not`, `never`, or related expressions receive a `NOT_` prefix. This helps the model distinguish between expressions such as `good` and `not good`.

## Required Task 2 components

The notebook includes all requirements from the internship brief:

- Selection of a movie-review text dataset
- Tokenization and text cleaning
- Removal of ordinary stopwords while preserving negation words
- TF-IDF vectorization
- Classification of reviews as negative, neutral, or positive
- Reporting of accuracy, precision, recall, macro-F1, and a confusion matrix
- Discussion of model limitations

## Evaluation protocol

The data is divided into 70% training, 15% validation, and 15% test partitions using stratification. The validation partition is used only to select the neutral-confidence margin. The test partition remains separate until final evaluation.

The executed notebook preserves its outputs, including the class-distribution chart, validation-tuning chart, metric table, classification report, confusion matrix, and uncertain-example table.

## Limitations

TF-IDF does not fully understand sarcasm, irony, long-range context, or genre-specific meaning. The mapping from five sentiment levels to three classes also reduces some detail. The model may therefore misclassify reviews whose sentiment depends on context rather than explicit word patterns. A future study could compare this method with a transformer model such as BERT, use repeated cross-validation, and calibrate the predicted probabilities.

## Running the notebook

1. Open `Wisdom_Kekeli_Task2_Movie_Review_Sentiment.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. If Kaggle authentication is available, the notebook loads the Kaggle competition data.
4. If Kaggle authentication is unavailable, the notebook automatically loads the public SST-5 fallback.
5. Review the saved metrics and confusion matrix.

## Repository contents

- `Wisdom_Kekeli_Task2_Movie_Review_Sentiment.ipynb` — the executed Google Colab notebook with preserved outputs.
- `README.md` — project explanation, dataset links, algorithm description, requirements, evaluation protocol, and limitations.

## Technologies

Python, Google Colab, Pandas, NumPy, Scikit-learn, SciPy, NLTK, Matplotlib, Seaborn, KaggleHub, and Hugging Face Datasets.

## Internship

**Daryl Tech & Educational Network — AI and Machine Learning Internship**

**Task completed:** Task 2 — Sentiment Analysis on Text Data
