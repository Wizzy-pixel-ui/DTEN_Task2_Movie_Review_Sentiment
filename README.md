# DTEN Task 2: Movie Review Sentiment Analysis

This project is completing **Task 2: Sentiment Analysis on Text Data** from the Daryl Tech & Educational Network AI and Machine Learning internship.

## Project objective

The project is classifying movie reviews into three categories:

- Negative
- Neutral
- Positive

The workflow is covering text cleaning, tokenization, stopword removal, negation-aware preprocessing, TF-IDF vectorization, model training, evaluation, visualization, and limitation analysis.

## Dataset sources

The primary dataset source is the [Kaggle Sentiment Analysis on Movie Reviews competition](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews). Its original labels range from 0 to 4:

| Original labels | Meaning | Project label |
|---|---|---|
| 0, 1 | Negative and somewhat negative | Negative |
| 2 | Neutral | Neutral |
| 3, 4 | Somewhat positive and positive | Positive |

For reproducible execution when Kaggle authentication is unavailable, the notebook is also supporting the public [SST-5 dataset on Hugging Face](https://huggingface.co/datasets/SetFit/sst5). SST-5 is a movie-review sentiment corpus with the same five-level sentiment structure.

## Project-specific algorithm

The main model is called **Negation-Aware Contrastive Centroid Classification**. It is a transparent, project-specific algorithm rather than a black-box neural network.

The algorithm is operating in five stages:

1. It is representing each sentiment class with a TF-IDF centroid.
2. It is normalizing the centroids for cosine-style similarity comparison.
3. It is calculating contrastive feature weights from the difference between class centroids.
4. It is comparing each new review with the negative, neutral, and positive centroids.
5. It is assigning ambiguous reviews to the neutral class when the top two class scores are too close.

The preprocessing is also preserving negation. Words appearing near `not`, `never`, or related expressions are receiving a `NOT_` prefix. This is helping the model distinguish between expressions such as `good` and `not good`.

## Required Task 2 components

The notebook is including all requirements from the internship brief:

- Selecting a movie-review text dataset
- Tokenizing and cleaning text
- Removing ordinary stopwords while preserving negation words
- Vectorizing reviews with TF-IDF
- Classifying reviews as negative, neutral, or positive
- Reporting accuracy, precision, recall, macro-F1, and a confusion matrix
- Discussing limitations of the model

## Evaluation protocol

The data is being split into 70% training, 15% validation, and 15% test partitions using stratification. The validation partition is being used only for selecting the neutral-confidence margin. The test partition is being kept separate until final evaluation.

The executed notebook is preserving its outputs, including the class-distribution chart, validation-tuning chart, metric table, classification report, confusion matrix, and uncertain-example table.

## Limitations

TF-IDF is not fully understanding sarcasm, irony, long-range context, or genre-specific meaning. The mapping from five sentiment levels to three classes is also reducing some detail. The model may therefore misclassify reviews whose sentiment depends on context rather than on explicit word patterns. A future study could compare the method with a transformer model such as BERT, use repeated cross-validation, and calibrate the predicted probabilities.

## Running the notebook

1. Open `Wisdom_Kekeli_Task2_Movie_Review_Sentiment.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. If Kaggle authentication is available, the notebook is loading the Kaggle competition data.
4. If Kaggle authentication is unavailable, the notebook is loading the public SST-5 fallback automatically.
5. Review the saved metrics and confusion matrix.

## Repository contents

- `Wisdom_Kekeli_Task2_Movie_Review_Sentiment.ipynb` — the executed Google Colab notebook with preserved outputs.
- `README.md` — project explanation, dataset links, algorithm description, requirements, evaluation protocol, and limitations.

## Technologies

Python, Google Colab, Pandas, NumPy, Scikit-learn, SciPy, NLTK, Matplotlib, Seaborn, KaggleHub, and Hugging Face Datasets.

## Internship

**Daryl Tech & Educational Network — AI and Machine Learning Internship**

**Task completed:** Task 2 — Sentiment Analysis on Text Data
