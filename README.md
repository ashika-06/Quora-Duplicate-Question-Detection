# Quora-Duplicate-Question-Detection
A Binary Classification system using NLP and XGBoost to identify duplicate questions on Quora. Features TF-IDF weighted Word2Vec (GloVe), FuzzyWuzzy matching, and advanced feature engineering to achieve a Log Loss of 0.36.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashika-06/Quora-Duplicate-Question-Detection/blob/main/EDA.ipynb)

### Project Overview
The goal is to identify if two questions have the same intent, even if they use different words or sentence structures.
* **Problem:** Standard "Exact Match" models fail on inputs like *"What is your age?"* vs *"How old are you?"*.
* **Solution:** A machine learning pipeline that combines advanced text preprocessing, 15+ engineered features, and weighted word embeddings to capture deep semantic meaning.

### Key Techniques & Features
* **Text Preprocessing:** HTML tag removal, stemming, stopword removal, and contraction expansion.
* **Advanced NLP Features:** Ratios like `cwc_min` (Common Word Count), `ctc_max` (Common Token Count), and `longest_substr_ratio`.
* **Fuzzy Matching:** Implemented `fuzzywuzzy` features (e.g., `token_set_ratio`, `fuzz_ratio`) to handle typos, partial matches, and different word orderings.
* **Vectorization:** Generated **TF-IDF Weighted Word2Vec** embeddings using Spacy's GloVe model (`en_core_web_lg`) to preserve semantic context.
* **Dimensionality:** Handled ~794 features (300 dim per question + NLP features) using L2 regularization and tree-based gradient boosting.

### Models & Performance
Evaluated using **Log Loss** (to penalize uncertainty) and Confusion Matrices.

| Model | Log Loss | Key Observation |
| :--- | :--- | :--- |
| **Random Model** | `0.88` | Baseline (Worst Case) |
| **Logistic Regression** | `~0.509` | Linear separation improvement |
| **Linear SVM** | `~0.502` | Optimized with Hinge Loss |
| **XGBoost** | **`0.36`** | **Best Performance.** Captures non-linear feature interactions. |

**Performance Highlights:**
* **Precision:** Achieved **0.789** for the duplicate class (Class 1), meaning high confidence when flagging duplicates.
* **Recall:** Achieved **0.700**, successfully identifying 70% of real duplicates in the dataset.

### Tech Stack
* **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn.
* **NLP:** Spacy (GloVe), NLTK, FuzzyWuzzy.
* **Model:** XGBoost.

### To Run

**Method 1: Quick Start (Google Colab)**
Click the "Open in Colab" badge at the top of this file.
* *Note:* You will need to upload your `kaggle.json` token to the Colab environment when prompted to download the dataset.

**Method 2: Local Setup**
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashika-06/Quora-Duplicate-Question-Detection.git
    cd Quora-Duplicate-Question-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy xgboost spacy fuzzywuzzy python-Levenshtein
    python -m spacy download en_core_web_lg
    ```

3.  **Download Dataset (Kaggle API):**
    Place your `kaggle.json` API token in `~/.kaggle/` and run:
    ```bash
    pip install kaggle
    kaggle competitions download -c quora-question-pairs
    unzip quora-question-pairs.zip
    ```
   
4.  **Run the Notebook:**
    ```bash
    jupyter notebook EDA.ipynb
    ```


