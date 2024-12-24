# CSCI303: Data Science - Data Preprocessing

#### *Melody Goldanloo*

## Overview
This project demonstrates structured and unstructured data preprocessing techniques as part of the CSCI303 (Data Science) course. The project focuses on data cleaning, feature engineering, exploratory data analysis (EDA), dimensionality reduction, and text preprocessing using Python libraries.

## Key Features

### Structured Data Preprocessing
Working with a dataset of flight information to analyze and preprocess numerical and categorical data for machine learning applications.

- #### EDA and Visualization:
  Generated descriptive statistics and visualized relationships between features (e.g., delays vs. distance).

- #### Data Cleaning:
  Handles missing values and outliers for robust preprocessing.

- #### Feature Engineering:
  Created new features (e.g., normalized columns and one-hot encoding for categorical variables).

- #### Dimensionality Reduction:
  Applied PCA to reduce dimentionality while retaining 95% variance.

- #### Data Splitting:
  Prepared training, testing, and validation datasets for machine learning models.

### Unstructured Data Preprocessing
Cleaning and preprocessing text data, including tokenization, removing punctuations, and handling stopwords for Natural Language Processing (NLP).

- #### Text Preprocessing:
  Cleaned and tokenized text data, removed punctuation, and filtered out stopwords.

- #### TF-IDF Transformation:
  Converted text data into numerical vectors for modeling and analysis.

### Data Files
- `flights-small.csv` - Dataset for structured data analysis

## Technologies Used

**Programming Language:** Python

### Libraries
#### Data Preprocessing and Analysis:
- `pandas` - Data manipulation, cleaning, and anlysis.
- `numpy` - Numerical computations and handling arrays.

#### Visualization:
- `matplotlib.pyplot` - Creating static plots and visualizations.
- `seaborn` - Built on Matplotlib, used for advanced statistical visualization.

#### Preprocessing:
- `sklearn.preprocessing` -
  - `StandardScaler` and `MinMaxScaler` - Scaling numerical data.
  - `LabelEncoder` - Encoding categorical data.
- `sklearn.decomposition.PCA` - Principal Component Analysis for dimensionality reduction.
- `sklearn.model_selection.train_test_split` - Splitting data into training and testing sets.

#### Text Processing:
- `nltk` -
  - `word_tokenize` - Tokenizes text into words for further processing.
  - `stopwords` - Provides common stopwords to filter out.
- `string` - Built-in Python library for text manipulation and handling punctuation.
- `sklearn.feature_extraction.text.TfidfVectorizer` - Converts text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF)

## Usage
1. Open the `Project - Data Preprocessing.ipynb` file.
2. Follow the steps outlined and run the corresponding scripts to preprocess the data.

## Acknowledgements
This project was developed as part of CSCI303 - Data Science at the Colorado School of Mines. Special thanks to Professor Morgan Cox and Dr. Wendy Fisher for guidance and course resources and materials.
