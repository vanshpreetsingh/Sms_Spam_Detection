# SMS Spam Detection Project

### Note: The project is deployed and fully functional
### URL: https://smsspamdetection1.streamlit.app/

This repository contains the complete code and documentation for an SMS spam detection project, which includes both a machine learning model and a web application. The primary goal of this project is to classify SMS messages as spam or not spam using various machine learning algorithms.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Text Preprocessing](#text-preprocessing)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [Model Improvement](#model-improvement)
  - [Final Model Selection](#final-model-selection)
- [Web Application](#web-application)
  - [Web App Functionality](#web-app-functionality)
  - [Web App Code](#web-app-code)
- [Dependencies](#dependencies)

## Overview
The project aims to develop an **SMS spam detection** system that can effectively classify messages as spam or non-spam. This system utilizes natural language processing (NLP) techniques and machine learning algorithms to analyze and predict the nature of SMS messages.
The project consists of two main components: 
- A **Jupyter Notebook** for building and evaluating the machine learning model
- A **Streamlit Web Application** for real-time spam detection.

## Dataset
The dataset used for this project is the **"SMS Spam Collection" dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)**. It consists of 5,572 SMS messages in English, categorized as either 'ham' (legitimate) or 'spam'. The dataset contains two columns:
- `v1`: The label (ham or spam)
- `v2`: The message content

## Machine Learning Model

### Data Cleaning
The data cleaning process involves:
1. **Loading the Dataset**: Read the `spam.csv` file into a pandas DataFrame.
2. **Removing Unnecessary Columns**: Drop columns `Unnamed: 2`, `Unnamed: 3`, and `Unnamed: 4` as they contain no useful information.
3. **Renaming Columns**: Rename `v1` to `target` and `v2` to `text` for better clarity.
4. **Encoding the Target Variable**: Convert 'ham' to 0 and 'spam' to 1 using LabelEncoder.
5. **Removing Duplicates**: Eliminate duplicate entries to ensure data quality and integrity.

### Exploratory Data Analysis (EDA)
EDA provides insights into the dataset:
1. **Target Distribution**: Visualize the distribution of 'ham' and 'spam' messages.
2. **Feature Extraction**: Extract features such as the number of characters, words, and sentences in each message.
3. **Visualization**: Use histograms and pair plots to visualize feature distributions for spam and ham messages. Heatmaps are used to visualize correlations between features.

### Text Preprocessing
Text preprocessing involves transforming the text data into a format suitable for machine learning algorithms:
1. **Lowercasing**: Convert all text to lowercase to maintain uniformity.
2. **Tokenization**: Split the text into individual words.
3. **Removing Non-Alphanumeric Characters**: Keep only alphanumeric words.
4. **Removing Stop Words and Punctuation**: Filter out common stop words and punctuation marks.
5. **Stemming**: Reduce words to their root forms using PorterStemmer.
6. **Transform Function**: Apply these preprocessing steps to the text data.

### Model Building
The model building process includes:
1. **Vectorization**: Transform the text data into numerical vectors using TF-IDF vectorization.
2. **Data Splitting**: Split the dataset into training and testing sets.
3. **Training Models**: Train various machine learning models including Naive Bayes, SVM, Decision Trees, Random Forests, and ensemble methods.
4. **Hyperparameter Tuning**: Experiment with different parameters to optimize model performance.

### Model Evaluation
Evaluate the models using accuracy and precision metrics. The performance of various models is summarized in the table below:

| Model                   | Accuracy | Precision |
|-------------------------|----------|-----------|
| Gaussian Naive Bayes    | 87.37%   | 72.73%    |
| Multinomial Naive Bayes | 96.34%   | 96.00%    |
| Bernoulli Naive Bayes   | 97.30%   | 95.68%    |
| Support Vector Classifier | 98.79% | 97.53%    |
| K-Nearest Neighbors     | 91.51%   | 87.50%    |
| Decision Tree Classifier| 92.02%   | 82.76%    |
| Logistic Regression     | 96.82%   | 95.00%    |
| Random Forest Classifier| 98.42%   | 97.06%    |
| AdaBoost Classifier     | 96.82%   | 93.88%    |
| Bagging Classifier      | 97.48%   | 95.74%    |
| Extra Trees Classifier  | 98.60%   | 97.06%    |
| Gradient Boosting Classifier | 96.64% | 93.75% |
| XGBoost Classifier      | 97.48%   | 94.74%    |

### Model Improvement
To enhance the model performance, the following steps are taken:
1. **Parameter Tuning**: Experiment with different values for the `max_features` parameter in TF-IDF vectorization.
2. **Feature Scaling**: Apply scaling to the features.
3. **Ensemble Methods**: Use ensemble techniques like Voting Classifier and Stacking Classifier to combine the strengths of multiple models.

### Final Model Selection
After evaluating and improving various models, the final model is selected based on its performance metrics. The final model used for deployment is a Voting Classifier, which combines the predictions of the Support Vector Classifier, Multinomial Naive Bayes, and Extra Trees Classifier to achieve high accuracy and precision.

## Web Application

### Web App Functionality
The **Streamlit web application** provides an interactive interface for users to classify SMS messages. The app:
1. Accepts user input via a text area.
2. Preprocesses the input text using the same steps as in the machine learning model.
3. Vectorizes the input text using the saved TF-IDF vectorizer.
4. Predicts the class of the message using the saved machine learning model.
5. Displays the prediction result as either "Spam" or "Not Spam".

### Web App Code
The web app code (`app.py`) includes the following sections:
1. **Imports and NLTK Data Download**: Import necessary libraries and download required NLTK data.
2. **Text Preprocessing Function**: Define the `transform_text` function to preprocess input text.
3. **Load Saved Models**: Load the TF-IDF vectorizer and the trained machine learning model using `pickle`.
4. **Streamlit Interface**: Create the Streamlit interface for user input and prediction display.

## Dependencies

This project requires the following Python packages:

- Python 3.x
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud
- xgboost
- streamlit
- pickle

## How to Run

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook

1. Open `spam_detection.ipynb` in Jupyter.
2. Run all cells to train and evaluate the model.
3. Save the trained model and vectorizer using `pickle`.

### Run the Streamlit Web App

```bash
streamlit run app.py
```

## Outputs

### Not Spam:
<img src="https://github.com/vanshpreetsingh/Sms_Spam_Detection/assets/84062550/aff22351-e26d-4d87-8a9e-274b012fd291" width="600" />

### Spam:
<img src="https://github.com/vanshpreetsingh/Sms_Spam_Detection/assets/84062550/d7c34c31-9d4e-47b4-b936-8882cd446685" width="600" />

