Ticket Classification and Entity Extraction System
This project implements a complete pipeline for ticket text classification and entity extraction, focusing on issue type classification, urgency level prediction, and named entity extraction from customer support tickets. The system includes data preprocessing, model training with hyperparameter tuning, evaluation, and a Gradio-based web app for real-time inference.


Project Overview
Customer support systems generate large volumes of textual tickets requiring automatic classification to route issues and prioritize responses efficiently. This project tackles:

Issue Type Classification: Categorizing tickets into predefined issue types.

Urgency Level Prediction: Predicting ticket urgency (e.g., low, medium, high).

Entity Extraction: Extracting product names, complaint keywords, and relevant dates from ticket texts.

Interactive UI: Deploying a web app interface via Gradio to input ticket texts and get instant predictions.

Features
Robust Text Preprocessing:

Tokenization, lemmatization, stopword removal using spaCy.

Custom text cleaning for normalization.

Feature Engineering:

TF-IDF vectorization with controlled vocabulary size.

Numerical features like sentiment polarity (TextBlob) and token count.

Modeling:

Random Forest classifiers for both issue type and urgency level prediction.

Extensive hyperparameter tuning with GridSearchCV.

Pipeline integration with sklearn’s ColumnTransformer and Pipeline.

Entity Extraction:

Rule-based extraction of products, complaint keywords, and date expressions via regex.

Evaluation:

Classification reports with precision, recall, F1-scores.

Cross-validation accuracy scores.

Deployment:

Gradio-based interactive app for user-friendly inference.

Data Preparation
Loaded ticket data from Excel file (ai_dev_assignment_tickets_complex_1000.xls).

Dropped rows missing critical columns (ticket_text, issue_type, urgency_level).

Applied spaCy’s English model (en_core_web_sm) for text preprocessing:

Lowercasing, lemmatization, removal of stopwords and non-alpha tokens.

Added features:

Cleaned text version for modeling.

Sentiment polarity using TextBlob.

Ticket length (word count).

Defined domain-specific lists for products and complaint keywords to support entity extraction.

Modeling
Issue Type Classifier:

TF-IDF vectorizer limited to 300 features.

RandomForestClassifier with hyperparameter tuning over estimators, depth, splits, leaf sizes, and class weights.

Evaluated with classification report and cross-validation.

Urgency Level Classifier:

Features: TF-IDF (3000 features, unigrams & bigrams), sentiment, ticket length.

Used sklearn’s ColumnTransformer for mixed feature types.

RandomForestClassifier with GridSearchCV for hyperparameter tuning.

Stratified train-test split to preserve class distribution.

Evaluation by macro F1-score and classification report.

Entity Extraction
Extracted products, complaints, and dates from ticket text using:

Case-insensitive substring matching for products and complaints.

Regex patterns for common date formats and expressions like "just 7 days," "after just 9 days," "expected by 04 March," etc.

Evaluation
Issue type classification showed strong accuracy and F1 scores after tuning.

Urgency level prediction balanced performance across classes.

Cross-validation confirmed model stability.

Entity extraction provides structured context to complement predictions.

Example evaluation queries:

"Order #30903 for Vision LED TV is 13 days late."

"Login not working on SmartWatch V2."

"Delayed shipment of RoboChef Blender expected by 04 March."

Web Application
Built with Gradio for easy deployment and user interaction.

Accepts free-form ticket text input.

Outputs:

Predicted issue type (textbox).

Predicted urgency level (textbox).

Extracted entities (JSON formatted).

Launch the app locally with:

bash
Copy
Edit
python app.py
Access the interface via the local URL provided on launch.

Installation and Usage
Prerequisites
Python 3.8+

Recommended: Create and activate a virtual environment.

Required Python Packages
bash
Copy
Edit
pip install -r requirements.txt
Sample requirements.txt includes:

nginx
Copy
Edit
pandas
numpy
scikit-learn
spacy
textblob
joblib
dateutil
gradio
xlrd
Setup
Download the dataset Excel file ai_dev_assignment_tickets_complex_1000.xls and place it in the project directory.

Run the modeling notebook or script to:

Preprocess data

Train and tune models

Save vectorizer and models as .pkl

Run app.py to launch the Gradio web app.



