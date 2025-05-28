import joblib
import pandas as pd
import re
from textblob import TextBlob
from dateutil import parser

# Load models
tfidf = joblib.load('tfidf_vectorizer.pkl')
best_rf_issue = joblib.load('rf_issue_model.pkl')
best_model = joblib.load('urgency_model.pkl')

# Your existing helper functions
product_list = [
    'RoboChef Blender', 'ProTab X1', 'EcoBreeze AC', 'PhotoSnap Cam', 'Vision LED TV',
    'FitRun Treadmill', 'PowerMax Battery', 'SmartWatch V2', 'SoundWave 300', 'UltraClean Vacuum'
]

product_list_lower = [p.lower() for p in product_list]

complaint_keywords = [
    'broken', 'late', 'error', 'not working', 'damaged', 'defective',
    'missing', 'delay', 'issue', 'problem', 'no response', 'malfunction',
    'not refunded', 'charged twice', 'wrong product', 'lost', 'blocked',
    'stopped working', 'login not working', 'installation issue', 'not here'
]

def extract_entities(text):
    entities = {'products': [], 'dates': [], 'complaints': []}
    if pd.isnull(text): return entities

    text_lower = text.lower()

    # Product match
    for product in product_list_lower:
        if product in text_lower:
            entities['products'].append(product)

    # Complaint keyword match (substring style)
    for kw in complaint_keywords:
        if kw in text_lower:
            entities['complaints'].append(kw)

    # Date match
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text_lower)
        found_dates.extend(matches)

    entities['dates'] = found_dates

    return entities
    

date_patterns = [
    r'\b(?:\d{1,2}[/-])?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-]?\d{1,2}?\b',
    r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',
    r'\b(?:ordered|expected by|on)\s+\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
    r'after just \d+ days',
    r'just \d+ days',
    r'\d+ days late'
]

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

def process_ticket(ticket_text):
    text_clean = clean_text(ticket_text)
    sentiment = TextBlob(text_clean).sentiment.polarity
    ticket_length = len(text_clean.split())

    # Issue type prediction
    X_issue = tfidf.transform([text_clean])
    predicted_issue = best_rf_issue.predict(X_issue)[0]

    # Urgency prediction
    input_df = pd.DataFrame([{
        'ticket_text_clean': text_clean,
        'sentiment': sentiment,
        'ticket_length': ticket_length,
        'product': None
    }])
    predicted_urgency = best_model.predict(input_df)[0]

    # Entity extraction
    entities = extract_entities(ticket_text)

    return {
        'predicted_issue_type': predicted_issue,
        'predicted_urgency_level': predicted_urgency,
        'entities': entities
    }


import gradio as gr


def process_ticket_gradio(ticket_text):
    result = process_ticket(ticket_text)
    return (
        result['predicted_issue_type'],   # first output -> textbox
        result['predicted_urgency_level'],# second output -> textbox
        result['entities']                # third output -> JSON
    )


iface = gr.Interface(
    fn=process_ticket_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Enter ticket text here..."),
    outputs=[
        gr.Textbox(label="Predicted Issue Type"),
        gr.Textbox(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Ticket Classifier"
)

if __name__ == "__main__":
    iface.launch()
