import streamlit as st
from transformers import BertTokenizer,BertForSequenceClassification
from sephora_functions import train_and_predict_sentiment

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def app(df, x, y):
    # User input for review
    review = st.text_area("Enter your review:")

    x_train = df['text']
    y_train = df['label']

    if st.button("Predict Sentiment"):
        if review:
            predictions = train_and_predict_sentiment(review, x_train, y_train, tokenizer, model)
            # Display the result
            if predictions[0] == 1:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
                
            st.success(f"The sentiment of the review is {sentiment}.")
        else:
            st.warning("Please enter a review before predicting.")