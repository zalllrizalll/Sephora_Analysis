import streamlit as st
import torch
from transformers import BertTokenizer,BertForSequenceClassification
from sephora_functions import perform_sentiment_analysis
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def app(df, x, y):
    
    # Prediction section
    st.header("Predict Sentiment")

    review_text = st.text_area("Enter a review:")

    if st.button("Submit"):
        if review_text:
            sentiment = perform_sentiment_analysis(review_text)
            if (sentiment == 'Positive'):
                st.success(f"{sentiment} Review")
            else:
                st.error(f"{sentiment} Review")
        else:
            st.warning("Please enter a review.")