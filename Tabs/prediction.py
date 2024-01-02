from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from sephora_functions import train_model

def app(df, x, y):
    st.title("Prediction")

    
