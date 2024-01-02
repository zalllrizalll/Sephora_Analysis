import streamlit as st
from streamlit_option_menu import option_menu
from sephora_functions import load_dataset
from Tabs import about, help, overview, prediction, sentiment_analysis

Pages = {
    "Overview" : overview,
    "Sentiment Analysis" : sentiment_analysis,
    "Prediction" : prediction,
    "Help" : help,
    "About" : about
}

# Option Menu
with st.sidebar:
    selected = option_menu("Sephora Analysis", list(Pages.keys()), 
        icons=['eye', 'emoji-smile', 'file-earmark-text', 'question-circle', 'info-circle'], menu_icon=" ", default_index=0)
    selected

# Load Dataset
df, x, y = load_dataset()

# Call app function
if selected in ["Sentiment Analysis", "Prediction"]:
    Pages[selected].app(df, x, y)
else:
    Pages[selected].app()