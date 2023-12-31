from seaborn import load_dataset
import streamlit as st
from streamlit_option_menu import option_menu
from sephora_functions import load_dataset
from Tabs import overview, sentiment_analysis, topic_modelling, account

Pages = {
    "Overview" : overview,
    "Sentiment Analysis" : sentiment_analysis,
    "Topic Modelling": topic_modelling,
    "My Profile": account
}

# Option Menu
with st.sidebar:
    selected = option_menu(" ", list(Pages.keys()), 
        icons=['eye', 'emoji-smile', 'file-earmark-text', 'person-circle'], menu_icon=" ", default_index=0)
    selected

# Load Dataset
df, x, y = load_dataset()

# Call app function
if selected in ["Sentiment Analysis", "Topic Modelling"]:
    Pages[selected].app(df, x, y)
else:
    Pages[selected].app()