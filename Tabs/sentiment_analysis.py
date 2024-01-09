import streamlit as st
import io
import base64
from sephora_functions import generate_wordcloud,preprocessing_data,create_feedback_plot_streamlit,plot_helpfulness_vs_recommendation

def app(df, x, y):
    st.title("Sentiment Analysis")

    tab1, tab2, = st.tabs(["Vizualizations", "Review by Sentiment"])
   
    with tab1:

        col1, col2 = st.columns(2)

        # Add content to the first column
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
            inner_row1 = st.container()
            with inner_row1:
                st.text("Total Feedback")
                create_feedback_plot_streamlit(df)

            # Inner row 2 within col1
            inner_row2 = st.container()
            with inner_row2:
                st.text("Total Score")
                plot_helpfulness_vs_recommendation(df)
                
        # Add content to the second column
        with col2:
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

            st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

            choose=st.radio("Choose Wordcloud :",("Positive Word", "Negative Word"))

            # Perform word cloud analysis
            # Display content based on the selected option
            if choose == "Positive Word":
                wordcloud_figure_positive = generate_wordcloud(df['text'].tolist(), sentiment_label=1)
                if wordcloud_figure_positive is not None:
                     st.image(io.BytesIO(base64.b64decode(wordcloud_figure_positive)), use_column_width=True)
            elif choose == "Negative Word":
                wordcloud_figure_negative = generate_wordcloud(df['text'].tolist(), sentiment_label=0)
                if wordcloud_figure_negative is not None:
                     st.image(io.BytesIO(base64.b64decode(wordcloud_figure_negative)), use_column_width=True)
    with tab2:
        review_label = st.selectbox("Select review type", ['Choose your choice','positive', 'negative'])

        label_mapping = {'positive': 1, 'negative': 0}

        # Get the corresponding label value
        selected_label = label_mapping.get(review_label) if review_label is not None else None

        selected_reviews = df[df['label'] == selected_label]['text']

        if not selected_reviews.empty:
            selected_review = selected_reviews.iloc[0]
            # Use selected_review as needed
            st.info(selected_review)
        else:
            st.warning("Please choose your choice label!") 