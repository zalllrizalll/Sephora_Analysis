import streamlit as st

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
                st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

            # Inner row 2 within col1
            inner_row2 = st.container()
            with inner_row2:
                st.text("Total Score")
                st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
            
        # Add content to the second column
        with col2:
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

            st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

            choose=st.radio("Choose Wordcloud :",("Positive Word", "Negative Word"))
            # Display content based on the selected option
            if choose == "Positive Word":
                show_option_1_content()
            elif choose == "Negative Word":
                show_option_2_content()

    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

def show_option_1_content():
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

def show_option_2_content():
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
