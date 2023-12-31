import streamlit as st
from sephora_functions import preproses

def app(df):
    st.title("Overview Sephora")

    # Display the image
    image = "Images/sephora.jpg"
    st.image(image, caption="Logo Brand Sephora", use_column_width=True)

    st.markdown("""
                Sentimen Analisis dari review produk dari brand Sephora dengan dataset yang 
                sudah dipreprosesing seperti dibawah
                """)

    preprocessed_df = preproses(df)

    st.markdown("<h3>Preprossesed DataFrame Overview</h3>", unsafe_allow_html=True)
    st.write(preprocessed_df.head(10))

    st.markdown("""
                Label x dan y yang digunakan sebagai acuan dari training 
                """)
    # Display x and y side by side

    col1, col2 = st.columns(2)

    with col1:
        columns_to_display = ["text"]  # Replace with your column names
        st.header("Label x")
        st.write(preprocessed_df[columns_to_display])
        

    with col2:
        columns_to_display = ["label"]  # Replace with your column names
        st.header("Label y")
        st.write(preprocessed_df[columns_to_display])
    