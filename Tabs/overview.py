import streamlit as st

def app():
    st.title("Overview Sephora")

    st.markdown("""
        Di era digital saat ini, perusahaan kecantikan, khususnya yang bergerak di industri perawatan kulit, menghadapi tantangan besar dalam memahami kebutuhan dan persepsi pelanggannya. 
        Penggunaan produk perawatan kulit sudah menjadi bagian penting dari rutinitas sehari-hari banyak orang, dan dengan beragamnya produk yang beredar di pasaran, konsumen kini lebih selektif dan berhati-hati dalam memilih produk yang akan digunakan. 
        Kebanyakan konsumen kini mendasarkan keputusannya pada review produk yang mereka baca di internet.
    """)

    image = "Images/sephora.jpg"
    st.image(image, caption="Logo Brand Sephora", use_column_width=True)

    st.markdown("""
        Dengan menggunakan dataset skincare review dari Kaggle yang berisi review produk perawatan kulit, proyek ini bertujuan untuk melakukan analisis sentimen terhadap produk skincare yang diproduksi oleh perusahaan dengan brand Sephora. 
        Data ini akan dianalisis menggunakan pendekatan CNN Layer untuk analisis sentimen dan dengan pemodelan menggunakan BERT.
    """)