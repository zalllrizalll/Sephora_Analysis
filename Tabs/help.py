import streamlit as st

def app():
    st.title("Frequently Asked Questions by Users")

    # Menampilkan beberapa pertanyaan umum dari para pengguna
    with st.expander("Apa saja merek kecantikan yang tersedia di Sephora dan apa yang membuatnya unik?"):
        st.write("Sephora menawarkan berbagai merek terkemuka seperti Fenty Beauty, Urban Decay, dan Dior, yang dikenal karena inovasi produk dan kualitasnya.")

    with st.expander("Bagaimana program loyalitas pelanggan Sephora berfungsi dan apa keuntungan yang dapat diperoleh pelanggan yang terdaftar?"):
        st.write("Program loyalitas Sephora, Beauty Insider, memberikan poin setiap pembelian yang dapat ditukar dengan sampel gratis, hadiah eksklusif, dan diskon.")

    with st.expander("Apa produk kecantikan terbaru atau tren yang sedang ditonjolkan oleh Sephora?"):
        st.write("Sephora sering menonjolkan produk kecantikan terbaru, seperti koleksi terbaru dari merek-merek terkemuka atau peluncuran produk eksklusif.")

    with st.expander("Bagaimana Sephora mendekati keberagaman dalam menyediakan produk kecantikan untuk berbagai jenis kulit dan warna?"):
        st.write("Sephora memiliki beragam produk untuk berbagai jenis kulit dan warna, dan mereka aktif dalam mendukung keberagaman dan inklusivitas dalam industri kecantikan.")

    with st.expander("Apa layanan konsultasi atau uji produk yang ditawarkan oleh Sephora di toko atau online?"):
        st.write("Sephora menyediakan layanan konsultasi kecantikan dan uji produk di toko untuk membantu pelanggan memilih produk yang sesuai.")

    with st.expander("Bagaimana program Sephora Beauty Insider berbeda dari program loyalitas toko kecantikan lainnya?"):
        st.write("Beauty Insider menawarkan tingkatan keanggotaan dengan berbagai manfaat, seperti diskon, akses ke penawaran eksklusif, dan acara khusus bagi anggota.")
    
    with st.expander("Apakah Sephora memiliki kebijakan pengembalian yang unik, dan bagaimana prosesnya?"):
        st.write("Sephora memiliki kebijakan pengembalian yang fleksibel, memungkinkan pelanggan untuk mengembalikan produk dengan syarat tertentu.")
    
    with st.expander("Apa yang bisa diharapkan dari acara atau promosi khusus yang diadakan oleh Sephora secara berkala?"):
        st.write("Sephora secara berkala mengadakan acara khusus seperti VIB Sale dan promosi eksklusif untuk merayakan peluncuran produk baru atau perayaan khusus.")
    
    with st.expander("Bagaimana Sephora mengintegrasikan teknologi dalam pengalaman berbelanja kecantikan, baik di toko fisik maupun online?"):
        st.write("Sephora menggunakan teknologi seperti augmented reality untuk uji produk virtual dan aplikasi seluler untuk memberikan pengalaman berbelanja yang lebih interaktif.")
    
    with st.expander("Bagaimana Sephora berkontribusi terhadap keberlanjutan dan tanggung jawab sosial perusahaan di industri kecantikan?"):
        st.write("Sephora berkomitmen pada keberlanjutan dan tanggung jawab sosial, termasuk program peningkatan keberagaman, dan berbagai inisiatif lingkungan.")
    
    