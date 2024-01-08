import streamlit as st

def app():
    st.title("About Us")

    info_brand = "https://www.sephora.co.id/"
    contact = "https://www.sephora.co.id/contact-us"
    st.markdown("""
        Sephora adalah destinasi utama Anda untuk eksplorasi kecantikan tanpa batas. Sejak didirikan pada tahun 1970, kami telah menjadi destinasi terpercaya bagi para pecinta kecantikan yang mencari inovasi terbaru, merek terkemuka, dan pengalaman berbelanja yang tak terlupakan.
    """)

    st.markdown("""
        Misi kami adalah memberdayakan setiap individu untuk mengekspresikan diri mereka melalui kecantikan. Kami berkomitmen untuk menyediakan produk-produk berkualitas tinggi yang memenuhi berbagai kebutuhan kecantikan, dari makeup hingga perawatan kulit dan parfum.
    """)

    st.markdown("""
        Di Sephora, kami percaya bahwa kecantikan adalah bentuk ekspresi diri yang paling intim. Oleh karena itu, kami berusaha memberikan pengalaman berbelanja yang unik dan penuh inspirasi. Dengan berbagai merek terkemuka dan inovatif, kami menyediakan ruang di mana setiap pelanggan dapat menemukan produk yang sesuai dengan gaya dan kebutuhan mereka.
    """)

    st.markdown("""
        Kami merayakan keberagaman dalam segala bentuknya. Sephora adalah rumah bagi semua jenis kulit, warna, dan identitas. Kami berkomitmen untuk menyediakan produk dan layanan yang mengakomodasi kebutuhan setiap pelanggan, memastikan bahwa semua orang merasa dihargai dan diwakili.
    """)

    st.markdown("""
        Jika Anda memiliki pertanyaan, umpan balik, atau ingin berbagi pengalaman Anda dengan kami, tim layanan pelanggan kami siap membantu. Kunjungi halaman [info@sephora.co.id](%s) untuk informasi lebih lanjut.
    """ % info_brand)

    st.markdown("""
        Terima kasih telah memilih Sephora sebagai mitra kecantikan Anda. Bersama-sama, mari jelajahi kecantikan tanpa batas!
    """)

    st.markdown("")
    st.subheader("Contact Us")
    st.subheader("Alamat Kantor Cabang")
    st.markdown("""
        DP Mall Semarang Unit G-30 ,31,32 & 36, 
        Jalan Pemuda no. 150 50132 
        Kota Semarang Central Java Jawa Tengah
    """)

    st.subheader("Customer Service")
    st.markdown("Contact: [contactus@sephora.co.id](%s)" % contact)
    st.markdown("Telephone: +62 24 86041904")

    st.subheader("Social Media")
    facebook = "https://www.facebook.com/sephoraindonesia"
    instagram = "https://www.instagram.com/sephoraidn"
    twitter = "https://www.twitter.com/Sephora"

    st.markdown("Facebook: [facebook@sephora.co.id](%s)" % facebook)
    st.markdown("Instagram: [instagram@sephora.co.id](%s)" % instagram)
    st.markdown("Twitter: [twitter@sephora.co.id](%s)" % twitter)