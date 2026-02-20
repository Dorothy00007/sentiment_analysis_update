import streamlit as st
import home
import aboutus

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "ℹ️ About Us"],
    key="navigation"
)

# Page routing
if page == "🏠 Home":
    home.show()  
else:
    aboutus.show()  

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Sentiment Analysis App")