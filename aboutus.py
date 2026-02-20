import streamlit as st

def show():
    st.title("ℹ️ About Our Sentiment Analysis System")
    
    # ညီမရဲ့ မူရင်း about us content ကိုဒီမှာထည့်
    st.markdown("""
    ### 🤖 Model Information
    - **Model Type:** Logistic Regression
    - **Algorithm:** Scikit-learn
    - **Vectorization:** TF-IDF with 3000 features
    - **Training Data:** 3,534 tweets
    
    ### 📚 Libraries Used
    - Streamlit (v1.28.0)
    - Scikit-learn (v1.3.0)
    - Pandas (v2.0.3)
    - NLTK (v3.8.1)
    - NumPy (v1.24.3)
    
    ### 🎯 How It Works
    1. **Text Cleaning** - Remove special characters, lowercase
    2. **Vectorization** - Convert text to numbers (TF-IDF)
    3. **Prediction** - Logistic Regression model
    4. **Result** - Show sentiment with confidence score
    """)

# For direct execution
if __name__ == "__main__":
    show()
