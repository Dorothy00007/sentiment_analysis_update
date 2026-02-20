import streamlit as st
import pickle
import re
import pandas as pd

# Page setup
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ About Us"])

# About Us Page
if page == "ℹ️ About Us":
    st.title("ℹ️ About Us")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    
    st.markdown("""
    ### 🎯 Our Mission
    To provide easy-to-use sentiment analysis tools for everyone.
    
    ### 👥 Our Team
    We are a team of NLP enthusiasts passionate about understanding human emotions through text.
    
    ### 📊 About This App
    - **Version:** 1.0.0
    - **Model:** Logistic Regression
    - **Accuracy:** ~85%
    - **Training Data:** 3,500+ tweets
    
    ### 📞 Contact Us
    - 📧 Email: sentiment.analyzer@gmail.com
    - 🐦 Twitter: @sentiment_ai
    - 💻 GitHub: sentiment-analyzer
    
    ### 📝 How It Works
    1. Enter your text
    2. Our AI model analyzes the sentiment
    3. Get instant results with confidence scores
    
    ### ⚙️ Technical Details
    - **Frontend:** Streamlit
    - **ML Library:** Scikit-learn
    - **Text Processing:** NLTK, Regex
    - **Deployment:** Streamlit Cloud
    
    ### 🙏 Special Thanks
    To all our users who provide valuable feedback!
    
    ### 📜 License
    MIT License - Free to use and modify
    """)
    
    st.markdown("---")
    st.markdown("### ⭐ Rate Us")
    rating = st.slider("How would you rate this app?", 1, 5, 3)
    if st.button("Submit Rating"):
        st.success(f"Thanks for rating us {rating} stars! ⭐" * rating)
    
    st.markdown("---")
    st.markdown("### 💬 Feedback")
    feedback = st.text_area("Have suggestions? Let us know:")
    if st.button("Send Feedback"):
        st.success("Thanks for your feedback! 🙏")
    
    st.markdown("---")
    st.markdown("#### © 2024 Sentiment Analyzer. All rights reserved.")

# Home Page
else:
    st.title("😊 Sentiment Analysis")
    st.write("Enter text to analyze its sentiment (Positive/Negative/Neutral)")

    # Try to load model
    try:
        with open("sentiment_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        st.success("✅ Model loaded successfully!")
    except:
        st.error("❌ Model file not found. Please make sure 'sentiment_model.pkl' is in the same folder.")
        st.stop()

    # Text cleaning
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    # Text input
    user_text = st.text_area("📝 Enter your text:", height=100, 
                            placeholder="Example: 'I love this product! It's amazing!'")

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_text.strip():
            # Clean and predict
            clean_txt = clean_text(user_text)
            text_vec = vectorizer.transform([clean_txt])
            
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            confidence = max(probabilities) * 100
            
            # Show results
            st.markdown("---")
            st.subheader("📊 Results:")
            
            # Color-coded result
            if prediction == "positive":
                st.markdown("### 🟢 POSITIVE")
            elif prediction == "negative":
                st.markdown("### 🔴 NEGATIVE")
            else:
                st.markdown("### 🔵 NEUTRAL")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probabilities
            st.subheader("Probabilities:")
            prob_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": [f"{p:.1%}" for p in probabilities]
            })
            st.table(prob_df)
            
            # Bar chart
            st.subheader("Visualization:")
            chart_data = pd.DataFrame({
                "sentiment": model.classes_,
                "probability": probabilities
            }).set_index("sentiment")
            st.bar_chart(chart_data)
        else:
            st.warning("⚠️ Please enter some text!")

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ Natural Language Processing")
    
    # Quick about link in footer
    st.markdown("""
    <div style='text-align: center'>
        <a href='#' onclick='alert("Go to sidebar and click About Us")' style='text-decoration: none; color: #888;'>
            ℹ️ About Us | 📧 Contact | 📝 Feedback
        </a>
    </div>
    """, unsafe_allow_html=True)