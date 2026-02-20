import streamlit as st
import pickle
import re
import pandas as pd

def show():
    st.title("😊 Sentiment Analysis")
    st.write("Enter text to analyze its sentiment (Positive/Negative/Neutral)")

    # Load model
    try:
        with open("sentiment_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        st.success("✅ Model loaded successfully!")
    except:
        st.error("❌ Model file not found.")
        st.stop()

    # Text cleaning
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    # Text input
    user_text = st.text_area("📝 Enter your text:", height=100)

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_text.strip():
            clean_txt = clean_text(user_text)
            text_vec = vectorizer.transform([clean_txt])
            
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            st.subheader("📊 Results:")
            
            if prediction == "positive":
                st.markdown("### 🟢 POSITIVE")
            elif prediction == "negative":
                st.markdown("### 🔴 NEGATIVE")
            else:
                st.markdown("### 🔵 NEUTRAL")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probabilities
            prob_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": [f"{p:.1%}" for p in probabilities]
            })
            st.table(prob_df)
            
            # Bar chart
            chart_data = pd.DataFrame({
                "sentiment": model.classes_,
                "probability": probabilities
            }).set_index("sentiment")
            st.bar_chart(chart_data)
        else:
            st.warning("⚠️ Please enter some text!")

# For direct execution
if __name__ == "__main__":
    show()
