import streamlit as st
import pickle
import re
import pandas as pd

def show():
    st.title("😊 Sentiment Analysis")
    st.write("Enter text to analyze its sentiment (Positive/Negative/Neutral)")
    
    # Twitter character limit
    MAX_TWEET_LENGTH = 280

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

    # Text input with character limit
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_text = st.text_area(
            "📝 Enter your text:", 
            height=100,
            max_chars=MAX_TWEET_LENGTH,
            placeholder=f"Type your text here... (max {MAX_TWEET_LENGTH} characters)",
            key="text_input"
        )
    
    with col2:
        st.markdown("### 📊 Limit")
        st.info(f"Max: {MAX_TWEET_LENGTH} chars")
        
        if user_text:
            chars_used = len(user_text)
            remaining = MAX_TWEET_LENGTH - chars_used
            word_count = len(user_text.split())
            
            # Character counter with color
            if remaining > 50:
                st.metric("Remaining", f"{remaining} chars", delta_color="off")
                st.caption(f"📝 Words: {word_count}")
                st.success("✅ Good length")
            elif remaining > 20:
                st.metric("Remaining", f"{remaining} chars", delta_color="off")
                st.caption(f"📝 Words: {word_count}")
                st.warning("⚠️ Getting long")
            elif remaining >= 0:
                st.metric("Remaining", f"{remaining} chars", delta_color="inverse")
                st.caption(f"📝 Words: {word_count}")
                st.error("🔴 Almost at limit")
            else:
                st.error(f"❌ Over by {abs(remaining)} chars")

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text!")
        elif len(user_text) > MAX_TWEET_LENGTH:
            st.error(f"❌ Text exceeds {MAX_TWEET_LENGTH} characters! Please shorten it.")
        else:
            clean_txt = clean_text(user_text)
            text_vec = vectorizer.transform([clean_txt])
            
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            st.subheader("📊 Results:")
            
            # Result with color
            col1, col2 = st.columns(2)
            with col1:
                if prediction == "positive":
                    st.markdown("### 🟢 POSITIVE")
                elif prediction == "negative":
                    st.markdown("### 🔴 NEGATIVE")
                else:
                    st.markdown("### 🔵 NEUTRAL")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
                st.progress(confidence/100)
            
            # Text stats
            st.caption(f"📊 Text stats: {word_count} words, {chars_used} characters")
            
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
    
    # Example texts section
    with st.expander("📋 Try these examples"):
        examples = {
            "Positive": "I absolutely love this product! It's amazing! 😍",
            "Negative": "Very disappointed with the service today 😠",
            "Neutral": "The weather is nice today. Nothing special."
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("😊 Positive Example", use_container_width=True):
                st.session_state.text_input = examples["Positive"]
                st.rerun()
        
        with col2:
            if st.button("😠 Negative Example", use_container_width=True):
                st.session_state.text_input = examples["Negative"]
                st.rerun()
        
        with col3:
            if st.button("😐 Neutral Example", use_container_width=True):
                st.session_state.text_input = examples["Neutral"]
                st.rerun()

# For direct execution
if __name__ == "__main__":
    show()
