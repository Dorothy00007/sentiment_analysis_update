import streamlit as st
import pickle
import re
import pandas as pd
import emoji

def show():
    st.title("😊 Sentiment Analysis with Emoji Support")
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

    # Emoji to text converter
    def emoji_to_text(text):
        """Convert emoji to text description"""
        # Dictionary of common emojis and their meanings
        emoji_map = {
            '😊': ' smiling ',
            '😍': ' love ',
            '🥰': ' love ',
            '😘': ' love ',
            '❤️': ' heart love ',
            '💕': ' love ',
            '💖': ' love ',
            '💗': ' love ',
            '💓': ' love ',
            '😭': ' crying sad ',
            '😢': ' sad ',
            '😠': ' angry ',
            '😡': ' angry ',
            '🤬': ' angry ',
            '😤': ' frustrated angry ',
            '😞': ' disappointed sad ',
            '😔': ' sad ',
            '😟': ' worried ',
            '😕': ' confused ',
            '🙁': ' sad ',
            '☹️': ' sad ',
            '🎉': ' celebration happy ',
            '✨': ' magic happy ',
            '🌟': ' star happy ',
            '⭐': ' star ',
            '💫': ' happy ',
            '🔥': ' fire awesome ',
            '💯': ' perfect ',
            '✅': ' check yes ',
            '❌': ' wrong no ',
            '👍': ' like good ',
            '👎': ' dislike bad ',
            '🙏': ' thank you please ',
            '😂': ' laughing happy ',
            '🤣': ' laughing happy ',
            '😅': ' awkward ',
            '😁': ' happy ',
            '☀️': ' sun sunny ',
            '🌧️': ' rain rainy ',
            '⛈️': ' storm stormy ',
            '🌈': ' rainbow happy ',
            '🍕': ' pizza food ',
            '🍔': ' burger food ',
            '☕': ' coffee ',
            '🍺': ' beer drink ',
            '🍷': ' wine drink ',
            '🏆': ' trophy win ',
            '🥇': ' gold win ',
            '💼': ' work job ',
            '📚': ' books study ',
            '📱': ' phone mobile ',
            '💻': ' computer ',
            '✈️': ' travel flight ',
            '🚗': ' car drive ',
            '🏠': ' home house ',
            '🐶': ' dog pet ',
            '🐱': ' cat pet ',
            '🌸': ' flower beautiful ',
            '🌺': ' flower beautiful ',
            '🎵': ' music ',
            '🎶': ' music ',
            '⚽': ' sports ',
            '🏀': ' sports ',
            '🎮': ' gaming ',
            '⌛': ' time waiting ',
            '⏰': ' time alarm ',
            '💔': ' heartbroken sad ',
            '💪': ' strong power ',
            '🤞': ' hope ',
            '🤷': ' whatever ',
            '🥺': ' pleading sad ',
            '😴': ' sleepy tired ',
            '🤒': ' sick ',
            '🤢': ' disgusted ',
            '🥳': ' party happy ',
            '😎': ' cool ',
            '🤔': ' thinking ',
            '🤨': ' suspicious ',
            '😏': ' smirk ',
            '😬': ' awkward ',
            '🥱': ' bored tired ',
            '😷': ' sick mask ',
            '🤕': ' hurt ',
            '🤑': ' money rich ',
            '🤮': ' disgusted vomit ',
            '😈': ' evil ',
            '👿': ' evil angry ',
            '💀': ' dead ',
            '☠️': ' dead danger ',
            '👻': ' ghost ',
            '🤖': ' robot ',
            '🎃': ' halloween ',
            '😺': ' cat happy ',
            '😸': ' cat happy ',
            '😹': ' cat laughing ',
            '😻': ' cat love ',
            '😼': ' cat smirk ',
            '😽': ' cat kiss ',
            '🙀': ' cat shock ',
            '😿': ' cat cry ',
            '😾': ' cat angry '
        }
        
        # Replace emojis with text
        for emoji_char, text_replacement in emoji_map.items():
            if emoji_char in text:
                text = text.replace(emoji_char, text_replacement)
        
        # Also try using emoji library for any missed emojis
        try:
            text = emoji.demojize(text)
            # Convert :smile: format to readable text
            text = text.replace(':', ' ').replace('_', ' ')
        except:
            pass
            
        return text

    # Text cleaning with emoji support
    def clean_text_with_emoji(text):
        # First convert emojis to text
        text_with_emoji_text = emoji_to_text(text)
        
        # Then do normal cleaning
        text = str(text_with_emoji_text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    # Text input with character limit
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_text = st.text_area(
            "📝 Enter your text (emojis supported!):", 
            height=100,
            max_chars=MAX_TWEET_LENGTH,
            placeholder=f"Type your text here... emojis will be understood! (max {MAX_TWEET_LENGTH} characters)",
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
    if st.button("🔍 Analyze Sentiment (with Emoji Support)", type="primary"):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text!")
        elif len(user_text) > MAX_TWEET_LENGTH:
            st.error(f"❌ Text exceeds {MAX_TWEET_LENGTH} characters! Please shorten it.")
        else:
            # Show original text with emojis
            st.markdown("### 📝 Original Text:")
            st.write(user_text)
            
            # Show emoji conversion
            with st.expander("🔍 View emoji conversion"):
                converted = emoji_to_text(user_text)
                st.write("**After emoji conversion:**")
                st.code(converted)
                
                cleaned = clean_text_with_emoji(user_text)
                st.write("**Final cleaned text (sent to model):**")
                st.code(cleaned)
            
            # Clean and predict using emoji-aware cleaning
            clean_txt = clean_text_with_emoji(user_text)
            text_vec = vectorizer.transform([clean_txt])
            
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            st.subheader("📊 Results:")
            
            # Result with color and matching emoji
            col1, col2 = st.columns(2)
            with col1:
                if prediction == "positive":
                    st.markdown("### 🟢 POSITIVE 😊")
                elif prediction == "negative":
                    st.markdown("### 🔴 NEGATIVE 😠")
                else:
                    st.markdown("### 🔵 NEUTRAL 😐")
            
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
    
    # Example texts section with emojis
    # with st.expander("📋 Try these examples with emojis"):
    #     examples = {
    #         "Positive": "I absolutely love this product! It's amazing! 😍❤️🎉",
    #         "Negative": "Very disappointed with the service today 😠😤💔",
    #         "Neutral": "The weather is nice today. Nothing special. ☁️🌤️"
    #     }
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         if st.button("😊 Positive + Emoji", use_container_width=True):
    #             st.session_state.text_input = examples["Positive"]
    #             st.rerun()
        
    #     with col2:
    #         if st.button("😠 Negative + Emoji", use_container_width=True):
    #             st.session_state.text_input = examples["Negative"]
    #             st.rerun()
        
    #     with col3:
    #         if st.button("😐 Neutral + Emoji", use_container_width=True):
    #             st.session_state.text_input = examples["Neutral"]
    #             st.rerun()
    
    # Show emoji support info
    # with st.expander("ℹ️ About Emoji Support"):
    #     st.info("""
    #     **Supported Emojis:**
    #     - 😊😍🥰😘❤️ - Love/Positive
    #     - 😠😡🤬😤 - Anger/Negative
    #     - 😭😢😞😔 - Sad/Negative
    #     - 🎉✨🌟🔥 - Celebration/Positive
    #     - 👍👎✅❌ - Like/Dislike
    #     - And many more!
        
    #     Emojis are converted to text descriptions before analysis.
    #     """)

# For direct execution
if __name__ == "__main__":
    show()
