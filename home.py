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

    # Emoji to text converter with sentiment intensity
    def emoji_to_text_with_sentiment(text):
        """Convert emoji to text with sentiment intensity"""
        
        # ဒီမှာ သတိထားရမှာက - negative emoji တွေကို positive words တွေနဲ့ မပြောင်းမိဖို့ပါ
        
        # Dictionary of common emojis and their meanings with sentiment intensity
        emoji_map = {
            # Positive emojis 
            '😊': ' happy positive good great wonderful ',
            '😍': ' love amazing wonderful fantastic ',
            '🥰': ' love adore amazing wonderful ',
            '😘': ' love kiss affection ',
            '❤️': ' love heart positive good ',
            '💕': ' love affection positive ',
            '💖': ' love sparkle happy ',
            '💗': ' love happy positive ',
            '💓': ' love excited positive ',
            '🎉': ' celebration party happy congratulations ',
            '✨': ' sparkle special wonderful amazing ',
            '🌟': ' star excellent amazing wonderful ',
            '⭐': ' star good excellent great ',
            '💫': ' happy excited amazing ',
            '🔥': ' fire awesome amazing excellent ',
            '💯': ' perfect excellent complete awesome ',
            '✅': ' correct yes good approved positive ',
            '👍': ' like good great positive ',
            '😂': ' laughing funny happy joyful ',
            '🤣': ' laughing funny hilarious amazing ',
            '😁': ' smile happy cheerful positive ',
            '🌈': ' rainbow beautiful colorful happy ',
            '🏆': ' trophy winner champion victory best ',
            '🥇': ' gold medal winner champion best ',
            '💪': ' strong powerful confident great ',
            '🤞': ' hope wishful positive ',
            '🥳': ' party celebration birthday happy excited ',
            '😎': ' cool stylish awesome great ',
            
            # Negative emojis (ဒါတွေကို သေချာခွဲထားရမယ်)
            '😡': ' angry mad furious frustrated upset terrible bad negative horrible awful ',
            '😠': ' angry mad furious frustrated upset negative bad ',
            '🤬': ' swear curse angry mad furious frustrated negative bad ',
            '😤': ' frustrated angry upset negative bad ',
            '😭': ' cry sad heartbroken upset terrible negative awful ',
            '😢': ' sad tearful unhappy depressed negative ',
            '😞': ' disappointed sad upset unhappy negative ',
            '😔': ' sad depressed unhappy gloomy negative ',
            '😟': ' worried concerned anxious nervous uneasy ',
            '😕': ' confused unsure puzzled doubtful uncertain ',
            '🙁': ' sad unhappy disappointed negative ',
            '☹️': ' very sad unhappy negative terrible ',
            '💔': ' heartbroken sad devastated negative terrible awful ',
            '👎': ' dislike bad terrible poor negative horrible ',
            '❌': ' wrong no bad incorrect negative ',
            '😈': ' evil mischievous bad negative ',
            '👿': ' devil evil angry negative bad ',
            '💀': ' dead death scary frightening negative ',
            '☠️': ' danger death poisonous negative ',
            '👻': ' ghost scary spooky negative ',
            '🤢': ' disgusted sick negative horrible ',
            '🤮': ' vomit disgusted sick very negative terrible ',
            '😷': ' sick ill unwell negative ',
            '🤕': ' hurt injured pain negative ',
            '🥺': ' pleading sad begging emotional ',
            '😤': ' frustrated angry upset negative ',
            '😫': ' tired frustrated upset negative ',
            '😩': ' exhausted frustrated upset negative ',
            '😒': ' annoyed unsatisfied negative ',
            '😓': ' stressed tired negative ',
            '😥': ' sad disappointed negative ',
            '😰': ' worried anxious nervous negative ',
            '😨': ' scared frightened negative ',
            '😱': ' scream shocked frightened negative ',
            
            # Neutral emojis
            '😅': ' awkward nervous embarrassed ',
            '😴': ' tired sleepy bored ',
            '🤒': ' sick ill unwell ',
            '🤔': ' thinking thoughtful pondering ',
            '🤨': ' skeptical suspicious ',
            '😏': ' smirking smug confident ',
            '😬': ' awkward nervous uncomfortable ',
            '🥱': ' tired bored sleepy ',
            '🤑': ' money rich wealthy ',
            '🤷': ' shrug whatever indifferent unsure ',
            
            # Food and objects (neutral)
            '🍕': ' pizza food ',
            '🍔': ' burger food ',
            '☕': ' coffee drink ',
            '🍺': ' beer drink ',
            '🍷': ' wine drink ',
            '💼': ' work job business ',
            '📚': ' books study reading ',
            '📱': ' phone mobile ',
            '💻': ' computer laptop ',
            '✈️': ' airplane travel flight ',
            '🚗': ' car vehicle drive ',
            '🏠': ' house home building ',
            '🐶': ' dog pet animal ',
            '🐱': ' cat pet animal ',
            '🌸': ' flower beautiful ',
            '🌺': ' flower beautiful ',
            '🎵': ' music song ',
            '🎶': ' music song ',
            '⚽': ' soccer sports ',
            '🏀': ' basketball sports ',
            '🎮': ' game gaming ',
            '⌛': ' time waiting ',
            '⏰': ' time alarm ',
            '☀️': ' sun sunny weather ',
            '🌧️': ' rain rainy weather ',
            '⛈️': ' thunder storm stormy ',
            '🌨️': ' snow snowy cold ',
        }
        
        # Replace emojis with text
        for emoji_char, text_replacement in emoji_map.items():
            if emoji_char in text:
                text = text.replace(emoji_char, text_replacement)
        
        # Also try using emoji library for any missed emojis
        try:
            demojized = emoji.demojize(text)
            if ':' in demojized:
                # Get the emoji name
                emoji_name = demojized.split(':')[1].replace('_', ' ')
                
                # Add sentiment based on emoji name
                if any(word in emoji_name for word in ['angry', 'rage', 'mad', 'furious', 'pout']):
                    demojized = demojized.replace(':', ' ') + ' angry mad furious upset negative bad terrible '
                elif any(word in emoji_name for word in ['sad', 'cry', 'tear', 'disappoint', 'frown']):
                    demojized = demojized.replace(':', ' ') + ' sad unhappy depressed negative terrible '
                elif any(word in emoji_name for word in ['smile', 'joy', 'heart', 'love', 'happy', 'grin']):
                    demojized = demojized.replace(':', ' ') + ' happy positive good great '
                else:
                    demojized = demojized.replace(':', ' ')
                
                text = demojized
        except:
            pass
            
        return text

    # Text cleaning with emoji support
    def clean_text_with_emoji(text):
        # First convert emojis to text with sentiment
        text_with_emoji_text = emoji_to_text_with_sentiment(text)
        
        # Then do normal cleaning
        text = str(text_with_emoji_text).lower()
        
        # Keep important words but remove special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
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
            
            if remaining > 50:
                st.metric("Remaining", f"{remaining} chars")
                st.caption(f"📝 Words: {word_count}")
                st.success("✅ Good length")
            elif remaining > 20:
                st.metric("Remaining", f"{remaining} chars")
                st.caption(f"📝 Words: {word_count}")
                st.warning("⚠️ Getting long")
            elif remaining >= 0:
                st.metric("Remaining", f"{remaining} chars")
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
                converted = emoji_to_text_with_sentiment(user_text)
                st.write("**After emoji conversion:**")
                st.code(converted)
                
                cleaned = clean_text_with_emoji(user_text)
                st.write("**Final cleaned text (sent to model):**")
                st.code(cleaned)
            
            # Clean and predict
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
   
   
    
    # Show current emoji mapping for debugging
    with st.expander("🔧 Debug: See Emoji Mapping"):
        st.write("**Negative Emojis Mapping:**")
        negative_emojis = {
            '😡': ' angry mad furious frustrated upset terrible bad negative horrible awful ',
            '😢': ' sad tearful unhappy depressed negative ',
            '😞': ' disappointed sad upset unhappy negative ',
            '👎': ' dislike bad terrible poor negative horrible ',
            '❌': ' wrong no bad incorrect negative ',
            '💔': ' heartbroken sad devastated negative terrible awful ',
        }
        
        for emoji_char, mapping in negative_emojis.items():
            st.write(f"{emoji_char} -> {mapping}")
        
        st.write("\n**Test with just 😡 emoji:**")
        test_text = "😡"
        converted = emoji_to_text_with_sentiment(test_text)
        cleaned = clean_text_with_emoji(test_text)
        st.write(f"Original: {test_text}")
        st.write(f"Converted: {converted}")
        st.write(f"Cleaned: {cleaned}")

# For direct execution
if __name__ == "__main__":
    show()
