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
            
            st.caption(f"📊 Text stats: {word_count} words, {chars_used} characters")
            
            st.subheader("Probabilities:")
            prob_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": [f"{p:.1%}" for p in probabilities]
            })
            st.table(prob_df)
            
            st.subheader("Visualization:")
            chart_data = pd.DataFrame({
                "sentiment": model.classes_,
                "probability": probabilities
            }).set_index("sentiment")
            st.bar_chart(chart_data)
    
    # Example texts section
    # with st.expander("📋 Try these examples"):
    #     examples = {
    #         "Positive": "I absolutely love this product! It's amazing! 😍",
    #         "Negative": "Very disappointed with the service today 😠",
    #         "Neutral": "The weather is nice today. Nothing special."
    #     }
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         if st.button("😊 Positive Example", use_container_width=True):
    #             st.session_state.text_input = examples["Positive"]
    #             st.rerun()
        
    #     with col2:
    #         if st.button("😠 Negative Example", use_container_width=True):
    #             st.session_state.text_input = examples["Negative"]
    #             st.rerun()
        
    #     with col3:
    #         if st.button("😐 Neutral Example", use_container_width=True):
    #             st.session_state.text_input = examples["Neutral"]
    #             st.rerun()

if name == "main":
    show()
