import pandas as pd
import nltk
import re
import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
import pickle

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .sentiment-result {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
    }
    .positive {
        background-color: #90EE90;
        color: #006400;
    }
    .negative {
        background-color: #FFB6C1;
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

# Header with emoji and description
st.markdown("# 🎭 Sentiment Analysis Tool")
st.markdown("### Analyze the emotional tone of your text")

# Divider
st.markdown("---")

# Model loading with error handling
@st.cache_resource  # Add caching to prevent reloading
def load_models():
    try:
        model = load_model('models/ANN_FILE.h5')
        with open('models/bow.pkl', 'rb') as file:
            bow = pickle.load(file)
        return model, bow
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, bow = load_models()

def text_preprocessing_bow(text):
    try:
        regex_text = re.sub(r'[^a-zA-Z]', ' ', text)
        regex_text = regex_text.lower()
        vectorized = bow.transform([regex_text])
        return vectorized
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return None

# Input section with placeholder
text = st.text_area(
    'Enter your text for sentiment analysis:',
    placeholder="Type something like 'I love this product!' or 'This service was disappointing...'",
    height=100
)

# Centered button with custom styling
col1, col2, col3 = st.columns([1,1,1])
with col2:
    predict_button = st.button('Analyze Sentiment', use_container_width=True)

if predict_button and text:
    if model is not None and bow is not None:
        # Add a spinner during prediction
        with st.spinner('Analyzing sentiment...'):
            try:
                input_qurie = text_preprocessing_bow(text)
                if input_qurie is not None:
                    prediction = model.predict(input_qurie)
                    
                    # Display result with custom styling
                    if prediction > 0.6:
                        st.markdown("""
                            <div class="sentiment-result positive">
                                ✨ Positive Sentiment! 😊
                            </div>
                        """, unsafe_allow_html=True)
                    elif prediction < 0.4:
                        st.markdown("""
                            <div class="sentiment-result negative">
                                😔 Negative Sentiment
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="sentiment-result" style="background-color: #F0F0F0; color: #666666;">
                                😐 Neutral Sentiment
                            </div>
                        """, unsafe_allow_html=True)

                    # Fix confidence score calculation and display
                    confidence = float(abs(prediction[0][0] - 0.5) * 2)
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Models failed to load. Please check your model files.")

# Footer
st.markdown("---")
st.markdown("### How it works")
st.write("This tool uses Artifical Neural Network to analyze the emotional tone of text. "
         "It can detect whether the sentiment is positive or negative based on the words and phrases used.")

