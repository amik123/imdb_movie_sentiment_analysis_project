import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()

model=load_model('imdb_model_file.h5')
#Step 1: Function to preprocess data
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_text=preprocess_text(review)
    prediction=model.predict(preprocessed_text)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

##Streamlit app

st.title("IMDB Movies Review Sentiment Analysis")
st.write("Enter a movie review to classify it either Positve or negative")

#User input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,prediction=predict_sentiment(user_input)
    st.write(f"The sentiment is:{sentiment}")
    st.write(f"Probability is {prediction}")
