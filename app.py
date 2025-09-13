import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load Model
model = load_model('next_word_lstm_100epochs.h5')

model_lstm = load_model('next_word_lstm_100epochs.h5')

model_gru = load_model('next_word_gru_100epochs.h5')

# Load Tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # ensure the sequence length matches max_sequence_len
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## Streamlit App
st.title('Next word prediction with LSTM and EarlyStoping')

# User input
user_input = st.text_area("Enter the sequence of words","To be or not to")

if st.button('Predict Next Word (LSTM)'):
    max_sequence_len = model_lstm.input_shape[1] + 1 # Retrive the max sequence length from the model input shape
    next_word_lstm = predict_next_word(model_lstm,tokenizer,user_input,max_sequence_len)
    st.write(f'Next word (LSTM): {next_word_lstm}')

if st.button('Predict Next Word (GRU)'):
    max_sequence_len = model_gru.input_shape[1] + 1 # Retrive the max sequence length from the model input shape
    next_word_gru = predict_next_word(model_gru,tokenizer,user_input,max_sequence_len)
    st.write(f'Next word (GRU): {next_word_gru}')
