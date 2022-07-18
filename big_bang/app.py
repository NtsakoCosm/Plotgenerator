import streamlit as st
from computation import unique_lines

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from pandas import read_csv
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_m():
  model = load_model('model',compile=False)
  return model

st.title('The Big Bang Theory  plot Generator.')
st.subheader('By Ntsako Ngwana')
st.subheader('Improvements will be made!')

st.image('https://m.media-amazon.com/images/M/MV5BY2FmZTY5YTktOWRlYy00NmIyLWE0ZmQtZDg2YjlmMzczZDZiXkEyXkFqcGdeQXVyNjg4NzAyOTA@._V1_FMjpg_UX1000_.jpg',width=256)

st.success('Enter a sentence to give the neural network a start, choose how many words to output , and Apply the Ai magic juice')

seq_input = st.text_input('Give the ai a start','Penny does not understand why Sheldon plays games')


lines,unique = unique_lines()

t = Tokenizer(num_words=len(unique))

t.fit_on_texts(lines)
sequences = t.texts_to_sequences(lines)





def generate_text_seq(model, tokenizer, seed_text,n_words=50):
  text = []
  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    
    encoded = pad_sequences([encoded],maxlen = 10, truncating='pre')
    
    predict_x=model.predict(encoded) 
  
    classes_x=np.argmax(predict_x,axis=1)
    
    
    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == classes_x[0]:
        
        break
    seed_text = seed_text + ' ' + word
    text.append(word)
  return ' '.join(text)

n_words = st.number_input('How many words to output?',max_value=500,value=50, min_value=5)
    
if st.button('Ai magic juice ') and len(seq_input) > 0 :
  text = generate_text_seq(load_m(),t,seq_input,n_words=n_words)
  f" : {seq_input} {text}"

if len(seq_input) == 0:
  st.warning('The Neural network needs something to work with dude... :)')



