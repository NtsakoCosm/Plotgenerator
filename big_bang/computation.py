import string
from tensorflow.keras.utils import to_categorical
from pandas import read_csv
import numpy as np




frame = read_csv('The Big Bang Theory.csv')
plot = frame['plot']
everything = []
words = []
for i in plot :
    everything.append(f' {i}')
    
for word in everything:
  words_ = word.split()
  words = words +words_


def clean(doc):
  table = str.maketrans('','',string.punctuation)
  tokens = [w.translate(table) for w in doc]
  tokens = [word.lower() for word in tokens]
  return tokens

unique = list(set(words))
length = 10 + 1

lines = []
for i in range(length,len(words)):
  seq = words[i - length:i]
  line = ' '.join(seq)
  lines.append(line)

lines = clean(lines)

def unique_lines():
    return lines, unique