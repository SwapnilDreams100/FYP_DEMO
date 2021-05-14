import copy
from keras.preprocessing.sequence import pad_sequences

def vectorize(text, tokenizer, pad='pre'):
  texts = copy.deepcopy(text_array)
  texts = tokenizer.texts_to_sequences(texts)
  padded_seq = pad_sequences(texts, maxlen = 500, padding = pad, truncating = pad)
  return padded_seq
