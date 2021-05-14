import nltk
nltk.download('punkt')
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def create_embeddings(data_file = "/content/FYP_DEMO/training_set_rel3.tsv",glove_name= "glove.6B.300d.txt"):
  MAX_NB_WORDS=4000
  EMBEDDING_DIM= 300
  MAX_SEQUENCE_LENGTH = 500
  
  texts=[]
  labels=[]
  sentences=[]
  
  fp1=open(glove_name,"r", encoding="utf-8")
  glove_emb={}
  for line in fp1:
    temp=line.split(" ")
    try:
      glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])
    except Exception as e:
      pass

#   fp=open(data_file,'r', encoding="ascii", errors="ignore")
#   fp.readline()
#   originals = []
#   for line in fp:
#       temp=line.split("\t")
#       if(temp[1]==essay_type):
#           originals.append(float(temp[6]))
#   fp.close()
  range_min = 0
  range_max = 30
  fp=open(data_file,'r', encoding="ascii", errors="ignore")
  fp.readline()
  for line in fp:
      temp=line.split("\t")
      if(temp[1]=='7'): 
          texts.append(temp[2])
          labels.append((float(temp[6])-range_min)/(range_max-range_min))
          line=temp[2].strip()
          sentences.append(nltk.tokenize.word_tokenize(line))

  fp.close()
  
  for i in sentences:
    temp1=np.zeros((1, EMBEDDING_DIM))
    for w in i:
      if(w in glove_emb):
        temp1+=glove_emb[w]
    temp1/=len(i)
 
  tokenizer=Tokenizer(nb_words = MAX_NB_WORDS) #num_words=MAX_NB_WORDS) #limits vocabulory size
  tokenizer.fit_on_texts(texts) #encoding the text
  sequences=tokenizer.texts_to_sequences(texts) #returns list of sequences
  word_index=tokenizer.word_index #dictionary mapping, word and specific token for that word...
  print('Found %s unique tokens.' % len(word_index))

  data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #padding to max_length
  
  embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
  for word,i in word_index.items():
    if(i>=len(word_index)):
      continue
    if word in glove_emb:
        embedding_matrix[i]=glove_emb[word]
    
  return data, labels, tokenizer, embedding_matrix

def get_vocab_size(tokenizer):
  return len(tokenizer.word_index)

def create_train_val_set(data, labels):
  VALIDATION_SPLIT=0.20
  
  np.random.seed(0)
  indices=np.arange(data.shape[0])
  np.random.shuffle(indices)
  data=data[indices]
  labels=np.asarray(labels)
  labels=labels[indices]
  validation_size=int(VALIDATION_SPLIT*data.shape[0])

  x_train=data[:-validation_size]
  y_train=labels[:-validation_size]
  x_val=data[-validation_size:]
  y_val=labels[-validation_size:]
  
  return x_train, x_val, y_train, y_val
