import keras.layers as klayers 
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, Embedding, GlobalAveragePooling1D, Concatenate, Activation, Lambda, BatchNormalization, Convolution1D, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import cohen_kappa_score
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers
import pickle
from scipy import stats
  
class Neural_Tensor_layer(Layer):
  def __init__(self,output_dim,input_dim=None, **kwargs):
    self.output_dim=output_dim
    self.input_dim=input_dim
    if self.input_dim:
      kwargs['input_shape']=(self.input_dim,)
    # 		print("YAYY", input_dim, output_dim)
    super(Neural_Tensor_layer,self).__init__(**kwargs)

  def call(self,inputs,mask=None):
    e1=inputs[0]
    e2=inputs[1]
    batch_size=K.shape(e1)[0]
    k=self.output_dim

    feed_forward=K.dot(K.concatenate([e1,e2]),self.V)

    bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]

    for i in range(k)[1:]:	
      btp=K.sum((e2*K.dot(e1,self.W[i]))+self.b,axis=1)
      bilinear_tensor_products.append(btp)

    result=K.tanh(K.reshape(K.concatenate(bilinear_tensor_products,axis=0),(batch_size,k))+feed_forward)

    return result

  def build(self,input_shape):
    mean=0.0
    std=1.0
    k=self.output_dim
    d=self.input_dim
    ##truncnorm generate continuous random numbers in given range
    W_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
    V_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
    self.W=K.variable(W_val)
    self.V=K.variable(V_val)
    self.b=K.zeros((self.input_dim,))
    self.trainable_weights.append([self.W,self.V,self.b])

  def compute_output_shape(self, input_shape):
    batch_size=input_shape[0][0]
    return(batch_size,self.output_dim)
  

class Temporal_Mean_Pooling(Layer): # conversion from (samples,timesteps,features) to (samples,features)
  def __init__(self, **kwargs):
    super(Temporal_Mean_Pooling,self).__init__(**kwargs)
    # masked values in x (number_of_samples,time)
    self.supports_masking=True
    # Specifies number of dimensions to each layer
    self.input_spec=InputSpec(ndim=3)

  def call(self,x,mask=None):
    if mask is None:
      mask=K.mean(K.ones_like(x),axis=-1)

    mask=K.cast(mask,K.floatx())
        #dimension size single vec/number of samples
    return K.sum(x,axis=-2)/K.sum(mask,axis=-1,keepdims=True)        

  def compute_mask(self,input,mask):
    return None

  def compute_output_shape(self,input_shape):
    return (input_shape[0],input_shape[2])

def SKIPFLOW(lstm_dim=50, lr=1e-4, lr_decay=1e-6, k=4, eta=3, delta=50, activation="relu", maxlen=500, seed=None, embedding_matrix = None, vocab_size = None):
    EMBEDDING_DIM=300
    e = Input(name='essay',shape=(maxlen,))
    embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
              input_length=maxlen,
              mask_zero=True,
              trainable=False)
    side_embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
                  input_length=maxlen,
                  mask_zero=False,
                  trainable=False)

    embed = embedding_layer(e)
    lstm_layer=LSTM(lstm_dim,return_sequences=True)
    hidden_states=lstm_layer(embed)
    htm=Temporal_Mean_Pooling()(hidden_states)    
    side_embed = side_embedding_layer(e)
    side_hidden_states=lstm_layer(side_embed)    
    tensor_layer=Neural_Tensor_layer(output_dim=k,input_dim=500)
    pairs = [((eta + i * delta) % maxlen, (eta + i * delta + delta) % maxlen) for i in range(maxlen // delta)]
    hidden_pairs = [ (Lambda(lambda t: t[:, p[0], :])(side_hidden_states), Lambda(lambda t: t[:, p[1], :])(side_hidden_states)) for p in pairs]
    sigmoid = Dense(1, activation="sigmoid", kernel_initializer=initializers.glorot_normal(seed=seed))
    coherence = [sigmoid(tensor_layer([hp[0], hp[1]])) for hp in hidden_pairs]
    co_tm=Concatenate()(coherence[:]+[htm])
    dense = Dense(256, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(co_tm)
    dense = Dense(128, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    dense = Dense(64, activation=activation,kernel_initializer=initializers.glorot_normal(seed=seed))(dense)
    out = Dense(1, activation="sigmoid")(dense)
    model = Model(inputs=[e], outputs=[out])
    adam = Adam(lr=lr, decay=lr_decay)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["MSE"])
    print('MODEL READY')
    return model

def load_model(vocab_size, embedding_matrix):
  earlystopping = EarlyStopping(monitor="val_mean_squared_error", patience=5)
  MAX_SEQUENCE_LENGTH=500
  sf = SKIPFLOW(lstm_dim=500, lr=2e-4, lr_decay=2e-6, k=4, eta=13, delta=50, activation="relu", maxlen = MAX_SEQUENCE_LENGTH, seed=None, embedding_matrix = embedding_matrix, vocab_size = vocab_size)
  pklfile= "/content/drive/MyDrive/sf_models/7_weights.pkl"
  fpkl= open(pklfile, 'rb')
  sf.set_weights(pickle.load(fpkl))
  print('Weights Loaded')
  fpkl.close()
  return sf
