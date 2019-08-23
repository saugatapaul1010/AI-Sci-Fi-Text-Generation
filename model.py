import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding, CuDNNLSTM, Bidirectional


MODEL_DIR = './model'

def save_weights(epoch, model, docname):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}_{}.h5'.format(docname,epoch)))

def load_weights(epoch, model, docname):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}_{}.h5'.format(docname,epoch)))

def build_model(batch_size, seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))

    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.20))

    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.20))

    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.20))

    model.add(TimeDistributed(Dense(vocab_size))) 
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = build_model(32, 128, 75)
    model.summary()
