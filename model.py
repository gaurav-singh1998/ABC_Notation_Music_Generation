import os
from keras.models import Sequential, load_model
from keras.layers import Activation, Embedding, Dense, TimeDistributed, LSTM, Dropout

MODEL_DIR = './model'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'epoch.{}.h5'.format(epoch)))

def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'epoch.{}.h5'.format(epoch)))

def build_model(batch_size, seq_length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_length)))
    for i in range(3):
        model.add(LSTM(126, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = build_model(16, 64, 8104)
    model.summary()
