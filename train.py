import os
import json
import argparse
import numpy as np
from model import build_model, save_weights
from datetime import datetime as dt


DATA_DIR = './data'
LOG_DIR = './logs'

BATCH_SIZE = 32
SEQ_LENGTH = 128

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

def read_batches(T, vocab_size):
    length = T.shape[0]; #1,493,263
    batch_chars = int(length / BATCH_SIZE); # 46,664

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # (0, 46664, 128)
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 32X128
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) # 32X128X75
        for batch_idx in range(0, BATCH_SIZE): # (0,32)
            for i in range(0, SEQ_LENGTH): #(0,128)
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

def train(text, epochs=100, save_freq=5):

    text = text[0:len(text)//100]
    # character to index and vice-versa mappings
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of unique characters: " + str(len(char_to_idx))) #75

    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    #model_architecture
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #Train data generation
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) #Convert complete text into numerical indices

    print("Length of text:" + str(T.size)) #149,326,361

    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        st = dt.now()
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            #print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))

        print("Time taken to run epoch {} = {}".format(epoch+1,dt.now()-st))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='internet_archive_scifi_v3.txt', help='Name of the text file to train from.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train for.')
    parser.add_argument('--freq', type=int, default=5, help='Checkpoint save frequency.')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs, args.freq)
