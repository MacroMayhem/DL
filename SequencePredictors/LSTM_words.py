import os
from keras.preprocessing.text import  Tokenizer
import re

DATA_PATH = 'C:\\Users\\Aditya\\PycharmProjects\\DeepLearning\\data\\PTB'

def sentence_read(filename):
    with open(filename) as f:
        text = f.read()
    return re.split(r' *[\.\?!][\'"\)\]]* *', text)

def load_data():
    # get the data paths
    train_path = os.path.join(DATA_PATH, "ptb.train.txt")
    valid_path = os.path.join(DATA_PATH, "ptb.valid.txt")
    test_path = os.path.join(DATA_PATH, "ptb.test.txt")

    train_sentences =sentence_read(train_path)
    valid_sentences =sentence_read(valid_path)
    test_sentences  =sentence_read(test_path)

    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n'
                          , lower=True, split=' ', char_level=False, oov_token=None)
    tokenizer.fit_on_texts(train_sentences)
    print(tokenizer.word_counts)
    return tokenizer, train_sentences, valid_sentences, test_sentences

tokenizer, train_sentences, valid_sentences, test_senteces = load_data()
print(tokenizer.word_counts)
