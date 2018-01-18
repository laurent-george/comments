
import os.path
import pandas as pd
import string
import gensim
import logging
import numpy as np
import pickle
import random
import tqdm

def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove upper case
    text = text.lower()
    printable = set(string.printable)
    # remove non ascii character.. (non printable in fact)
    text = ''.join(filter(lambda x: x in printable, text))
    return text

def to_words(text):
    # remove one letter word except I/O/A
    sentence = text.split()

    res = []
    for word in sentence:
        #if len(word) == 1 and word != 'i' and word != 'o' and word != 'a':
        #    continue
        res.append(word)
    return res



def word_set(sentences):
    """

    :param sentences:
    :return: a dict with unique words in all comments
    """
    words = set()
    for sentence in sentences:
        for word in sentence:
            words.add(word)
    return words


def preprocess(data_csv, comment_col='comment_text'):
    df = pd.read_csv(data_csv, na_filter = False)
    df['clean_comment'] = df[comment_col].map(clean_text)
    df['sentence'] = df['clean_comment']
    #df['sentence'] = df['clean_comment'].map(to_words)
    return df

def constructLabeledSentences(data):
    """from https://www.kaggle.com/alyosama/doc2vec-with-keras-0-77/notebook"""
    from gensim.models.doc2vec import LabeledSentence
    from gensim import utils

    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(row, ['Text' + '_%s' % str(index)]))
    return sentences

def sentence_to_list_of_vec(sentence, text_model):
    res = []
    for word in sentence.split():
        try:
            vec = text_model.word_vec(word)
            res.append(vec)
        except Exception as e:
            logging.debug("ignoring word {}".format(word))
    if len(res) == 0:
        res = None
    return res


def create_dataset():


    # ignoring unknown word
    print("start preprocess")
    df = preprocess("~/data/comments/train.csv")


    # create word 2 index

    word_2_index = {}
    index_2_word = {}
    # word at index 0 in the matrix == unknown word
    count = 0
    for sentence in df['sentence']:
        for word in sentence.split():
            if word in word_2_index:
                continue
            word_2_index[word] = count
            index_2_word[count] = word
            count += 1


    # creating embedding matrix

    vector_length_per_word = 300
    matrix = np.zeros((count+1, vector_length_per_word))


    # filling the matrix
    text_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    for word, num in word_2_index.items():
        try:
            matrix[num] = text_model.word_vec(word)
        except KeyError:
            # word that are not in pretrained dict are kept as separate index but assigned value 0
            matrix[num] = np.zeros((1, vector_length_per_word))
    with open('data/embedding.pkl', 'wb') as f:
        pickle.dump({'matrix': matrix, 'word_2_index': word_2_index, 'index_2_word': index_2_word}, f)

    # longuest sentence = 1396 words  # max( result['sentence_in_vectors'].map(lambda x: len(x)))


    # word 2 index in sentence
    #df['train_vector'] = []
    # bon faire un truc super simple ou revoir ce site:     https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # quel format, le plus simple je pense


    # TODO: idee ajouter une feature en plus genre: nombre de lettre en majuscule/nombre de lettre ? et/ou nombre de point d'exclamation etc
    # enfin avoir une sorte de word2vec pour les mots non reconnu dans le dict ca peut valoir le coup aussi (en gros ne pas perdre d'information)

    # puis regarder ça: https://stats.stackexchange.com/questions/202544/handling-unknown-words-in-language-modeling-tasks-using-lstm
    # https://arxiv.org/abs/1508.02096

    #toxic severe_toxic obscene threat insult identity_hate


def get_y(input_csv):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    df = pd.read_csv(input_csv)

    labels = {}
    for index, row in df.iterrows():
        vec = np.zeros((len(class_names)))
        for class_num, class_name in enumerate(class_names):
            vec[class_name] = row[class_names]

        labels[row['id']] = vec

    return labels, class_names


def get_x(input_csv, embeding_file='data/embedding.pkl', max_lenght=100, comment_col='comment_text'):
    with open(embeding_file, 'rb') as f:
        saved_embedding = pickle.load(f)

    word_index = saved_embedding['word_2_index']
    index_2_word = saved_embedding['index_2_word']
    matrix = saved_embedding['matrix']

    df = preprocess(input_csv, comment_col=comment_col)

    #x = np.zeros((len(df), max_lenght, matrix.shape[1]))
    x = {}

    for index, row in df.iterrows():
        for num, word in enumerate(row['sentence'].split()):
            vec = np.zeros((max_lenght, matrix.shape[1]))
            if num >= max_lenght:
                break
            if word in word_index.keys():
                word_idx = word_index[word]
                vec[num] = matrix[word_idx]
            x[row['id']] = vec

    return x

from collections import namedtuple



def generate_x_y(input_csv,
                  embeding_file='data/embedding.pkl',
                  max_lenght=100):

    Entry = namedtuple('comments_entry', ['id', 'X', 'Y', 'raw_sentence', 'clean_sentence', 'vec_sentence'])
    with open(embeding_file, 'rb') as f:
        saved_embedding = pickle.load(f)

    word_index = saved_embedding['word_2_index']
    index_2_word = saved_embedding['index_2_word']
    matrix = saved_embedding['matrix']
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    df = pd.read_csv(input_csv, na_filter = False)
    for idx, row in df.iterrows():
        Y = None
        try:
            Y = np.array(row[class_names].values, dtype=np.float32)
        except Exception as e:
            pass

        text = clean_text(row['comment_text'])
        X = np.zeros((max_lenght, matrix.shape[1]), dtype=np.float32)
        vec_sentence = []
        for num, word in enumerate(text.split()):
            if num >= max_lenght:
                break
            if word in word_index.keys():
                word_idx = word_index[word]
                X[num] = matrix[word_idx]
                vec_sentence.append(word)

        yield Entry(id=row['id'], X=X, Y=Y, raw_sentence=row['comment_text'], clean_sentence=text,
                    vec_sentence=vec_sentence)



import tensorflow as tf

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def generate_train_tfrecord():
    # not using tfrecord for now
    validation_percentage = 0.2
    train_writer = tf.python_io.TFRecordWriter('data/train.tfrecord')
    validation_writer = tf.python_io.TFRecordWriter('data/validation.tfrecord')

    g = generate_x_y('~/data/comments/train.csv')
    for entry in g:

        example = tf.train.Example(features=tf.train.Features(
        feature={
            'length': _int64_feature([len(entry.Y)]),
            'id': _int64_feature([1]),
            'X': _float_feature(entry.X.flatten()),
            'Y': _float_feature(entry.Y.flatten())}))

        print("example is ok")



        if random.random() > validation_percentage:
            print("Saving to train")
            train_writer.write(example.SerializeToString())
        else:
            print("Saving to validation")
            validation_writer.write(example.SerializeToString())







# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# BOn je ne m'en sort pas suivre ce tuto ici et refaire la meme chose mais avec notre dataset

if __name__ == "__main__":

    #test_file = '~/data/comments/test.csv'
    #x_test = get_x(test_file)  # -> fail car trop gros d'allouer un tableau numpy avec autant d'entree

    embedding_data = 'data/embedding.pkl'
    if not(os.path.isfile(embedding_data)):
        create_dataset(embedding_data)


    generate_train_tfrecord()

#  labels = get_labels('~/data/comments/train.csv')
#  print(labels)
   # train_file = '~/data/comments/train.csv'
   # x_train = get_x(train_file)
   # y_train, y_names = get_y(train_file)


   # print("Creating test data")

   # with open('data/train.pkl', 'wb') as f:
   #     pickle.dump({'x_train': x_train,
   #                  'y_train': y_train,
   #                  'label_names': y_names
   #                  })






