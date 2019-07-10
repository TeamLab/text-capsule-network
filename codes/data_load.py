import numpy as np
import os
import pickle
import pandas as pd


def checkdirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_text(filename):
    with open(filename, 'r') as f:
        best_epoch = int(f.read())
        return best_epoch


def write_text(text, filename):
    with open(filename, 'w') as f:
        f.write(text)


def load_pickle_data(pickle_dir, dataset):
    assert os.path.exists(pickle_dir)
    pickle_data = pickle.load(open(pickle_dir+dataset, "rb"))
    return pickle_data


def get_idx_from_sent(sent, word_idx_map, max_length):
    x = []
    words = sent.split()[:max_length]
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_length:
        x.append(0)
    return x


def make_idx_data(raw_datas, word_idx_map, len_train, max_length):
    data = []
    for raw_data in raw_datas:
        sent = get_idx_from_sent(raw_data["text"], word_idx_map, max_length)
        sent.append(raw_data["y"])
        data.append(sent)
    split = len_train
    train = np.array(data[:split], dtype="int")
    test = np.array(data[split:], dtype="int")
    return train, test


def preprocessing(data_path, dataset, long_sent=800):
    """

    :param data_path: base directory
    :param dataset: select dataset {'20news', 'mr', 'trec', 'mpqa'}
    :param long_sent: if dataset has long sentences, set to be constant length value
    :return: seq_length, num_classes, vocab_size, x_train, y_train, x_test, y_test, pre-train_word (GloVe 840b),
    word_idx
    """

    assert os.path.exists(data_path) is True

    x = load_pickle_data(data_path, dataset)
    data_frame, pretrain_word, len_train, n_exist_word, vocab, word_idx = x

    max_l = int(np.max(pd.DataFrame(data_frame)["num_words"]))

    if dataset in ["reuters", "20news", "imdb", 'mr']:
        train, test = make_idx_data(data_frame, word_idx, len_train, long_sent)
    else:
        train, test = make_idx_data(data_frame, word_idx, len_train, max_l)

    # train[:, :-1] = word idx
    # train[:, -1] = true label
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = test[:, :-1]
    y_test = test[:, -1]
    sequence_length = len(x_train[0])

    # make one-hot
    labels = sorted(list(set(y_train)))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_train = np.eye(len(label_dict))[y_train]
    num_class = y_train.shape[1]

    y_test = np.eye(len(label_dict))[y_test]
    vocab_size = pretrain_word.shape[0]

    print("sequence length :", sequence_length)
    print("vocab size :", vocab_size)
    print("num classes :", num_class)

    return sequence_length, num_class, vocab_size, x_train, y_train, x_test, y_test, pretrain_word, word_idx
