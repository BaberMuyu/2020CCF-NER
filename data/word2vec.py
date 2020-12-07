import pickle
from global_config import *

ROOT_LOCAL_DATA = ROOT_DATA + 'w2v/'


def conv2pkl():
    raw_file = ROOT_LOCAL_DATA + 'sgns.merge.word'
    vocab_file = ROOT_LOCAL_DATA + 'w2v_vocab.pkl'
    vec_file = ROOT_LOCAL_DATA + 'w2v_vector.pkl'
    raw_fp = open(raw_file, 'r', encoding='utf-8')
    vocab_fp = open(vocab_file, 'wb+')
    vec_fp = open(vec_file, 'wb+')
    raw_data = raw_fp.readlines()[1:]
    vocab_list = {'PAD': 0}
    vec_list = [[0.0] * 300]
    for index, d in enumerate(raw_data):
        d = d.split()
        vocab_list[d[0]] = index + 1
        vec = [float(s) for s in d[1:]]
        vec_list.append(vec)

    pickle.dump(vocab_list, vocab_fp)
    pickle.dump(vec_list, vec_fp)


def get_w2v_vocab():
    return pickle.load(open(ROOT_LOCAL_DATA + 'w2v_vocab.pkl', 'rb'))


def get_w2v_vector():
    return pickle.load(open(ROOT_LOCAL_DATA + 'w2v_vector.pkl', 'rb'))


if __name__ == '__main__':
    conv2pkl()


