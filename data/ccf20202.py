from torch.utils.data import Dataset
from util.mytokenizer import MyBertTokenizer
import csv
import os
import pkuseg
import pandas
import copy
import random
import numpy
import re

from global_config import ROOT_DATA
from data.word2vec import get_w2v_vocab
from util.tool import *

ROOT_LOCAL_DATA = ROOT_DATA + 'ccf2020/'


class DataConfig(object):
    def __init__(self):
        self.max_len = 150
        self.num_types = 14
        self.tokenizer = 'hfl/chinese-roberta-wwm-ext'

    def __str__(self):
        string = ""
        string += "max_len: {}\n".format(self.max_len)
        string += "num_types: {}\n".format(self.num_types)
        return string


class CCF2020DataSet(Dataset):
    def __init__(self, data, test_mode):
        self.data = data
        if test_mode:
            self.len = 100 if len(self.data) > 100 else len(self.data)
        else:
            self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


class CCF2020DataSetFactory(object):
    def __init__(self, config):
        self.config = config
        self.train_size = 7596
        self.train_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_train_data.json')
        self.test_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_test_data.json')
        self.train_tree = build_tree(self.train_data, 'id')
        if os.path.exists(ROOT_LOCAL_DATA + 'token_seg_train_data_loss_mask.json'):
            self.train_data_with_mask = json_load_by_line(ROOT_LOCAL_DATA + 'token_seg_train_data_loss_mask.json')
            self.train_data_with_mask_tree = build_tree(self.train_data_with_mask, 'id')
        else:
            self.train_data_with_mask = None
            self.train_data_with_mask_tree = None

        if self.config.reshuffle or not os.path.exists(ROOT_LOCAL_DATA + 'train_map.json'):
            self.train_map = [i for i in range(self.train_size)]
            random.shuffle(self.train_map)
            json_dump(self.train_map, ROOT_LOCAL_DATA + 'train_map.json')
        else:
            self.train_map = json_load(ROOT_LOCAL_DATA + 'train_map.json')

        self.part_size = int(len(self.train_map) / self.config.fold_num)
        self.train_size = int(len(self.train_map) * (1 - self.config.dev_rate))

    def __call__(self, inputs):
        type_data = inputs['type_data']
        fold_index = inputs['fold_index']
        if type_data in ('train', 'train_loss_mask'):
            data_tree = self.train_tree if type_data == 'train' else self.train_data_with_mask_tree
            if self.config.en_cross:
                data_map = self.train_map[:fold_index * self.part_size]
                if fold_index + 1 < self.config.fold_num:
                    data_map += self.train_map[(fold_index + 1) * self.part_size:]
            else:
                data_map = self.train_map[:self.train_size]
            data = []
            for dindex in data_map:
                data.extend(data_tree[dindex])
        elif type_data in ('dev', 'dev_loss_mask'):
            data_tree = self.train_tree if type_data == 'dev' else self.train_data_with_mask_tree
            if self.config.en_cross:
                if fold_index + 1 < self.config.fold_num:
                    data_map = self.train_map[fold_index * self.part_size: (fold_index + 1) * self.part_size]
                else:
                    data_map = self.train_map[fold_index * self.part_size:]
            else:
                data_map = self.train_map[self.train_size:]
            data = []
            for dindex in data_map:
                data.extend(data_tree[dindex])
        elif type_data == 'test':
            data = self.test_data
        else:
            assert 0
        return CCF2020DataSet(data, self.config.test_mode)


class CCF2020DataProcess(object):
    def __init__(self, config):
        self.config = config
        self.token_max_len = self.config.max_len
        self.max_len = self.config.max_len - 2
        self.thu_lac = None
        self.pku_seg = None
        self.tokenizer = None
        self.word_w2v = None
        self.train_file = 'train_data.json'
        self.test_file = 'test_data.json'
        self.type_list = ['position', 'name', 'movie', 'organization', 'company', 'game', 'book', 'address', 'scene',
                          'government', 'email', 'mobile', 'QQ', 'vx']
        assert self.config.num_types == len(self.type_list)
        self.type2id = dict((pred, i) for i, pred in enumerate(self.type_list))
        self.id2type = dict((i, pred) for i, pred in enumerate(self.type_list))

    def format_data(self, data_type='train', num_works=1):
        if self.thu_lac is None or self.pku_seg is None or self.tokenizer is None or self.word_w2v is None:
            self.pku_seg = pkuseg.pkuseg()
            self.tokenizer = MyBertTokenizer.from_pretrained(self.config.tokenizer)
            self.word_w2v = get_w2v_vocab()
            self.word_w2v = dict([(word, index) for index, word in enumerate(self.word_w2v)])

        clue_data = json_load_by_line(ROOT_DATA + 'clue/train_dev_test_data.json')
        len_map = [[] for _ in range(55)]
        for dd in clue_data:
            len_map[len(dd['text'])].append(dd)

        def split_text(sample):

            def split_by_len(sample_len):
                if len(sample_len['text']) <= self.max_len:
                    return [sample_len]
                out_samples = []
                right_limit = 0
                rest_text = sample_len['text']
                while len(rest_text) > self.max_len:
                    new_sample = copy.deepcopy(sample_len)
                    new_sample['entities'] = []
                    for char_index in range(self.max_len - 1, -1, -1):
                        if (rest_text[char_index] in ('，', '。', '!', '?')) or char_index == 0:
                            if char_index == 0:
                                char_index = self.max_len - 1
                            left_limit = right_limit
                            right_limit += char_index + 1
                            new_sample['text'] = rest_text[:char_index + 1]
                            new_sample['sub_id'] = sample_len['sub_id'] + left_limit

                            for entity in sample_len['entities']:
                                if entity['pos_b'] >= left_limit and entity['pos_e'] < right_limit:
                                    new_entity = copy.deepcopy(entity)
                                    new_entity['pos_b'] = entity['pos_b'] - left_limit
                                    new_entity['pos_e'] = entity['pos_e'] - left_limit
                                    new_sample['entities'].append(new_entity)

                            rest_text = rest_text[char_index + 1:]
                            out_samples.append(new_sample)
                            break
                else:
                    left_limit = right_limit
                    new_sample = copy.deepcopy(sample_len)
                    new_sample['text'] = rest_text
                    new_sample['entities'] = []
                    new_sample['sub_id'] = sample_len['sub_id'] + left_limit
                    for entity in sample_len['entities']:
                        if entity['pos_b'] >= left_limit:
                            new_entity = copy.deepcopy(entity)
                            new_entity['pos_b'] = entity['pos_b'] - left_limit
                            new_entity['pos_e'] = entity['pos_e'] - left_limit
                            new_sample['entities'].append(new_entity)
                    out_samples.append(new_sample)
                return out_samples

            def split_by_clue(sample_clue):
                out_samples = []

                def split_(sample_):
                    nonlocal out_samples
                    a = len_map[:len(sample_['text']) + 1]
                    a.reverse()
                    for a_len_map in a:
                        for clue_d in a_len_map:
                            start_index = sample_['text'].find(clue_d['text'])
                            if start_index == -1 and ',' in clue_d['text']:
                                if 'text_2' not in clue_d.keys():
                                    clue_d['text_2'] = re.sub(r',', '，', clue_d['text'])
                                start_index = sample_['text'].find(clue_d['text_2'])
                            if start_index != -1:
                                end_index = start_index + len(clue_d['text']) - 1
                                new_sample = {'id': sample_['id'], 'sub_id': sample_['sub_id'] + start_index,
                                              'clue_id': clue_d['id'],
                                              'text': sample_['text'][start_index:end_index + 1], 'entities': []}
                                for e in sample_['entities']:
                                    if e['pos_b'] >= start_index and e['pos_e'] <= end_index:
                                        new_e = copy.deepcopy(e)
                                        new_e['pos_b'] = e['pos_b'] - start_index
                                        new_e['pos_e'] = e['pos_e'] - start_index
                                        new_sample['entities'].append(new_e)
                                out_samples.append(new_sample)
                                if start_index > 0:
                                    rest_sample_0 = {'id': sample_['id'], 'sub_id': sample_['sub_id'],
                                                     'text': sample_['text'][:start_index], 'entities': []}
                                    for e in sample_['entities']:
                                        if e['pos_e'] < start_index:
                                            new_e = copy.deepcopy(e)
                                            rest_sample_0['entities'].append(new_e)
                                    split_(rest_sample_0)
                                if end_index + 1 < len(sample_['text']):
                                    rest_sample_1 = {'id': sample_['id'], 'sub_id': sample_['sub_id'] + end_index + 1,
                                                     'text': sample_['text'][end_index + 1:], 'entities': []}
                                    for e in sample_['entities']:
                                        if e['pos_b'] > end_index:
                                            new_e = copy.deepcopy(e)
                                            new_e['pos_b'] = e['pos_b'] - end_index - 1
                                            new_e['pos_e'] = e['pos_e'] - end_index - 1
                                            rest_sample_1['entities'].append(new_e)
                                    split_(rest_sample_1)
                                return

                    new_sample = {'id': sample_['id'], 'sub_id': sample_['sub_id'], 'clue_id': -1,
                                  'text': sample_['text'], 'entities': copy.deepcopy(sample_['entities'])}
                    out_samples.append(new_sample)

                sample_clue['sub_id'] = 0
                split_(sample_clue)
                return out_samples

            new_samples = split_by_clue(sample)
            new_samples_2 = []
            for ppp in new_samples:
                new_samples_2.extend(split_by_len(ppp))
            new_samples_2.sort(key=lambda x: x['sub_id'])
            for index, ppp in enumerate(new_samples_2):
                ppp['sub_id'] = index
            return new_samples_2

        def get_lattice_word(sample):
            def is_all_chinese(word_str):
                for c in word_str:
                    if not '\u4e00' <= c <= '\u9fa5':
                        return False
                return True

            def pkuseg_cut(text):
                index = 0
                word_list = []
                for word in self.pku_seg.cut(text):
                    word_len = len(word)
                    if word_len > 1 and is_all_chinese(word):
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            cut_func = [pkuseg_cut]

            lattice_word = set()
            for func in cut_func:
                words = func(sample['text'])
                lattice_word |= set(words)
            lattice_word = [w for w in lattice_word]
            lattice_word.sort(key=lambda x: len(x[0]))
            lattice_word.sort(key=lambda x: x[2])
            sample['lattice'] = lattice_word
            return sample

        def convert_to_token(sample):
            text = self.tokenizer.my_encode(sample['text'], max_length=self.token_max_len, add_special_tokens=True,
                                            truncation=True)
            text_map, raw2decode = self.tokenizer.get_token_map(sample['text'])
            lattice = []
            for lword in sample['lattice']:
                if lword[0] in self.word_w2v:
                    lword_index = self.word_w2v[lword[0]]
                    lattice.append([raw2decode[lword[1]], raw2decode[lword[2]], lword_index])

            entities = []
            if 'entities' in sample.keys():
                for entity in sample['entities']:
                    entities.append({"category": self.type2id[entity['category']],
                                     "pos_b": raw2decode[entity['pos_b']],
                                     "pos_e": raw2decode[entity['pos_e']]})
            mask_seq = None
            if 'loss_mask' in sample.keys():
                mask_seq = numpy.ones(len(text))
                for e in sample['loss_mask']:
                    e_s = raw2decode[e[0]]
                    e_e = raw2decode[e[1]]
                    mask_seq[e_s:e_e + 1] = 0
                mask_seq = mask_seq.tolist()
            return {'id': sample['id'], 'sub_id': sample['sub_id'], 'text': text, 'entities': entities,
                    'lattice': lattice, 'loss_mask': mask_seq}, \
                   {'id': sample['id'], 'sub_id': sample['sub_id'], 'text': sample['text'], 'text_map': text_map}

        def main_process(data, results):
            for i, d in enumerate(data):
                print(i, len(data))
                samples = split_text(d)
                for sample in samples:
                    sample = get_lattice_word(sample)
                    text_token, text_map = convert_to_token(sample)
                    results['token'].append(text_token)
                    results['map'].append(text_map)
                results['seg'].extend(samples)

        def add_loss_mask(data, results):
            for i, d in enumerate(data):
                print(i, len(data))
                text_token, text_map = convert_to_token(d)
                results['token'].append(text_token)

        if data_type in ('train', 'test'):
            in_file = self.train_file if data_type == 'train' else self.test_file
            run_func = main_process
            out_data = {'seg': [],
                        'token': [],
                        'map': []}
        elif data_type == 'train_loss_mask':
            run_func = add_loss_mask
            in_file = 'seg_train_data_loss_mask.json'
            out_data = {'token': []}
        else:
            print('format_data', data_type)
            return None

        raw_data = json_load_by_line(ROOT_LOCAL_DATA + in_file)
        if num_works > 1:
            simple_multiprocess(run_func, raw_data, out_data, num_works)
        else:
            run_func(raw_data, out_data)

        if data_type == 'train':
            clue_train_data = clue_data[:12082]
            ccf_data = out_data['seg']
            clue_out_ccf_data = []
            ccf_in_clue_data = []
            for dd in ccf_data:
                if dd['clue_id'] < 12082:
                    ccf_in_clue_data.append(dd['clue_id'])
            index = ccf_data[-1]['id'] + 1
            for dd in clue_train_data:
                if dd['id'] not in ccf_in_clue_data:
                    new_d = {'id': index, 'sub_id': 0, 'clue_id': dd['id'],
                             'text': dd['text'], 'entities': dd['entities']}
                    index += 1
                    clue_out_ccf_data.append(new_d)
            out_data2 = {'seg': [],
                         'token': [],
                         'map': []}
            if num_works > 1:
                simple_multiprocess(main_process, clue_out_ccf_data, out_data2, num_works)
            else:
                main_process(clue_out_ccf_data, out_data2)
            for key in out_data.keys():
                out_data[key].extend(out_data2[key])

        for key in out_data.keys():
            json_dump_by_line(out_data[key], ROOT_LOCAL_DATA + '{}_'.format(key) + in_file)

    def generate_results(self, pred, data_type, output_file):
        data_set = {"train": "map_train_data.json",
                    'dev': 'map_train_data.json',
                    'test': 'map_test_data.json'}
        token_data_set = {"train": "token_train_data.json",
                          'dev': 'token_train_data.json',
                          'test': 'token_test_data.json'}
        seg_data_set = {"train": "seg_train_data.json",
                        "dev": "seg_train_data.json",
                        "test": "seg_test_data.json"}
        data = json_load_by_line(ROOT_LOCAL_DATA + data_set[data_type])
        id_tree = build_tree(data, 'id')

        token_data = json_load_by_line(ROOT_LOCAL_DATA + token_data_set[data_type])
        seg_data = json_load_by_line(ROOT_LOCAL_DATA + seg_data_set[data_type])
        id_tree_token = build_tree(token_data, 'id')
        id_tree_seg = build_tree(seg_data, 'id')
        result = []
        for p in pred:
            d = id_tree[p['id']][p['sub_id']]
            td = id_tree_token[p['id']][p['sub_id']]
            sd = id_tree_seg[p['id']][p['sub_id']]
            sample = {'id': d['id'], 'sub_id': d['sub_id'], 'clue_id': sd['clue_id'],
                      'text': d['text'], 'entities': []}
            for entity in p['entities']:
                pos_b = d['text_map'][entity[1]]
                if entity[2] + 1 == len(d['text_map']):
                    privacy = d['text'][pos_b: d['text_map'][entity[2]] + 1].strip()
                else:
                    privacy = d['text'][pos_b: d['text_map'][entity[2] + 1]].strip()
                raw_entity = {
                    'privacy': privacy,
                    'category': self.id2type[entity[0]],
                    'pos_b': pos_b,
                    'pos_e': pos_b + len(privacy) - 1}
                sample['entities'].append(raw_entity)
                token_entity = {'category': entity[0], 'pos_b': entity[1], 'pos_e': entity[2]}
                if raw_entity not in sd['entities'] and token_entity in td['entities']:
                    pass
                    # print(token_entity)
                    # print(raw_entity)
                    # print(sd['text'])
                    # print(sd['entities'])

            result.append(sample)
        json_dump_by_line(result, output_file)

    @staticmethod
    def combine_by_vote(file_list, file_out, min_num=None):
        min_map = {'position': 5, 'name': 4, 'movie': 4, 'organization': 5,
                   'company': 5, 'game': 5, 'book': 4, 'address': 5,
                   'scene': 5, 'government': 5, 'email': 5, 'mobile': 5,
                   'QQ': 5, 'vx': 5}
        data_list = [json_load_by_line(f) for f in file_list]
        data = []
        for sample_index in range(len(data_list[0])):
            new_sample = copy.deepcopy(data_list[0][sample_index])
            new_sample['entities'] = []
            if 'emission' in new_sample:
                del new_sample['emission']
            entities = dict()
            for data_index in range(len(data_list)):
                for e in data_list[data_index][sample_index]['entities']:
                    ekey = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
                    if ekey not in entities:
                        entities[ekey] = [e, 0]
                    entities[ekey][1] += 1
            for ekey in entities.keys():
                if min_num is None:
                    min_num = min_map[entities[ekey][0]['category']]
                if entities[ekey][1] >= min_num:
                    if entities[ekey][0] not in new_sample['entities']:
                        new_sample['entities'].append(entities[ekey][0])
                    else:
                        continue
            data.append(new_sample)
        json_dump_by_line(data, file_out)


# ==================================================================================
def get_qq_strict(text):
    pattern = re.compile(r"q{1,2}[^0-9.]{,4}[0-9]{5,12}", re.IGNORECASE)
    h = pattern.finditer(text)
    qq = []
    for i in h:
        qq_num = re.search(r"[0-9]{5,12}$", i.group()).group()
        if len(qq_num) >= 12 or text[i.span()[1]] == '@':
            continue
        qq.append({'privacy': qq_num,
                   'category': 'QQ',
                   'pos_b': i.span()[1] - len(qq_num),
                   'pos_e': i.span()[1] - 1})
    return qq


def get_vx(text):
    pattern = re.compile(r"微信[^0-9.]{,6}[0-9a-pr-z_-]{4,20}", re.IGNORECASE)
    h = pattern.finditer(text)
    qq = []
    for i in h:
        qq_num = re.search(r"[0-9a-pr-z_-]{4,20}$", i.group(), re.IGNORECASE).group()
        qq.append({'privacy': qq_num,
                   'category': 'vx',
                   'pos_b': i.span()[1] - len(qq_num),
                   'pos_e': i.span()[1] - 1})
    return qq


def convert2json(text_dir, output_file, label_dir=None):
    text_file_list = os.listdir(text_dir)
    data = []

    for text_file in text_file_list:
        d = {'id': None, 'text': None, 'entities': []}
        with open(text_dir + text_file, 'r', encoding='utf-8') as fp:
            d['text'] = fp.read()
            d['id'] = int(text_file[:-4])
        data.append(d)

    if label_dir is not None:
        for d in data:
            label_file = label_dir + "{}.csv".format(d['id'])
            reader = csv.reader(open(label_file, "r", encoding='utf-8'))
            for item in reader:
                # 忽略第一行
                if reader.line_num == 1:
                    continue
                if item[4] != d['text'][int(item[2]):int(item[3]) + 1]:
                    print('entity error', d, item)
                entity = {'privacy': item[4],
                          'category': item[1],
                          'pos_b': int(item[2]),
                          'pos_e': int(item[3])}
                if not d['id'] == int(item[0]):
                    print(d['id'], int(item[0]))
                d['entities'].append(entity)

    data.sort(key=lambda x: x['id'])

    json_dump_by_line(data, output_file)


def convert2csv(infile, outfile):
    data = json_load_by_line(infile)
    csv_data = []
    base = 0
    for d in data:
        if d['sub_id'] == 0:
            base = 0
        for e in d['entities']:
            if '\n' in e['privacy']:
                print(d)
            pos_b = base + e['pos_b']
            pos_e = pos_b + len(e['privacy']) - 1
            csv_data.append([d['id'], e['category'], pos_b, pos_e, str(e['privacy'])])

        base += len(d['text'])

    csv_data.sort(key=lambda x: x[3])
    csv_data.sort(key=lambda x: x[2])
    csv_data.sort(key=lambda x: x[0])
    csv_data = pandas.DataFrame(csv_data, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    csv_data.to_csv(outfile, index=False)


def analyze_dev_data(pred_file, target_file, verbose=False):
    """
        没有去重
    """

    def split_by_category(entities):
        result = dict((key, []) for key in
                      ['position', 'name', 'movie', 'organization', 'company', 'game', 'book', 'address', 'scene',
                       'government', 'email', 'mobile', 'QQ', 'vx'])
        for ee in entities:
            result[ee['category']].append(ee)
        return result

    pred_data = json_load_by_line(pred_file)
    target_data = json_load_by_line(target_file)
    target_tree = {}
    for t in target_data:
        if t['id'] not in target_tree:
            target_tree[t['id']] = {}
        if t['sub_id'] not in target_tree[t['id']]:
            target_tree[t['id']][t['sub_id']] = t
    type_list = ['position', 'name', 'movie', 'organization', 'company', 'game', 'book', 'address', 'scene',
                 'government', 'email', 'mobile', 'QQ', 'vx']
    f1_result = dict((key, {'cn': 0, 'pn': 0, 'tn': 0}) for key in type_list + ['all'])

    for p in pred_data:
        target = target_tree[p['id']][p['sub_id']]

        pred_c = split_by_category(p['entities'])
        target_c = split_by_category(target['entities'])
        precise_list = []
        recall_list = []
        for key in type_list:
            for e in pred_c[key]:
                if e in target_c[key]:
                    f1_result[key]['cn'] += 1
                    f1_result['all']['cn'] += 1
                else:
                    precise_list.append(e)
            for e in target_c[key]:
                if e not in pred_c[key]:
                    recall_list.append(e)

            f1_result[key]['pn'] += len(pred_c[key])
            f1_result[key]['tn'] += len(target_c[key])
            f1_result['all']['pn'] += len(pred_c[key])
            f1_result['all']['tn'] += len(target_c[key])

        if verbose and (precise_list or recall_list):
            print('\nprecise')
            print(precise_list)
            print('recall')
            print(recall_list)

    for key in f1_result.keys():
        p, r, f1 = calculate_f1(f1_result[key]['cn'],
                                f1_result[key]['pn'],
                                f1_result[key]['tn'], verbose=True)
        print('{:<12s}: precise {:0.6f} - recall {:0.6f} - f1 {:0.6f} - num {}'.format(key, p, r, f1,
                                                                                       f1_result[key]['tn']))


def mix_clue_result(infile, clue_file, outfile):
    data = json_load_by_line(infile)
    clue_data = json_load_by_line(clue_file)
    for d in data:
        if -1 < d['clue_id'] < 12091:
            clue_d = clue_data[d['clue_id']]
            d['entities'] = clue_d['entities']
            for e in d['entities']:
                e['privacy'] = d['text'][e['pos_b']:e['pos_e'] + 1]
    json_dump_by_line(data, outfile)


def cal_len(infile):
    data = json_load_by_line(infile)
    count_len = []
    for d in data:
        count_len.append(len(d['text']))
    a = count_item(count_len)
    for aa in a:
        print(aa)


def combine_dev_result(dev_dir, fold_num, outfile):
    pred_data = []
    for i in range(fold_num):
        pd = json_load_by_line(dev_dir + 'dev_result_{}.json'.format(i))
        pred_data.extend(pd)
    pred_data.sort(key=lambda x: x['sub_id'])
    pred_data.sort(key=lambda x: x['id'])
    json_dump_by_line(pred_data, outfile)


def prediction_count(files, outfile):
    data_list = [json_load_by_line(f) for f in files]
    new_data = []
    for sample_index in range(len(data_list[0])):
        sample = copy.deepcopy(data_list[0][sample_index])
        sample['entities'] = []
        entities = {}
        for data_index in range(len(data_list)):
            if data_index >= 1:
                assert data_list[data_index][sample_index]['id'] == data_list[data_index - 1][sample_index]['id']
                assert data_list[data_index][sample_index]['sub_id'] == data_list[data_index - 1][sample_index][
                    'sub_id']
            for e in data_list[data_index][sample_index]['entities']:
                ekey = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
                if ekey not in entities:
                    entities[ekey] = [e, 0]
                entities[ekey][1] += 1
        sample['entities'] = entities
        new_data.append(sample)
    json_dump_by_line(new_data, outfile)


def generate_loss_mask(pu_file, train_file, outfile):
    train_data = json_load_by_line(train_file)
    pu_data = json_load_by_line(pu_file)
    book_count = 0
    for sample_index in range(len(train_data)):
        if sample_index >= len(pu_data):
            break
        d = train_data[sample_index]
        pd = pu_data[sample_index]
        assert d['id'] == pd['id']
        assert d['sub_id'] == pd['sub_id']
        if 'loss_mask' not in d.keys():
            d['loss_mask'] = []
        for e in d['entities']:
            ekey = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
            if e['category'] in ('name', 'movie', 'organization', 'company', 'game', 'book', 'government'):
                if ekey not in pd['entities'] or pd['entities'][ekey][1] < 2:
                    pair = [e['pos_b'], e['pos_e']]
                    if pair not in d['loss_mask']:
                        d['loss_mask'].append(pair)

            elif e['category'] in ('position', 'address', 'scene'):
                if ekey not in pd['entities'] or pd['entities'][ekey][1] < 2:
                    pair = [e['pos_b'], e['pos_e']]
                    if pair not in d['loss_mask']:
                        d['loss_mask'].append(pair)

        # for ekey in pd['entities'].keys():
        #     e = pd['entities'][ekey][0]
        #     if e not in d['entities'] and pd['entities'][ekey][1] > 1 and e['category'] in ('name', 'book', 'game'):
        #         pair = [e['pos_b'], e['pos_e']]
        #         if pair not in d['loss_mask']:
        #             d['loss_mask'].append(pair)
        #             book_count += 1
        #             print(e)
    json_dump_by_line(train_data, outfile)
    print(book_count)


def conv2json():
    convert2json(ROOT_LOCAL_DATA + 'train_data/', ROOT_LOCAL_DATA + 'train_data.json',
                 label_dir=ROOT_LOCAL_DATA + 'train_label/')
    convert2json(ROOT_LOCAL_DATA + 'test_data/', ROOT_LOCAL_DATA + 'test_data.json')


if __name__ == '__main__':
    pass
    # prediction_count([
    #     ROOT_RESULT + 'dev_result_all.json',
    #     ROOT_LOCAL_DATA + 'seg_train_data.json'],
    #     ROOT_LOCAL_DATA + 'voted_dev_data.json')
    #
    # generate_loss_mask(
    #     ROOT_LOCAL_DATA + 'voted_dev_data.json',
    #     ROOT_LOCAL_DATA + 'seg_train_data.json',
    #     ROOT_LOCAL_DATA + 'seg_train_data_loss_mask.json'
    # )
