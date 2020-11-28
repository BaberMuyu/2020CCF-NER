from torch.utils.data import Dataset
from util.mytokenizer import MyBertTokenizer
import csv
import os
import jieba
import thulac
import pkuseg
import pandas
import copy
import random
import numpy
import pickle
import torch
import re

from global_config import ROOT_DATA, ROOT_RESULT, ROOT_WEIGHT, ROOT_PROJECT
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
        self.train_data_with_mask = json_load_by_line(ROOT_LOCAL_DATA + 'token_seg_train_data_loss_mask.json')
        # self.clue_train_data = json_load_by_line(ROOT_DATA + 'clue/token_train_data.json')
        # self.clue_dev_data = json_load_by_line(ROOT_DATA + 'clue/token_dev_data.json')
        self.test_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_test_data.json')
        self.train_tree = build_tree(self.train_data, 'id')
        self.train_data_with_mask_tree = build_tree(self.train_data_with_mask, 'id')

        if self.config.reshuffle:
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
            # data.extend(self.clue_train_data)
            # data.extend(self.clue_dev_data)
        elif type_data in ('dev', 'dev_loss_mask'):
            data_tree = self.train_tree if type_data == 'train' else self.train_data_with_mask_tree
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
        elif type_data == 'combine':
            data = json_load_by_line(ROOT_RESULT + 'combine_emission.json')
        elif type_data == 'combine_type':
            data = json_load_by_line(ROOT_RESULT + 'combine_type_emission.json')
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

    @staticmethod
    def fix_entity(sample):
        for e in sample['entities']:
            if e['privacy'][0] == '《' and e['pos_e'] + 1 < len(sample['text']) and \
                    sample['text'][e['pos_e']] != '》' and sample['text'][e['pos_e'] + 1] == '》':
                print(e)
                print(sample['text'])
                e['privacy'] += '》'
                e['pos_e'] += 1
            if e['privacy'][-1] == '》' and e['pos_b'] > 0 and \
                    sample['text'][e['pos_b']] != '《' and sample['text'][e['pos_b'] - 1] == '《':
                print(e)
                print(sample['text'])
                e['privacy'] += '《'
                e['pos_b'] -= 1

    def format_data(self, data_type='train', num_works=1):
        if self.thu_lac is None or self.pku_seg is None or self.tokenizer is None or self.word_w2v is None:
            # self.thu_lac = thulac.thulac(seg_only=True)
            self.pku_seg = pkuseg.pkuseg()
            self.tokenizer = MyBertTokenizer.from_pretrained(self.config.tokenizer)
            self.word_w2v = pickle.load(open(ROOT_PROJECT + '../data/w2v/w2v_vocab.pkl', 'rb'))
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
                                    print(sample_['text'])
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

            def jieba_cut(text):
                index = 0
                word_list = []
                for word in jieba.cut(text):
                    word_len = len(word)
                    if word_len > 1 and is_all_chinese(word):
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            def pkuseg_cut(text):
                index = 0
                word_list = []
                for word in self.pku_seg.cut(text):
                    word_len = len(word)
                    if word_len > 1 and is_all_chinese(word):
                        word_list.append((word, index, index + word_len - 1))
                    index += word_len
                return word_list

            def thulac_cut(text):
                index = 0
                word_list = []
                for word, _ in self.thu_lac.cut(text):
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
            # index = 2514
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

    def generate_results(self, pred, data_type, output_file, keep_token=False):
        # json_dump_by_line(pred, ROOT_RESULT + 'temp_result.json')
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
            if 'emission' in p:  # entity emission
                sample = {'id': d['id'], 'sub_id': d['sub_id'], 'clue_id': sd['clue_id'],
                          'text': td['text'], 'emission': p['emission']}
            else:
                if keep_token:
                    sample = td
                    sample['entities'] = []
                else:
                    sample = {'id': d['id'], 'sub_id': d['sub_id'], 'clue_id': sd['clue_id'],
                              'text': d['text'], 'entities': []}
                for entity in p['entities']:
                    if keep_token:
                        sample['entities'].append({'category': entity[0], 'pos_b': entity[1], 'pos_e': entity[2]})
                    else:
                        pos_b = d['text_map'][entity[1]]
                        if entity[2] + 1 == len(d['text_map']):
                            privacy = d['text'][pos_b: d['text_map'][entity[2]] + 1].strip()
                        else:
                            privacy = d['text'][pos_b: d['text_map'][entity[2] + 1]].strip()
                        raw_entity = {
                            'privacy': privacy,
                            'category': None,
                            'pos_b': pos_b,
                            'pos_e': pos_b + len(privacy) - 1}
                        if type(entity[0]) is int:
                            raw_entity['category'] = self.id2type[entity[0]]
                        else:
                            raw_entity['category'] = entity[0].tolist()
                        sample['entities'].append(raw_entity)
                        token_entity = {'category': entity[0], 'pos_b': entity[1], 'pos_e': entity[2]}
                        if raw_entity not in sd['entities'] and token_entity in td['entities']:
                            print(token_entity)
                            print(raw_entity)
                            print(sd['text'])
                            print(sd['entities'])

            result.append(sample)
        json_dump_by_line(result, output_file)

    @staticmethod
    def combine_emission(file_list, crf_param, crf_out):
        outfile = ROOT_RESULT + 'combine_emission.json'
        data_list = [json_load_by_line(f) for f in file_list]
        enum = len(file_list)
        for sample_index in range(len(data_list[0])):
            sum_array = numpy.array(data_list[0][sample_index]['emission'])
            for eindex in range(1, enum):
                sum_array += numpy.array(data_list[eindex][sample_index]['emission'])
            sum_array /= enum
            data_list[0][sample_index]['emission'] = sum_array.tolist()
        json_dump_by_line(data_list[0], outfile)

        crf_param = json_load_by_line(crf_param)
        crf_s = numpy.array(crf_param[0][0])
        crf_e = numpy.array(crf_param[0][1])
        crf_t = numpy.array(crf_param[0][2])
        for crf in crf_param[1:]:
            crf_s += numpy.array(crf[0])
            crf_e += numpy.array(crf[1])
            crf_t += numpy.array(crf[2])
        crf_s /= len(crf_param)
        crf_e /= len(crf_param)
        crf_t /= len(crf_param)
        json_dump_by_line([crf_s.tolist(), crf_e.tolist(), crf_t.tolist()], crf_out)

    def combine_type_emission(self, file_list, outfile):
        data_list = [json_load_by_line(f) for f in file_list]
        enum = len(file_list)
        for sample_index in range(len(data_list[0])):
            for entity_index in range(len(data_list[0][sample_index]['entities'])):
                sum_array = torch.zeros(self.config.num_types).double()
                for file_index in range(enum):
                    type_emission = numpy.array(
                        data_list[file_index][sample_index]['entities'][entity_index]['category'])
                    sum_array += torch.softmax(torch.tensor(type_emission), dim=0)
                type_id = torch.argmax(sum_array, dim=0)
                data_list[0][sample_index]['entities'][entity_index]['category'] = self.id2type[int(type_id)]
        json_dump_by_line(data_list[0], outfile)


# ================================================================================================
def get_email(text):
    if '@' not in text and '#' not in text:
        return []
    # pattern = re.compile(
    #     r"(([a-z]+[0-9]*[-_.]?[0-9]*){1,4}|[0-9]{4,16})(\@|\#)[a-z0-9-]*(\.[a-z0-9]{0,8}){,4}(?:\.cn|\.com|\.tw|\.net|\.asia|\.org|\.edu|\.hk|\.au|\.tv|\.sg|\.mo|\.ca)+",
    #     re.IGNORECASE)
    pattern = None
    h = pattern.finditer(text)
    emails = []
    for i in h:
        emails.append({'privacy': i.group(),
                       'category': 'email',
                       'pos_b': i.span()[0],
                       'pos_e': i.span()[1] - 1})
    return emails


def get_qq(text):
    pattern = re.compile(r"(q{1,2}|[^微]?群)[^0-9]{,2}[0-9]?[^0-9]{,2}[0-9]{6,12}", re.IGNORECASE)
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
    pattern = re.compile(r"微信[^0-9.]{,6}[0-9a-z_-]{4,20}", re.IGNORECASE)
    h = pattern.finditer(text)
    qq = []
    for i in h:
        qq_num = re.search(r"[0-9a-z_-]{4,20}$", i.group(), re.IGNORECASE).group()
        qq.append({'privacy': qq_num,
                   'category': 'vx',
                   'pos_b': i.span()[1] - len(qq_num),
                   'pos_e': i.span()[1] - 1})
    return qq


def clean_result(infile, outfile):
    data = json_load_by_line(infile)
    for sample in data:
        new_entities = []
        qq = get_qq_strict(sample['text'])
        for q in qq:
            if q not in sample['entities']:
                sample['entities'].append(q)
        for e in sample['entities']:
            if "，" == e['privacy'][-1]:
                e['privacy'] = e['privacy'][:-1]
                e['pos_e'] -= 1
            if '""' == e['privacy'][:2]:
                e['privacy'] = e['privacy'][2:]
                e['pos_b'] += 2
            if '""' == e['privacy'][-2:]:
                e['privacy'] = e['privacy'][:-2]
                e['pos_b'] -= 2
            if "(（”" == e['privacy'][0] and e['pos_e'] + 1 < len(sample['text']) and sample['text'][
                e['pos_e'] + 1] == ')）”':
                e['privacy'] = e['privacy'][1:]
                e['pos_b'] += 1
            if ")）”" == e['privacy'][-1] and sample['text'][e['pos_e'] - 1] == '(（”':
                e['privacy'] = e['privacy'][:-1]
                e['pos_e'] -= 1
            if '(' in e['privacy'] and e['pos_e'] + 1 < len(sample['text']) and sample['text'][e['pos_e'] + 1] == ')':
                e['privacy'] = sample['text'][e['pos_b']:e['pos_e'] + 2]
                e['pos_e'] += 1
            if ')' in e['privacy'] and e['pos_e'] + 1 < len(sample['text']) and sample['text'][e['pos_e'] + 1] == '(':
                e['privacy'] = sample['text'][e['pos_b'] - 1:e['pos_e'] + 1]
                e['pos_b'] -= 1
            if e['category'] == 'email':
                if '@' not in e['privacy'] and '#' not in e['privacy']:
                    continue
                h = re.search(r"[0-9a-z._-]+(\@|\#)[0-9a-z_-]+\.[a-z0-9._-]+", e['privacy'], re.IGNORECASE)
                if not h:
                    continue
                e['privacy'] = h.group()
                e['pos_b'] = e['pos_b'] + h.span()[0]
                e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
            elif e['category'] == 'QQ':
                if len(e['privacy']) > 15 and e['privacy'].isdigit():
                    h = re.search(r"1[0-9]{10}$", e['privacy'], re.IGNORECASE)
                    if h:
                        new_entity = {'category': 'mobile',
                                      'privacy': h.group(),
                                      'pos_b': e['pos_b'] + h.span()[0],
                                      'pos_e': e['pos_b'] + len(e['privacy']) - 1}
                        new_entities.append(new_entity)
                        if 5 < h.span()[0] < 11:
                            e['privacy'] = e['privacy'][:h.span()[0]]
                            e['pos_b'] = e['pos_b']
                            e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
                h = re.search(r"[0-9]{5,12}", e['privacy'], re.IGNORECASE)
                if not h or len(h.group()) >= 12:
                    continue
                if (not re.search('q', sample['text'], re.IGNORECASE)) and (
                        not re.search('群', sample['text'], re.IGNORECASE)):
                    continue
                e['privacy'] = h.group()
                e['pos_b'] = e['pos_b'] + h.span()[0]
                e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
            elif e['category'] == 'mobile':
                if len(e['privacy']) > 15 and e['privacy'].isdigit():
                    h = re.search(r"1[0-9]{10}$", e['privacy'], re.IGNORECASE)
                    if h:
                        e['privacy'] = h.group()
                        e['pos_b'] = e['pos_b'] + h.span()[0]
                        e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
                    else:
                        continue
            elif e['category'] == 'company':
                if e['privacy'][0] in '《【「':
                    e['privacy'] = e['privacy'][1:]
                    e['pos_b'] += 1
                if e['privacy'][-1] in '》】」':
                    e['privacy'] = e['privacy'][:-1]
                    e['pos_e'] -= 1
                if e['privacy'][0] in '（“' and e['privacy'][1] in '）”':
                    e['privacy'] = e['privacy'][1:-1]
                    e['pos_b'] += 1
                    e['pos_e'] -= 1
            elif e['category'] == 'name':
                if e['privacy'] in ['南征北战', '欧佩克', '专家', '金融', '俄罗斯', 'DotA', 'EA', '台灣', '《大众软件》', '2009', '820']:
                    continue
                if '《' in e['privacy'] or '》' in e['privacy']:
                    continue
                if '"' in e['privacy']:
                    print(e['privacy'])

            new_entities.append(e)
        sample['entities'] = new_entities
    json_dump_by_line(data, outfile)


def clean_result_2(infile, outfile):
    data = json_load_by_line(infile)
    for sample in data:
        new_entities = []
        qq = get_qq_strict(sample['text'])
        for q in qq:
            if q not in sample['entities']:
                sample['entities'].append(q)
        vx = get_vx(sample['text'])
        for v in vx:
            if v not in sample['entities']:
                sample['entities'].append(v)
        for e in sample['entities']:
            if e['category'] == 'email':
                if '@' not in e['privacy'] and '#' not in e['privacy']:
                    continue
                h = re.search(r"[0-9a-z._-]+(\@|\#)[0-9a-z_-]+\.[a-z0-9._-]+", e['privacy'], re.IGNORECASE)
                if not h:
                    continue
                e['privacy'] = h.group()
                e['pos_b'] = e['pos_b'] + h.span()[0]
                e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
            if e['category'] == 'QQ':
                if len(e['privacy']) <= 6:
                    continue
                if not re.search('(q|群)', sample['text'], re.IGNORECASE):
                    continue

                h = re.search(r"[0-9]{7,12}", e['privacy'], re.IGNORECASE)
                if not h or len(h.group()) >= 12:
                    continue
                else:
                    e['privacy'] = h.group()
                    e['pos_b'] = e['pos_b'] + h.span()[0]
                    e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
            if e['category'] == 'mobile':
                if len(e['privacy']) > 15 and e['privacy'].isdigit():
                    h = re.search(r"(1[3-9]{10}$|^1[3-9]{10})", e['privacy'], re.IGNORECASE)
                    if h:
                        e['privacy'] = h.group()
                        e['pos_b'] = e['pos_b'] + h.span()[0]
                        e['pos_e'] = e['pos_b'] + len(e['privacy']) - 1
                    else:
                        continue
                h = re.search(r"[a-np-z]", e['privacy'], re.IGNORECASE)
                if h:
                    continue
            if e['category'] == 'vx':
                if not re.search('(vx|微|wx)', sample['text'], re.IGNORECASE):
                    continue
            if e['category'] == 'name':
                if '"' in e['privacy']:
                    print(e['privacy'])
                    e['privacy'] = re.sub('"', '', e['privacy'])
                    print(e['privacy'])
            new_entities.append(e)
        sample['entities'] = new_entities
    json_dump_by_line(data, outfile)


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

    csv_data = pandas.DataFrame(csv_data, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    csv_data.to_csv(outfile, index=False)


def analyze_dev_data(pred_file, target_file):
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
    num = 0
    error_num = 0
    for p in pred_data:
        target = target_tree[p['id']][p['sub_id']]
        target_entities = [[x['category'], x['pos_b'], x['pos_e']] for x in target['entities']]
        target_span = [[x['pos_b'], x['pos_e']] for x in target['entities']]
        pred_entities = [[x['category'], x['pos_b'], x['pos_e']] for x in p['entities']]
        pred_span = [[x['pos_b'], x['pos_e']] for x in p['entities']]

        # for span, entity, iii in zip(pred_span, pred_entities, p['entities']):
        #     if span in target_span and entity not in target_entities:
        #         pass
        #         print(iii)
        #         print(target['entities'])
        #         print(target['text'], '\n')
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
                    if key == 'mobile':
                        precise_list.append(e)
            for e in target_c[key]:
                if e in pred_c[key]:
                    pass
                else:
                    if key == 'mobile':
                        recall_list.append(e)

            f1_result[key]['pn'] += len(pred_c[key])
            f1_result[key]['tn'] += len(target_c[key])
            f1_result['all']['pn'] += len(pred_c[key])
            f1_result['all']['tn'] += len(target_c[key])

        if precise_list or recall_list:
            error_num += 1
            # print('\nprecise')
            # print(precise_list)
            # print('recall')
            # print(recall_list)
            # print('clue_id: {}'.format(target['clue_id']), target['text'])
            # print(target['entities'])
        num += 1
    for key in f1_result.keys():
        p, r, f1 = calculate_f1(f1_result[key]['cn'],
                                f1_result[key]['pn'],
                                f1_result[key]['tn'], verbose=True)
        print('{:<12s}: precise {:0.6f} - recall {:0.6f} - f1 {:0.6f} - num {}'.format(key, p, r, f1,
                                                                                       f1_result[key]['tn']))
    print(error_num / num)


def analyze_train_data(train_file):
    data = json_load_by_line(train_file)
    count_len = []
    for d in data:
        name_list = []

        nested_list = []
        if d['clue_id'] == -1 or d['clue_id'] > 12090:
            for ei, e in enumerate(d['entities']):
                if e['category'] == 'address':  # and e['privacy'] not in name_list and len(e['privacy'])>1 and \
                    name_list.append(e)
        if name_list:
            for n in name_list:
                print(n)
            print(d['text'])
            print('\n')


def check_name(infile, name_file):
    name_list = json_load_by_line(name_file)
    data = json_load_by_line(infile)
    for d in data:
        d_name_list = []
        c_name_list = []
        for e in d['entities']:
            if e['category'] == 'name':
                d_name_list.append(e['privacy'])
        for name in name_list:
            if name in d['text'] and name not in d_name_list:
                c_name_list.append(name)
        if c_name_list:
            print(c_name_list)
            print(d['entities'])
            print(d['text'])
            print('\n')


def mix_clue_result(infile, clue_file, outfile):
    data = json_load_by_line(infile)
    clue_data = json_load_by_line(clue_file)
    for d in data:
        if -1 < d['clue_id'] < 12091:
            clue_d = clue_data[d['clue_id']]
            p_list = []
            r_list = []
            for p in d['entities']:
                if p not in clue_d['entities']:
                    p_list.append(p)
            for r in clue_d['entities']:
                if r not in d['entities']:
                    r_list.append(r)
            d['entities'] = clue_d['entities']
            for e in d['entities']:
                e['privacy'] = d['text'][e['pos_b']:e['pos_e'] + 1]
            if p_list or r_list:
                print('precise')
                print(p_list)
                print('recall')
                print(r_list)
                print(d['text'])
                print('\n')
        else:
            pass
            # d['entities'] = []

    json_dump_by_line(data, outfile)


def convert():
    convert2json('../../data/ccf2020_ner/train_data/', ROOT_LOCAL_DATA + 'train_data.json',
                 label_dir='../../data/ccf2020_ner/train_label/')
    convert2json('../../data/ccf2020_ner/test_data/', ROOT_LOCAL_DATA + 'test_data.json')


def analyze_0():
    data = json_load_by_line(ROOT_LOCAL_DATA + 'train_data.json')
    test_result = json_load_by_line(ROOT_RESULT + 'test_result.json')
    data = json_load_by_line(ROOT_RESULT + 'test_result.json')
    count = 0
    for d, t in zip(data, test_result):
        emails = get_vx(d['text'])
        t = d
        p_list = []
        r_list = []
        for i in emails:
            if i not in t['entities']:
                p_list.append(i)
        for i in t['entities']:
            if i['category'] == 'vx' and i not in emails:
                r_list.append(i)
        if p_list or r_list:
            print('p')
            print(p_list)
            print('r')
            print(r_list)
            print(d['text'])
    print(count)


def calculate_rate(infile):
    data = json_load_by_line(infile)
    num = 0
    count = 0
    for d in data:
        if d['clue_id'] != -1:
            count += 1
        num += 1
    print(count / num)
    # print(count)


def cal_len(infile):
    data = json_load_by_line(infile)
    count_len = []
    for d in data:
        # if len(d['text']) > 130:
        #     print(d['text'])
        count_len.append(len(d['text']))
    a = count_item(count_len)
    for aa in a:
        print(aa)


def find_same_entity(infile):
    data = json_load_by_line(infile)
    count = 0
    for d in data:
        entity_tree = dict()
        for e in d['entities']:
            e_str = e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
            if e_str not in entity_tree.keys():
                entity_tree[e_str] = []
            entity_tree[e_str].append(e)
        for key in entity_tree.keys():
            if len(entity_tree[key]) > 1:
                print(entity_tree[key])
                print(d['text'])
                print(d['entities'])
                print('\n')
                count += 1
    print(count)


def cal_dev_score(dev_dir):
    pred_data = []
    for i in range(5):
        pd = json_load_by_line(dev_dir + '/dev_result_{}.json'.format(i))
        pred_data.extend(pd)

    ccf_data = []
    for d in pred_data:
        # if d['clue_id'] == -1:
        # if d['clue_id'] > 12
        ccf_data.append(d)

    json_dump_by_line(ccf_data, ROOT_RESULT + 'temp_result.json')
    clean_result_2(ROOT_RESULT + 'temp_result.json', ROOT_RESULT + 'temp_result2.json')
    analyze_dev_data(ROOT_RESULT + 'temp_result2.json', ROOT_LOCAL_DATA + 'seg_train_data.json')


def generate_loss_mask(pred_dirs, train_file, outfile):
    train_data = json_load_by_line(train_file)
    for pred_dir, type_list in pred_dirs:
        pred_data = []
        for i in range(5):
            pd = json_load_by_line(pred_dir + '/dev_result_{}.json'.format(i))
            pred_data.extend(pd)

        pred_tree = build_tree(pred_data, 'id')
        for d in train_data:
            pd = pred_tree[d['id']][d['sub_id']]
            if pd['sub_id'] != d['sub_id']:
                print('error')
            if 'loss_mask' not in d.keys():
                d['loss_mask'] = []
            for e in d['entities']:
                if e not in pd['entities'] and e['category'] in type_list:
                    pair = [e['pos_b'], e['pos_e']]
                    if pair not in d['loss_mask']:
                        d['loss_mask'].append(pair)
    json_dump_by_line(train_data, outfile)


if __name__ == '__main__':
    pass
    analyze_dev_data(ROOT_RESULT + 'dev_result.json', ROOT_LOCAL_DATA + 'seg_train_data.json')
    # analyze_train_data(ROOT_LOCAL_DATA + 'seg_train_data.json')
    # analyze_train_data(ROOT_RESULT + 'test_result_1.json')
    # mix_clue_result(ROOT_RESULT + 'test_result.json', ROOT_DATA + 'clue/train_dev_test_data.json',
    #                 ROOT_RESULT + 'clean_test_result.json')
    # calculate_rate(ROOT_LOCAL_DATA + 'test1_data_s.json')
    # cal_len(ROOT_LOCAL_DATA + 'test2_data.json')
    # calculate_rate(ROOT_LOCAL_DATA + 'test1_data_s.json')
    # clean_result_2(ROOT_RESULT + 'combine_test_result.json', ROOT_RESULT + 'clean_combine_test_result.json')
    # mix_clue_result(ROOT_RESULT + 'clean_combine_test_result.json', ROOT_DATA + 'clue/train_dev_test_data.json',
    #                         ROOT_RESULT + 'clue_combine_test_result.json')
    # convert2csv(ROOT_RESULT + 'clean_combine_test_result.json',
    #                     ROOT_RESULT + 'noclue_combine_predict.csv')
    # analyze_train_data(ROOT_LOCAL_DATA + 'seg_train_data.json')
    cal_dev_score(ROOT_RESULT + 'none_linear_nobie_5fold_3_pu/')
    cal_dev_score(ROOT_RESULT + '/')
    # find_same_entity(ROOT_RESULT + "none_linear_nobie_5fold_3_pu/combine_test_result.json")
    # generate_loss_mask(
    #     [(ROOT_RESULT + 'w2v_flat_nobie_5fold/', ['position', 'movie', 'organization', 'company', 'book', 'address']),
    #      (ROOT_RESULT + 'none_linear_nobie_5fold_3_pu/', ['name', 'game', 'scene', 'government'])],
    #     ROOT_LOCAL_DATA + 'seg_train_data.json',
    #     ROOT_LOCAL_DATA + 'seg_train_data_loss_mask.json')
