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
        self.train_size = 15956
        self.train_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_train_data.json')
        # self.clue_train_data = json_load_by_line(ROOT_DATA + 'clue/token_train_data.json')
        # self.clue_dev_data = json_load_by_line(ROOT_DATA + 'clue/token_dev_data.json')
        self.test_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_test_data.json')
        self.train_tree = build_tree(self.train_data, 'id')

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
        if type_data == 'train':
            if self.config.en_cross:
                data_map = self.train_map[:fold_index * self.part_size]
                if fold_index + 1 < self.config.fold_num:
                    data_map += self.train_map[(fold_index + 1) * self.part_size:]
            else:
                data_map = self.train_map[:self.train_size]
            data = []
            for dindex in data_map:
                data.extend(self.train_tree[dindex])
            # data.extend(self.clue_train_data)
            # data.extend(self.clue_dev_data)
        elif type_data == 'dev':
            if self.config.en_cross:
                if fold_index + 1 < self.config.fold_num:
                    data_map = self.train_map[fold_index * self.part_size: (fold_index + 1) * self.part_size]
                else:
                    data_map = self.train_map[fold_index * self.part_size:]
            else:
                data_map = self.train_map[self.train_size:]
            data = []
            for dindex in data_map:
                data.extend(self.train_tree[dindex])
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
        self.train_file = 'train_data_mix.json'
        self.test_file = 'test_data_s.json'
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
            self.thu_lac = thulac.thulac(seg_only=True)
            self.pku_seg = pkuseg.pkuseg()
            self.tokenizer = MyBertTokenizer.from_pretrained(self.config.tokenizer)
            self.word_w2v = pickle.load(open(ROOT_PROJECT + '../data/w2v/w2v_vocab.pkl', 'rb'))
            self.word_w2v = dict([(word, index) for index, word in enumerate(self.word_w2v)])

        def split_text(sample):
            out_samples = []
            right_limit = 0
            sub_id = 0
            rest_text = sample['text']
            while len(rest_text) > self.max_len:
                new_sample = {'id': None, 'sub_id': None, 'text': None, 'entities': []}
                for char_index in range(self.max_len - 1, -1, -1):
                    if (rest_text[char_index] in ('，', '。', '!', '?')) or char_index == 0:
                        if char_index == 0:
                            char_index = self.max_len - 1
                        left_limit = right_limit
                        right_limit += char_index + 1
                        new_sample['text'] = rest_text[:char_index + 1]
                        new_sample['id'] = sample['id']
                        new_sample['sub_id'] = sub_id

                        if data_type in ('train', 'dev'):
                            for entity in sample['entities']:
                                if entity['pos_b'] >= left_limit and entity['pos_e'] < right_limit:
                                    new_entity = copy.deepcopy(entity)
                                    new_entity['pos_b'] = entity['pos_b'] - left_limit
                                    new_entity['pos_e'] = entity['pos_e'] - left_limit
                                    new_sample['entities'].append(new_entity)

                        sub_id += 1
                        rest_text = rest_text[char_index + 1:]
                        out_samples.append(new_sample)
                        break
            else:
                left_limit = right_limit
                new_sample = {'id': sample['id'], 'sub_id': sub_id, 'text': rest_text, 'entities': []}
                if data_type in ('train', 'dev'):
                    for entity in sample['entities']:
                        if entity['pos_b'] >= left_limit:
                            new_entity = copy.deepcopy(entity)
                            new_entity['pos_b'] = entity['pos_b'] - left_limit
                            new_entity['pos_e'] = entity['pos_e'] - left_limit
                            new_sample['entities'].append(new_entity)
                out_samples.append(new_sample)
            return out_samples

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

            cut_func = [jieba_cut, thulac_cut, pkuseg_cut]

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
                lword_index = 0
                if lword[0] in self.word_w2v:
                    lword_index = self.word_w2v[lword[0]]
                lattice.append([raw2decode[lword[1]], raw2decode[lword[2]], lword_index])

            entities = []
            if data_type in ('train', 'dev'):
                for entity in sample['entities']:
                    entities.append({"category": self.type2id[entity['category']],
                                     "pos_b": raw2decode[entity['pos_b']],
                                     "pos_e": raw2decode[entity['pos_e']]})
            return {'id': sample['id'], 'sub_id': sample['sub_id'], 'text': text, 'entities': entities,
                    'lattice': lattice}, \
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

        if data_type == 'train':
            in_file = self.train_file
            data_file = 'train_data.json'
        elif data_type == 'test':
            in_file = self.test_file
            data_file = 'test_data.json'
        else:
            print('format_data', data_type)
            return None

        raw_data = json_load_by_line(ROOT_LOCAL_DATA + in_file)
        out_data = {'seg': [],
                    'token': [],
                    'map': []}
        if num_works > 1:
            simple_multiprocess(main_process, raw_data, out_data, num_works)
        else:
            main_process(raw_data, out_data)

        json_dump_by_line(out_data['token'], ROOT_LOCAL_DATA + 'token_' + data_file)
        json_dump_by_line(out_data['map'], ROOT_LOCAL_DATA + 'map_' + data_file)
        json_dump_by_line(out_data['seg'], ROOT_LOCAL_DATA + 'seg_' + data_file)

    def generate_results(self, pred, data_type, output_file, keep_token=False):
        json_dump_by_line(pred, ROOT_RESULT+'temp_result.json')
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
                sample = {'id': d['id'], 'sub_id': d['sub_id'], 'text': td['text'], 'emission': p['emission'].tolist()}
            else:
                if keep_token:
                    sample = td
                    sample['entities'] = []
                else:
                    sample = {'id': d['id'], 'sub_id': d['sub_id'], 'text': d['text'], 'entities': []}
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
    pattern = re.compile(
        r"(([a-z]+[0-9]*[-_.]?[0-9]*){1,4}|[0-9]{4,16})(\@|\#)[a-z0-9-]*(\.[a-z0-9]{0,8}){,4}(?:\.cn|\.com|\.tw|\.net|\.asia|\.org|\.edu|\.hk|\.au|\.tv|\.sg|\.mo|\.ca)+",
        re.IGNORECASE)
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
    pattern = re.compile(r"微信[^0-9.]{,6}[0-9a-z_-]{2,20}", re.IGNORECASE)
    h = pattern.finditer(text)
    qq = []
    for i in h:
        qq_num = re.search(r"[0-9a-z_-]{2,20}$", i.group(), re.IGNORECASE).group()
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
            if "(（”" == e['privacy'][0] and e['pos_e'] + 1 < len(sample['text']) and sample['text'][e['pos_e'] + 1] == ')）”':
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
            csv_data.append([d['id'], e['category'], pos_b, pos_e, e['privacy']])

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

    f1_result = dict((key, {'cn': 0, 'pn': 0, 'tn': 0}) for key in
                     ['position', 'name', 'movie', 'organization', 'company', 'game', 'book', 'address', 'scene',
                      'government',
                      'email', 'mobile', 'QQ', 'vx', 'all'])

    for p in pred_data:
        # p['entities'] = get_email(p['text'])

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
        for key in pred_c.keys():

            for e in pred_c[key]:
                if e in target_c[key]:
                    f1_result[key]['cn'] += 1
                    f1_result['all']['cn'] += 1
                else:
                    precise_list.append(e)
            # if key == 'QQ':
            for e in target_c[key]:
                if e in pred_c[key]:
                    pass
                else:
                    recall_list.append(e)

            f1_result[key]['pn'] += len(pred_c[key])
            f1_result[key]['tn'] += len(target_c[key])
            f1_result['all']['pn'] += len(pred_c[key])
            f1_result['all']['tn'] += len(target_c[key])

        if (precise_list or recall_list):
            print('\nprecise')
            print(precise_list)
            print('recall')
            print(recall_list)
            print(target['text'])
            # print(target['entities'])

    for key in f1_result.keys():
        p, r, f1 = calculate_f1(f1_result[key]['cn'],
                                f1_result[key]['pn'],
                                f1_result[key]['tn'], verbose=True)
        print('{:<12s}: precise {:0.6f} - recall {:0.6f} - f1 {:0.6f} - num {}'.format(key, p, r, f1,
                                                                                       f1_result[key]['tn']))


def analyze_train_data(train_file):
    data = json_load_by_line(train_file)
    # data.extend(json_load_by_line(ROOT_DATA + 'clue/dev_data.json'))
    # data.extend(json_load_by_line(ROOT_DATA + 'clue/dev_data.json'))
    # max_lattice = 0
    count_len = []
    for d in data:
        name_list = []

        nested_list = []
        for ei, e in enumerate(d['entities']):
            if e['category'] == 'email':  # and e['privacy'] not in name_list and len(e['privacy'])>1 and \
                name_list.append(e['privacy'])
        if name_list:
            print(name_list)
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


if __name__ == '__main__':
    pass
    # convert2json('../../data/ccf2020_ner/train_data/', ROOT_LOCAL_DATA + 'train_data.json',
    #              label_dir='../../data/ccf2020_ner/train_label/')
    # convert2json('../../data/ccf2020_ner/test_data/', ROOT_LOCAL_DATA + 'test_data.json')

    # data = json_load_by_line(ROOT_LOCAL_DATA + 'train_data.json')
    # test_result = json_load_by_line(ROOT_RESULT + 'test_result.json')
    # data = json_load_by_line(ROOT_RESULT + 'test_result.json')
    # count = 0
    # for d, t in zip(data, test_result):
    #     emails = get_vx(d['text'])
    #     t = d
    #     p_list = []
    #     r_list = []
    #     for i in emails:
    #         if i not in t['entities']:
    #             p_list.append(i)
    #     for i in t['entities']:
    #         if i['category'] == 'vx' and i not in emails:
    #             r_list.append(i)
    #     if p_list or r_list:
    #         print('p')
    #         print(p_list)
    #         print('r')
    #         print(r_list)
    #         print(d['text'])
    # print(count)
    data = json_load_by_line(ROOT_LOCAL_DATA + 'train_data_s.json')
    data.extend(json_load_by_line(ROOT_LOCAL_DATA + 'test_data_s.json'))
    # data = json_load_by_line('../../data/clue_ner/test.json')
    count_len = []
    for d in data:
        if len(d['text']) < 5:
            print(d)
        count_len.append(len(d['text']))
    length, rate = count_item(count_len)
    for l, r in zip(length, rate):
        print(l, r)
