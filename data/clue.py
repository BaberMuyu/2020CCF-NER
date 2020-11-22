from torch.utils.data import Dataset
import jieba
import thulac
import pkuseg
import copy
import pickle

from global_config import ROOT_DATA, ROOT_PROJECT
from util.mytokenizer import MyBertTokenizer
from util.tool import *

ROOT_LOCAL_DATA = ROOT_DATA + 'clue/'


class CLUEDataSet(Dataset):
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


class CLUEDataSetFactory(object):
    def __init__(self, config):
        self.config = config
        self.train_size = 10748
        self.train_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_train_data.json')
        self.dev_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_dev_data.json')
        self.test_data = json_load_by_line(ROOT_LOCAL_DATA + 'token_test_data.json')

    def __call__(self, inputs):
        type_data = inputs['type_data']
        if type_data == 'train':
            data = self.train_data
        elif type_data == 'dev':
            data = self.dev_data
        elif type_data == 'test':
            data = self.test_data
        else:
            assert 0
        return CLUEDataSet(data, self.config.test_mode)


class CLUEDataProcess(object):
    def __init__(self, config):
        self.config = config
        self.token_max_len = self.config.max_len
        self.max_len = self.config.max_len - 2
        self.thu_lac = None
        self.pku_seg = None
        self.tokenizer = None
        self.word_w2v = None
        self.train_file = 'train_data.json'
        self.dev_file = 'dev_data.json'
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
                    if (rest_text[char_index] in ('，', '。', ':', '：', '）')) or char_index == 0:
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
        elif data_type == 'dev':
            in_file = self.dev_file
        elif data_type == 'test':
            in_file = self.test_file
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

        json_dump_by_line(out_data['token'], ROOT_LOCAL_DATA + 'token_' + in_file)
        json_dump_by_line(out_data['map'], ROOT_LOCAL_DATA + 'map_' + in_file)
        json_dump_by_line(out_data['seg'], ROOT_LOCAL_DATA + 'seg_' + in_file)

    def generate_results(self, pred, data_type, output_file):
        data_set = {"train": "map_train_data.json",
                    'dev': 'map_dev_data.json',
                    'test': 'map_test_data.json'}
        token_data_set = {"train": "token_train_data.json",
                          'dev': 'token_dev_data.json',
                          'test': 'token_test_data.json'}
        data = json_load_by_line(ROOT_LOCAL_DATA + data_set[data_type])
        id_tree = build_tree(data, 'id')

        token_data = json_load_by_line(ROOT_LOCAL_DATA + token_data_set[data_type])
        id_tree_token = build_tree(token_data, 'id')

        result = []
        for p in pred:
            d = id_tree[p['id']][p['sub_id']]
            td = id_tree_token[p['id']][p['sub_id']]
            sample = {'id': d['id'], 'sub_id': d['sub_id'], 'text': d['text'], 'entities': []}
            if 'emission' in p:
                sample['emission'] = p['emission']
                sample['text'] = td['text']
            else:
                for entity in p['entities']:
                    pos_b = d['text_map'][entity[1]]
                    if entity[2] + 1 == len(d['text_map']):
                        privacy = d['text'][pos_b: d['text_map'][entity[2]] + 1].strip()
                    else:
                        privacy = d['text'][pos_b: d['text_map'][entity[2] + 1]].strip()
                    sample['entities'].append({
                        'privacy': privacy,
                        'category': self.id2type[entity[0]],
                        'pos_b': pos_b,
                        'pos_e': pos_b + len(privacy) - 1})
            result.append(sample)
        json_dump_by_line(result, output_file)


# ================================================================================================

def convert2json(infile, outfile, cnum=1):
    data = json_load_by_line(infile)
    data.extend(json_load_by_line('../../data/clue_ner/dev.json'))
    data.extend(json_load_by_line('../../data/clue_ner/test.json'))
    outdata = []
    index = 0
    for d in data:
        if d['text'] in ('《蝙蝠侠》', '星际争霸2','星际2', '反恐精英','穿越火线','魔兽争霸3', '《超人》', '《变形金刚》', '英雄联盟'):
            continue
        sample = {'id': index, 'text': d['text'], 'entities': []}
        index += 1
        if 'label' in d:
            for category in d['label'].keys():
                for privacy in d['label'][category].keys():
                    for entity in d['label'][category][privacy]:
                        sample['entities'].append({'privacy': privacy,
                                                   'category': category,
                                                   'pos_b': entity[0],
                                                   'pos_e': entity[1]})
        outdata.append(sample)

    outdata2 = []
    base = 0
    sample = None
    for index, d in enumerate(outdata):
        if (index % cnum) == 0:
            if sample is not None:
                outdata2.append(sample)
            sample = {'id': int(index / cnum), 'text': d['text'], 'entities': d['entities'][:]}
            base = len(d['text'])
        else:
            sample['text'] += d['text']
            for e in d['entities']:
                sample['entities'].append({'privacy': e['privacy'],
                                           'category': e['category'],
                                           'pos_b': e['pos_b'] + base,
                                           'pos_e': e['pos_e'] + base})
            base += len(d['text'])

    json_dump_by_line(outdata2, outfile)


if __name__ == '__main__':
    convert2json('../../data/clue_ner/train.json', ROOT_LOCAL_DATA + 'train_dev_test_data.json', cnum=1)
    # convert2json('../../data/clue_ner/dev.json', ROOT_LOCAL_DATA + 'dev_data.json')
    # convert2json('../../data/clue_ner/test.json', ROOT_LOCAL_DATA + 'test_data.json')
    data = json_load_by_line(ROOT_LOCAL_DATA + 'train_dev_test_data.json')
    # data = json_load_by_line('../../data/clue_ner/test.json')
    count_len = []
    for d in data:
        if len(d['text']) < 10:
            print(d)
        count_len.append(len(d['text']))
    length, rate = count_item(count_len)
    for l, r in zip(length, rate):
        print(l, r)
