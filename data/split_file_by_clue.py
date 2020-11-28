from global_config import *
from util.tool import json_load_by_line, json_dump_by_line
import copy
import re


def split_text(clue_file, ccffile, outfile, max_len):
    data = json_load_by_line(clue_file)
    len_map = [[] for _ in range(55)]
    for d in data:
        len_map[len(d['text'])].append(d)


    def split_text_(sample):

        def split_by_len(sample_len):
            if len(sample_len['text']) <= max_len:
                return [sample_len]
            out_samples = []
            right_limit = 0
            rest_text = sample_len['text']
            while len(rest_text) > max_len:
                new_sample = copy.deepcopy(sample_len)
                new_sample['entities'] = []
                for char_index in range(max_len - 1, -1, -1):
                    if (rest_text[char_index] in ('，', '。', '!', '?')) or char_index == 0:
                        if char_index == 0:
                            char_index = max_len - 1
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

    data = json_load_by_line(ccffile)
    new_data = []
    for d in data:
        d['sub_id'] = 0
        new_data.extend(split_text_(d))

    json_dump_by_line(new_data, outfile)


def mix_data_set(clue_file, ccf_file, mixout_file):
    clue_data = json_load_by_line(clue_file)
    clue_train_data = clue_data[:12082]
    ccf_data = json_load_by_line(ccf_file)
    ccf_in_clue_list = []
    for d in ccf_data:
        if d['clue_id'] < 12082:
            ccf_in_clue_list.append(d['clue_id'])
    index = len(ccf_data)
    for d in clue_train_data:
        if d['id'] not in ccf_in_clue_list:
            new_d = {'id': index, 'sub_id': 0, 'clue_id': d['id'],
                     'text': d['text'], 'entities': d['entities']}
            index += 1
            ccf_data.append(new_d)
    json_dump_by_line(ccf_data, mixout_file)


if __name__ == '__main__':
    split_text(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/train_data.json',
               ROOT_DATA + 'ccf2020/train_data_s.json', 138)
    # split_text(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/test_data.json',
    #            ROOT_DATA + 'ccf2020/test_data_s.json', 138)
    # mix_data_set(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/train_data_s.json',
    #              ROOT_DATA + 'ccf2020/train_data_mix.json')
