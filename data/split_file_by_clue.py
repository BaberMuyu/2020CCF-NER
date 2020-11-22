from global_config import *
from util.tool import json_load_by_line, json_dump_by_line
import copy


def split_text(clue_file, ccffile, outfile):
    data = json_load_by_line(clue_file)
    len_map = [[] for _ in range(55)]
    for d in data:
        len_map[len(d['text'])].append(d)

    new_data = dict()

    def split_(sample):
        if sample['id'] not in new_data.keys():
            new_data[sample['id']] = []
        a = len_map[:len(sample['text'])]
        a.reverse()
        for a_len_map in a:
            for clue_d in a_len_map:
                start_index = sample['text'].find(clue_d['text'])
                if start_index != -1:
                    end_index = start_index + len(clue_d['text']) - 1
                    new_sample = {'id': sample['id'], 'sub_id': sample['sub_id'] + start_index, 'clue_id': clue_d['id'],
                                  'text': clue_d['text'], 'entities': []}
                    for e in sample['entities']:
                        if e['pos_b'] >= start_index and e['pos_e'] <= end_index:
                            new_e = copy.deepcopy(e)
                            new_e['pos_b'] = e['pos_b'] - start_index
                            new_e['pos_e'] = e['pos_e'] - start_index
                            new_sample['entities'].append(new_e)
                    new_data[sample['id']].append(new_sample)
                    if start_index > 0:
                        rest_sample_0 = {'id': sample['id'], 'sub_id': sample['sub_id'],
                                         'text': sample['text'][:start_index], 'entities': []}
                        for e in sample['entities']:
                            if e['pos_e'] < start_index:
                                new_e = copy.deepcopy(e)
                                rest_sample_0['entities'].append(new_e)
                        split_(rest_sample_0)
                    if end_index + 1 < len(sample['text']):
                        rest_sample_1 = {'id': sample['id'], 'sub_id': sample['sub_id'] + end_index + 1,
                                         'text': sample['text'][end_index + 1:], 'entities': []}
                        for e in sample['entities']:
                            if e['pos_b'] > end_index:
                                new_e = copy.deepcopy(e)
                                new_e['pos_b'] = e['pos_b'] - end_index - 1
                                new_e['pos_e'] = e['pos_e'] - end_index - 1
                                rest_sample_1['entities'].append(new_e)
                        split_(rest_sample_1)
                    return
        new_sample = {'id': sample['id'], 'sub_id': sample['sub_id'], 'clue_id': -1,
                      'text': sample['text'], 'entities': sample['entities']}
        new_data[sample['id']].append(new_sample)

    data = json_load_by_line(ccffile)
    for d in data:
        d['sub_id'] = 0
        split_(d)

    new_data_list = []
    keys = list(new_data.keys())
    keys.sort()
    for key in keys:
        new_data[key].sort(key=lambda x: x['sub_id'])
        for index, sample in enumerate(new_data[key]):
            sample['sub_id'] = index
        new_data_list.extend(new_data[key])
    json_dump_by_line(new_data_list, outfile)


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
    # split_text(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/train_data.json',
    #            ROOT_DATA + 'ccf2020/train_data_s.json')
    # split_text(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/test_data.json',
    #            ROOT_DATA + 'ccf2020/test_data_s.json')
    mix_data_set(ROOT_DATA + 'clue/train_dev_test_data.json', ROOT_DATA + 'ccf2020/train_data_s.json',
                 ROOT_DATA + 'ccf2020/train_data_mix.json')
