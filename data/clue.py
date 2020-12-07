from global_config import ROOT_DATA
from util.tool import *

ROOT_LOCAL_DATA = ROOT_DATA + 'clue/'


# ================================================================================================


def conv2json_(infiles, outfile, cnum=1):
    data = []
    if type(infiles) is list:
        for infile in infiles:
            data.extend(json_load_by_line(infile))
    else:
        data.extend(json_load_by_line(infiles))

    outdata = []
    index = 0
    for d in data:
        if d['text'] in ('《蝙蝠侠》', '星际争霸2', '星际2', '反恐精英', '穿越火线', '魔兽争霸3', '《超人》', '《变形金刚》', '英雄联盟'):
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


def conv2json():
    conv2json_([ROOT_LOCAL_DATA + 'train.json',
                ROOT_LOCAL_DATA + 'dev.json',
                ROOT_LOCAL_DATA + 'test.json'],
               ROOT_LOCAL_DATA + 'train_dev_test_data.json')


if __name__ == '__main__':
    pass
    # conv2json_(ROOT_LOCAL_DATA + 'train.json', ROOT_LOCAL_DATA + 'train_data.json')
    # conv2json_(ROOT_LOCAL_DATA + 'dev.json', ROOT_LOCAL_DATA + 'dev_data.json')
    # conv2json_(ROOT_LOCAL_DATA + 'test.json', ROOT_LOCAL_DATA + 'test_data.json')
    # conv2json_([ROOT_LOCAL_DATA + 'test.json', ROOT_LOCAL_DATA + 'dev.json', ROOT_LOCAL_DATA + 'test.json'],
    #              ROOT_LOCAL_DATA + 'train_dev_test_data.json')
