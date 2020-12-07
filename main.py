import torch
import data.ccf20202 as ccf2020
import data.clue as clue
import data.word2vec as w2v
from data.ccf20202 import DataConfig, CCF2020DataSetFactory, CCF2020DataProcess

from module.encoder import EncoderConfig, WordE
from module.fusion import FusionConfig, FusionE
from module.output import OutputConfig, OutputConfigObie, LabelE
from module.fitting import FittingConfig, NERModelFitting
from model.fast_prototype import FastPrototype

from global_config import ROOT_WEIGHT, ROOT_RESULT, ROOT_DATA

import os


class Config(object):
    def __init__(self):
        self.encoder = EncoderConfig()
        self.fusion = FusionConfig()
        self.output = OutputConfigObie()
        self.output_type = OutputConfig()
        self.data = DataConfig()
        self.fitting = FittingConfig()

        # self.encoder.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        # self.encoder.ptm_feat_size = 1024
        # self.encoder.num_ptm_layers = 24
        self.encoder.worde = WordE.w2v

        self.fusion.fusion = FusionE.flat
        self.fusion.en_ffd = False
        self.fusion.num_pos = 4 if self.fusion.fusion == FusionE.flat else 1
        self.fusion.in_feat_size = self.encoder.out_feat_size

        self.output.label = LabelE.Nobie
        self.output.num_types = 14
        self.output.in_feat_size = self.fusion.out_feat_size

        self.output_type.num_types = 14
        self.output_type.in_feat_size = self.encoder.out_feat_size

        self.data.max_len = 140  # max token length
        self.data.tokenizer = self.encoder.ptm_model
        self.data.num_types = 14

        self.fitting.test_mode = False
        self.fitting.reshuffle = False  # 记得去改下随机种子
        self.fitting.en_cross = True
        self.fitting.fold_num = 5
        self.fitting.dev_rate = 0.2
        self.fitting.en_fgm = True
        self.fitting.en_swa = True
        self.fitting.epochs = 3  # 20
        self.fitting.end_epoch = 3
        self.fitting.batch_size = 16
        self.fitting.lr = {'ptm': 0.00003,
                           'other': 0.00003}
        if self.output.label != LabelE.point:
            self.fitting.lr['crf'] = 0.005
        self.fitting.verbose = True

    def __str__(self):
        string = ""
        string += str(self.encoder)
        string += str(self.fusion)
        string += str(self.output)
        string += str(self.data)
        string += str(self.fitting)
        return string


# =====================================================================================================
if __name__ == '__main__':
    config = Config()
    print('\nCUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    print(config)
    en_prep = True
    en_format = True
    en_train = True
    en_eval = True
    en_test = True
    en_conv = True

    data_process = CCF2020DataProcess(config.data)
    if en_prep:
        clue.conv2json()
        ccf2020.conv2json()
        w2v.conv2pkl()

    if en_format:
        data_process.format_data('train', num_works=4)
        data_process.format_data('test', num_works=4)
        # data_process.format_data('train_loss_mask', num_works=4)

    data_factory = CCF2020DataSetFactory(config.fitting)

    model_fitting = NERModelFitting(config, data_process.generate_results)

    # train
    best_weight_list = []
    for fold_index in range(0, config.fitting.fold_num if config.fitting.en_cross else 1):
        if en_train or en_eval or en_test:
            model = FastPrototype(config).cuda()
        else:
            model = None
        # train_data = data_factory({'type_data': 'train_loss_mask', 'fold_index': fold_index})
        train_data = data_factory({'type_data': 'train', 'fold_index': fold_index})
        dev_data = data_factory({'type_data': 'dev', 'fold_index': fold_index})
        test_data = data_factory({'type_data': 'test', 'fold_index': fold_index})

        if en_train:
            inputs = {'model': model,
                      'train_data': train_data,
                      'dev_data': dev_data,
                      'test_data': test_data,
                      'dev_res_file': ROOT_RESULT + 'dev_result.json',
                      'test_res_file': ROOT_RESULT + 'test_result.json',
                      'epoch_start': 0}
            _ = model_fitting.train(inputs)
            if config.fitting.en_cross:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index))
            else:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model.ckpt')

        if en_eval:
            inputs = {'model': model, 'data': dev_data, 'type_data': 'dev'}
            if config.fitting.en_cross:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'dev_result_{}.json'.format(fold_index)
            else:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'dev_result.json'
            print(model_fitting.eval(inputs))

        if config.fitting.en_cross and fold_index == config.fitting.fold_num - 1 or False:
            ccf2020.combine_dev_result(ROOT_RESULT, config.fitting.fold_num,
                                       ROOT_RESULT + 'dev_result_all.json')
            ccf2020.analyze_dev_data(ROOT_RESULT + 'dev_result_all.json',
                                     ccf2020.ROOT_LOCAL_DATA + 'seg_train_data.json',
                                     verbose=True)

        if en_test:
            inputs = {'model': model, 'data': test_data, 'type_data': 'test'}
            if config.fitting.en_cross:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'test_result_{}.json'.format(fold_index)
            else:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'test_result.json'
            model_fitting.test(inputs)

    if en_conv:
        if config.fitting.en_cross:
            data_process.combine_by_vote([ROOT_RESULT + 'test_result_{}.json'.format(i) for i in range(5)],
                                         ROOT_RESULT + 'test_result_vote.json')
            ccf2020.mix_clue_result(ROOT_RESULT + 'test_result_vote.json',
                                    ROOT_DATA + 'clue/train_dev_test_data.json',
                                    ROOT_RESULT + 'test_result_vote_mix_clue.json')
            ccf2020.convert2csv(ROOT_RESULT + 'test_result_vote_mix_clue.json',
                                ROOT_RESULT + 'predict_vote.csv')
        else:
            ccf2020.mix_clue_result(ROOT_RESULT + 'test_result.json',
                                    ROOT_DATA + 'clue/train_dev_test_data.json',
                                    ROOT_RESULT + 'test_result_mix_clue.json')
            ccf2020.convert2csv(ROOT_RESULT + 'test_result_mix_clue.json',
                                ROOT_RESULT + 'predict.csv')
