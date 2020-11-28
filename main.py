import torch
import data.ccf20202 as ccf2020
from data.ccf20202 import DataConfig, CCF2020DataSetFactory, CCF2020DataProcess
from data.clue import CLUEDataProcess

from module.encoder import EncoderConfig, WordE
from module.fusion import FusionConfig, FusionE
from module.output import OutputConfig, OutputConfigObie,LabelE
from module.fitting import FittingConfig, NERModelFitting, CombineFitting, NERCasModelFitting, TypeModelFitting
from model.fast_prototype import FastPrototype, CombineModel, CasPrototype
from model.pipline import TypePrototype

from util.tool import json_dump_by_line, json_load_by_line
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

        self.data.max_len = 140  # 250 -> 150 -> 200 ->120->150 ->140 and clue split max token length
        self.data.tokenizer = self.encoder.ptm_model
        self.data.num_types = 14

        self.fitting.test_mode = False
        self.fitting.reshuffle = False
        self.fitting.en_cross = True
        self.fitting.fold_num = 5
        self.fitting.dev_rate = 0.2
        self.fitting.en_fgm = True
        self.fitting.en_swa = True
        self.fitting.entity_mask_rate = 0
        self.fitting.epochs = 8  # 20
        self.fitting.end_epoch = 8
        self.fitting.batch_size = 16
        self.fitting.lr = {'ptm': 0.00003,
                           'other': 0.00003}
        if self.output.label != LabelE.point:
            self.fitting.lr['crf'] = 0.005
        self.fitting.verbose = False

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
    en_preproc = en_train = en_eval = en_test = en_combine = en_conv = is_cas = False

    data_process = CCF2020DataProcess(config.data)
    # data_process = CLUEDataProcess(config.data)
    if en_preproc or False:
        # data_process.format_data('train', num_works=2)
        # data_process.format_data('test', num_works=2)
        data_process.format_data('train_loss_mask', num_works=4)
        exit(0)

    data_factory = CCF2020DataSetFactory(config.fitting)
    # data_factory = CLUEDataSetFactory(config.fitting)

    model_fitting = NERModelFitting(config, data_process.generate_results)
    # model_fitting = NERCasModelFitting(config, data_process.generate_results)
    # model_fitting = TypeModelFitting(config, data_process.generate_results)
    # train
    crf_param = []
    best_weight_list = []
    for fold_index in range(config.fitting.fold_num if config.fitting.en_cross else 1):
        model = FastPrototype(config).cuda()
        # model = CasPrototype(config).cuda()
        # model = TypePrototype(config).cuda()
        # model = None
        train_data = data_factory({'type_data': 'train_loss_mask', 'fold_index': fold_index})
        dev_data = data_factory({'type_data': 'dev', 'fold_index': fold_index})
        test_data = data_factory({'type_data': 'test', 'fold_index': fold_index})

        if en_train or True:
            inputs = {'model': model,
                      'train_data': train_data,
                      'dev_data': dev_data,
                      'test_data': test_data,
                      'dev_res_file': ROOT_RESULT + 'dev_result.json',
                      'test_res_file': ROOT_RESULT + 'test_result.json',
                      'epoch_start': 0}
            # model.load_state_dict(torch.load(ROOT_WEIGHT + 'auto_save_0.847587.ckpt'))
            best_score_weight = model_fitting.train(inputs)
            best_weight_list.append(best_score_weight)
            if config.fitting.en_cross:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index))
            else:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model.ckpt')

        if en_eval or True:
            inputs = {'model': model, 'data': dev_data, 'type_data': 'dev'}
            if config.fitting.en_cross:
                # inputs['weight'] = ROOT_WEIGHT + best_weight_list[0]
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'dev_result_{}.json'.format(fold_index)
            else:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                # inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'dev_result.json'
            print(model_fitting.eval(inputs))

        if en_test or True:
            inputs = {'model': model, 'data': test_data, 'type_data': 'test'}
            if config.fitting.en_cross:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'test_result_{}.json'.format(fold_index)
                # model.load_state_dict(torch.load(ROOT_WEIGHT + 'save/swa_model_{}.ckpt'.format(fold_index)))
            else:
                # inputs['weight'] = ROOT_WEIGHT + best_weight_list[0]
                inputs['weight'] = ROOT_WEIGHT + 'new_obie_swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'test_result.json'
            model_fitting.test(inputs)

            #一定要放在test之后, 运行之后才加载模型
            with torch.no_grad():
                model.eval()
                if config.fitting.en_cross:
                    crf_param.append([p.tolist() for p in model.get_params()['crf']])

    if crf_param:
        json_dump_by_line(crf_param, ROOT_WEIGHT + 'crf_weights.json')

    if en_combine or True:
        en_combine = True
        is_cas = False
        config.output.only_crf = True
        data_process.combine_emission(
            [ROOT_RESULT + 'test_result_{}.json'.format(i) for i in range(config.fitting.fold_num)],
            ROOT_WEIGHT + 'crf_weights.json',
            ROOT_WEIGHT + 'combine_crf_weight.json')
        model = CombineModel(config, json_load_by_line(ROOT_WEIGHT + 'combine_crf_weight.json')).cuda()
        model_fitting = CombineFitting(config, data_process.generate_results)
        test_data = data_factory({'type_data': 'combine',
                                  'fold_index': 0})
        inputs = {'model': model,
                  'data': test_data,
                  'type_data': 'test',
                  }
        if not is_cas:
            inputs['outfile'] = ROOT_RESULT + 'combine_test_result.json'
            model_fitting.test(inputs)
        else:
            inputs['outfile'] = ROOT_RESULT + 'combine_type_emission.json'  # don't change this filename
            model_fitting.test(inputs, keep_token=True)  #

            # type
            config.output.only_crf = False
            test_data = data_factory({'type_data': 'combine_type',
                                      'fold_index': 0})
            model = CasPrototype(config).cuda()
            model_fitting = NERCasModelFitting(config, data_process.generate_results)
            for index, weight in enumerate(best_weight_list):
                inputs = {'model': model,
                          'data': test_data,
                          'type_data': 'test',
                          'weight': ROOT_WEIGHT + weight,
                          'outfile': ROOT_RESULT + 'combine_type_test_result_{}.json'.format(index)}
                model_fitting.pred_type(inputs)
            data_process.combine_type_emission(
                [ROOT_RESULT + 'combine_type_test_result_{}.json'.format(index) for index in range(len(best_weight_list))],
                ROOT_RESULT + 'cas_combine_test_result.json')

    if en_conv or True:
        # en_combine = True
        if not en_combine:
            ccf2020.mix_clue_result(ROOT_RESULT + 'test_result.json', ROOT_DATA + 'clue/train_dev_test_data.json',
                                    ROOT_RESULT + 'del_clue_test_result.json')
            ccf2020.clean_result_2(ROOT_RESULT + 'del_clue_test_result.json',
                                   ROOT_RESULT + 'name_del_clue_test_result.json')
            ccf2020.convert2csv(ROOT_RESULT + 'name_del_clue_test_result.json',
                                ROOT_RESULT + 'name_del_clue_predict.csv')
        else:
            if is_cas:
                ccf2020.convert2csv(ROOT_RESULT + 'cas_combine_test_result.json',
                                    ROOT_RESULT + 'cas_combine_predict.csv')
            else:
                ccf2020.mix_clue_result(ROOT_RESULT + 'combine_test_result.json', ROOT_DATA + 'clue/train_dev_test_data.json',
                                        ROOT_RESULT + 'clue_combine_test_result.json')
                ccf2020.clean_result_2(ROOT_RESULT + 'clue_combine_test_result.json',
                                       ROOT_RESULT + 'clean_clue_combine_test_result.json')
                ccf2020.convert2csv(ROOT_RESULT + 'clean_clue_combine_test_result.json',
                                    ROOT_RESULT + 'clue_combine_predict3.csv')
