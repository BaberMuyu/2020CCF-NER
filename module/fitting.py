import torch
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

import shutil
import numpy as np
import pickle
import random
import copy

from util.callback import EarlyStopping, LearningSchedual, FGM
from util.log import ModelAux, MovingData
from util.tool import calculate_f1
from module.encoder import WordE
from module.output import LabelE
from global_config import ROOT_WEIGHT, ROOT_PROJECT


def random_mask(sample, mask_rate):
    if len(sample['entities']) > 0:
        index = random.randrange(len(sample['entities']))
        rate = random.random()
        new_sample = copy.deepcopy(sample)
        if rate < mask_rate and len(sample['entities']) >= 1:
            for pos in range(sample['entities'][index]['pos_b'], sample['entities'][index]['pos_e'] + 1):
                new_sample['text'][pos] = 103
        rate = random.random()
        if rate < mask_rate:
            rate = random.random()
            pos_e = sample['entities'][index]['pos_e']
            pos_b = sample['entities'][index]['pos_b']
            entity_len = pos_e - pos_b + 1
            start = int(rate * len(sample['text']))
            if start > pos_e + 1 or start + entity_len < pos_b:
                end = start + entity_len
                end = len(sample['text']) if end > len(sample['text']) else end
                for pos in range(start, end):
                    new_sample['text'][pos] = 103
        return new_sample
    return sample


class FittingConfig(object):
    def __init__(self):
        self.epochs = 8
        self.test_mode = False
        self.reshuffle = False
        self.en_cross = False
        self.fold_num = 5
        self.dev_rate = 0.2
        self.en_fgm = False
        self.en_swa = False
        self.entity_mask_rate = 0.0
        self._entity_mask_rate = 0.0
        self.epochs = 8
        self.end_epoch = 8
        self.batch_size = 16
        self.lr = {'ptm': 0.00003,
                   'other': 0.00003,
                   'crf': 0.009}
        self.verbose = False

    def __str__(self):
        string = ""
        string += "reshuffle: {}\n".format(self.reshuffle)
        string += "en_cross: {}\n".format(self.en_cross)
        string += "fold_num: {}\n".format(self.fold_num)
        string += "dev_rate: {}\n".format(self.dev_rate)
        string += "en_fgm: {}\n".format(self.en_fgm)
        string += "en_swa: {}\n".format(self.en_swa)
        string += "entity_mask_rate: {}\n".format(self.entity_mask_rate)
        string += "epochs: {}\n".format(self.epochs)
        string += "end_epoch: {}\n".format(self.end_epoch)
        string += "batch_size: {}\n".format(self.batch_size)
        string += "lr: {}\n".format(self.lr)
        return string


class NERModelFitting(object):
    def __init__(self, config, gen_res_fn):
        self.config = config
        self.gen_res = gen_res_fn
        self.w2v_array = None

    def collate_fn_test(self, batch):
        if self.config.encoder.worde == WordE.w2v and self.w2v_array is None:
            self.w2v_array = pickle.load(open(ROOT_PROJECT + '../data/w2v/w2v_vector.pkl', 'rb'))

        text = []
        text_mask = []

        iid = []
        sub_id = []
        raw_text = []

        result = [dict(), dict()]

        max_len = 0
        min_len = 999
        max_char_word_len = 0
        for sample in batch:
            max_len = max_len if max_len > len(sample['text']) else len(sample['text'])
            min_len = min_len if min_len < len(sample['text']) else len(sample['text'])
            char_word_len = len(sample['text']) + len(sample['lattice'])
            if self.config.encoder.worde in (WordE.w2v, WordE.cat, WordE.attn):
                char_word_len += len(sample['lattice']) + 1
            if max_char_word_len < char_word_len:
                max_char_word_len = char_word_len

            iid.append(sample['id'])
            sub_id.append(sample['sub_id'])
            raw_text.append(sample['text'])
        result[1] = {'id': iid, 'sub_id': sub_id, 'raw_text': raw_text}

        for sample in batch:
            text_length = len(sample['text'])
            text.append(sample['text'] + [0] * (max_len - text_length))
            text_mask.append([1] * text_length + [0] * (max_len - text_length))
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask).float()
        result[0] = {'text': [text, True], 'mask': [text_mask, True]}

        if self.config.encoder.worde in (WordE.w2v, WordE.cat, WordE.attn):
            word = []
            word_mask = []
            word_pos_b = []
            word_pos_e = []
            word_indice = []
            indice_base = 0
            for sample in batch:
                text_length = len(sample['text'])
                word_ = [0] * (text_length - min_len)
                word_mask_ = [0] * (text_length - min_len) + [1] * len(sample['lattice'])
                word_pos_b_ = [0] * (text_length - min_len)
                word_pos_e_ = [0] * (text_length - min_len)
                word_indice_ = [0] * (text_length - min_len) * 2
                for lattice in sample['lattice']:
                    word_.append(lattice[2])
                    word_pos_b_.append(lattice[0])
                    word_pos_e_.append(lattice[1])
                    word_indice_.extend([indice_base + lattice[0], indice_base + lattice[1]])
                tail_pad = [0] * (max_char_word_len - len(word_) - min_len)
                word_.extend(tail_pad)
                word_mask_.extend(tail_pad)
                word_pos_b_.extend(tail_pad)
                word_pos_e_.extend(tail_pad)
                word_indice_.extend(tail_pad)
                word_indice_.extend(tail_pad)
                indice_base += max_len

                if self.config.encoder.worde == WordE.w2v:
                    word.append([self.w2v_array[w] for w in word_])
                else:
                    word.append(word_)
                word_mask.append(word_mask_)
                word_pos_b.append(word_pos_b_)
                word_pos_e.append(word_pos_e_)
                word_indice.extend(word_indice_)
            word = torch.tensor(word)
            word_mask = torch.tensor(word_mask).float()
            word_pos_b = torch.tensor(word_pos_b).long()
            word_pos_e = torch.tensor(word_pos_e).long()
            word_indice = torch.tensor(word_indice).long()
            part_size = [min_len, max_len - min_len, max_char_word_len - max_len]
            result[0]['word_text'] = [word, True]
            result[0]['word_mask'] = [word_mask, True]
            result[0]['word_pos_b'] = [word_pos_b, True]
            result[0]['word_pos_e'] = [word_pos_e, True]
            result[0]['word_indice'] = [word_indice, True]
            result[0]['part_size'] = [part_size, False]
        return result

    def collate_fn_train(self, batch):
        def add_bieo(line, s, e, t=0):
            if self.config.output.label == LabelE.Cobie:
                pass
            else:
                if s == e:
                    if line[s] == 0:
                        line[s] = 1 + t * 3
                    return line
                else:
                    for bioe in line[s:e + 1]:
                        if bioe != 0:
                            return line
                    line[s] = 1 + t * 3
                    line[s + 1:e] = 2 + t * 3
                    line[e] = 3 + t * 3
                    return line

        inputs, others = self.collate_fn_test(batch)
        max_len = inputs['mask'][0].size(1)

        ner_label = []
        raw_entity = []

        for batch_index, sample in enumerate(batch):
            _ner_label = None
            if self.config.output.label == LabelE.Tobie:
                _ner_label = np.zeros([1, max_len])
            elif self.config.output.label == LabelE.Nobie:
                _ner_label = np.zeros([self.config.data.num_types, max_len])
            elif self.config.output.label == LabelE.Cobie:
                _ner_label = np.zeros([1, max_len])
            elif self.config.output.label == LabelE.point:
                _ner_label = np.zeros([self.config.data.num_types * 2 + 1, max_len])
            _raw_entity = []
            for entity in sample['entities']:
                _raw_entity.append([entity['category'], entity['pos_b'], entity['pos_e']])
                if self.config.output.label == LabelE.Tobie:
                    add_bieo(_ner_label[0], entity['pos_b'], entity['pos_e'], entity['category'])
                elif self.config.output.label == LabelE.Nobie:
                    add_bieo(_ner_label[entity['category']], entity['pos_b'], entity['pos_e'])
                elif self.config.output.label == LabelE.Cobie:
                    add_bieo(_ner_label[0], entity['pos_b'], entity['pos_e'], entity['category'])
                elif self.config.output.label == LabelE.point:
                    _ner_label[entity['category']][entity['pos_b']] = 1
                    _ner_label[entity['category'] + self.config.data.num_types][entity['pos_e']] = 1
                    _ner_label[-1][entity['pos_b'] + 1:entity['pos_e']] = 1
            ner_label.append(_ner_label)
            raw_entity.append(_raw_entity)

        if self.config.output.label in (LabelE.Nobie, LabelE.Tobie, LabelE.Cobie):
            ner_label = torch.tensor(ner_label).long()
        else:  # point
            ner_label = torch.tensor(ner_label).float()

        others.update({'raw_entity': raw_entity})
        return inputs, {'y_true': ner_label}, others

    def get_collate_fn(self, mode='train'):
        if mode == 'train' or mode == 'dev':
            return self.collate_fn_train
        elif mode == 'test':
            return self.collate_fn_test

    @classmethod
    def calculate_f1(self, ner_pred, ner_true):
        correct_num = pred_num = true_num = 0
        for batch_index in range(len(ner_pred)):
            for ner in ner_pred[batch_index]:
                # ner_span_true = [[a[1], a[2]] for a in ner_true[batch_index]]
                # if [ner[1], ner[2]] in ner_span_true:
                #     correct_num += 1
                if ner in ner_true[batch_index]:
                    correct_num += 1
            pred_num += len(ner_pred[batch_index])
            true_num += len(ner_true[batch_index])
        return correct_num, pred_num, true_num

    def train(self, train_inputs):
        config = self.config.fitting
        model = train_inputs['model']
        train_data = train_inputs['train_data']
        dev_data = train_inputs['dev_data']
        epoch_start = train_inputs['epoch_start']

        train_steps = int((len(train_data) + config.batch_size - 1) / config.batch_size)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      collate_fn=self.get_collate_fn('train'),
                                      shuffle=True)
        params_lr = []
        for key, value in model.get_params().items():
            if key in config.lr:
                params_lr.append({"params": value, 'lr': config.lr[key]})
        optimizer = torch.optim.Adam(params_lr)
        optimizer = SWA(optimizer)

        early_stopping = EarlyStopping(model, ROOT_WEIGHT, mode='max', patience=3)
        learning_schedual = LearningSchedual(optimizer, config.epochs, config.end_epoch, train_steps, config.lr)

        aux = ModelAux(self.config, train_steps)
        moving_log = MovingData(window=100)

        ending_flag = False
        detach_flag = False
        fgm = FGM(model)
        for epoch in range(epoch_start, config.epochs):
            for step, (inputs, targets, others) in enumerate(train_dataloader):
                inputs = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in inputs.items()])
                targets = dict([(key, value.cuda()) for key, value in targets.items()])
                if epoch > 0 and step == 0:
                    model.detach_ptm(False)
                    detach_flag = False
                if epoch == 0 and step == 0:
                    model.detach_ptm(True)
                    detach_flag = True
                # train ================================================================================================
                preds = model(inputs, en_decode=config.verbose)
                loss = model.cal_loss(preds, targets, inputs['mask'])
                loss.backward()

                # 对抗训练
                if (not detach_flag) and config.en_fgm:
                    fgm.attack(emb_name='word_embeddings')  # 在embedding上添加对抗扰动
                    preds_adv = model(inputs, en_decode=False)
                    loss_adv = model.cal_loss(preds_adv, targets, inputs['mask'])
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore(emb_name='word_embeddings')  # 恢复embedding参数

                # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logs = {}
                    if config.verbose and epoch > 0:
                        pred_entity_point = model.find_entity(preds['pred'], others['raw_text'])
                        cn, pn, tn = self.calculate_f1(pred_entity_point, others['raw_entity'])
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1,
                                        'correct_num': cn, 'pred_num': pn,
                                        'true_num': tn}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                        logs['precise'], logs['recall'], logs['f1'] = calculate_f1(moving_data['correct_num'],
                                                                                   moving_data['pred_num'],
                                                                                   moving_data['true_num'],
                                                                                   verbose=True)
                    else:
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                    # update lr
                    lr_data = learning_schedual.update_lr(epoch, step)
                    logs.update(lr_data)

                    if step + 1 == train_steps:
                        model.eval()
                        torch.save(model.state_dict(), ROOT_WEIGHT + 'temp_model.ckpt')
                        aux.new_line()
                        # dev ==========================================================================================
                        eval_inputs = {'model': model,
                                       'data': dev_data,
                                       'type_data': 'dev',
                                       'outfile': train_inputs['dev_res_file']}
                        dev_result = self.eval(eval_inputs)
                        logs['dev_loss'] = dev_result['loss']
                        logs['dev_precise'] = dev_result['precise']
                        logs['dev_recall'] = dev_result['recall']
                        logs['dev_f1'] = dev_result['f1']
                        # if logs['dev_f1'] > 0.88:
                        shutil.move(ROOT_WEIGHT + 'temp_model.ckpt',
                                    "{}/auto_save_{:.6f}.ckpt".format(ROOT_WEIGHT, logs['dev_f1']))
                        # if epoch > 10:
                        #     optimizer.update_swa()
                        early_stop, best_score = early_stopping(logs['dev_f1'])

                        # test =========================================================================================
                        if (epoch + 1 == config.epochs and step + 1 == train_steps) or early_stop:
                            ending_flag = True
                            model.load_state_dict(
                                torch.load("{}/auto_save_{:.6f}.ckpt".format(ROOT_WEIGHT, best_score)))
                            # optimizer.swap_swa_sgd()
                            # optimizer.bn_update(train_dataloader, model)

                        model.train()
                aux.show_log(epoch, step, logs)
                if ending_flag:
                    return

    def eval(self, eval_inputs):
        config = self.config.fitting
        model = eval_inputs['model']
        dev_data = eval_inputs['data']
        type_data = eval_inputs['type_data']
        outfile = eval_inputs['outfile']
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=config.batch_size,
                                    collate_fn=self.get_collate_fn('dev'))
        entity_result = []
        result = {}
        data_size = int((len(dev_data) + config.batch_size - 1) / config.batch_size)
        metrics_data = {"loss": 0, "correct_num": 0, "pred_num": 0, "true_num": 0, "sampled_num": 0}
        if 'weight' in eval_inputs:
            model.load_state_dict(torch.load(eval_inputs['weight']))

        with torch.no_grad():
            model.eval()
            batch_index = 1
            print('')
            for inputs, targets, others in dev_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                inputs = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in inputs.items()])
                targets = dict([(key, value.cuda()) for key, value in targets.items()])
                preds = model(inputs)
                loss = model.cal_loss(preds, targets, inputs['mask'])
                pred_entity_point = model.find_entity(preds['pred'], others['raw_text'])
                cn, pn, tn = self.calculate_f1(pred_entity_point, others['raw_entity'])
                metrics_data['correct_num'] += cn
                metrics_data['pred_num'] += pn
                metrics_data['true_num'] += tn
                metrics_data['loss'] += float(loss.cpu().numpy())
                metrics_data['sampled_num'] += 1

                for iid, sub_id, entities in zip(others['id'], others['sub_id'], pred_entity_point):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})

        result['loss'] = metrics_data['loss'] / metrics_data['sampled_num']
        result['precise'], result['recall'], result['f1'] = calculate_f1(
            metrics_data['correct_num'],
            metrics_data['pred_num'],
            metrics_data['true_num'], verbose=True)
        self.gen_res(entity_result, type_data, outfile)
        return result

    def test(self, test_inputs):
        config = self.config.fitting
        model = test_inputs['model']
        test_data = test_inputs['data']
        type_data = test_inputs['type_data']
        outfile = test_inputs['outfile']
        test_dataloader = DataLoader(test_data,
                                     batch_size=config.batch_size,
                                     collate_fn=self.get_collate_fn('test'))
        if 'weight' in test_inputs:
            model.load_state_dict(torch.load(test_inputs['weight']))
        # =================================
        with torch.no_grad():
            model.eval()
            entity_result = []
            data_size = int((len(test_data) + config.batch_size - 1) / config.batch_size)
            print('')
            batch_index = 0
            for inputs, others in test_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                inputs = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in inputs.items()])
                preds = model(inputs)
                pred_entity_point = model.find_entity(preds['pred'], others['raw_text'])

                for iid, sub_id, entities, emission in zip(others['id'], others['sub_id'], pred_entity_point,
                                                           preds['emission'].cpu().numpy().tolist()):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})
                    if config.en_cross:
                        entity_result[-1]['emission'] = emission
        self.gen_res(entity_result, type_data, outfile)
        return entity_result


class NERCasModelFitting(object):
    def __init__(self, config, gen_res_fn):
        self.config = config
        self.gen_res = gen_res_fn

    def collate_fn_test(self, batch):
        text = []
        text_mask = []

        iid = []
        sub_id = []
        raw_text = []

        result = [dict(), dict()]

        max_len = 0
        new_batch = []
        for sample in batch:
            max_len = max_len if max_len > len(sample['text']) else len(sample['text'])

            iid.append(sample['id'])
            sub_id.append(sample['sub_id'])
            raw_text.append(sample['text'])
            if self.config.fitting.entity_mask_rate > 0:
                new_batch.append(random_mask(sample, self.config.fitting._entity_mask_rate))
            else:
                new_batch.append(sample)
        batch = new_batch
        result[1] = {'id': iid, 'sub_id': sub_id, 'raw_text': raw_text}

        for sample in batch:
            text_length = len(sample['text'])
            text.append(sample['text'] + [0] * (max_len - text_length))
            text_mask.append([1] * text_length + [0] * (max_len - text_length))
        text = torch.tensor(text)
        text_mask = torch.tensor(text_mask).float()
        result[0] = {'entity': {'text': [text, True], 'mask': [text_mask, True]}}
        return result

    def collate_fn_train(self, batch):
        def add_bieo(line, s, e):
            if s == e:
                if line[s] == 0:
                    line[s] = 1
                return line
            else:
                for bioe in line[s:e + 1]:
                    if bioe != 0:
                        return line
                line[s] = 1
                line[s + 1:e] = 2
                line[e] = 3
                return line

        inputs, others = self.collate_fn_test(batch)
        max_len = inputs['entity']['mask'][0].size(1)

        ner_label = []
        entity_indice = []
        type_label = []
        raw_entity = []
        base = 0
        for batch_index, sample in enumerate(batch):
            if self.config.output.label == LabelE.Cobie:
                _ner_label = np.zeros([1, max_len])
            elif self.config.output.label == LabelE.Nobie:
                _ner_label = np.zeros([self.config.data.num_types, max_len])
            _raw_entity = []
            for entity in sample['entities']:
                _raw_entity.append([entity['category'], entity['pos_b'], entity['pos_e']])
                if self.config.output.label == LabelE.Cobie:
                    add_bieo(_ner_label[0], entity['pos_b'], entity['pos_e'])
                elif self.config.output.label == LabelE.Nobie:
                    add_bieo(_ner_label[entity['category']], entity['pos_b'], entity['pos_e'])
                type_label.append(entity['category'])
                entity_indice.extend([entity['pos_b'] + base, entity['pos_e'] + base])
            base += max_len
            ner_label.append(_ner_label)
            raw_entity.append(_raw_entity)

        ner_label = torch.tensor(ner_label).long()
        entity_indice = torch.tensor(entity_indice).long()
        type_label = torch.tensor(type_label).long()
        inputs['type'] = {'indice': [entity_indice, True]}
        targets = {'entity': {'y_true': ner_label},
                   'type': {'y_true': type_label}}
        others.update({'raw_entity': raw_entity})
        return inputs, targets, others

    def get_collate_fn(self, mode='train'):
        if mode == 'train' or mode == 'dev':
            return self.collate_fn_train
        elif mode == 'test':
            return self.collate_fn_test

    @classmethod
    def calculate_f1(self, entity_pred, type_pred, ner_true):
        e_cn = e_pn = e_tn = 0
        t_cn = t_n = 0
        a_cn = a_pn = a_tn = 0
        ner_type_true = []
        for batch_index in range(len(ner_true)):
            ner_type_true.extend([a[0] for a in ner_true[batch_index]])
            for ner in ner_true[batch_index]:
                ner_span_pred = [[a[1], a[2]] for a in entity_pred[batch_index]]
                if [ner[1], ner[2]] in ner_span_pred:
                    e_cn += 1
                if ner in entity_pred[batch_index]:
                    a_cn += 1
            e_pn += len(entity_pred[batch_index])
            e_tn += len(ner_true[batch_index])
        a_pn = e_pn
        a_tn = e_tn
        t_cn = (np.array(ner_type_true) == (np.array(type_pred))).sum()
        t_n = len(ner_type_true)
        return e_cn, e_pn, e_tn, t_cn, t_n, a_cn, a_pn, a_tn

    def train(self, train_inputs):
        config = self.config.fitting
        self.config.fitting._entity_mask_rate = self.config.fitting.entity_mask_rate
        model = train_inputs['model']
        train_data = train_inputs['train_data']
        dev_data = train_inputs['dev_data']
        epoch_start = train_inputs['epoch_start']

        train_steps = int((len(train_data) + config.batch_size - 1) / config.batch_size)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      collate_fn=self.get_collate_fn('train'),
                                      shuffle=True,)
                                      # num_workers=2)
        params_lr = []
        for key, value in model.get_params().items():
            if key in config.lr:
                params_lr.append({"params": value, 'lr': config.lr[key]})
        optimizer = torch.optim.Adam(params_lr)
        optimizer = SWA(optimizer)

        early_stopping = EarlyStopping(model, ROOT_WEIGHT, mode='max', patience=5)
        learning_schedual = LearningSchedual(optimizer, config.epochs, config.end_epoch, train_steps, config.lr)

        aux = ModelAux(self.config, train_steps)
        moving_log = MovingData(window=100)

        ending_flag = False
        detach_flag = False
        fgm = FGM(model)
        swa_start = False
        for epoch in range(epoch_start, config.epochs):
            for step, (inputs, targets, others) in enumerate(train_dataloader):
                cuda_inputs = dict()
                cuda_targets = dict()
                for key in inputs.keys():
                    cuda_inputs[key] = dict(
                        [(key0, value[0].cuda() if value[1] else value[0]) for key0, value in inputs[key].items()])
                    cuda_targets[key] = dict([(key0, value.cuda()) for key0, value in targets[key].items()])
                inputs = cuda_inputs
                targets = cuda_targets

                if epoch > 0 and step == 0:
                    model.detach_ptm(False)
                    detach_flag = False
                if epoch == 0 and step == 0:
                    model.detach_ptm(True)
                    detach_flag = True
                # train ================================================================================================
                preds = model(inputs, en_decode=config.verbose)
                loss = model.cal_loss(preds, targets, inputs['entity']['mask'])
                loss.backward()

                # 对抗训练
                if (not detach_flag) and config.en_fgm:
                    fgm.attack(emb_name='word_embeddings')  # 在embedding上添加对抗扰动
                    preds_adv = model(inputs, en_decode=False)
                    loss_adv = model.cal_loss(preds_adv, targets, inputs['entity']['mask'])
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore(emb_name='word_embeddings')  # 恢复embedding参数

                # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logs = {}
                    if config.verbose and epoch > 0:
                        pred_entity, pred_type = model.find_entity(preds)
                        e_cn, e_pn, e_tn, t_cn, t_n, a_cn, a_pn, a_tn = self.calculate_f1(pred_entity, pred_type,
                                                                                          others['raw_entity'])
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1,
                                        'e_cn': e_cn, 'e_pn': e_pn, 'e_tn': e_tn,
                                        't_cn': t_cn, 't_n': t_n,
                                        'a_cn': a_cn, 'a_pn': a_pn, 'a_tn': a_tn,
                                        }
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                        logs['e_p'], logs['e_r'], logs['e_f1'] = calculate_f1(moving_data['e_cn'],
                                                                              moving_data['e_pn'],
                                                                              moving_data['e_tn'],
                                                                              verbose=True)
                        logs['t_acc'] = moving_data['t_cn'] / moving_data['t_n']
                        logs['a_p'], logs['a_r'], logs['a_f1'] = calculate_f1(moving_data['a_cn'],
                                                                              moving_data['a_pn'],
                                                                              moving_data['a_tn'],
                                                                              verbose=True)
                    else:
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                    # update lr
                    lr_data = learning_schedual.update_lr(epoch, step)
                    logs.update(lr_data)

                    if step + 1 == train_steps:
                        model.eval()
                        torch.save(model.state_dict(), ROOT_WEIGHT + 'temp_model.ckpt')
                        aux.new_line()
                        # dev ==========================================================================================
                        eval_inputs = {'model': model,
                                       'data': dev_data,
                                       'type_data': 'dev',
                                       'outfile': train_inputs['dev_res_file']}
                        self.config.fitting._entity_mask_rate = 0
                        dev_result = self.eval(eval_inputs)
                        self.config.fitting._entity_mask_rate = self.config.fitting.entity_mask_rate
                        for key in dev_result.keys():
                            logs['dev_' + key] = dev_result[key]
                        if logs['dev_a_f1'] > 0.89 or True:
                            shutil.move(ROOT_WEIGHT + 'temp_model.ckpt',
                                        "{}/auto_save_{:.6f}.ckpt".format(ROOT_WEIGHT, logs['dev_a_f1']))
                        if (epoch > 14 or logs['dev_a_f1'] > 0.920 or swa_start) and config.en_swa:
                            swa_start = True
                            optimizer.update_swa()
                        early_stop, best_score = early_stopping(logs['dev_a_f1'])

                        # test =========================================================================================
                        if (epoch + 1 == config.epochs and step + 1 == train_steps) or early_stop:
                            ending_flag = True
                            if swa_start and config.en_swa:
                                optimizer.swap_swa_sgd()
                                optimizer.bn_update(train_dataloader, model)
                            # model.load_state_dict(
                            #     torch.load("{}/auto_save_{:.6f}.ckpt".format(ROOT_WEIGHT, best_score)))
                        model.train()
                aux.show_log(epoch, step, logs)
                if ending_flag:
                    return "auto_save_{:.6f}.ckpt".format(best_score)

    def eval(self, eval_inputs):
        config = self.config.fitting
        model = eval_inputs['model']
        dev_data = eval_inputs['data']
        type_data = eval_inputs['type_data']
        outfile = eval_inputs['outfile']
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=config.batch_size,
                                    collate_fn=self.get_collate_fn('dev'))
        entity_result = []
        result = {}
        data_size = int((len(dev_data) + config.batch_size - 1) / config.batch_size)
        metrics_data = {"loss": 0, "sampled_num": 0,
                        "e_cn": 0, "e_pn": 0, "e_tn": 0,
                        "t_cn": 0, "t_n": 0,
                        "a_cn": 0, "a_pn": 0, "a_tn": 0}

        if 'weight' in eval_inputs:
            model.load_state_dict(torch.load(eval_inputs['weight']))

        with torch.no_grad():
            model.eval()
            batch_index = 1
            print('')
            for inputs, targets, others in dev_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                cuda_inputs = dict()
                cuda_targets = dict()
                for key in inputs.keys():
                    cuda_inputs[key] = dict(
                        [(key0, value[0].cuda() if value[1] else value[0]) for key0, value in inputs[key].items()])
                    cuda_targets[key] = dict([(key0, value.cuda()) for key0, value in targets[key].items()])
                inputs = cuda_inputs
                targets = cuda_targets
                preds = model(inputs, mode='eval')
                loss = model.cal_loss(preds, targets, inputs['entity']['mask'])
                pred_entity, pred_type = model.find_entity(preds)
                e_cn, e_pn, e_tn, t_cn, t_n, a_cn, a_pn, a_tn = self.calculate_f1(pred_entity, pred_type,
                                                                                  others['raw_entity'])
                metrics_data['e_cn'] += e_cn
                metrics_data['e_pn'] += e_pn
                metrics_data['e_tn'] += e_tn
                metrics_data['t_cn'] += t_cn
                metrics_data['t_n'] += t_n
                metrics_data['a_cn'] += a_cn
                metrics_data['a_pn'] += a_pn
                metrics_data['a_tn'] += a_tn
                metrics_data['loss'] += float(loss.cpu().numpy())
                metrics_data['sampled_num'] += 1

                for iid, sub_id, entities in zip(others['id'], others['sub_id'], pred_entity):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})

        result['loss'] = metrics_data['loss'] / metrics_data['sampled_num']
        result['e_p'], result['e_r'], result['r_f1'] = calculate_f1(metrics_data['e_cn'],
                                                                    metrics_data['e_pn'],
                                                                    metrics_data['e_tn'], verbose=True)
        result['t_acc'] = metrics_data['t_cn'] / metrics_data['t_n']
        result['a_p'], result['a_r'], result['a_f1'] = calculate_f1(metrics_data['a_cn'],
                                                                    metrics_data['a_pn'],
                                                                    metrics_data['a_tn'], verbose=True)
        self.gen_res(entity_result, type_data, outfile)
        return result

    def test(self, test_inputs):
        config = self.config.fitting
        model = test_inputs['model']
        test_data = test_inputs['data']
        type_data = test_inputs['type_data']
        outfile = test_inputs['outfile']
        test_dataloader = DataLoader(test_data,
                                     batch_size=config.batch_size,
                                     collate_fn=self.get_collate_fn('test'))
        if 'weight' in test_inputs:
            model.load_state_dict(torch.load(test_inputs['weight']))
        # =================================
        with torch.no_grad():
            model.eval()
            entity_result = []
            data_size = int((len(test_data) + config.batch_size - 1) / config.batch_size)
            print('')
            batch_index = 0
            for inputs, others in test_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                cuda_inputs = dict()
                for key in inputs.keys():
                    cuda_inputs[key] = dict(
                        [(key0, value[0].cuda() if value[1] else value[0]) for key0, value in inputs[key].items()])
                inputs = cuda_inputs
                preds = model(inputs, mode='test')
                pred_entity_point, _ = model.find_entity(preds)

                for iid, sub_id, entities, entity_emission in zip(others['id'], others['sub_id'], pred_entity_point,
                                                                  preds['entity']['emission']):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})
                    if config.en_cross:
                        entity_result[-1]['emission'] = entity_emission

        self.gen_res(entity_result, type_data, outfile)
        return entity_result

    def pred_type(self, eval_inputs):
        config = self.config.fitting
        model = eval_inputs['model']
        dev_data = eval_inputs['data']
        type_data = eval_inputs['type_data']
        outfile = eval_inputs['outfile']
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=config.batch_size,
                                    collate_fn=self.get_collate_fn('dev'))
        entity_result = []
        result = {}
        data_size = int((len(dev_data) + config.batch_size - 1) / config.batch_size)

        if 'weight' in eval_inputs:
            model.load_state_dict(torch.load(eval_inputs['weight']))

        with torch.no_grad():
            model.eval()
            batch_index = 1
            print('')
            for inputs, targets, others in dev_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                cuda_inputs = dict()
                for key in inputs.keys():
                    cuda_inputs[key] = dict(
                        [(key0, value[0].cuda() if value[1] else value[0]) for key0, value in inputs[key].items()])
                inputs = cuda_inputs
                preds = model(inputs, mode='train')

                entity_count = 0
                for iid, sub_id, entities in zip(others['id'], others['sub_id'], others['raw_entity']):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})
                    for e in entities:
                        e[0] = preds['type_gt']['emission'][entity_count]
                        entity_count += 1

        self.gen_res(entity_result, type_data, outfile)
        return result


class CombineFitting(object):
    def __init__(self, config, gen_res_fn):
        self.config = config
        self.gen_res = gen_res_fn

    @staticmethod
    def collate_fn_test(batch):
        text = []
        text_emission = []
        text_mask = []

        iid = []
        sub_id = []
        raw_text = []

        result = [dict(), dict()]

        max_len = 0
        for sample in batch:
            max_len = max_len if max_len > len(sample['text']) else len(sample['text'])

            iid.append(sample['id'])
            sub_id.append(sample['sub_id'])
            raw_text.append(sample['text'])

        result[1] = {'id': iid, 'sub_id': sub_id, 'raw_text': raw_text}

        for sample in batch:
            text_length = len(sample['text'])
            text.append(sample['text'] + [0] * (max_len - text_length))
            text_emission.append(sample['emission'])
            text_mask.append([1] * text_length + [0] * (max_len - text_length))
        text = torch.tensor(text)
        text_emission = torch.tensor(text_emission)
        text_mask = torch.tensor(text_mask).float()
        result[0] = {'text': [text, True], 'emission': [text_emission, True], 'mask': [text_mask, True]}
        return result

    def test(self, test_inputs, keep_token=False):
        config = self.config.fitting
        model = test_inputs['model']
        test_data = test_inputs['data']
        type_data = test_inputs['type_data']
        outfile = test_inputs['outfile']
        test_dataloader = DataLoader(test_data,
                                     batch_size=config.batch_size,
                                     collate_fn=self.collate_fn_test)
        # =================================
        with torch.no_grad():
            model.eval()
            entity_result = []
            data_size = int((len(test_data) + config.batch_size - 1) / config.batch_size)
            print('')
            batch_index = 0
            for inputs, others in test_dataloader:
                print('\r {}/{}'.format(batch_index, data_size), end='')
                batch_index += 1
                inputs = dict([(key, value[0].cuda() if value[1] else value[0]) for key, value in inputs.items()])
                preds = model(inputs)
                pred_entity_point = model.find_entity(preds['pred'], others['raw_text'])

                for iid, sub_id, entities in zip(others['id'], others['sub_id'], pred_entity_point):
                    entity_result.append({'id': iid,
                                          'sub_id': sub_id,
                                          'entities': entities})
        self.gen_res(entity_result, type_data, outfile, keep_token=keep_token)
        return entity_result
