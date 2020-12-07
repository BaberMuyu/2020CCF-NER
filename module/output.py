import torch.nn as nn
from pytorchcrf import CRF
from abc import abstractmethod, ABCMeta
from enum import Enum
from global_config import *

LabelE = Enum('LabelE', ('Nobie', 'Tobie', 'Cobie', 'point'))


class OutputConfig(object):
    def __init__(self):
        self.num_types = None
        self.in_feat_size = None

    def __str__(self):
        string = ""
        string += "num_types: {}\n".format(self.num_types)
        return string


class OutputConfigObie(OutputConfig):
    def __init__(self):
        super(OutputConfigObie, self).__init__()
        self.label = LabelE.Nobie
        self.only_crf = False
        self.crf_strict = True

    @property
    def num_tags(self):
        if self.label == LabelE.Nobie:
            return 4
        elif self.label == LabelE.Tobie:
            return 1 + 3 * self.num_types
        elif self.label == LabelE.Cobie:
            return 4

    def __str__(self):
        string = super(OutputConfigObie, self).__str__()
        string += "label: {}\n".format(self.label)
        string += "crf_strict: {}\n".format(self.crf_strict)
        return string


class OutputConfigPoint(OutputConfig):
    def __init__(self):
        super(OutputConfigPoint, self).__init__()
        self.label = LabelE.point
        self.only_crf = False

    def __str__(self):
        string = super(OutputConfigPoint, self).__str__()
        string += "label: {}\n".format(self.label)
        return string


class Output(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, emission, mask):
        pass

    @abstractmethod
    def cal_emission(self, text_vec):
        pass

    @abstractmethod
    def cal_loss(self, preds, targets, mask):
        pass

    @abstractmethod
    def find_entity(self, pred, text=None):
        pass

    def get_params(self):
        return self.params

    def forward(self, inputs, en_pred=True):
        text_vec, mask = inputs['text_vec'], inputs['mask']
        emission = self.cal_emission(text_vec)
        if en_pred:
            pred = self.decode(emission, mask)
        else:
            pred = None
        return {'emission': emission,
                'pred': pred}


class OutputOBIE(Output):
    def __init__(self, config):
        super(OutputOBIE, self).__init__()
        self.config = config
        self.params = {'other': [], 'crf': []}

        if not config.only_crf:
            if self.config.label == LabelE.Nobie:
                self.emission_linear = nn.Linear(config.in_feat_size, config.num_types * config.num_tags)
            elif self.config.label == LabelE.Tobie:
                self.emission_linear = nn.Linear(config.in_feat_size, config.num_tags)
            elif self.config.label == LabelE.Cobie:
                self.emission_linear = nn.Linear(config.in_feat_size, config.num_tags)
            self.params['other'].extend([p for p in self.emission_linear.parameters()])
            self.crf = CRF(self.config.num_tags, batch_first=True)
            self.params['crf'].extend([p for p in self.crf.parameters()])
        else:
            self.crf = CRF(self.config.num_tags, batch_first=True)
            self.params['crf'].extend([p for p in self.crf.parameters()])

    def decode(self, emission, mask):
        """
        emission: B T L F
        """
        emission_shape = emission.size()
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, emission_shape[1], 1)
        mask = mask.reshape([-1, mask.size(2)])
        emission = emission.reshape([-1, emission_shape[2], emission.size(3)])
        result = self.crf.decode(emission, mask)
        result = result.reshape([-1, emission_shape[1], mask.size(1)])
        result = result.tolist()
        return result

    def cal_emission(self, text_vec):
        if self.config.only_crf:
            return text_vec
        emission = self.emission_linear(text_vec)
        if self.config.label in (LabelE.Tobie, LabelE.Cobie):
            emission = emission.reshape(list(emission.size()[:2]) + [1, self.config.num_tags])
        elif self.config.label == LabelE.Nobie:
            emission = emission.reshape(list(emission.size()[:2]) + [self.config.num_types, self.config.num_tags])
        emission = emission.permute([0, 2, 1, 3])  # B L T F -> B T L F
        return emission

    def cal_loss(self, preds, targets, mask):
        if 'loss_mask' in targets:
            mask = targets['loss_mask']
        emission = preds['emission']
        y_true = targets['y_true']
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, emission.size(1), 1)
        mask = mask.reshape([-1, mask.size(2)])
        emission = emission.reshape([-1, emission.size(2), emission.size(3)])  # B*T L F
        y_true = y_true.reshape([-1, y_true.size(2)])
        _loss = -self.crf(emission, y_true, mask, reduction='token_mean')
        return _loss

    def find_entity(self, pred, text=None):
        def find_entitiy_nobie(ps):
            entities_point_line = []

            def add_entity(es, ee):
                entities_point_line.append([es, ee])

            e_s = -1
            e_i = -1
            for pos, token in enumerate(ps):
                if self.config.crf_strict:
                    if token == 0:  # O
                        if e_s != -1:
                            add_entity(e_s, pos - 1)
                            e_s = -1
                    if token == 1:  # B
                        if e_s != -1:
                            add_entity(e_s, pos - 1)
                        e_s = pos
                    if token == 3:  # E
                        if e_s != -1:
                            add_entity(e_s, pos)
                            e_s = -1
                else:
                    if token == 1:
                        if e_s != -1:
                            add_entity(e_s, e_i)
                        e_s = pos
                        e_i = pos
                    elif token == 2:
                        if e_s == -1:
                            e_s = pos
                            e_i = pos
                        else:
                            e_i = pos
                    elif token == 3:
                        if e_s != -1:
                            add_entity(e_s, pos)
                        else:
                            add_entity(pos, pos)
                        e_s = e_i = -1
                    elif token == 0:
                        if e_s != -1:
                            add_entity(e_s, e_i)
                            e_s = e_i = -1

            return entities_point_line

        def find_entitiy_tobie(ps):
            entities_point = []

            def add_entity(cate, es, ee):
                entities_point.append([cate, es, ee])

            s, e, c = -1, -1, -1
            for pos, token in enumerate(ps):
                if token == 0:
                    if s != -1:
                        add_entity(c, s, e)
                    s, e, c = -1, -1, -1
                else:
                    e_cate = int((token - 1) / 3)
                    e_token = (token - 1) % 3
                    if e_token == 0:  # B
                        if s != -1:
                            add_entity(c, s, e)
                            s, e, c = -1, -1, -1
                        else:
                            s, e, c = pos, pos, e_cate
                    elif e_token == 1:  # I
                        if s != -1:  # and c == e_cate:
                            e = pos
                        else:
                            s, e, c = -1, -1, -1
                    elif e_token == 2:
                        if s != -1 and c == e_cate:
                            e = pos
                            add_entity(c, s, e)
                        s, e, c = -1, -1, -1
            return entities_point

        entity = []
        for a_pred in pred:
            if self.config.label == LabelE.Tobie:
                entity_line = find_entitiy_tobie(a_pred[0])
            elif self.config.label == LabelE.Cobie:
                entity_line = find_entitiy_tobie(a_pred[0])
            elif self.config.label == LabelE.Nobie:
                entity_line = []
                for type_index in range(self.config.num_types):
                    entity_point_temp = find_entitiy_nobie(a_pred[type_index])
                    entity_line.extend([[type_index] + point for point in entity_point_temp])
            else:
                exit(1)
            entity.append(entity_line)
        return entity


class OutputPoint(Output):
    def __init__(self, config):
        super(OutputPoint, self).__init__()
        self.config = config
        self.params = {'other': []}
        self.emission_linear = nn.Linear(self.config.in_feat_size, self.config.num_types * 2 + 1)
        self.params['other'].extend([p for p in self.emission_linear.parameters()])

    def decode(self, emission, mask):
        emission = emission * mask.unsqueeze(dim=-1)
        emission = emission.permute([0, 2, 1])  # B L F -> B F L
        # result = emission > 0.5
        max_n, _ = emission.max(dim=1)
        result = emission >= max_n.unsqueeze(dim=1)
        result = result.cpu().numpy()
        return result

    def cal_emission(self, text_vec):
        emission = self.emission_linear(text_vec)
        # emission = nn.functional.sigmoid(emission)
        # emission = nn.functional.softmax(emission, dim=1)
        # emission = emission.permute([0, 2, 1])  # B L F -> B F L
        return emission

    def cal_loss(self, preds, targets, mask):
        emission = preds['emission']
        y_true = targets['y_true'].argmax(dim=1)
        # c = 1.5
        # loss
        # emission = torch.log(emission)
        # _loss = nn.functional.nll_loss(emission.reshape([-1, self.config.num_types*2+1]),
        #                                y_true.reshape([-1]),
        #                                reduction='none')
        _loss = nn.functional.cross_entropy(emission.reshape([-1, self.config.num_types * 2 + 1]),
                                            y_true.reshape([-1]),
                                            reduction='none')
        _loss = _loss.reshape(mask.size())
        _loss = (mask * _loss).sum() / mask.sum()
        return _loss

    def find_entity(self, pred, text=None):
        def find_entitiy_point(text_line, ps, pe, pi):
            entities_point_line = []
            seq_length = len(text_line)
            start_index = -999
            end_index = 999
            ps_map = np.zeros(seq_length, dtype=np.int)
            pe_map = np.zeros(seq_length, dtype=np.int)
            start_list = []
            end_list = []
            for index in range(seq_length):
                if ps[index]:
                    start_index = index
                    start_list.append(start_index)
                if pe[seq_length - index - 1]:
                    end_index = seq_length - index - 1
                    end_list.append(end_index)
                ps_map[index] = start_index
                pe_map[seq_length - index - 1] = end_index

            for start_index in start_list:
                end_index = pe_map[start_index]
                entity_point = [start_index, end_index]
                if entity_point[0] != -999 and entity_point[1] != 999:
                    if entity_point not in entities_point_line:
                        entities_point_line.append(entity_point)

            for end_index in end_list:
                start_index = ps_map[end_index]
                entity_point = [start_index, end_index]
                if entity_point[0] != -999 and entity_point[1] != 999:
                    if entity_point not in entities_point_line:
                        entities_point_line.append(entity_point)

            return entities_point_line

        entity = []
        for a_text, a_pred in zip(text, pred):
            a_pred_s = a_pred[:self.config.num_types]
            a_pred_e = a_pred[self.config.num_types:self.config.num_types * 2]

            entity_line = []
            for type_index in range(self.config.num_types):
                entity_point_temp = find_entitiy_point(a_text, a_pred_s[type_index], a_pred_e[type_index], a_pred[-1])
                entity_line.extend([[type_index] + point for point in entity_point_temp])

            entity.append(entity_line)
        return entity
